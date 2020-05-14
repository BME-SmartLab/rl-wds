# -* coding: utf-8 -*-
import os
import yaml
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from epynet import Network
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines import DQN
from wdsEnv import wds
from scipy.optimize import minimize as nm

import panel as pn
import param
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.models import ColumnDataSource as cds
from bokeh.models import ColorBar, LinearColorMapper, Button
pn.extension()

def assemble_plot_data(junc_prop):
    plot_data = cds(
        data = {
            'x':    wrapper.junc_coords['x'],
            'y':    wrapper.junc_coords['y'],
            'junc_prop':    junc_prop
            }
        )
    return plot_data

def build_plot_from_data(data, min_prop=None, max_prop=None):
    if not min_prop:
        min_prop = min(data.data['junc_prop'])
    if not max_prop:
        max_prop = max(data.data['junc_prop'])

    mapper = linear_cmap(
        field_name  = 'junc_prop',
        palette     = "Viridis10",
        low         = min_prop,
        high        = max_prop
        )
    cmapper = LinearColorMapper(
        palette = "Viridis10",
        low     = min_prop,
        high    = max_prop
        )

    fig         = figure()
    edges       = fig.line(wrapper.pipe_coords['x'], wrapper.pipe_coords['y'])
    nodes       = fig.circle(x='x', y='y', color=mapper, source=data, size=12)
    color_bar   = ColorBar(color_mapper=cmapper)
    fig.add_layout(color_bar, 'right')
    fig = pn.pane.Bokeh(fig, width=400, height=300)
    return fig

class environment_wrapper(param.Parameterized):
    sel_wds     = param.ObjectSelector(
        default = "Anytown",
        objects = ["Anytown", "D-Town"]
        )
    sel_dmd     = param.ObjectSelector(
        default = "Original demands",
        objects = ['Original demands', 'Randomized demands']
        )
    sel_spd     = param.ObjectSelector(
        default = "Original speeds",
        objects = ['Original speeds', 'Randomized speeds']
        )
    act_load    = param.Action(
        lambda x: x.param.trigger('act_load'), label='Load water distribution system')
    
    def __init__(self):
        self.loaded_wds = ''
        self.head_lmt_lo= 15
        self.head_lmt_hi= 120

    def _assemble_junc_coordinates(self, wds):
        junc_x = []
        junc_y = []
        junc_z = []
        for junc in wds.junctions:
            junc_x.append(junc.coordinates[0])
            junc_y.append(junc.coordinates[1])
            junc_z.append(junc.elevation)
        return {'x': junc_x, 'y': junc_y, 'z': junc_z}
    
    def _assemble_pipe_coords(self, wds):
        pipe_x = []
        pipe_y = []
        pipe_z = []
        for pipe in wds.pipes:
            if (pipe.from_node.index in list(wds.junctions.index)) and (pipe.to_node.index in list(wds.junctions.index)):
                pipe_x.append(pipe.from_node.coordinates[0])
                pipe_x.append(pipe.to_node.coordinates[0])
                pipe_x.append(float('nan'))

                pipe_y.append(pipe.from_node.coordinates[1])
                pipe_y.append(pipe.to_node.coordinates[1])
                pipe_y.append(float('nan'))

                pipe_z.append(pipe.from_node.elevation)
                pipe_z.append(pipe.to_node.elevation)
                pipe_z.append(float('nan'))
        return {'x': pipe_x, 'y': pipe_y, 'z': pipe_z}

    def load_env(self, wds_name, resetOrigDemands, resetOrigPumpSpeeds):
        if wds_name != self.loaded_wds:
            if wds_name == 'Anytown':
                hyperparams_fn  = 'anytownMaster'
                model_fn        = 'anytownHO1-best'
                self.dmd_lo     = 0
                self.dmd_hi     = 250
            elif wds_name == 'D-Town':
                hyperparams_fn  = 'dtownMaster'
                model_fn        = 'dtownHO1-best'
                self.dmd_lo     = 0
                self.dmd_hi     = 10

            pathToParams = os.path.join(
                'experiments',
                'hyperparameters',
                hyperparams_fn+'.yaml'
                )
            with open(pathToParams, 'r') as fin:
                self.hparams = yaml.load(fin, Loader=yaml.Loader)
            self.pathToModel = os.path.join('experiments', 'models', model_fn+'.zip')
            self.model = DQN.load(wrapper.pathToModel)
            
            self.loaded_wds = wds_name

        self.env = wds(
            wds_name        = self.hparams['env']['waterNet']+'_master',
            speed_increment = self.hparams['env']['speedIncrement'],
            episode_len     = self.hparams['env']['episodeLen'],
            pump_groups     = self.hparams['env']['pumpGroups'],
            total_demand_lo = self.hparams['env']['totalDemandLo'],
            total_demand_hi = self.hparams['env']['totalDemandHi'],
            reset_orig_pump_speeds  = resetOrigPumpSpeeds,
            reset_orig_demands      = resetOrigDemands
            )
        self.junc_coords = self._assemble_junc_coordinates(self.env.wds)
        self.pipe_coords = self._assemble_pipe_coords(self.env.wds)

    @param.depends('act_load')
    def load_wds(self):
        self.load_env(
            self.sel_wds,
            self.sel_dmd == 'Original demands',
            self.sel_spd == 'Original speeds'
        )
        self.env.reset(training=True)

        plot_data   = assemble_plot_data(wrapper.env.wds.junctions.head)
        self.plot   = build_plot_from_data(plot_data, self.dmd_lo, self.dmd_hi)
        return self.plot

class optimize_speeds(param.Parameterized):
    act_opti    = param.Action(
        lambda x: x.param.trigger('act_opti'),
        label='Optimize pump speeds'
        )
    play_button = param.Action(
        lambda x: x.param.trigger('play_button'),
        label   = 'Play'
        )

    def __init__(self):
        self.rew_dqn    = 0
        self.rew_nm     = 0
        self.hist_dqn   = []
        self.hist_nm    = []
        self.hist_idx_nm    = 0
        self.hist_idx_dqn   = 0
        self.nm_plot_id     = 0
        self.dqn_plot_id     = 0
        self.playing = False

        self.store_bc()
        self.call_nm()
        self.rew_nm = wrapper.env.get_state_value()
        self.nm_dta = assemble_plot_data(wrapper.env.wds.junctions.head)
        self.nm_plot= build_plot_from_data(self.nm_dta)
        self.restore_bc()

    def call_dqn(self):
        wrapper.env.wds.solve()
        wrapper.env.steps   = 0
        wrapper.env.done    = False
        obs             = wrapper.env.get_observation()
        self.hist_dqn   = [wrapper.env.wds.junctions.head]
        while not wrapper.env.done:
            act, _              = wrapper.model.predict(obs, deterministic=True)
            obs, reward, _, _   = wrapper.env.step(act, training=False)
            self.hist_dqn.append(wrapper.env.wds.junctions.head)

    def callback_nm(self, fun):
        self.hist_nm.append(wrapper.env.wds.junctions.head)

    def call_nm(self):
        init_guess  = wrapper.env.dimensions * [1.]
        options     = {
            'maxfev': 100,
            'xatol' : .005,
            'fatol' : .01}
        wrapper.env.wds.solve()
        self.hist_nm    = [wrapper.env.wds.junctions.head]
        result  = nm(
            wrapper.env.reward_to_scipy,
            init_guess,
            method  = 'Nelder-Mead',
            options = options,
            callback= self.callback_nm
            )
        self.nm_evals   = result.nit

    def store_bc(self):
        self.orig_demands   = wrapper.env.wds.junctions.basedemand
        self.orig_speeds    = wrapper.env.wds.pumps.speed

    def restore_bc(self):
        wrapper.env.wds.junctions.basedemand    = self.orig_demands
        wrapper.env.wds.pumps.speed             = self.orig_speeds

    @param.depends('act_opti')
    def plot_dqn(self):
        self.store_bc()
        self.call_dqn()
        self.rew_dqn    = wrapper.env.get_state_value()
        plot_data       = assemble_plot_data(wrapper.env.wds.junctions.head)
        plot            = build_plot_from_data(plot_data)
        self.restore_bc()
        return plot

    @param.depends('act_opti')
    def plot_nm(self):
        self.store_bc()
        self.call_nm()
        self.rew_nm = wrapper.env.get_state_value()
        self.nm_dta = assemble_plot_data(wrapper.env.wds.junctions.head)
        plot        = build_plot_from_data(self.nm_dta)
        self.restore_bc()
        return plot





    def animate_nm_plot(self):
        if self.hist_idx_nm == 0:
            self.store_bc()
            self.call_nm()
            self.rew_nm = wrapper.env.get_state_value()
            self.nm_dta = assemble_plot_data(wrapper.env.wds.junctions.head)
            self.nm_plot= build_plot_from_data(self.nm_dta)
            self.restore_bc()

        self.nm_dta.data = {
            'x': wrapper.junc_coords['x'],
            'y': wrapper.junc_coords['y'],
            'junc_prop': optimizer.hist_nm[self.hist_idx_nm]
        }
        self.hist_idx_nm    += 1
        if self.hist_idx_nm == len(optimizer.hist_nm):
            self.hist_idx_nm    = 0
            curdoc().remove_periodic_callback(self.nm_plot_id)
            self.playing    = False
#        return self.nm_plot

    @param.depends('play_button')
    def play_animation_nm(self):
#        if self.play_button.label == 'Play':
        if not self.playing:
            self.playing = True
#            self.play_button.label = 'Pause'
            self.nm_plot_id = curdoc().add_periodic_callback(self.animate_nm_plot, 500)
        else:
            #self.play_button.label = 'Play'
            self.playing = False
            curdoc().remove_periodic_callback(self.nm_plot_id)
        return self.nm_plot






    @param.depends('act_opti')
    def read_dqn_rew(self):
        return self.rew_dqn

    @param.depends('act_opti')
    def read_nm_rew(self):
        return self.rew_nm

    @param.depends('act_opti')
    def read_dqn_evals(self):
        return wrapper.env.steps

    @param.depends('act_opti')
    def read_nm_evals(self):
        return self.nm_evals

wrapper = environment_wrapper()
pn.Column(
    '# Loading the water distribution system',
    pn.Row(
        pn.Column(
            pn.panel(
                wrapper.param,
                show_labels = False,
                show_name   = False,
                margin      = 0,
                widgets = {
                    'sel_dmd': pn.widgets.RadioButtonGroup,
                    'sel_spd': pn.widgets.RadioButtonGroup
                    }
                ),
            ),
        wrapper.load_wds
        )
).servable()

optimizer = optimize_speeds()
pn.Column(
    '# Optimizing pump speeds',
    pn.panel(optimizer.param, show_labels=False, show_name=False, margin=0),
    pn.Row(
        optimizer.plot_dqn,
        optimizer.plot_nm,
        ),
    pn.Row(
        pn.WidgetBox(optimizer.read_dqn_rew, width=200),
        pn.WidgetBox(optimizer.read_dqn_evals, width=200),
        pn.WidgetBox(optimizer.read_nm_rew, width=200),
        pn.WidgetBox(optimizer.read_nm_evals, width=200),
        ),
    pn.Row(
        #optimizer.animate_nm_plot
        #optimizer.play_animation_nm
        )
).servable()

#data = cds(data={
#    'x': wrapper.junc_coords['x'],
#    'y': wrapper.junc_coords['y'],
#    'head': optimizer.hist_nm[0]
#    }
#)
#mapper = linear_cmap(
#    field_name = 'head',
#    palette = "Viridis10",
#    low = 40,
#    high = 80
#)
#fig = figure()
#edges = fig.line(wrapper.pipe_coords['x'], wrapper.pipe_coords['y'])
#nodes = fig.circle(x='x', y='y', color=mapper, source=data, size=12)
#

mlp = 0
call_id = 0
def animate_plot():
    global mlp, data, call_id
    optimizer.nm_dta.data = {
        'x': wrapper.junc_coords['x'],
        'y': wrapper.junc_coords['y'],
        'junc_prop': optimizer.hist_nm[mlp]
    }
    mlp += 1
    if mlp == len(optimizer.hist_nm):
        mlp = 0
        curdoc().remove_periodic_callback(call_id)
        button.label = 'Play'

def play_animation():
    global call_id
    if button.label == 'Play':
        button.label = 'Pause'
        call_id = curdoc().add_periodic_callback(animate_plot, 500)
    else:
        button.label = 'Play'
        curdoc().remove_periodic_callback(call_id)

button = Button(label='Play', width=60)
button.on_click(play_animation)
pn.Row(button).servable()
