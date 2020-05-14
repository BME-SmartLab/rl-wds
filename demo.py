# -* coding: utf-8 -*-
import os
import yaml
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
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

    def __init__(self):
        self.rew_dqn    = 0
        self.rew_nm     = 0
        self.hist_dqn   = []
        self.hist_nm    = []
        self.hist_val_nm= []
        self_hist_fail_counter_nm   = []

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
        self.hist_val_nm.append(wrapper.env.get_state_value())
        invalid_heads_count = (np.count_nonzero(wrapper.env.wds.junctions.head < wrapper.head_lmt_lo) +
            np.count_nonzero(wrapper.env.wds.junctions.head > wrapper.head_lmt_hi))
        self.hist_fail_counter_nm.append(invalid_heads_count)

    def call_nm(self):
        init_guess  = wrapper.env.dimensions * [1.]
        options     = {
            'maxfev': 100,
            'xatol' : .005,
            'fatol' : .01}
        wrapper.env.wds.solve()
        self.hist_nm    = [wrapper.env.wds.junctions.head]
        self.hist_val_nm= [wrapper.env.get_state_value()]
        invalid_heads_count = (np.count_nonzero(wrapper.env.wds.junctions.head < wrapper.head_lmt_lo) +
            np.count_nonzero(wrapper.env.wds.junctions.head > wrapper.head_lmt_hi))
        self.hist_fail_counter_nm = [invalid_heads_count]
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
        )
    ).servable()

hist_idx_nm = 0
call_id_nm  = 0
nm_idx_widget   = pn.widgets.TextInput(value='Step: ', width=400)
nm_val_widget   = pn.widgets.TextInput(value='Value: ', width=400)
nm_fail_widget  = pn.widgets.TextInput(value='Invalid heads: ', width=400)
def animate_nm_plot():
    global hist_idx_nm, call_id_nm
    global nm_idx_widget, nm_val_widget
    optimizer.nm_dta.data = {
        'x': wrapper.junc_coords['x'],
        'y': wrapper.junc_coords['y'],
        'junc_prop': optimizer.hist_nm[hist_idx_nm]
    }
    nm_idx_widget.value = 'Step: ' + str(hist_idx_nm+1)
    nm_val_widget.value = 'Value: ' + str(optimizer.hist_val_nm[hist_idx_nm])
    nm_fail_widget.value = 'INvalid heads: ' + str(optimizer.hist_fail_counter_nm[hist_idx_nm])
    hist_idx_nm += 1
    if hist_idx_nm == len(optimizer.hist_nm):
        hist_idx_nm = 0
        curdoc().remove_periodic_callback(call_id_nm)
        button_nm.label = 'Play optimization sess'

def play_animation_nm():
    global call_id_nm
    if button_nm.label == 'Play optimization sess':
        button_nm.label = 'Pause'
        call_id_nm      = curdoc().add_periodic_callback(animate_nm_plot, 500)
    else:
        button_nm.label = 'Play optimization sess'
        curdoc().remove_periodic_callback(call_id_nm)


button_nm   = Button(label='Play optimization sess', width=400)
button_nm.on_click(play_animation_nm)
pn.Row(
    pn.Column(
        button_nm,
        pn.Row(
            pn.Column(
                nm_idx_widget,
                nm_val_widget,
                nm_fail_widget
                )
            )
        ),
).servable()

