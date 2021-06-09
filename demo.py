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

def build_plot_from_data(data, min_prop=None, max_prop=None, title=None, figtitle=None):
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

    fig         = figure(title=figtitle)
    edges       = fig.line(wrapper.pipe_coords['x'], wrapper.pipe_coords['y'])
    nodes       = fig.circle(x='x', y='y', color=mapper, source=data, size=12)
    color_bar   = ColorBar(color_mapper=cmapper, title=title)
    fig.add_layout(color_bar, 'right')

    fig.toolbar.logo        = None
    fig.toolbar_location    = None
    fig.xaxis.major_tick_line_color = None
    fig.xaxis.minor_tick_line_color = None
    fig.yaxis.major_tick_line_color = None
    fig.yaxis.minor_tick_line_color = None
    fig.xaxis.major_label_text_font_size = '0pt'
    fig.yaxis.major_label_text_font_size = '0pt'

    fig = pn.pane.Bokeh(fig, width=600, height=500)
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
        lambda x: x.param.trigger('act_load'),
        label   = 'Load water distribution system'
        )

    def __init__(self):
        self.loaded_wds = ''
        self.head_lmt_lo= 15
        self.head_lmt_hi= 150

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
        for pump in wds.pumps:
            if (pipe.from_node.index in list(wds.junctions.index)) and (pipe.to_node.index in list(wds.junctions.index)):
                pipe_x.append(pump.from_node.coordinates[0])
                pipe_x.append(pump.to_node.coordinates[0])
                pipe_x.append(float('nan'))

                pipe_y.append(pump.from_node.coordinates[1])
                pipe_y.append(pump.to_node.coordinates[1])
                pipe_y.append(float('nan'))

                pipe_z.append(pump.from_node.elevation)
                pipe_z.append(pump.to_node.elevation)
                pipe_z.append(float('nan'))
        for valve in wds.valves:
            if (pipe.from_node.index in list(wds.junctions.index)) and (pipe.to_node.index in list(wds.junctions.index)):
                pipe_x.append(valve.from_node.coordinates[0])
                pipe_x.append(valve.to_node.coordinates[0])
                pipe_x.append(float('nan'))

                pipe_y.append(valve.from_node.coordinates[1])
                pipe_y.append(valve.to_node.coordinates[1])
                pipe_y.append(float('nan'))

                pipe_z.append(valve.from_node.elevation)
                pipe_z.append(valve.to_node.elevation)
                pipe_z.append(float('nan'))
        return {'x': pipe_x, 'y': pipe_y, 'z': pipe_z}

    def load_env(self, wds_name, resetOrigDemands, resetOrigPumpSpeeds):
        if wds_name != self.loaded_wds:
            if wds_name == 'Anytown':
                hyperparams_fn  = 'anytownMaster'
                model_fn        = 'anytownHO1-best'
                self.dmd_lo     = 30
                self.dmd_hi     = 80
            elif wds_name == 'D-Town':
                hyperparams_fn  = 'dtownMaster'
                model_fn        = 'dtownHO1-best'
                self.dmd_lo     = 50
                self.dmd_hi     = 150

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
        global response_load, res_box_opti
        global num_of_juncs, num_of_pipes, num_of_pumps
        response_load.value     = 'Loading, please wait.'
        res_box_load.background = '#FF0000'
        self.load_env(
            self.sel_wds,
            self.sel_dmd == 'Original demands',
            self.sel_spd == 'Original speeds'
        )
        self.env.reset(training=True)

        plot_data   = assemble_plot_data(wrapper.env.wds.junctions.head)
        self.plot   = build_plot_from_data(plot_data, self.dmd_lo, self.dmd_hi, 'm^3/h', figtitle='Nodal demand')
        response_load.value     = 'Ready.'
        res_box_load.background = '#FFFFFF'

        num_of_juncs.value  = 'Number of junctions: {}'.format(str(len(wrapper.env.wds.junctions.uid)))
        num_of_pipes.value  = 'Number of pipes: {}'.format(str(len(wrapper.env.wds.pipes.uid)))
        num_of_pumps.value  = 'Number of pump stations: {}'.format(str(len(wrapper.env.pumpGroups)))
        return self.plot

class optimize_speeds(param.Parameterized):
    act_opti    = param.Action(
        lambda x: x.param.trigger('act_opti'),
        label   = 'Optimize pump speeds'
        )

    def __init__(self):
        self.rew_nm     = 0
        self.hist_nm    = []
        self.hist_val_nm= [0]
        self.hist_fail_counter_nm   = []
        self.failed_steps_nm    = 0
        self.rew_dqn        = 0
        self.hist_dqn       = []
        self.hist_val_dqn   = [0]
        self.hist_fail_counter_dqn  = []
        self.cpu_time_nm    = 0
        self.cpu_time_dqn   = 0

    def call_dqn(self):
        wrapper.env.wds.solve()
        wrapper.env.steps   = 0
        wrapper.env.done    = False
        obs                 = wrapper.env.get_observation()
        self.hist_dqn       = [wrapper.env.wds.junctions.head]
        self.hist_val_dqn   = [wrapper.env.get_state_value()]
        invalid_heads_count = (np.count_nonzero(wrapper.env.wds.junctions.head < wrapper.head_lmt_lo) +
            np.count_nonzero(wrapper.env.wds.junctions.head > wrapper.head_lmt_hi))
        self.hist_fail_counter_dqn= [invalid_heads_count]
        self.cpu_time_dqn   = []
        while not wrapper.env.done:
            start_time  = time.time()
            act, _              = wrapper.model.predict(obs, deterministic=True)
            self.cpu_time_dqn.append(time.time()-start_time)
            obs, reward, _, _   = wrapper.env.step(act, training=False)
            self.hist_dqn.append(wrapper.env.wds.junctions.head)
            self.hist_val_dqn.append(wrapper.env.get_state_value())
            invalid_heads_count = (np.count_nonzero(wrapper.env.wds.junctions.head < wrapper.head_lmt_lo) +
                np.count_nonzero(wrapper.env.wds.junctions.head > wrapper.head_lmt_hi))
            self.hist_fail_counter_dqn.append(invalid_heads_count)
        self.hist_dqn       = self.hist_dqn[:-3]
        self.hist_val_dqn   = self.hist_val_dqn[:-3]
        self.hist_fail_counter_dqn  = self.hist_fail_counter_dqn[:-3]
        self.cpu_time_dqn   = sum(self.cpu_time_dqn[:-3])
        fail_counter    = np.array(self.hist_fail_counter_dqn)
        if np.isin(0, fail_counter):
            trunc   = fail_counter[np.where(fail_counter    == 0)[0][0]:]
            self.failed_steps_dqn   = np.sum(trunc != 0)
        else:
            self.failed_steps_dqn   = len(self.hist_fail_counter_dqn)

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
        self.cpu_time_nm    = time.time()
        result  = nm(
            wrapper.env.reward_to_scipy,
            init_guess,
            method  = 'Nelder-Mead',
            options = options,
            callback= self.callback_nm
            )
        self.cpu_time_nm    = time.time()-self.cpu_time_nm
        self.nm_evals   = result.nit
        fail_counter    = np.array(self.hist_fail_counter_nm)
        if np.isin(0, fail_counter):
            trunc   = fail_counter[np.where(fail_counter    == 0)[0][0]:]
            self.failed_steps_nm    = np.sum(trunc != 0)
        else:
            self.failed_steps_nm    = len(self.hist_fail_counter_nm)

    def store_bc(self):
        self.orig_demands   = wrapper.env.wds.junctions.basedemand
        self.orig_speeds    = wrapper.env.wds.pumps.speed

    def restore_bc(self):
        wrapper.env.wds.junctions.basedemand    = self.orig_demands
        wrapper.env.wds.pumps.speed             = self.orig_speeds

    @param.depends('act_opti')
    def plot_dqn(self):
        response_opti.value     = 'Computing, please wait.'
        res_box_opti.background = '#FF0000'
        self.store_bc()
        self.call_dqn()
        self.rew_dqn    = wrapper.env.get_state_value()
        self.dqn_dta    = assemble_plot_data(wrapper.env.wds.junctions.head)
        if wrapper.loaded_wds == 'Anytown':
            plot    = build_plot_from_data(self.dqn_dta, 30, 90, title='m', figtitle='Nodal head')
        else:
            plot    = build_plot_from_data(self.dqn_dta, wrapper.head_lmt_lo, wrapper.head_lmt_hi, title='m', figtitle='Nodal head')
        self.restore_bc()
        response_opti.value     = 'Ready.'
        res_box_opti.background = '#FFFFFF'
        return plot

    @param.depends('act_opti')
    def plot_nm(self):
        response_opti.value     = 'Computing, please wait.'
        res_box_opti.background = '#FF0000'
        self.store_bc()
        self.call_nm()
        self.rew_nm = wrapper.env.get_state_value()
        self.nm_dta = assemble_plot_data(wrapper.env.wds.junctions.head)
        if wrapper.loaded_wds == 'Anytown':
            plot    = build_plot_from_data(self.nm_dta, 30, 90, title='m', figtitle='Nodal head')
        else:
            plot    = build_plot_from_data(self.nm_dta, wrapper.head_lmt_lo, wrapper.head_lmt_hi, title='m', figtitle='Nodal head')
        self.restore_bc()
        response_opti.value     = 'Ready.'
        res_box_opti.background = '#FFFFFF'
        return plot

    @param.depends('act_opti')
    def read_dqn_rew(self):
        return 'Final state value: {:.3f}'.format(self.hist_val_dqn[-1])

    @param.depends('act_opti')
    def read_nm_rew(self):
        return 'Final state value: {:.3f}'.format(self.hist_val_nm[-1])

    @param.depends('act_opti')
    def read_dqn_evals(self):
        return 'Total steps: {}'.format(len(self.hist_val_dqn))

    @param.depends('act_opti')
    def read_nm_evals(self):
        play_animation_dqn()
        play_animation_nm()
        return 'Total steps: {}'.format(len(self.hist_val_nm))

def animate_nm_plot():
    global hist_idx_nm, call_id_nm
    global nm_idx_widget, nm_val_widget
    global nm_widget
    optimizer.nm_dta.data = {
        'x': wrapper.junc_coords['x'],
        'y': wrapper.junc_coords['y'],
        'junc_prop': optimizer.hist_nm[hist_idx_nm]
    }
    nm_idx_widget.value = 'Step: ' + str(hist_idx_nm+1)
    nm_val_widget.value = 'State value: {:.3f}'.format(optimizer.hist_val_nm[hist_idx_nm])
    nm_fail_widget.value= 'High-pressure nodes: ' + str(optimizer.hist_fail_counter_nm[hist_idx_nm])
    if optimizer.hist_fail_counter_nm[hist_idx_nm]:
        nm_widget.background    = '#FF0000'
    else:
        nm_widget.background    = '#FFFFFF'
    hist_idx_nm += 1
    if hist_idx_nm == len(optimizer.hist_nm):
        hist_idx_nm = 0
        curdoc().remove_periodic_callback(call_id_nm)
        button_nm.label = 'Replay optimization'

def play_animation_nm():
    global call_id_nm
    cpu_time_nm.value   = 'CPU time (simulation required): {:.3f} s'.format(optimizer.cpu_time_nm)
    nm_fail_sum.value   = 'Failed steps: {}'.format(optimizer.failed_steps_nm)
    if button_nm.label == 'Replay optimization':
        button_nm.label = 'Pause'
        call_id_nm      = curdoc().add_periodic_callback(animate_nm_plot, 500)
    else:
        button_nm.label = 'Replay optimization'
        curdoc().remove_periodic_callback(call_id_nm)
    comp_time.value     = 'Speedup compared to Nelder-Mead: {:.1f}%'.format(optimizer.cpu_time_nm/optimizer.cpu_time_dqn*100)
    performance.value   = 'Final state value compared to Nelder-Mead: {:.1f}%'.format(optimizer.hist_val_dqn[-1]/optimizer.hist_val_nm[-1]*100)

def animate_dqn_plot():
    global hist_idx_dqn, call_id_dqn
    global dqn_idx_widget, dqn_val_widget
    global dqn_widget
    optimizer.dqn_dta.data = {
        'x': wrapper.junc_coords['x'],
        'y': wrapper.junc_coords['y'],
        'junc_prop': optimizer.hist_dqn[hist_idx_dqn]
    }
    dqn_idx_widget.value = 'Step: ' + str(hist_idx_dqn+1)
    dqn_val_widget.value = 'State value: {:.3f}'.format(optimizer.hist_val_dqn[hist_idx_dqn])
    dqn_fail_widget.value = 'High-pressure nodes: ' + str(optimizer.hist_fail_counter_dqn[hist_idx_dqn])
    if optimizer.hist_fail_counter_dqn[hist_idx_dqn]:
        dqn_widget.background    = '#FF0000'
    else:
        dqn_widget.background    = '#FFFFFF'
    hist_idx_dqn    += 1
    if hist_idx_dqn == len(optimizer.hist_dqn):
        curdoc().remove_periodic_callback(call_id_dqn)
        hist_idx_dqn        = 0
        button_dqn.label    = 'Replay optimization'

def play_animation_dqn():
    global call_id_dqn
    cpu_time_dqn.value  = 'CPU time (simulation not required): {:.3f} s'.format(optimizer.cpu_time_dqn)
    dqn_fail_sum.value  = 'Failed steps: {}'.format(optimizer.failed_steps_dqn)
    if button_dqn.label == 'Replay optimization':
        button_dqn.label= 'Pause'
        call_id_dqn     = curdoc().add_periodic_callback(animate_dqn_plot, 500)
    else:
        button_dqn.label= 'Replay optimization'
        curdoc().remove_periodic_callback(call_id_dqn)

demo_introduction   = pn.pane.Markdown("""
    ### Introduction
    This is a showcase of agents controlling pumps in water distribution systems (WDSs).
    Two benchmark WDSs
    ([Anytown](http://emps.exeter.ac.uk/engineering/research/cws/resources/benchmarks/expansion/anytown.php)
    and
    [D-Town](http://emps.exeter.ac.uk/engineering/research/cws/resources/benchmarks/expansion/d-town.php))
    are available for the demo with agents trained by the deep Q-network (DQN) algorithm.
    Nelder-Mead method serves as a baseline optimization technique that can find optimum pump speeds in a confined number of iteration steps.

    The quality of a setting is measured by the state value that is a weighted sum of multiple objective values.
    Both algorithms are competing to find the optimum speed setting (where the state value is at its peak) for the pumps under given nodal demands and initial speed settings.

    Benefits of the DQN-agent over the conventional optimization algorithm are that DQN-agent

    - makes its steps monotonously towards the optimum,
    - avoids speed settings where pressure is dangerously high,
    - relies only on nodal pressure data (no computer simulation needed).

    These properties make the DQN-agent a good candidate to use as a continuous pump speed controller in WDSs.
    Presently, the drawback is that this type of agent handles discrete pump speeds, hence the achieved optimum is slightly smaller compared to a conventional optimization algorithm.
    """,
    width   = 600
    )
demo_usage  = pn.pane.Markdown("""
    ### Usage
    The nodal demand (in cubic meters per hour) and the nodal pressure as head (in meters) is visualized on the topology of the selected water distribution system.
    Pumps and tanks are not shown in the figures.
    To play an optimization session, do the following.

    1. Select a water distribution system.
    2. Select whether the demands and the pump speeds should be set according to the original values or randomly.
    3. Press the **Load water distribution system** button to load the WDS and the DQN-agent. It can take a few seconds.
    4. The nodal demands are depicted in the upper right figure.
    5. Press the **Optimize pump speeds** button. The pump speeds are set by the DQN-agent and the Nelder-Mead method.
    6. The nodal heads are depicted in the figures in the bottom while the state value and the number of steps needed to reach the optimum is shown beneath the figures.
    7. Each step of the optimization process can be replayed by pressing the **Replay optimization** button. The followings are printed for the current step:
        - the number of the step,
        - the state value,
        - the number of nodes in danger due to high pressure.

    During optimization replay, the nodal heads corresponding to the actual step are colored in the figures.
    In the Anytown case there is nearly no variation in the nodal pressures during replay as the single pump station has little effect on the heads compared to the tank.
    """,
    width   = 600
    )
demo_repo   = pn.pane.Markdown("""
    ### Code repository
    [https://github.com/BME-SmartLab/rl-wds](https://github.com/BME-SmartLab/rl-wds)
    """,
    width   = 400
    )
demo_paper  = pn.pane.Markdown("""
    ### Paper
    Hajgató, G.; Gyires-Tóth, B.; Paál, G. 2020. "Deep Reinforcement Learning for Real-Time Optimization of Pumps in Water Distribution Systems."
    *Journal of Water Resources Planning and Management*. 146 (11): 04020079.
    DOI: [10.1061/(ASCE)WR.1943-5452.0001287](https://doi.org/10.1061/(ASCE)WR.1943-5452.0001287).

    (preprint on arXiv: [2010.06460](https://arxiv.org/abs/2010.06460))
    """,
    width   = 800
    )
demo_acknowledgment = pn.pane.Markdown("""
    ### Acknowledgment
    The research has been supported by the BME-Artificial Intelligence FIKP grant of Ministry of Human Resources (BME FIKP-MI/SC).
    """,
    width   = 1200
    )

wrapper     = environment_wrapper()
optimizer   = optimize_speeds()

hist_idx_nm = 0
call_id_nm  = 0
hist_idx_dqn= 0
call_id_dqn = 0
nm_idx_widget   = pn.widgets.TextInput(value='Step: ', width=300)
nm_val_widget   = pn.widgets.TextInput(value='State value: ', width=300)
nm_fail_widget  = pn.widgets.TextInput(value='High-pressure nodes: ', width=300)
nm_fail_sum     = pn.widgets.TextInput(value='Failed steps: ', width=300)
dqn_idx_widget  = pn.widgets.TextInput(value='Step: ', width=300)
dqn_val_widget  = pn.widgets.TextInput(value='State value: ', width=300)
dqn_fail_widget = pn.widgets.TextInput(value='High-pressure nodes: ', width=300)
dqn_fail_sum    = pn.widgets.TextInput(value='Failed steps: ', width=300)
cpu_time_nm     = pn.widgets.TextInput(value='CPU time (simulation required): ', width=300)
cpu_time_dqn    = pn.widgets.TextInput(value='CPU time (simulation not required): ', width=300)
comp_time       = pn.widgets.TextInput(value='Computation time compared to Nelder-Mead: ', width=400)
performance     = pn.widgets.TextInput(value='Final state value compared to Nelder-Mead: ', width=400)
response_load   = pn.widgets.TextInput(value='', width=200)
response_opti   = pn.widgets.TextInput(value='', width=200)
button_nm   = Button(label='Replay optimization', width=600)
button_nm.on_click(play_animation_nm)
button_dqn  = Button(label='Replay optimization', width=600)
button_dqn.on_click(play_animation_dqn)
num_of_juncs    = pn.widgets.TextInput(value='Number of junctions: ', width=230)
num_of_pipes    = pn.widgets.TextInput(value='Number of pipes: ', width=230)
num_of_pumps    = pn.widgets.TextInput(value='Number of pump stations: ', width=230)

dqn_widget  = pn.WidgetBox(
                dqn_idx_widget,
                dqn_val_widget,
                dqn_fail_widget,
                width       = 600,
                background  = '#FFFFFF'
                )
nm_widget   = pn.WidgetBox(
                nm_idx_widget,
                nm_val_widget,
                nm_fail_widget,
                width       = 600,
                background  = '#FFFFFF'
                )
nm_sum_wid  = pn.WidgetBox(
                nm_fail_sum,
                cpu_time_nm,
                width       = 600,
                background  = '#FFFFFF'
                )
dqn_sum_wid = pn.WidgetBox(
                dqn_fail_sum,
                cpu_time_dqn,
                comp_time,
                performance,
                width       = 600,
                background  = '#FFFFFF'
                )
res_box_load= pn.WidgetBox(
                '',
                width   = 30,
                height  = 30,
                background  = '#FFFFFF'
                )
res_box_opti= pn.WidgetBox(
                '',
                width   = 30,
                height  = 30,
                background  = '#FFFFFF'
                )
pn.Column(
    "# Optimal pump operation with reinforcement learning",
    pn.Row(
        demo_introduction,
        demo_usage
        ),
    '## Loading the water distribution system',
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
            pn.Row(
                response_load,
                res_box_load
                ),
            pn.Column(
                '### Properties',
                num_of_juncs,
                num_of_pipes,
                num_of_pumps,
                )
            ),
        wrapper.load_wds
        ),
    '## Optimizing pump speeds',
    pn.Row(
        pn.panel(optimizer.param, show_labels=False, show_name=False, margin=0, width=500),
        response_opti,
        res_box_opti
        ),
    pn.Row(
        pn.Column(
            '### Deep Q-Network',
            pn.Row(
                optimizer.plot_dqn
                ),
            pn.Row(
                pn.WidgetBox(optimizer.read_dqn_rew, width=300),
                pn.WidgetBox(optimizer.read_dqn_evals, width=300)
                )
            ),
        pn.Column(
            '### Nelder-Mead method',
            pn.Row(
                optimizer.plot_nm
                ),
            pn.Row(
                pn.WidgetBox(optimizer.read_nm_rew, width=300),
                pn.WidgetBox(optimizer.read_nm_evals, width=300)
                )
            )
        ),
    pn.Row(
        pn.Column(
            button_dqn,
            pn.Row(
                dqn_widget
                )
            ),
        pn.Column(
            button_nm,
            pn.Row(
                nm_widget
                )
            )
        ),
    pn.pane.Markdown("""
        ### Summary
        #####Definition of failed steps
        Initial conditions generated randomly can lead to initial states where some of the junctions are in the undesirable high-pressure region.
        The algorithms fail in a timestep when they govern the system from a healthy state to a high-pressure state.
        All of these kind of steps are summed up.
        If one of the algorithms cannot set not even one non-high-pressure scene during optimization, then all of the steps are summed up.

        The Nelder-Mead method is an optimization algorithm (not a controller) with a combined objective function that penalizes high-pressure states.
        Meanwhile, the DQN agent was penalized during training for high-pressure states.
        Hence, the expectation is that the DQN agent can keep the number of failed steps low even when Nelder-Mead cannot.
        """
        ),
    pn.Row(
        dqn_sum_wid,
        nm_sum_wid,
        ),
    pn.Row(
        demo_repo,
        demo_paper
        ),
    demo_acknowledgment
    ).servable()
