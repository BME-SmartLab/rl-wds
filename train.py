# -* coding: utf-8 -*-
import argparse
import os
import glob
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
from stable_baselines.common.schedules import PiecewiseSchedule
from wdsEnv import wds

parser  = argparse.ArgumentParser()
parser.add_argument('--params', default='anytownMaster', help="Name of the YAML file.")
parser.add_argument('--seed', default=None, type=int, help="Random seed for the optimization methods.")
parser.add_argument('--nproc', default=1, type=int, help="Number of processes to raise.")
parser.add_argument('--tstsplit', default=20, type=int, help="Ratio of scenes moved from vld to tst scene in percentage.")
args    = parser.parse_args()

pathToRoot      = os.path.dirname(os.path.realpath(__file__))
pathToExp       = os.path.join(pathToRoot, 'experiments')
pathToHistory   = os.path.join(pathToExp, 'history')
pathToParams    = os.path.join(pathToExp, 'hyperparameters', args.params+'.yaml')
with open(pathToParams, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)
pathToSceneDB   = os.path.join(pathToExp, hparams['evaluation']['dbName']+'_db.h5')
history_files   = [f for f in glob.glob(os.path.join(pathToHistory, '*.h5'))]
runId   = 1
while os.path.join(pathToHistory, args.params+str(runId)+'_vld.h5') in history_files:
    runId += 1
runId   = args.params+str(runId)
pathToVldHistoryDB  = os.path.join(pathToHistory, runId+'_vld.h5')
pathToBestModel = os.path.join(pathToExp, 'models', runId+'-best')
pathToLastModel = os.path.join(pathToExp, 'models', runId+'-last')
pathToLog       = os.path.join(pathToExp, 'tensorboard_logs')
vldFreq = hparams['training']['totalSteps'] // 25

env = wds(
        wds_name        = hparams['env']['waterNet']+'_master',
        speed_increment = hparams['env']['speedIncrement'],
        episode_len     = hparams['env']['episodeLen'],
        pump_groups     = hparams['env']['pumpGroups'],
        total_demand_lo = hparams['env']['totalDemandLo'],
        total_demand_hi = hparams['env']['totalDemandHi'],
        reset_orig_pump_speeds  = hparams['env']['resetOrigPumpSpeeds'],
        reset_orig_demands      = hparams['env']['resetOrigDemands']
)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
            layers      = hparams['model']['layers'],
            dueling     = True,
            layer_norm  = False,
            act_fun     = tf.nn.relu,
            feature_extraction  = 'mlp')

def init_trn_history():
    hist_header = ['episodeId', 'stepId', 'reward']
    init_array  = np.empty(
                    (hparams['training']['totalSteps'], len(hist_header)),
                    dtype=np.float32)
    init_array.fill(np.nan)
    trn_history = pd.DataFrame(init_array, columns = hist_header)
    trn_history.to_hdf(pathToTrnHistoryDB, key=runId, mode='w')
    return trn_history

def init_vldtst_history(scenes, tst=False):
    hist_header = [
        'lastReward', 'bestReward', 'worstReward',
        'nFail', 'nBump', 'nSiesta', 'nStep',
        'explorationFactor']
    for i in range(env.dimensions):
        hist_header.append('speedOfGrp'+str(i))
    scene_ids   = np.arange(len(scenes))
    step_ids    = np.arange(
                    vldFreq,
                    hparams['training']['totalSteps']+1,
                    vldFreq)
    if not tst:
        hist_index  = pd.MultiIndex.from_product(
                        [step_ids, scene_ids],
                        names = ['step_id', 'scene_id'])
    else:
        hist_index  = pd.Index(scene_ids, name='scene_id')
    init_array  = np.empty((len(hist_index), len(hist_header)), dtype=np.float32)
    init_array.fill(np.nan)
    history     = pd.DataFrame(
                    init_array,
                    index   = hist_index,
                    columns = hist_header)
    return history

def init_vld_history():
    hist_header = [
        'lastReward', 'bestReward', 'worstReward',
        'nFail', 'nBump', 'nSiesta', 'nStep',
        'explorationFactor']
    for i in range(env.dimensions):
        hist_header.append('speedOfGrp'+str(i))
    scene_ids   = np.arange(len(vld_scenes))
    step_ids    = np.arange(
                    vldFreq,
                    hparams['training']['totalSteps']+1,
                    vldFreq)
    hist_index  = pd.MultiIndex.from_product(
                    [step_ids, scene_ids],
                    names = ['step_id', 'scene_id'])
    init_array  = np.empty((len(hist_index), len(hist_header)), dtype=np.float32)
    init_array.fill(np.nan)
    vld_history = pd.DataFrame(
                    init_array,
                    index   = hist_index,
                    columns = hist_header)
    vld_history.to_hdf(pathToVldHistoryDB, key=runId, mode='w')
    return vld_history

def play_scenes(scenes, history, path_to_history, tst=False):
    global best_metric
    cummulated_reward   = 0
    for scene_id in range(len(scenes)):
        env.wds.junctions.basedemand    = scenes.loc[scene_id]
        obs     = env.reset(training=False)
        rewards = np.empty(
                    shape = (env.episodeLength,),
                    dtype = np.float32)
        rewards.fill(np.nan)
        pump_speeds = np.empty(
                        shape   = (env.episodeLength, env.dimensions),
                        dtype   = np.float32)
        while not env.done:
            act, _              = model.predict(obs, deterministic=True)
            obs, reward, _, _   = env.step(act, training=False)
            pump_speeds[env.steps-1, :] = env.get_pump_speeds()
            rewards[env.steps-1]        = reward
        cummulated_reward   += reward

        if not tst:
            df_view = history.loc[step_id].loc[scene_id].copy(deep=False)
        else:
            df_view = history.loc[scene_id].copy(deep=False)
        df_view['lastReward']   = rewards[env.steps-1]
        df_view['bestReward']   = np.nanmax(rewards)
        df_view['worstReward']  = np.nanmin(rewards)
        df_view['nFail']        = np.count_nonzero(rewards==0)
        df_view['nBump']        = env.n_bump
        df_view['nSiesta']      = env.n_siesta
        df_view['nStep']        = env.steps
        df_view['explorationFactor']= model.exploration.value(step_id)
        for i in range(env.dimensions):
            df_view['speedOfGrp'+str(i)] = pump_speeds[env.steps-1, i]
    avg_reward  = cummulated_reward / (scene_id+1)
    print('Average reward for {:} scenes: {:.3f}.'.format(scene_id+1, avg_reward))
    if (not tst) and (avg_reward > best_metric):
        print('Average reward improved {:.3f} --> {:.3f}.\nSaving...'
            .format(best_metric, avg_reward))
        best_metric = avg_reward
        model.save(pathToBestModel)
    obs = env.reset(training=True)
    history.to_hdf(path_to_history, key=runId, mode='a')
    return avg_reward

def callback(_locals, _globals):
    global step_id, vld_history, best_metric
    step_id += 1
    if step_id % vldFreq == 0:
        if args.tstsplit != 100:
            print('{}. step, validating.'.format(step_id))
            avg_reward  = play_scenes(vld_scenes, vld_history, pathToVldHistoryDB)
            if avg_reward > best_metric:
                print('Cummulated reward improved {:.3f} --> {:.3f}.\nSaving...'
                    .format(best_metric, avg_reward))
                best_metric = avg_reward
                model.save(pathToBestModel)
            obs = env.reset(training=True)
#        vld_history.to_hdf(pathToVldHistoryDB, key=runId, mode='a')
    return True

step_id     = 0
best_metric = 0
vldtst_scenes   = pd.read_hdf(pathToSceneDB, 'scenes')
if args.tstsplit:
    assert ((args.tstsplit >= 0) and (args.tstsplit <= 100))
    print('Splitting scene db to {:}% validation and {:}% test data.\n'
        .format(100-args.tstsplit, args.tstsplit))
    cut_idx     = int(len(vldtst_scenes) * (100 - args.tstsplit)*0.01)
    vld_scenes  = vldtst_scenes[:cut_idx].copy(deep=False)
    tst_scenes  = vldtst_scenes[cut_idx:].copy(deep=False)
    tst_scenes.index    = tst_scenes.index - tst_scenes.index[0]
else:
    vld_scenes  = vldtst_scenes.copy(deep=False)

vld_history = init_vldtst_history(vld_scenes)
vld_history.to_hdf(pathToVldHistoryDB, key=runId, mode='w')

totalSteps  = hparams['training']['totalSteps']
initLrnRate = hparams['training']['initLrnRate']
lr_schedule = PiecewiseSchedule(([
                (0, initLrnRate),
                (1*totalSteps // 2, initLrnRate * .1),
                (3*totalSteps // 4, initLrnRate * .01)
]))
model   = DQN(
    policy                  = CustomPolicy,
    env                     = env,
    verbose                 = 1,
    #learning_rate           = lr_schedule.value(step_id),
    learning_rate           = initLrnRate,
    buffer_size             = hparams['training']['bufferSize'],
    gamma                   = hparams['training']['gamma'],
    batch_size              = hparams['training']['batchSize'],
    learning_starts         = hparams['training']['learningStarts'],
    exploration_fraction    = .95,
    exploration_final_eps   = .0,
    param_noise             = False,
    prioritized_replay      = False,
    tensorboard_log         = pathToLog,
    full_tensorboard_log    = True,
    seed                    = args.seed,
    n_cpu_tf_sess           = args.nproc)
model.learn(
    total_timesteps = hparams['training']['totalSteps'],
    log_interval    = hparams['training']['totalSteps'] // 50,
    callback        = callback,
    tb_log_name     = args.params)
model.save(pathToLastModel)

if args.tstsplit:
    print('End of training, testing.\n')
    tst_history = init_vldtst_history(tst_scenes, tst=True)
    pathToTstHistoryDB  = os.path.join(pathToHistory, runId+'_tst.h5')
    tst_history.to_hdf(pathToTstHistoryDB, key=runId, mode='w')
    play_scenes(tst_scenes, tst_history, pathToTstHistoryDB, tst=True)
