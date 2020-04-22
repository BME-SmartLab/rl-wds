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
args    = parser.parse_args()

pathToRoot      = os.path.dirname(os.path.realpath(__file__))
pathToExp       = os.path.join(pathToRoot, 'experiments')
pathToParams    = os.path.join(pathToExp, 'hyperparameters', args.params+'.yaml')
with open(pathToParams, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)
pathToSceneDB   = os.path.join(pathToExp, hparams['evaluation']['dbName']+'_db.h5')

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

model   = DQN.load('dummy')
