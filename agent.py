# -* coding: utf-8 -*-
import os
import yaml
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from stable_baselines import DQN
from wdsEnv import wds

hyperparams_fn  = 'anytownMaster'
model_fn        = 'anytownHO1-best'

pathToRoot  = os.path.dirname(os.path.realpath(__file__))
pathToExp   = os.path.join(pathToRoot, 'experiments')
pathToParams= os.path.join(pathToExp, 'hyperparameters', hyperparams_fn+'.yaml')
with open(pathToParams, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)
pathToModel = os.path.join(pathToExp, 'models', model_fn+'.zip')

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

model   = DQN.load(pathToModel)

obs = env.reset()
while not env.done:
    act, _              = model.predict(obs, deterministic=True)
    obs, reward, _, _   = env.step(act, training=False)
    print(reward)
