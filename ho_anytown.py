# -* coding: utf-8 -*-
import os
import gc
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import DQN
from stable_baselines.common.schedules import PiecewiseSchedule
from wdsEnv import wds
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

pathToRoot      = os.path.dirname(os.path.realpath(__file__))
pathToExp       = os.path.join(pathToRoot, 'experiments')
pathToParams    = os.path.join(pathToExp, 'hyperparameters', 'anytownRRR.yaml')
with open(pathToParams, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)

def optimize_dqn(trial):
    lr_exp      = trial.suggest_int('lr_exp', 0, 2)
    gamma_exp   = trial.suggest_int('gamma_exp', 0, 3)
    gamma       = .9
    for i in range(gamma_exp):
        gamma   += .9 * 10**(-i-1)
    batch_mpl   = trial.suggest_int('batch_mpl', 0, 3)
    return {
        'learning_rate' : 1e-4 * 10**lr_exp,
        'gamma'         : gamma,
        'batch_size'    : 2**(3+batch_mpl)
    }

def optimize_arch(trial):
    # Predicting the number of parameters:
    # input (# of nodes): 22
    # output (# of actions): 3
    # data (# of samples): 25000
    # total product: 1650000
    nn_arch = trial.suggest_categorical('nn_arch', [0, 1, 2, 3, 4])
    if nn_arch == 0:
        nn_layers   = [24] # 24
    elif nn_arch == 1:
        nn_layers   = [24, 6] # 144
    elif nn_arch == 2:
        nn_layers   = [48, 12] # 576
    elif nn_arch == 3:
        nn_layers   = [24, 12, 6] # 1728
    elif nn_arch == 4:
        nn_layers   = [48, 32, 12] # 18432
    return dict(layers=nn_layers)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
            dueling     = True,
            layer_norm  = False,
            act_fun     = tf.nn.relu,
            feature_extraction  = 'mlp')

def optimize_agent(trial):
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

    model_params    = optimize_dqn(trial)
    dict_layers     = optimize_arch(trial)
    model   = DQN(
        policy                  = CustomPolicy,
        policy_kwargs           = dict_layers,
        env                     = env,
        verbose                 = 0,
        train_freq              = 1,
        learning_starts         = 1000,
        buffer_size             = 25000,
        exploration_fraction    = .95,
        exploration_final_eps   = .0,
        param_noise             = False,
        prioritized_replay      = False,
        tensorboard_log         = None,
        n_cpu_tf_sess           = 1,
        **model_params)
    model.learn(
        total_timesteps = 50000)

    rewards = []
    n_episodes, reward_sum = 0, 0.0

    env.randomize_demands()
    obs = env.reset(training=False)
    while n_episodes < 50:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action, training=False)

        if done:
            rewards.append(reward)
            n_episodes += 1
            env.randomize_demands()
            obs = env.reset(training=False)

    mean_reward = np.mean(rewards)
    trial.report(-1 * mean_reward)
    del env, model
    gc.collect()
    return -1 * mean_reward

if __name__ == '__main__':
    study = optuna.create_study(study_name='anytown', storage='sqlite:///anytown_ho.db', load_if_exists=True) # local db
    #study = optuna.create_study(study_name='v4', storage='postgres://ghajgato@domino.hds.bme.hu:5432/anytown_ho', load_if_exists=True)
    study.optimize(optimize_agent, n_trials=160, n_jobs=1)
