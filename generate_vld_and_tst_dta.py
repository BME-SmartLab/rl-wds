# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import multiprocessing
import array
import operator
import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize as neldermead
from deap import base
from deap import creator
from deap import tools
from wdsEnv import wds

parser  = argparse.ArgumentParser()
parser.add_argument('--params', default='anytownMaster', type=str, help="Name of the YAML file.")
parser.add_argument('--nscenes', default=100, type=int, help="Number of the scenes to generate.")
parser.add_argument('--seed', default=None, type=int, help="Random seed for the optimization methods.")
parser.add_argument('--dbname', default=None, type=str, help="Name of the generated database.")
parser.add_argument('--nproc', default=1, type=int, help="Number of processes to raise.")
args    = parser.parse_args()

pathToRoot      = os.path.dirname(os.path.realpath(__file__))
pathToParams    = os.path.join(
                    pathToRoot,
                    'experiments',
                    'hyperparameters',
                    args.params+'.yaml')
with open(pathToParams, 'r') as fin:
    hparams = yaml.load(fin, Loader=yaml.Loader)

reset_orig_demands  = hparams['env']['resetOrigDemands']
test_db_name        = hparams['evaluation']['dbName']
wds_name            = hparams['env']['waterNet']+'_master'
if args.dbname:
    db_name = args.dbname
else:
    db_name     = test_db_name+'_db'
pathToDB    = os.path.join(pathToRoot, 'experiments', db_name+'.h5')
n_scenes    = args.nscenes
seed        = args.seed
n_proc      = args.nproc

if seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
else:
    random.seed()
    np.random.seed()

if n_scenes > 10:
    verbosity   = n_scenes // 10
else:
    verbosity   = 1

env = wds(
        wds_name        = hparams['env']['waterNet']+'_master',
        speed_increment = hparams['env']['speedIncrement'],
        episode_len     = hparams['env']['episodeLen'],
        pump_groups     = hparams['env']['pumpGroups'],
        total_demand_lo = hparams['env']['totalDemandLo'],
        total_demand_hi = hparams['env']['totalDemandHi'],
        seed            = args.seed
)

def generate_scenes(reset_orig_demands, n_scenes):
    junction_ids    = list(env.wds.junctions.uid)
    demand_db = pd.DataFrame(
        np.empty(shape = (n_scenes, len(junction_ids))),
        columns = junction_ids)
    if reset_orig_demands:
        for i in range(n_scenes):
            env.restore_original_demands()
            demand_db.loc[i]    = env.wds.junctions.basedemand
    else:
        for i in range(n_scenes):
            env.randomize_demands()
            demand_db.loc[i]    = env.wds.junctions.basedemand
    return demand_db

def reward_to_scipy(pump_speeds):
    """Only minimization allowed."""
    return -env.get_state_value_to_opti(pump_speeds)

def reward_to_deap(pump_speeds):
    return env.get_state_value_to_opti(np.asarray(pump_speeds)),

class nelder_mead_method():
    def __init__(self):
        self.options    = { 'maxfev': 1000,
                            'xatol' : .005,
                            'fatol' : .01}

    def maximize(self, scene_id):
        if seed:
            random.seed(args.seed)
            init_guess  = []
            for i in range(env.dimensions):
                init_guess.append(random.uniform(env.speedLimitLo, env.speedLimitHi))
        else:
            random.seed()
            init_guess  = []
            for i in range(env.dimensions):
                init_guess.append(random.uniform(env.speedLimitLo, env.speedLimitHi))
    
        env.wds.junctions.basedemand    = scene_df.loc[scene_id]
        options     = { 'maxfev': 1000,
                        'xatol' : .005,
                        'fatol' : .01}
        result  = neldermead(
            reward_to_scipy,
            init_guess,
            method  = 'Nelder-Mead',
            options = options)
    
        result_df   = pd.DataFrame(
                        np.empty(shape=(1, len(df_header))),
                        columns = df_header)
        result_df['index']     = scene_id
        result_df['reward']    = -result.fun
        result_df['evals']     = result.nit
        for i in range(env.dimensions):
            result_df['speedOfGrp'+str(i)] = result.x[i]
        return result_df

class differential_evolution():
    def __init__(self):
        pass

    def maximize(self, scene_id):
        """Optimizes a specific scene according to scene id."""
        CR      = 0.25
        F       = 1  
        MU      = 30
        NGEN    = 20
    
        if seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
        else:
            random.seed()
            np.random.seed()

        creator.create("FitnessMax",
            base.Fitness,
            weights=(+1.0,))
        creator.create("Individual",
            array.array,
            typecode    = 'd',
            fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_float",
            random.uniform,
            env.speedLimitLo,
            env.speedLimitHi)
        toolbox.register("individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            env.dimensions)
        toolbox.register("population",
            tools.initRepeat,
            list,
            toolbox.individual)
        toolbox.register("select",
            tools.selRandom,
            k   = 3)
        toolbox.register("evaluate",
            reward_to_deap)
    
        env.wds.junctions.basedemand    = scene_df.loc[scene_id]
        pop = toolbox.population(n=MU);
        hof = tools.HallOfFame(1)
    
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    
        for g in range(1, NGEN):
            for k, agent in enumerate(pop):
                a,b,c = toolbox.select(pop)
                y = toolbox.clone(agent)
                index = random.randrange(env.dimensions)
                for i, value in enumerate(agent):
                    if i == index or random.random() < CR:
                        candidate   = a[i] + F*(b[i]-c[i])
                        if candidate < env.speedLimitLo:
                            y[i]    = env.speedLimitLo
                        elif candidate > env.speedLimitHi:
                            y[i]    = env.speedLimitHi
                        else:
                            y[i] = candidate
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
            hof.update(pop)
    
        result_df   = pd.DataFrame(
                        np.empty(shape=(1, len(df_header))),
                        columns = df_header)
        result_df['index']     = scene_id
        result_df['reward']    = hof[0].fitness.values[0]
        result_df['evals']     = NGEN*MU
        for i in range(env.dimensions):
            result_df['speedOfGrp'+str(i)] = hof[0][i]
        del creator.FitnessMax, creator.Individual
        return result_df

class particle_swarm_optimization():
    def __init__(self):
        pass

    def generate_particle(self, size, pmin, pmax, smin, smax):
        part    = creator.Particle(
                    random.uniform(pmin, pmax) for _ in range(size))
        part.speed  = [random.uniform(smin, smax) for _ in range(size)]
        part.smin   = smin
        part.smax   = smax
        return part
    
    def update_particle(self, part, best, phi1, phi2, limitLo, limitHi):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(
                        operator.add,
                        part.speed,
                        map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if speed < part.smin:
                part.speed[i] = part.smin
            elif speed > part.smax:
                part.speed[i] = part.smax
        tmp = np.asarray(list(map(operator.add, part, part.speed)))
        np.clip(a=tmp, a_min=limitLo, a_max=limitHi, out=tmp)
        part[:] = tmp.tolist()

    def maximize(self, scene_id):
        MU      = 30
        NGEN    = 20
    
        if seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
        else:
            random.seed()
            np.random.seed()
    
        # Toolbox setup
        creator.create("FitnessMax",
            base.Fitness,
            weights = (+1.0,))
        creator.create("Particle",
            list,
            fitness = creator.FitnessMax,
            speed   = list,
            smin    = None,
            smax    = None,
            best    = None)
        toolbox = base.Toolbox()
        toolbox.register("particle",
            self.generate_particle,
            size    = env.dimensions,
            pmin    = env.speedLimitLo,
            pmax    = env.speedLimitHi,
            smin    = -.3,
            smax    = +.3)
        toolbox.register("population",
            tools.initRepeat,
            list,
            toolbox.particle)
        toolbox.register("update",
            self.update_particle,
            phi1    = .2,
            phi2    = .2,
            limitLo = env.speedLimitLo,
            limitHi = env.speedLimitHi)
        toolbox.register("evaluate",
            reward_to_deap)
    
        env.wds.junctions.basedemand    = scene_df.loc[scene_id]
        pop     = toolbox.population(n=MU)
        best    = None
        for g in range(NGEN):
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                toolbox.update(part, best)
    
        result_df   = pd.DataFrame(
                        np.empty(shape=(1, len(df_header))),
                        columns = df_header)
        result_df['index']     = scene_id
        result_df['reward']    = best.fitness.values[0]
        result_df['evals']     = NGEN*MU
        for i in range(env.dimensions):
            result_df['speedOfGrp'+str(i)] = best[i]
        del creator.FitnessMax, creator.Particle
        return result_df

class fixed_step_size_random_search():
    def __init__(self, target, dims, limit_lo, limit_hi, step_size, maxfev=1000):
        self.feval      = target
        self.dims       = dims
        self.limitLo    = limit_lo
        self.limitHi    = limit_hi
        self.stepSize   = step_size
        self.maxIter    = maxfev

    def sampling_hypersphere(self, origin):
        step_vector = np.random.uniform(
                        low     = -1.,
                        high    = +1.,
                        size    = (self.dims, 1))
        step_vector /= np.linalg.norm(step_vector, axis=0)
        step_vector *= self.stepSize
        new_origin  = np.add(origin, step_vector)
        break_bound = ( (np.min(new_origin) < self.limitLo) or
                        (np.max(new_origin) > self.limitHi))
        while break_bound:
            step_vector = np.random.uniform(
                            low     = -1.,
                            high    = +1.,
                            size    = (self.dims, 1))
            step_vector /= np.linalg.norm(step_vector, axis=0)
            step_vector *= self.stepSize
            new_origin  = np.add(origin, step_vector)
            break_bound = ( (np.min(new_origin) < self.limitLo) or
                            (np.max(new_origin) > self.limitHi))
        return new_origin

    def maximize(self, scene_id):
        if seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
        else:
            random.seed()
            np.random.seed()

        env.wds.junctions.basedemand    = scene_df.loc[scene_id]
        candidate   = np.random.uniform(
                        low     = self.limitLo,
                        high    = self.limitHi,
                        size    = (self.dims, 1))
        performance = self.feval(candidate)[0]
        for i in range(self.maxIter-1):
            new_candidate   = self.sampling_hypersphere(candidate)
            new_performance = self.feval(new_candidate)[0]
            if new_performance >= performance:
                candidate   = new_candidate
                performance = new_performance
        result_df   = pd.DataFrame(
                        np.empty(shape=(1, len(df_header))),
                        columns = df_header)
        result_df['index']     = scene_id
        result_df['reward']    = performance
        result_df['evals']     = self.maxIter
        for i in range(env.dimensions):
            result_df['speedOfGrp'+str(i)] = candidate[i]
        return result_df

def optimize_scenes(scene_df, method=None):
    pool        = multiprocessing.Pool(n_proc)
    result_df   = pool.map(method, range(len(scene_df)))
    result_df   = pd.concat(result_df)
    result_df.set_index('index', inplace=True)
    result_df.rename_axis(None, inplace=True)
    return result_df

scene_df    = generate_scenes(reset_orig_demands, n_scenes)
scene_df.to_hdf(pathToDB, key='scenes', mode='w')

df_header   = ['index', 'reward', 'evals']
for i in range(env.dimensions):
    df_header.append('speedOfGrp'+str(i))

nm      = nelder_mead_method()
de      = differential_evolution()
pso     = particle_swarm_optimization()
fssrs   = fixed_step_size_random_search(
    target      = reward_to_deap,
    dims        = env.dimensions,
    limit_lo    = env.speedLimitLo,
    limit_hi    = env.speedLimitHi,
    step_size   = env.speedIncrement,
    maxfev      = 500)
oneshot = fixed_step_size_random_search(
    target      = reward_to_deap,
    dims        = env.dimensions,
    limit_lo    = env.speedLimitLo,
    limit_hi    = env.speedLimitHi,
    step_size   = env.speedIncrement,
    maxfev      = 1)
subdf_nm    = optimize_scenes(scene_df, nm.maximize)
subdf_de    = optimize_scenes(scene_df, de.maximize)
subdf_pso   = optimize_scenes(scene_df, pso.maximize)
subdf_fssrs = optimize_scenes(scene_df, fssrs.maximize)
subdf_rnd   = optimize_scenes(scene_df, oneshot.maximize)

subdfs      = {'nm': subdf_nm, 'de': subdf_de, 'pso': subdf_pso, 'fssrs': subdf_fssrs, 'oneshot': subdf_rnd}
result_df   = pd.concat(subdfs.values(), axis=1, keys=subdfs.keys())
result_df.to_hdf(pathToDB, key='results', mode='a')
