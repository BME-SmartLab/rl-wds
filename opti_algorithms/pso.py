# -*- coding: utf-8 -*-
import operator
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools

MU      = 30
NGEN    = 20

class pso():
    def __init__(self, target, dims, limit_lo, limit_hi, printLog=True):
        self.target     = target
        self.dims       = dims
        self.limitLo    = limit_lo
        self.limitHi    = limit_hi
        self.printLog   = printLog
        
        creator.create("FitnessMax", base.Fitness,
            weights = (+1.0,))
        creator.create("Particle", list,
            fitness = creator.FitnessMax,
            speed   = list,
            smin    = None,
            smax    = None,
            best    = None)
        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.generate,
            size    = self.dims,
            pmin    = self.limitLo,
            pmax    = self.limitHi,
            smin    = -.3,
            smax    = +.3)
        self.toolbox.register("population", tools.initRepeat,
            list,
            self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle,
            phi1    = .2,
            phi2    = .2)
        self.toolbox.register("evaluate", target)

    def generate(self, size, pmin, pmax, smin, smax):
        part    = creator.Particle(
                    random.uniform(pmin, pmax) for _ in range(size))
        part.speed  = [random.uniform(smin, smax) for _ in range(size)]
        part.smin   = smin
        part.smax   = smax
        return part

    def updateParticle(self, part, best, phi1, phi2):
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
        np.clip(a=tmp, a_min=self.limitLo, a_max=self.limitHi, out=tmp)
        part[:] = tmp.tolist()

    def maximize(self):
        pop = self.toolbox.population(n=MU)

        if self.printLog:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            logbook = tools.Logbook()
            logbook.header = ["gen", "evals"] + stats.fields

        best = None

        for g in range(NGEN):
            for part in pop:
                part.fitness.values = self.toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in pop:
                self.toolbox.update(part, best)

            if self.printLog:
                logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
                print(logbook.stream)
        return best, best.fitness.values[0], NGEN*MU
