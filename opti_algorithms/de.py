# -*- coding: utf-8 -*-
import random
import array
import numpy as np
from deap import base
from deap import creator
from deap import tools

CR      = 0.25
F       = 1  
MU      = 30
NGEN    = 20
        
class de():
    def __init__(self, target, dims, limit_lo, limit_hi, printLog=True):
        self.target     = target
        self.dims       = dims
        self.limitLo    = limit_lo
        self.limitHi    = limit_hi
        self.printLog   = printLog
        
        creator.create("FitnessMax", base.Fitness, weights=(+1.0,))
        creator.create("Individual",
            array.array, typecode='d', fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register(  "attr_float",
                                random.uniform,
                                self.limitLo,
                                self.limitHi)
        self.toolbox.register("individual",
            tools.initRepeat, creator.Individual,
            self.toolbox.attr_float, self.dims)
        self.toolbox.register("population",
            tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("select", tools.selRandom, k=3)
        self.toolbox.register("evaluate", self.target)

    def maximize(self):
        pop = self.toolbox.population(n=MU);
        hof = tools.HallOfFame(1)
        
        # Evaluate the individuals
        fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
    
        if self.printLog:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"
            record = stats.compile(pop)
            logbook.record(gen=0, evals=len(pop), **record)
            print(logbook.stream)
    
        for g in range(1, NGEN):
            for k, agent in enumerate(pop):
                a,b,c = self.toolbox.select(pop)
                y = self.toolbox.clone(agent)
                index = random.randrange(self.dims)
                for i, value in enumerate(agent):
                    if i == index or random.random() < CR:
                        candidate   = a[i] + F*(b[i]-c[i])
                        if candidate < self.limitLo:
                            y[i]    = self.limitLo
                        elif candidate > self.limitHi:
                            y[i]    = self.limitHi
                        else:
                            y[i] = candidate
                y.fitness.values = self.toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
            hof.update(pop)
    
            if self.printLog:
                record = stats.compile(pop)
                logbook.record(gen=g, evals=len(pop), **record)
                print(logbook.stream)
        return hof[0], hof[0].fitness.values[0], NGEN*MU
