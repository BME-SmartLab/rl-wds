# -*- coding: utf-8 -*-
import numpy as np

class rs():
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

    def maximize(self):
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
        return candidate, performance, self.maxIter
