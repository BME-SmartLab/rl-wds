# -* coding: utf-8 -*-
import os
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import gym.spaces
from epynet import Network
from opti_algorithms import nm, rs

class wds():
    """Gym-like environment for water distribution systems."""
    def __init__(self,
            wds_name        = 'anytown_master',
            speed_increment = .05,
            episode_len     = 10,
            pump_groups     = [['78', '79']],
            total_demand_lo = .3,
            total_demand_hi = 1.1,
            reset_orig_pump_speeds  = False,
            reset_orig_demands      = False,
            seed    = None):

        self.seedNum   = seed
        if self.seedNum:
            np.random.seed(self.seedNum)
        else:
            np.random.seed()

        pathToRoot  = os.path.dirname(os.path.realpath(__file__))
        pathToWDS   = os.path.join(pathToRoot, 'water_networks', wds_name+'.inp')

        self.wds        = Network(pathToWDS)
        self.demandDict = self.build_demand_dict()
        self.pumpGroups = pump_groups
        self.pump_speeds= np.ones(shape=(len(self.pumpGroups)), dtype=np.float32)
        self.pumpEffs   = np.empty(shape=(len(self.pumpGroups)), dtype=np.float32)

        nomHCurvePtsDict, nomECurvePtsDict = self.get_performance_curve_points()
        nomHCurvePoliDict       = self.fit_polinomials(
                                    nomHCurvePtsDict,
                                    degree=2,
                                    encapsulated=True)
        self.nomECurvePoliDict  = self.fit_polinomials(
                                    nomECurvePtsDict,
                                    degree=4,
                                    encapsulated=True)
        self.sumOfDemands       = sum(
                            [demand for demand in self.wds.junctions.basedemand])
        self.demandRandomizer   = self.build_truncnorm_randomizer(
                                    lo=.7, hi=1.3, mu=1.0, sigma=1.0)

        # Theoretical bounds of {head, efficiency}
        peak_heads   = []
        for key in nomHCurvePoliDict.keys():
            max_q       = np.max(nomHCurvePtsDict[key][:,0])
            opti_result = minimize(
                -nomHCurvePoliDict[key], x0=1, bounds=[(0, max_q)])
            peak_heads.append(nomHCurvePoliDict[key](opti_result.x[0]))
        peak_effs  = []
        for key in nomHCurvePoliDict.keys():
            max_q       = np.max(nomHCurvePtsDict[key][:,0])
            q_list      = np.linspace(0, max_q, 10)
            head_poli   = nomHCurvePoliDict[key]
            eff_poli    = self.nomECurvePoliDict[key]
            opti_result = minimize(-eff_poli, x0=1, bounds=[(0, max_q)])
            peak_effs.append(eff_poli(opti_result.x[0]))
        self.peakTotEff = np.prod(peak_effs)

        # Reward control
        self.dimensions     = len(self.pumpGroups)
        self.episodeLength  = episode_len
        self.headLimitLo    = 15
        self.headLimitHi    = 120
        self.maxHead        = np.max(peak_heads)
        self.rewScale       = [5,8,3] # eff, head, pump
        self.baseReward     = +1
        self.bumpPenalty    = -1
        self.distanceRange  = .5
        self.wrongMovePenalty   = -1
        self.lazinessPenalty    = -1
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
        #maxReward   = 5
        # ----- ----- ----- ----- -----
        self.maxReward   = +1
        self.minReward   = -1

        # Inner variables
        self.spec           = None
        self.metadata       = None
        self.totalDemandLo  = total_demand_lo
        self.totalDemandHi  = total_demand_hi
        self.speedIncrement = speed_increment
        self.speedLimitLo   = .7
        self.speedLimitHi   = 1.2
        self.validSpeeds   = np.arange(
                                self.speedLimitLo,
                                self.speedLimitHi+.001,
                                self.speedIncrement,
                                dtype=np.float32)
        self.resetOrigPumpSpeeds= reset_orig_pump_speeds
        self.resetOrigDemands   = reset_orig_demands
        self.optimized_speeds   = np.empty(shape=(len(self.pumpGroups)),
                                    dtype=np.float32)
        self.optimized_speeds.fill(np.nan)
        self.optimized_value    = np.nan
        self.previous_distance  = np.nan
        # initialization of {observation, steps, done}
        observation = self.reset(training=False)
        self.action_space   = gym.spaces.Discrete(2*self.dimensions+1)
        self.observation_space  = gym.spaces.Box(
                                    low     = -1,
                                    high    = +1,
                                    shape   = (len(self.wds.junctions)+len(self.pumpGroups),),
                                    dtype   = np.float32)

        # for one-shot tests
        self.one_shot   = rs.rs(
            target      = self.reward_to_deap,
            dims        = self.dimensions,
            limit_lo    = self.speedLimitLo,
            limit_hi    = self.speedLimitHi,
            step_size   = self.speedIncrement,
            maxfev      = 1)

    def step(self, action, training=True):
        """ Reward computed from the Euclidean distance between the speed of the pumps
            and the optimized speeds."""
        self.steps  += 1
        self.done   = (self.steps == self.episodeLength)
        group_id    = action // 2
        command     = action % 2
        if training:
            if group_id != self.dimensions:
                self.n_siesta       = 0
                first_pump_in_grp   = self.wds.pumps[self.pumpGroups[group_id][0]]
                if command == 0:
                    if first_pump_in_grp.speed < self.speedLimitHi:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  += self.speedIncrement
                        self.update_pump_speeds()
                        distance    = np.linalg.norm(self.optimized_speeds-self.pump_speeds)
                        if distance < self.previous_distance:
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
                            #reward  = distance * self.baseReward / self.distanceRange
                            reward  = distance * self.baseReward / self.distanceRange / self.maxReward
        # ----- ----- ----- ----- -----
                        else:
                            reward  = self.wrongMovePenalty
                        self.previous_distance  = distance
                    else:
                        self.n_bump += 1
                        reward  = self.bumpPenalty
                else:
                    if first_pump_in_grp.speed > self.speedLimitLo:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  -= self.speedIncrement
                        self.update_pump_speeds()
                        distance    = np.linalg.norm(self.optimized_speeds-self.pump_speeds)
                        if distance < self.previous_distance:
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
                            #reward  = distance * self.baseReward / self.distanceRange
                            reward  = distance * self.baseReward / self.distanceRange /self.maxReward
        # ----- ----- ----- ----- -----
                        else:
                            reward  = self.wrongMovePenalty
                        self.previous_distance  = distance
                    else:
                        self.n_bump += 1
                        reward  = self.bumpPenalty
            else:
                self.n_siesta   += 1
                value   = self.get_state_value()
                if self.n_siesta == 3:
                    self.done   = True
                    if value/self.optimized_value > .98:
        # ----- ----- ----- ----- -----
        # Tweaking reward
        # ----- ----- ----- ----- -----
                        #reward  = 5
                        reward  = 5/self.maxReward
        # ----- ----- ----- ----- -----
                    else:
                        reward = self.lazinessPenalty
                else:
                    if value/self.optimized_value > .98:
                        reward  = self.n_siesta * self.baseReward
                    else:
                        reward = self.lazinessPenalty
            self.wds.solve()
        else:
            if group_id != self.dimensions:
                self.n_siesta       = 0
                first_pump_in_grp   = self.wds.pumps[self.pumpGroups[group_id][0]]
                if command == 0:
                    if first_pump_in_grp.speed < self.speedLimitHi:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  += self.speedIncrement
                    else:
                        self.n_bump += 1
                else:
                    if first_pump_in_grp.speed > self.speedLimitLo:
                        for pump in self.pumpGroups[group_id]:
                            self.wds.pumps[pump].speed  -= self.speedIncrement
                    else:
                        self.n_bump += 1
            else:
                self.n_siesta   += 1
                if self.n_siesta == 3:
                    self.done   = True
            self.wds.solve()
            reward  = self.get_state_value()
        observation = self.get_observation()
        return observation, reward, self.done, {}

    def reset(self, training=True):
        if training:
            if self.resetOrigDemands:
                self.restore_original_demands()
            else:
                self.randomize_demands()
            self.optimize_state()
## One-shot begins
#            self.optimize_state_with_one_shot()
#            if self.optimized_value == 0:
#                self.optimized_value    = .01
## One-shot ends

            if self.resetOrigPumpSpeeds:
                initial_speed   = 1.
                for pump in self.wds.pumps:
                    pump.speed  = initial_speed
            else:
                for pump_grp in self.pumpGroups:
                    initial_speed   = np.random.choice(self.validSpeeds)
                    for pump in pump_grp:
                        self.wds.pumps[pump].speed  = initial_speed
        else:
            if self.resetOrigPumpSpeeds:
                initial_speed   = 1.
                for pump in self.wds.pumps:
                    pump.speed  = initial_speed
            else:
                for pump_grp in self.pumpGroups:
                    initial_speed   = np.random.choice(self.validSpeeds)
                    for pump in pump_grp:
                        self.wds.pumps[pump].speed  = initial_speed
        self.wds.solve()
        observation = self.get_observation()
        self.done   = False
        self.steps  = 0
        self.n_bump = 0
        self.n_siesta   = 0
        return observation

    def seed(self, seed=None):
        """Collecting seeds."""
        return [seed]

    def optimize_state(self):
        speeds, target_val, _   = nm.minimize(
            self.reward_to_scipy, self.dimensions)
        self.optimized_speeds   = speeds
        self.optimized_value    = -target_val

    def optimize_state_with_one_shot(self):
        speeds, target_val, _   = self.one_shot.maximize()
        self.optimized_speeds   = speeds
        self.optimized_value    = target_val

    def fit_polinomials(self, pts_dict, degree, encapsulated=False):
        """Fitting polinomials to points stored in dict."""
        polinomials = dict()
        if encapsulated:
            for curve in pts_dict:
                polinomials[curve]  = np.poly1d(np.polyfit(
                    pts_dict[curve][:,0], pts_dict[curve][:,1], degree))
        else:
            for curve in pts_dict:
                polinomials[curve]  = np.polyfit(
                    pts_dict[curve][:,0], pts_dict[curve][:,1], degree)
        return polinomials

    def get_performance_curve_points(self):
        """Reader for H(Q) and P(Q) curves."""
        head_curves = dict()
        eff_curves  = dict()

        # Loading data to dictionary
        for curve in self.wds.curves:
            if curve.uid[0] == 'H': # this is an H(Q) curve
                head_curves[curve.uid[1:]]  = np.empty([len(curve.values), 2], dtype=np.float32)
                for i, op_pnt in enumerate(curve.values):
                    head_curves[curve.uid[1:]][i, 0]    = op_pnt[0]
                    head_curves[curve.uid[1:]][i, 1]    = op_pnt[1]
        for curve in self.wds.curves:
            if curve.uid[0] == 'E': # this is an E(Q) curve
                eff_curves[curve.uid[1:]]   = np.empty([len(curve.values), 2], dtype=np.float32)
                for i, op_pnt in enumerate(curve.values):
                    eff_curves[curve.uid[1:]][i, 0] = op_pnt[0]
                    eff_curves[curve.uid[1:]][i, 1] = op_pnt[1]

        # Checking consistency
        for head_key in head_curves.keys():
            if all(head_key != eff_key for eff_key in eff_curves.keys()):
                print('\nInconsistency in H(Q) and P(Q) curves.\n')
                raise IndexError
        return head_curves, eff_curves

    def get_junction_heads(self):
        junc_heads = np.empty(
                        shape   = (len(self.wds.junctions),),
                        dtype   = np.float32)
        for junc_id, junction in enumerate(self.wds.junctions):
            junc_heads[junc_id] = junction.head
        return junc_heads

    def get_observation(self):
        heads   = (2*self.get_junction_heads() / self.maxHead) - 1
        self.update_pump_speeds()
        speeds  = self.pump_speeds / self.speedLimitHi
        return np.concatenate([heads, speeds])

    def restore_original_demands(self):
        for junction in self.wds.junctions:
            junction.basedemand = self.demandDict[junction.uid]

    def build_truncnorm_randomizer(self, lo, hi, mu, sigma):
        randomizer = stats.truncnorm(
                        (lo-mu)/sigma, (hi-mu)/sigma, loc=mu, scale=sigma)
        return randomizer

    def randomize_demands(self):
        target_sum_of_demands   = self.sumOfDemands * (self.totalDemandLo +
                np.random.rand()*(self.totalDemandHi-self.totalDemandLo))
        sum_of_random_demands   = 0
        if self.seedNum:
            for junction in self.wds.junctions:
                junction.basedemand     = (self.demandDict[junction.uid] *
                    self.demandRandomizer.rvs(random_state=self.seedNum *
                        int(np.abs(np.floor(junction.coordinates[0])))))
                sum_of_random_demands   += junction.basedemand
        else:
            for junction in self.wds.junctions:
                junction.basedemand     = (self.demandDict[junction.uid] *
                    self.demandRandomizer.rvs())
                sum_of_random_demands   += junction.basedemand
        for junction in self.wds.junctions:
            junction.basedemand *= target_sum_of_demands / sum_of_random_demands

    def calculate_pump_efficiencies(self):
        for i, group in enumerate(self.pumpGroups):
            pump        = self.wds.pumps[group[0]]
            curve_id    = pump.curve.uid[1:]
            pump_head   = pump.downstream_node.head - pump.upstream_node.head
            eff_poli    = self.nomECurvePoliDict[curve_id]
            self.pumpEffs[i]   = eff_poli(pump.flow / pump.speed)

    def build_demand_dict(self):
        demand_dict = dict()
        for junction in self.wds.junctions:
            demand_dict[junction.uid]   = junction.basedemand
        return demand_dict

    def get_state_value_separated(self):
        self.calculate_pump_efficiencies()
        pump_ok = (self.pumpEffs < 1).all() and (self.pumpEffs > 0).all()
        if pump_ok:
            heads   = np.array([head for head in self.wds.junctions.head])
            invalid_heads_count = (np.count_nonzero(heads < self.headLimitLo) +
                np.count_nonzero(heads > self.headLimitHi))
            valid_heads_ratio   = 1 - (invalid_heads_count / len(heads))

            total_demand    = sum(
                [junction.basedemand for junction in self.wds.junctions])
            total_tank_flow = sum(
                [tank.inflow+tank.outflow for tank in self.wds.tanks])
            demand_to_total = total_demand / (total_demand+total_tank_flow)

            total_efficiency    = np.prod(self.pumpEffs)
            eff_ratio           = total_efficiency / self.peakTotEff
        else:
            eff_ratio           = 0
            valid_heads_ratio   = 0
            demand_to_total     = 0
        return eff_ratio, valid_heads_ratio, demand_to_total

    def get_state_value(self):
        self.calculate_pump_efficiencies()
        pump_ok = (self.pumpEffs < 1).all() and (self.pumpEffs > 0).all()
        if pump_ok:
            heads   = np.array([head for head in self.wds.junctions.head])
            invalid_heads_count = (np.count_nonzero(heads < self.headLimitLo) +
                np.count_nonzero(heads > self.headLimitHi))
            valid_heads_ratio   = 1 - (invalid_heads_count / len(heads))

            total_demand    = sum(
                [junction.basedemand for junction in self.wds.junctions])
            total_tank_flow = sum(
                [tank.inflow+tank.outflow for tank in self.wds.tanks])
            demand_to_total = total_demand / (total_demand+total_tank_flow)

            total_efficiency    = np.prod(self.pumpEffs)
            reward  = ( self.rewScale[0] * total_efficiency / self.peakTotEff +
                        self.rewScale[1] * valid_heads_ratio + 
                        self.rewScale[2] * demand_to_total) / sum(self.rewScale)
        else:
            reward  = 0
        return reward

    def get_state_value_to_opti(self, pump_speeds):
        np.clip(a   = pump_speeds,
            a_min   = self.speedLimitLo,
            a_max   = self.speedLimitHi,
            out     = pump_speeds)
        for group_id, pump_group in enumerate(self.pumpGroups):
            for pump in pump_group:
                self.wds.pumps[pump].speed  = pump_speeds[group_id]
        self.wds.solve()
        return self.get_state_value()

    def reward_to_scipy(self, pump_speeds):
        """Only minimization allowed."""
        return -self.get_state_value_to_opti(pump_speeds)

    def reward_to_deap(self, pump_speeds):
        """Return should be tuple."""
        return self.get_state_value_to_opti(np.asarray(pump_speeds)),

    def update_pump_speeds(self):
        for i, pump_group in enumerate(self.pumpGroups):
            self.pump_speeds[i] = self.wds.pumps[pump_group[0]].speed
        return self.pump_speeds

    def get_pump_speeds(self):
        self.update_pump_speeds()
        return self.pump_speeds
