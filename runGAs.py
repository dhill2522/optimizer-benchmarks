#!/usr/bin/env python3

'''
Testing some GAs

Nicholas Cooper
'''

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize, NonlinearConstraint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from geneticOpt import GA
from ScipyBaseModel import model, model_obj_only, model_con_max_T, model_con_min_T, model_con_max_ramp, get_T
from utils import get_data, results, gen_report
from default_config import config

def save_iters(x, savepath):
    all_iters = np.array(x).T
    df = pd.DataFrame(all_iters)
    df.to_csv(savepath, index=False)

def runCustom(animate=False):
    time, load = get_data(config['month'], config['year'])
    # config['max_ramp_rate'] = 3000

    def my_con_max_temp(gen):
        inequalities = model_con_max_T(gen, time, load, config)
        for i, con in enumerate(inequalities):
            if con >= 0:
                inequalities[i] = 0

        return np.sum(inequalities)

    def my_con_min_temp(gen):
        inequalities = model_con_min_T(gen, time, load, config)
        for i, con in enumerate(inequalities):
            if con >= 0:
                inequalities[i] = 0
        return np.sum(inequalities)

    def my_con_max_ramp(gen):
        inequalities = model_con_max_ramp(gen, config)
        for i, con in enumerate(inequalities):
            if con >= 0:
                inequalities[i] = 0
        return np.sum(inequalities)

    # def my_con_max_ramp(X):
    #     '''For custom GA.
    #     Max ramp up or down does not exceed 2000 MW/hr'''
    #     dEdt = []
    #     max_ramp = 2000
    #     for i in range(len(X)-1):
    #         slope = abs(X[i+1] - X[i])
    #         if slope > max_ramp:
    #             dEdt.append(slope)
    #         else:
    #             dEdt.append(0)
        
    #     return np.sum(dEdt)

    populations = []
    def callback(gen):
        populations.append(gen)

    guess = np.ones(len(time))*config['guess_coef']

    bounds = [(1e3, 8e4) for i in range(len(time))]

    constraints = ({'fun':my_con_max_temp, 'type':'ineq', 'scale':100}, 
                   {'fun':my_con_min_temp, 'type':'ineq', 'scale':100}, 
                   {'fun':my_con_max_ramp, 'type':'ineq', 'scale':1000})

    ga = GA(model_obj_only, bounds = bounds, maxiter=100, mscale=100, tol=1e-3, 
            constraints=constraints, pmutate=0.5, callback=callback)
    sol = ga.optimize(verbose=True)

    xstar = sol[0]
    nfev = ga.fevals

    print(sol)
    print(results(sol[0], config))
    gen_report([xstar,nfev], "Custom GA", "Constrained", config, notes="lb 1e3 ub 8e4", gen_plot=True, guess=guess)
    # save_iters(populations, "GA_iters3.csv")

    if animate:
        def update(i):
            fig.clear()
            plt.xlabel('Time')
            plt.ylabel("Generation")
            plt.plot(time, load)
            plt.plot(time, populations[i])

        fig = plt.figure()
        plt.xlabel("Time")
        plt.ylabel("Generation")

        anim = animation.FuncAnimation(fig, update, frames=len(populations), interval = 500)
        plt.show()

# Scipy differential_evolution
def runScipy():
    time, load = get_data(config['month'], config['year'])
    guess = np.ones(len(time))*config['guess_coef']

    strategies = ('best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 
                  'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 
                  'rand2bin', 'rand1bin')

    def penaltyobj(gen):
        return model(gen, time, load, config)[0]

    bounds = [(1e3, 1e5) for i in range(len(time))]
    # Try polish=False - if True then takes the best population and uses L-BFGS-B to finish

    # polish = True
    opt = differential_evolution(penaltyobj, bounds=bounds, polish=True, disp=True)
    print(opt)
    fstar = opt.fun
    xstar = opt.x
    nfev = opt.nfev
    print(results(xstar, config))
    gen_report([xstar, nfev], "Scipy GA Polished", "Penalized", config, guess=guess, 
               notes="lb 1e3 ub 8e4 Polished with L-BFGS-B, "+opt.message, gen_plot=True)

    # polish = False
    opt = differential_evolution(penaltyobj, bounds=bounds, polish=False, disp=True)
    print(opt)
    fstar = opt.fun
    xstar = opt.x
    nfev = opt.nfev
    print(results(xstar, config))
    gen_report([xstar, nfev], "Scipy GA", "Penalized", config, guess=guess, 
               notes="lb 1e3 ub 8e4, "+opt.message, gen_plot=True)

def testStrategies():
    time, load = get_data(config['month'], config['year'])
    guess = np.ones(len(time))*config['guess_coef']

    strategies = ('best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 
                  'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 
                  'rand2bin', 'rand1bin')

    def penaltyobj(gen):
        return model(gen, time, load, config)[0]

    bounds = [(1e3, 1e5) for i in range(len(time))]
    # Try polish=False - if True then takes the best population and uses L-BFGS-B to finish
    best = {'strategy': 'none', 
            'results': 'none', 
            'besty': 1e32}
    all_results = {}
    maxiter = 100
    print(f"Finding best over {maxiter} iterations")
    for s in strategies:
        print(f"{s}...")
        opt = differential_evolution(penaltyobj, bounds=bounds, polish=False, 
            disp=False, strategy=s, maxiter=maxiter)
        all_results[s] = opt

        if opt.fun < best['besty']:
            best['strategy'] = s
            best['results'] = opt
            best['besty'] = opt.fun

    print(f"Best Strategy: {best['strategy']}")
    print(f"Best Function Value: {best['besty']}")
    print("Results:")
    print(best['results'])

    print("All other strategies:\n")

    for key, val in all_results.items():
        print(f"{key}: {val.fun}, {val.nfev}, {val.message}")


def runScipyCon():
    global fevals
    time, load = get_data(config['month'], config['year'])
    fevals = 0
    def obj(gen):
        global fevals
        fevals += 1
        return model_obj_only(gen)

    def tempCon(gen):
        return np.array(get_T(gen, time, load, config))/100

    def rampCon(X):
        ''' Max ramp up or down does not exceed 2000 MW/hr'''
        dEdt = []

        for i in range(len(X)-1):
            slope = abs(X[i+1] - X[i])
            dEdt.append(slope)

        return np.array(dEdt)/1000


    bounds = [(1e3, 1e5) for i in range(len(time))]
    Temp_Con = NonlinearConstraint(tempCon, lb=config['tes_min_t']/100, ub=config['tes_max_t']/100)
    Ramp_Con = NonlinearConstraint(rampCon, lb=0, ub=config['max_ramp_rate']/1000)

    # Try polish=False - if True then takes the best population and uses L-BFGS-B to finish
    opt = differential_evolution(obj, bounds=bounds, constraints={Temp_Con, Ramp_Con}, 
                                 polish=False, disp=True)

    print(opt)
    fstar = opt.fun
    xstar = opt.x
    nfev = opt.nfev
    print(results(xstar, config))
    print("fevals:", fevals)
    gen_report([xstar, nfev], "Scipy GA", "Constrained", config, notes="lb 1e3 ub 8e4, Scaled, "+opt.message, gen_plot=True)

if __name__ == "__main__":
    # runCustom()
    # runScipy()
    runScipyCon()
    # testStrategies()
