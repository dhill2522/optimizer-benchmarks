import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyoptsparse
from scipy.integrate import odeint

import utils
from ScipyBaseModel import config, model, model_obj_only, get_T

# Makes matplotlib happy plotting pandas data arrays
# pd.plotting.register_matplotlib_converters()

time, net_load = utils.get_data(config['month'], config['year'])

def obj(input):
    objVal = model(input['xvars'], time, net_load, config)[0]
    return {'obj': objVal}

def obj_constrained(input):
    gen = input['xvars']
    objVal = model_obj_only(gen, config)
    T_hist = get_T(gen, time, net_load, config)
    max_ramp = 0
    for i, val in enumerate(gen[:-1]):
        ramp = abs(val - gen[i+1])
        if ramp > max_ramp:
            max_ramp = ramp
    return {
        'obj': objVal,
        'max_ramp': max_ramp,
        'max_T': max(T_hist),
        'min_T': min(T_hist)
    }


def opt_penalty(guess):
    n = len(time)
    optProb = pyoptsparse.Optimization('NHES-internal-penalty', obj)
    optProb.addVarGroup('xvars', n, 'c', value=guess, 
                        lower=np.zeros(n), upper=np.ones(n)*config['capacity'])
    optProb.addObj('obj')
    print('Problem', optProb)
    opt = pyoptsparse.OPT('snopt')
    sol = opt(optProb)
    return sol

def opt_constrained(guess):
    n = len(time)

    optProb = pyoptsparse.Optimization('NHES-constrained', obj_constrained)
    optProb.addVarGroup('xvars', n, 'c', value=guess, 
                        lower=np.zeros(n), upper=np.ones(n)*config['capacity'])
    optProb.addCon('max_T', upper=config['tes_max_t'])
    optProb.addCon('min_T', lower=config['tes_min_t'])
    optProb.addCon('max_ramp', upper=config['max_ramp_rate'])
    optProb.addObj('obj')
    print('Problem', optProb)
    opt = pyoptsparse.OPT('snopt')
    sol = opt(optProb)
    return sol

if __name__ == "__main__":
    guess = config['capacity']*np.ones(len(time))*0.95
    sol = opt_penalty(guess)
    utils.gen_report([sol.xStar['xvars'], sol.userObjCalls], 'SNOPT', 
                        'Penalized', config, gen_plot=True, guess=guess)
    sol = opt_constrained(guess)
    utils.gen_report([sol.xStar['xvars'], sol.userObjCalls], 'SNOPT', 
                        'Constrained', config, gen_plot=True, guess=guess)
    