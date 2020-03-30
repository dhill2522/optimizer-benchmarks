import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyoptsparse
from scipy.integrate import odeint

import utils
from ScipyBaseModel import config, model

# Makes matplotlib happy plotting pandas data arrays
pd.plotting.register_matplotlib_converters()

time, net_load = utils.get_data(config['month'], config['year'])

def obj(input):
    objVal = model(input['xvars'], time, net_load, config)[0]
    print('objVal:', objVal)
    return {
        'obj': objVal
    }

if __name__ == "__main__":

    n = len(time)
    guess = config['capacity']*np.ones(n)*0.95

    optProb = pyoptsparse.Optimization('NHES-internal-penalty', obj)
    optProb.addVarGroup('xvars', n, 'c', value=guess, 
                        lower=np.zeros(n), upper=np.ones(n)*config['capacity'])
    optProb.addObj('obj')
    print('Problem', optProb)
    opt = pyoptsparse.OPT('snopt')
    sol = opt(optProb)
    print('Solution', sol)
    print('Stuff', type(sol))
    solx = np.zeros(n)
    idx = 0
    for varname, val in sol.variables.items():
        for var in val:
            solx[idx] = var.value
            idx += 1

    cost, T_hist = model(solx, time, net_load, config)
    print(f'Cost optimized: ${cost}')

    cost_compare, T_hist_compare = model(guess, time, net_load, config)
    print(f'Cost comparison: ${cost_compare}')

    plt.subplot(211)
    plt.plot(time, net_load, label='Net Load')
    plt.plot(time, solx, label='Nuclear optimized')
    plt.plot(time, guess, label='Nuclear Comparison')
    plt.ylabel('Energy (MW)')
    plt.legend()
    plt.subplot(212)
    plt.plot(time, T_hist, label='TES optimized')
    plt.plot(time, T_hist_compare, label='TES Comparison')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Time')
    plt.legend()
    plt.show()
