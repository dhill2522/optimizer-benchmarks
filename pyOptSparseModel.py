import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sqlalchemy import create_engine
import pyoptsparse

# Makes matplotlib happy plotting pandas data arrays
pd.plotting.register_matplotlib_converters()

# Create the connection to the database
con = create_engine('sqlite:///data/ercot_data.db')

# Create the sql query
query = "SELECT * FROM Load WHERE HourEnding BETWEEN date('2019-10-01') AND date('2019-10-02')"

# Load the data from the database
data = pd.read_sql(query, con, parse_dates=['HourEnding'])

time = data['HourEnding']
load = data['ERCOT']

def thermal_storage(t, T, x, load, mass_salt, Cp):
    # Energy difference between load and generation is handled by TES
    return 3.6e9*(x - load)/(mass_salt*Cp)


def model(gen, verbose=False):
    mass_salt = 6e8  # kg of salt for thermal energy storage
    cost_nuclear = 0.021  # $/KWh
    cost_salt = 10.98    # $/kg
    cost_blackout = 1e10
    cost_oversupply = 1e10
    T_next = 350  # K
    Cp = 1530  # J/kg K, heat capacity of the salt
    T_hist = []
    tes_min_t = 300
    tes_max_t = 700

    cost_total = cost_salt*mass_salt

    for i in range(len(time)):
        step = odeint(thermal_storage, T_next, [0, 1],
                      args=(gen[i], load[i], mass_salt, Cp))
        T_next = step[1]
        if T_next < tes_min_t:
            if verbose:
                print('Warning: TES too cold.')
            cost_total += cost_blackout*(tes_min_t-T_next)
            T_next = tes_min_t

        if T_next > tes_max_t:
            if verbose:
                print('Warning: TES too hot.')
            cost_total += cost_oversupply*(T_next-tes_max_t)
            T_next = tes_max_t

        T_hist.append(T_next)

    cost_total += np.sum(gen*cost_nuclear)

    return cost_total, T_hist


def obj(input):
    objVal = model(input['xvars'])[0]
    print('objVal:', objVal)
    return {
        'obj': objVal
    }

if __name__ == "__main__":
    n = len(time)
    capacity = 54000
    guess = capacity*np.ones(n)*0.95

    optProb = pyoptsparse.Optimization('NHES', obj)
    optProb.addVarGroup('xvars', n, 'c', value=guess, lower=np.zeros(n), upper=np.ones(n)*capacity)
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

    cost, T_hist = model(solx, verbose=True)
    print(f'Cost optimized: ${cost}')

    cost_compare, T_hist_compare = model(guess, verbose=True)
    print(f'Cost comparison: ${cost_compare}')

    plt.subplot(211)
    plt.plot(time, load, label='Load')
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
