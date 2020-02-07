import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
from sqlalchemy import create_engine

# Makes matplotlib happy plotting pandas data arrays
pd.plotting.register_matplotlib_converters()


def thermal_storage(t, T, x, load, mass_salt, Cp):
    # Energy difference between load and generation is handled by TES
    return 3.6e9*(x - load)/(mass_salt*Cp)


def model(gen, time, load, verbose=False):
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


def obj(gen, time, load):
    return model(gen, time, load)[0]


if __name__ == "__main__":
    # Create the connection to the database
    con = create_engine('sqlite:///data/ercot_data.db')

    # Create the sql query
    query = "SELECT * FROM Load WHERE HourEnding BETWEEN date('2019-10-01') AND date('2019-10-02')"

    # Load the data from the database
    data = pd.read_sql(query, con, parse_dates=['HourEnding'])

    time = data['HourEnding']
    load = data['ERCOT']

    nuclear_capacity = 54000

    guess = np.ones(len(time))*nuclear_capacity*0.95
    sol = minimize(obj, guess, args=(time, load), method='Nelder-mead')
    print('Success:', sol['success'])
    print('x: ', sol['x'])
    print(sol)

    cost, T_hist = model(sol['x'], time, load, verbose=True)
    print(f'Cost optimized: ${cost}')

    guess = np.ones(len(time))*nuclear_capacity*0.97
    cost_compare, T_hist_compare = model(guess, time, load, verbose=True)
    print(f'Cost comparison: ${cost_compare}')

    plt.subplot(211)
    plt.plot(time, load, label='Load')
    plt.plot(time, sol['x'], label='Nuclear optimized')
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
