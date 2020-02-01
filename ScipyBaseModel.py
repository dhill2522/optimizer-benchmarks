import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
from sqlalchemy import create_engine

# Makes matplotlib happy plotting pandas data arrays
pd.plotting.register_matplotlib_converters()

def thermal_storage(t, T, x, load, mass_salt, Cp):
    return 3.6e9*(x - load)/(mass_salt*Cp)

def model(gen, time, load):
    mass_salt = 6e6 # kg of salt for thermal energy storage
    cost_nuclear = 0.021 # $/KWh
    cost_salt = 10.98    # $/kg
    T0 = 350 #K
    Cp = 1530 # J/kg K, heat capacity of the salt
    dt = 3600 # seconds/hr
    T_hist = []

    for i in range(len(time)):
        step = odeint(thermal_storage, T0, [0, 1], 
                args=(gen[i], load[i], mass_salt, Cp))
        print(step)
        T_hist.append(step[1])
        T0 = step[1]

    return cost_salt*mass_salt + np.sum(gen*cost_nuclear)


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

    cost = model(np.ones(len(time))*nuclear_capacity, time, load)
    print('Cost', cost)

    # sol = minimize(model, np.ones(len(time))*nuclear_capacity, args=(time, load))
    # print(sol)

    # plt.plot(time, load, label='Load')
    # # plt.plot(time, sol['x']*1200, label='Nuclear')
    # plt.legend()
    # plt.show()
