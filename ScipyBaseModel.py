import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint

n = 40
data = pd.read_csv('./data/ERCOT_load_2019.csv', parse_dates=['HourEnding'])[:n]
print(data.head())
print(len(data))
time = data['HourEnding']
load = data['WEST']

def thermal_storage(t, T, x, load, mass_salt, Cp):
    return 3.6e9*(x - load)/(mass_salt*Cp)

def model(x, time, load):
    mass_salt = 6e6 # kg of salt for thermal energy storage
    cost_nuclear = 0.021 # $/KWh
    cost_salt = 10.98    # $/kg
    T0 = 350 #K
    Cp = 1530 # J/kg K, heat capacity of the salt
    dt = 3600 # seconds/hr
    gen_nuclear = x*1200 # operating nuclear capacity
    T_hist = []

    for i, _ in enumerate(time):
        step = odeint(thermal_storage, T0, [0, dt], 
                args=(gen_nuclear[i], load[i], mass_salt, Cp))
        T_hist.append(step[1])
        T0 = step[1]

    return cost_salt*mass_salt + np.sum(gen_nuclear*cost_nuclear)

if __name__ == "__main__":
    sol = minimize(model, np.ones(len(time))*0.5, args=(time, load))
    print(sol)

