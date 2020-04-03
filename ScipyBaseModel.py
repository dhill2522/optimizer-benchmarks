import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from scipy.optimize import minimize
from scipy.integrate import odeint

# Makes matplotlib happy plotting pandas data arrays
pd.plotting.register_matplotlib_converters()

config = {
    'cost_nuclear': 0.021,      # $/KWh
    'cost_blackout': 1e10,      # $, cost of not supplying sufficient power, for penalty method
    'cost_oversupply': 1e10,    # $, cost of supplying too much power, for penalty method
    'cost_ramp': 1,             # $/MW/hr, Cost of ramping up and down the reactor core
    'cost_overramp': 1e10,      # $ per MW/hr overage, cost of ramping too quickly
    'max_ramp_rate': 2000,      # MW/hr, Max rate of ramping the reactor core
    'Cp': 1530,                 # J/kg K, heat capacity of the salt
    'tes_min_t': 300,           # K, Minimum temperature of thermal storage unit
    'tes_max_t': 700,           # K, Maximum temperature of thermal storage unit
    'mass_salt': 6e8,           # kg, mass of salt for thermal energy storage
    'capacity': 54000,          # MW, Total amount of potential nuclear output
    'year': '2019',             # Year being examined
    'month': '10',              # Month being examined
    'guess_coef': 54000*0.95    # Initial guess (multiplied by an array of 1s)
}

def thermal_storage(t, T, x, load, mass_salt, Cp):
    '''Determines how much heat will be stored/removed 
    by the salt in the reactor.  Meant to be integrated 
    using scipy.integrate.odeint.
    
    Params:
    -------
    t : 1D array
        time array
    T : 1D array
        Difference in temperature
    x : 1D array
        Energy generation
    load : 1D array
        Energy demand
    mass_salt : int
        Amount of salt available to the reactor
    Cp : int
        Heat capacity of salt
        
    Returns:
    --------
    ODE : 1D array
        Difference between generation and demand
    '''

    # Energy difference between load and generation is handled by TES
    return 3.6e9*(x - load)/(mass_salt*Cp)

def model(gen, time, load, cfg):
    '''Models the total cost of the system based on energy demand (load?), 
    a time interval, and how much energy is generated. This is a penalty 
    method and includes the constraints directly in the objective cost.
    
    Params:
    --------
    gen : 1D array
        represents energy to generate at each point in time
    time : 1D array
        time intervals
    load : 1D array
        energy demand at each point in time
    cfg : dict
        a dict of system paramters
    
    Returns:
    ---------
    cost_total : int
        total cost of running the system
    T_hist : 2D array
        Temperature of reactor at each point in time
    
    '''
    mass_salt = cfg['mass_salt']
    cost_blackout = cfg['cost_blackout']
    cost_oversupply = cfg['cost_oversupply']
    tes_min_t = cfg['tes_min_t']
    tes_max_t = cfg['tes_max_t']
    cost_nuclear = cfg['cost_nuclear']
    cost_ramp = cfg['cost_ramp']

    T_next = 350  # K
    T_hist = []
    cost_total = 0

    for i in range(len(time)):
        # Get next temperature by integrating difference between 
        # generation and demand
        step = odeint(thermal_storage, T_next, [0, 1],
                      args=(gen[i], load[i], mass_salt, cfg['Cp']))
        T_next = step[1][0]

        # Constraints using a penalty method
        if T_next < tes_min_t: # TES lower temp limit
            cost_total += cost_blackout*(tes_min_t-T_next)
            T_next = tes_min_t

        if T_next > tes_max_t: # TES upper temp limit
            cost_total += cost_oversupply*(T_next-tes_max_t)
            T_next = tes_max_t

        if i > 0 and abs(gen[i] - gen[i-1]) > cfg['max_ramp_rate']: # ramp rate limit
            cost_total += cfg['cost_overramp'] * (abs(gen[i] - gen[i-1]) - cfg['max_ramp_rate'])

        T_hist.append(T_next)

    # Include cost of ramping the reactor
    for i, val in enumerate(gen[:-1]):
        cost_total += cost_ramp * abs(val - gen[i+1])
    cost_total += np.sum(gen*cost_nuclear)
    return cost_total, T_hist

def model_obj_only(gen, cfg):
    '''Model objective without calculating temperatures.'''
    cost_nuclear = cfg['cost_nuclear']
    cost_ramp = cfg['cost_ramp']

    # Include cost of power generation
    cost_total = np.sum(gen*cost_nuclear)

    # Include cost of ramping the reactor
    for i, val in enumerate(gen[:-1]):
        cost_total += cost_ramp*abs(val - gen[i+1])

    return cost_total

def get_T(gen, time, load, cfg):
    '''General equation for getting the temperature list.'''
    
    mass_salt = cfg['mass_salt']  # kg of salt for thermal energy storage
    Cp = cfg['Cp']  # J/kg K, heat capacity of the salt
    T_next = 350  # K
    
    T_hist = []
    
    for i in range(len(time)):
        # Get next temperature by integrating difference between 
        # generation and demand

        step = odeint(thermal_storage, T_next, [0, 1],
                      args=(gen[i], load[i], mass_salt, Cp))
        T_next = step[1][0]

        T_hist.append(T_next)
        
    return T_hist

# Scipy wants constraints like this - other frameworks might let you use bounds
def model_con_max_T(gen, time, load, cfg):

    tes_max_t = 700
    inequalities = []
    
    T_hist = get_T(gen, time, load, cfg)
    
    resolution = 1e-6       # optional   
    for temp in T_hist:
        inequalities.append(tes_max_t - temp - resolution)  # >= 0
        
    return inequalities

def model_con_min_T(gen, time, load, cfg):
    
    tes_min_t = 300
    inequalities = []
    
    T_hist = get_T(gen, time, load, cfg)
    
    resolution=1e-6       # optional
    for temp in T_hist:
        inequalities.append(temp - tes_min_t + resolution)
        
    return inequalities

def model_con_max_ramp(gen, cfg):
    'A constraint to ensure the reactor does not ramp too quickly'
    cost_ramp = cfg['cost_ramp']
    max_ramp_rate = cfg['max_ramp_rate']
    inequalities = []
    for i, val in enumerate(gen[:-1]):
        inequalities.append(max_ramp_rate - abs(val - gen[i+1]))
    return inequalities


def obj(gen, time, load, cfg):
    '''Wrapper to minimize cost only.'''
    return model(gen, time, load, cfg)[0]


if __name__ == "__main__":
    time, net_load = utils.get_data(config['month'], config['year'])

    guess = np.ones(len(time))*config['capacity']*0.95

    # Optimize generation to minimize cost
    cons = [
        {
            'type': 'ineq',
            'fun': model_con_max_T,
            'args': [time, net_load, config]
        },
        {
            'type': 'ineq',
            'fun': model_con_min_T,
            'args': [time, net_load, config]
        },
        {
            'type': 'ineq',
            'fun': model_con_max_ramp,
            'args': [config]
        }
    ]

    opts = {'maxiter': 10000}

    # Penalized Nelder-Mead method
    sol = minimize(obj, guess, method='Nelder-Mead', args=(time, net_load, config), options=opts)
    utils.gen_report([sol['x'], sol['nfev']], 'Nelder-Mead', 'Penalized', 
                        config, gen_plot=True, guess=guess)
    
    # Constrained SLSQP Method
    sol = minimize(model_obj_only, guess, constraints=cons, method='SLSQP', args=(config))
    utils.gen_report([sol['x'], sol['nfev']], 'SLSQP', 'Constrained', 
                        config, gen_plot=True, guess=guess)
    
    # Penalized SLSQP Method
    sol = minimize(obj, guess, method='SLSQP', args=(time, net_load, config), options=opts)
    utils.gen_report([sol['x'], sol['nfev']], 'SLSQP', 'Penalized', 
                        config, gen_plot=True, guess=guess)