import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
from sqlalchemy import create_engine

# Makes matplotlib happy plotting pandas data arrays
pd.plotting.register_matplotlib_converters()

config = {
    'cost_nuclear': 0.021,      # $/KWh
    'cost_blackout': 1e10,      # $, cost of not supplying sufficient power, for penalty method
    'cost_oversupply': 1e10,    # $, cost of supplying too much power, for penalty method
    'Cp': 1530,                 # J/kg K, heat capacity of the salt
    'tes_min_t': 300,           # K, Minimum temperature of thermal storage unit
    'tes_max_t': 700,           # K, Maximum temperature of thermal storage unit
    'mass_salt': 6e8,           # kg, mass of salt for thermal energy storage
    'nuclear_capacity': 54000,  # MW, Total amount of potential nuclear output
    'cost_ramp': 1,             # $/MW/hr, Cost of ramping up and down the reactor core
    'max_ramp_rate': 1000       # MW/hr, Max rate of ramping the reactor core
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


def model(gen, time, load, cfg, verbose=False):
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
    cfg: dict
        a dict of system paramters
    verbose : bool
        prints warning messages
    
    Returns:
    ---------
    cost_total : int
        total cost of running the system
    T_hist : 2D array
        Temperature of reactor at each point in time
    
    '''
    mass_salt = cfg['mass_salt']
    Cp = cfg['Cp']
    cost_blackout = cfg['cost_blackout']
    cost_oversupply = cfg['cost_oversupply']
    tes_min_t = cfg['tes_min_t']
    tes_max_t = cfg['tes_max_t']
    cost_nuclear = cfg['cost_nuclear']

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

def load_query(year: str, month: str):
    return f'''
        SELECT HourEnding, ERCOT as Load
        FROM Load 
        WHERE HourEnding > datetime("{year}-{month}-01") 
            AND HourEnding < datetime("{year}-{month}-02 01:00:00")
        '''

def gen_query(fuelType: str, year: str, month: str):
    return f'''
        SELECT Generation, Date_Time 
        FROM Generation 
        WHERE Fuel = "{fuelType}" 
            AND Date_Time > datetime("{year}-{month}-01") 
            AND Date_Time < datetime("{year}-{month}-02") 
            AND Resolution = "Hourly"
        '''


if __name__ == "__main__":
    year = '2019'
    month = '10'

    # Create the connection to the database
    con = create_engine('sqlite:///data/ercot_data.db')


    # Load the data from the database
    data = pd.read_sql(load_query(year, month), con, parse_dates=['HourEnding'])
    data['Wind'] = pd.read_sql(gen_query('Wind', year, month), con, parse_dates=[
                               'Date_Time'])['Generation']
    data['Solar'] = pd.read_sql(gen_query('Solar', year, month), con, parse_dates=[
                               'Date_Time'])['Generation']

    time = data['HourEnding']
    load = data['Load']
    net_load = data['Load'] - data['Wind'] - data['Solar']

    guess = np.ones(len(time))*config['nuclear_capacity']*0.95

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
    sol = minimize(obj, guess, method='Nelder-Mead', args=(time, load, config))
    # sol = minimize(model_obj_only, guess, constraints=cons, method='SLSQP', args=(config))
    print(sol)

    cost = model_obj_only(sol['x'], config)
    T_hist = get_T(sol['x'], time, net_load, config)
    print(f'Cost optimized:  ${cost}')

    cost_compare = model_obj_only(guess, config)
    T_hist_compare = get_T(guess, time, net_load, config)
    print(f'Cost comparison: ${cost_compare}')

    plt.subplot(211)
    plt.plot(time, load, label='Load')
    plt.plot(time, net_load, label='Net Load')
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
