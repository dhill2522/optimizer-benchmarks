import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
from sqlalchemy import create_engine

# Makes matplotlib happy plotting pandas data arrays
pd.plotting.register_matplotlib_converters()


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


def model(gen, time, load, verbose=False):
    '''Models the total cost of the system based on energy demand (load?), 
    a time interval, and how much energy is generated.
    
    Params:
    --------
    gen : 1D array
        represents energy to generate at each point in time
    time : 1D array
        time intervals
    load : 1D array
        energy demand at each point in time
    verbose : bool
        prints warning messages
    
    Returns:
    ---------
    cost_total : int
        total cost of running the system
    T_hist : 2D array
        Temperature of reactor at each point in time
    
    '''
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
        # Get next temperature by integrating difference between 
        # generation and demand

        step = odeint(thermal_storage, T_next, [0, 1],
                      args=(gen[i], load[i], mass_salt, Cp))
        T_next = step[1]

        # Constraints - consider constrained optimization?
        if T_next < tes_min_t:
            if verbose:
                print('Warning: TES too cold.')
            cost_total += cost_blackout*(tes_min_t-T_next)      # FIXME cost_total becomes a list?
            T_next = tes_min_t

        if T_next > tes_max_t:
            if verbose:
                print('Warning: TES too hot.')
            cost_total += cost_oversupply*(T_next-tes_max_t)
            T_next = tes_max_t

        T_hist.append(T_next)

    cost_total += np.sum(gen*cost_nuclear)

    return cost_total, T_hist                          # FIXME T_hist is 2D, not necessary?


def obj(gen, time, load):
    '''Wrapper to minimize cost only.'''
    return model(gen, time, load)[0]


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

    nuclear_capacity = 54000
    guess = np.ones(len(time))*nuclear_capacity*0.95

    # Optimize generation to minimize cost
    sol = minimize(obj, guess, args=(time, net_load), method='SLSQP')
    print('Success:', sol['success'])
    print('x: ', sol['x'])
    print(sol)

    cost, T_hist = model(sol['x'], time, net_load, verbose=True)
    print(f'Cost optimized: ${cost}')

    guess = np.ones(len(time))*nuclear_capacity*0.97
    cost_compare, T_hist_compare = model(guess, time, net_load, verbose=True)
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
