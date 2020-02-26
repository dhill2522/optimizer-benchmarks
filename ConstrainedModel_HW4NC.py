'''To see how the constrained model performs compared to the penalty method model 
run `main_con()` then `main_uncon()`.

'''
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
        T_next = step[1][0]

        # Constraints - consider constrained optimization?
        if T_next < tes_min_t:
            if verbose:
                print('Warning: TES too cold.')
            cost_total += cost_blackout*(tes_min_t-T_next)      # FIXME cost_total becomes a list?
            T_next = tes_min_t

        if T_next > tes_max_t:
            if verbose:
                print('Warning: TES too hot.')
                print(T_next)
            cost_total += cost_oversupply*(T_next-tes_max_t)
            T_next = tes_max_t

        T_hist.append(T_next)

    cost_total += np.sum(gen*cost_nuclear)

    return cost_total, T_hist                          

def model_constrained(gen, time, load, verbose=False):
    mass_salt = 6e8  # kg of salt for thermal energy storage
    cost_nuclear = 0.021  # $/KWh
    cost_salt = 10.98    # $/kg
    T_next = 350  # K
    Cp = 1530  # J/kg K, heat capacity of the salt
    T_hist = []

    cost_total = cost_salt*mass_salt

    for i in range(len(time)):
        # Get next temperature by integrating difference between 
        # generation and demand

        step = odeint(thermal_storage, T_next, [0, 1],
                      args=(gen[i], load[i], mass_salt, Cp))
        T_next = step[1][0]

        T_hist.append(T_next)

    cost_total += np.sum(gen*cost_nuclear)

    return cost_total, T_hist                          

def con_max_temp(X):
    tes_max_t = 700
    inequalities = []
    
    T = model_constrained(X, time, load)[1]
    resolution = 1e-8
    
    for temp in T:
        inequalities.append(tes_max_t - temp - resolution)
        
    return inequalities

def con_min_temp(X):
    tes_min_t = 300
    inequalities = []
    resolution = 1e-8
    
    T = model_constrained(X, time, load)[1]
    
    for temp in T:
        inequalities.append(temp - tes_min_t + resolution)
        
    return inequalities

def obj(gen, time, load):
    '''Wrapper to minimize cost only.'''
    return model(gen, time, load)[0]

def obj_constrained(gen, time, load):
    return model_constrained(gen, time, load)[0]

def get_data(date):
    file = pd.read_csv("data/ERCOT_load_2019.csv")

    dates = []
    times = []
    for hr in file["HourEnding"]:
        split = hr.split(' ')
        dates.append(split[0])
        times.append(split[1])

    file["Date"] = dates
    file["Time"] = times

    data = file.loc[file["Date"] == date]
    data = data.reset_index(drop=True)

    return data

def optimize_only():
    my_date = "2019-10-01"

    data = get_data(my_date)

    time = data['HourEnding']
    load = data['ERCOT']

#    hours = data["Time"]

    nuclear_capacity = 54000

    # Optimize generation to minimize cost
    
    constraints = ({'type':'ineq', 'fun':con_max_temp}, {'type':'ineq', 'fun':con_min_temp})
    
    guess = np.ones(len(time))*nuclear_capacity*0.95
    sol = minimize(obj_constrained, guess, args=(time, load), method='SLSQP', constraints = constraints)


    print('Success:', sol['success'])
    print('x: ', sol['x'])
    print(sol)
    
    return sol


def main_uncon():
    # Create the connection to the database
    # con = create_engine('sqlite:///data/ercot_data.db')

    # # Create the sql query
    # query = "SELECT * FROM Load WHERE HourEnding BETWEEN date('2019-10-01') AND date('2019-10-02')"

    # # Load the data from the database
    # data = pd.read_sql(query, con, parse_dates=['HourEnding'])

    my_date = "2019-10-02"

    data = get_data(my_date)

    time = data['HourEnding']
    load = data['ERCOT']

#    hours = data["Time"]

    nuclear_capacity = 54000

    # Optimize generation to minimize cost
    guess = np.ones(len(time))*nuclear_capacity*0.95
    sol = minimize(obj, guess, args=(time, load), method='Nelder-Mead')


    print('Success:', sol['success'])
    print('x: ', sol['x'])
    print(sol)

    # Check out results
    cost, T_hist = model(sol['x'], time, load, verbose=True)
    print(f'Cost optimized: ${cost}')

    guess = np.ones(len(time))*nuclear_capacity*0.97
    cost_compare, T_hist_compare = model(guess, time, load, verbose=True)
    print(f'Cost comparison: ${cost_compare}')

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9,7))
    ax1.set_title(f"Unconstrained {my_date}")

    # plt.subplot(211)
    ax1.plot(time, load, label='Load')
    ax1.plot(time, sol['x'], label='Nuclear optimized')
    ax1.plot(time, guess, label='Nuclear Comparison')
    ax1.set_ylabel('Energy (MW)')
    ax1.legend()
    # plt.subplot(212)
    ax2.plot(time, T_hist, label='TES optimized')
    ax2.plot(time, T_hist_compare, label='TES Comparison')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_xlabel('Time')
    ax2.legend()

    plt.show()
    
def main_con():
    global time, load
    # Create the connection to the database
    # con = create_engine('sqlite:///data/ercot_data.db')

    # # Create the sql query
    # query = "SELECT * FROM Load WHERE HourEnding BETWEEN date('2019-10-01') AND date('2019-10-02')"

    # # Load the data from the database
    # data = pd.read_sql(query, con, parse_dates=['HourEnding'])

    my_date = "2019-10-04"

    data = get_data(my_date)

    time = data['HourEnding']
    load = data['ERCOT']

#    hours = data["Time"]

    nuclear_capacity = 54000
    
    constraints = ({'type':'ineq', 'fun':con_max_temp}, {'type':'ineq', 'fun':con_min_temp})

    # Optimize generation to minimize cost
    guess = np.ones(len(time))*nuclear_capacity*0.95
    sol = minimize(obj_constrained, guess, args=(time, load), method='SLSQP', constraints=constraints)


    print('Success:', sol['success'])
    print('x: ', sol['x'])
    print(sol)

    # Check out results
    cost, T_hist = model(sol['x'], time, load, verbose=True)
    print(f'Cost optimized: ${cost}')

    guess = np.ones(len(time))*nuclear_capacity*0.97
    cost_compare, T_hist_compare = model(guess, time, load, verbose=True)
    print(f'Cost comparison: ${cost_compare}')

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(9,7))
    ax1.set_title(f"Constrained {my_date}")

    # plt.subplot(211)
    ax1.plot(time, load, label='Load')
    ax1.plot(time, sol['x'], label='Nuclear optimized')
    ax1.plot(time, guess, label='Nuclear Comparison')
    ax1.set_ylabel('Energy (MW)')
    ax1.legend()
    # plt.subplot(212)
    ax2.plot(time, T_hist, label='TES optimized')
    ax2.plot(time, T_hist_compare, label='TES Comparison')
    ax2.set_ylabel('Temperature (K)')
    ax2.set_xlabel('Time')
    ax2.legend()

    plt.show()
    
if __name__ == "__main__":
    main_con()

    