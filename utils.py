import datetime
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from sqlalchemy import create_engine

from ScipyBaseModel import model, get_T, model_con_max_ramp

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

def get_data(month, year):
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
    return time, net_load

def results(xstar, config):
    '''Generate results from the NHES optimization.'''

    time, net_load = get_data(config['month'], config['year'])

    # Get optimized cost
    fstar = model(xstar, time, net_load, config)[0]

    # Get temperature violations
    T = get_T(xstar, time, net_load, config)

    T_violations = 0
    for temp in T:
        if temp > config['tes_max_t']:
            T_violations += abs(temp-config['tes_max_t'])
        elif temp < config['tes_min_t']:
            T_violations += abs(temp-config['tes_min_t'])

    # Get ramping violations
    ramp = model_con_max_ramp(xstar, config)

    ramp_violations = 0
    for val in ramp:
        if val < 0:
            ramp_violations += val

    return {'fstar':fstar, 'T':T, 'T_violations':T_violations, 'ramp_violations':ramp_violations}

def gen_report(out, optimizer, opt_type, config, notes="", filetype='csv', date=None, gen_plot=False, guess=None):
    '''Save info about a set of optimization results.
    Info for creating LaTex tables from a csv: 
        https://texblog.org/2012/05/30/generate-latex-tables-from-csv-files-excel/

    Inputs:
    -------
    out : array-like, two elements [xstar, fevals]
    optimizer : str, name of optimizer used
    opt_type : str, specify constrained/penalized
    config : dict, parameters used in optimization
    notes : str, additional notes about optimization
    filetype : str, only option right now is csv
    date : str, defaults to YYYY-MM-DD HH:MM:SS
    gen_plot : bool, generate a figure of results
    '''
    # https://texblog.org/2012/05/30/generate-latex-tables-from-csv-files-excel/
    xstar = out[0]
    fevals = out[1]

    res = results(xstar, config)
    fstar = res['fstar']
    T_violations = res['T_violations']
    ramp_violations = res['ramp_violations']

    if date is None:
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d %H:%M:%S")

    # Check for file and generate dataframe
    if filetype == 'csv':
        report_path = "report_data.csv"
        new_data_dic = {"Optimizer":[optimizer], "Type":[opt_type], "Optimized Cost":[fstar], 
                        "Function Calls":[fevals], "Temperature Violation":[T_violations], 
                        "Ramping Violation":[ramp_violations], "Date":[date], "Notes":[notes]}
        new_data_df = pd.DataFrame(new_data_dic)

        if exists(report_path):
            df = pd.read_csv(report_path)
            update = df.append(new_data_df, ignore_index=True)

        else:
            update = new_data_df

        update.to_csv(report_path, index=False)

        # print(update.to_string(index=False))

    if gen_plot:
        time, net_load = get_data(config['month'], config['year'])
        T_hist = get_T(xstar, time, net_load, config)
        plt.subplot(211)
        plt.plot(time.values, net_load, label='Net Load')
        plt.plot(time.values, xstar, label='Nuclear optimized')
        if guess is not None:
            plt.plot(time.values, guess, label='Nuclear Initial')
        plt.ylabel('Energy (MW)')
        plt.legend(loc='upper left')
        plt.subplot(212)
        time_limits = [time.values[0], time.values[-1]]
        plt.plot(time_limits, [config['tes_min_t'], config['tes_min_t']], 
                    '--r', label='Temperature Constraints')
        plt.plot(time_limits, [config['tes_max_t'], config['tes_max_t']], '--r')
        plt.plot(time.values, T_hist, label='Optimized TES')
        if guess is not None:
            T_hist_compare = get_T(guess, time, net_load, config)
            plt.plot(time.values, T_hist_compare, label='TES Initial')
        plt.ylabel('Temperature (K)')
        plt.xlabel('Time')
        plt.legend(loc='upper left')
        plt.gcf().autofmt_xdate()
        plt.savefig(f'saved_plots/{optimizer}-{opt_type}-{date}.png')
        plt.show()