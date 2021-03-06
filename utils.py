"""Helper functions for NHES optimization"""

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import exists
from sqlalchemy import create_engine
from default_config import config

import ScipyBaseModel as spbm
# from ScipyBaseModel import model, get_T, model_con_max_ramp


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
    net_load = data['Load'] - data['Wind'] - data['Solar']
    return time, net_load

def results(xstar, config=config):
    '''Generate results from the NHES optimization.'''

    time, net_load = get_data(config['month'], config['year'])

    # Get optimized cost
    fstar = spbm.model(xstar, time, net_load, config)[0]

    # Get temperature violations
    T = spbm.get_T(xstar, time, net_load, config)

    T_violations = 0
    for temp in T:
        if temp > config['tes_max_t']:
            T_violations += abs(temp-config['tes_max_t'])
        elif temp < config['tes_min_t']:
            T_violations += abs(temp-config['tes_min_t'])

    # Get ramping violations
    ramp = spbm.model_con_max_ramp(xstar, config)

    ramp_violations = 0
    for val in ramp:
        if val < 0:
            ramp_violations += abs(val)

    return {'fstar':fstar, 'T':T, 'T_violations':T_violations, 'ramp_violations':ramp_violations}

def gen_report(out, optimizer, opt_type, config=config, notes="", filetype='csv', date=None, gen_plot=False, guess=None):
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
    print(f"Data saved at: {report_path}")

    if gen_plot:
        plt.figure()
        time, net_load = get_data(config['month'], config['year'])
        T_hist = spbm.get_T(xstar, time, net_load, config)
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
            T_hist_compare = spbm.get_T(guess, time, net_load, config)
            plt.plot(time.values, T_hist_compare, label='TES Initial')
        plt.ylabel('Temperature (K)')
        plt.xlabel('Time')
        plt.legend(loc='upper left')
        plt.gcf().autofmt_xdate()
        plt.savefig(f'saved_plots/{optimizer}-{opt_type}-{date}.png')
        print(f"Figure saved at: saved_plots/{optimizer}-{opt_type}-{date}.png")
        plt.show()

def save_iters(x, savepath):
    all_iters = np.array(x).T
    df = pd.DataFrame(all_iters)
    df.to_csv(savepath, index=False)