from sqlalchemy import create_engine
import pandas as pd

from ScipyBaseModel import model, get_T, model_con_max_ramp
from os.path import exists
import datetime

# import csv

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

def gen_report(out, optimizer, opt_type, config, notes="", filetype='csv', date=None, gen_plot=False):
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

    