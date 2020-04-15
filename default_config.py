config = {
    'cost_nuclear': 0.021,      # $/KWh
    # $, cost of not supplying sufficient power, for penalty method
    'cost_blackout': 1e10,
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
    'guess_coef': 54000*0.85,   # Initial guess (multiplied by an array of 1s)
    'T0': 350                   # Initial temperature K
}
