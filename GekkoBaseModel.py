from gekko import GEKKO
import numpy as np
from utils import get_data, config, gen_report

time, net_load = get_data(config['month'], config['year'])
n = len(time)

m = GEKKO(remote=True)
m.time = np.linspace(0, n-1, n)

load = m.Param(value=net_load.values)
Cp = m.Param(value=config['Cp'])
mass = m.Param(value=config['mass_salt'])
cost = m.Var(value=0)

gen_nuclear = m.MV(value=config['guess_coef'], lb=0, ub=config['capacity'])
# gen_nuclear.STATUS = 1
gen_nuclear.DMAX = config['max_ramp_rate']

T = m.Var(value=config['T0'], lb=config['tes_min_t'], ub=config['tes_max_t'])

m.Equation(cost == 3.6e9*(gen_nuclear - load)/(mass*Cp))
#   total_cost = m.Intermediate(gen_nuclear*config['cost_nuclear']) # FIXME: Need to Add ramping costs

m.Equation(T.dt() == 1*(gen_nuclear - load)/(mass*Cp))
# m.Equation(T.dt() == 3.6e9*(gen_nuclear - load)/(mass*Cp))
m.Obj(cost)

m.options.IMODE = 6
m.options.SOLVER = 3 # 1: APOPT, 2: BPOPT, 3: IPOPT
m.solve()

xstar = np.array(gen_nuclear.VALUE.value)

gen_report([xstar, 100], 'Gekko IPOPT', 'Constrained', 
            config=config, notes='Test, not properly constrained', gen_plot=True)
