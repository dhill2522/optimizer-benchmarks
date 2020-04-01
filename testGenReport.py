import numpy as np

from utils import gen_report, results
from ScipyBaseModel import config

xstar = guess = np.ones(24)*config['capacity']*0.95

fevals = 1200

out = [xstar, fevals]

print(results(xstar, config))
gen_report(out, optimizer="Test", opt_type="Penalty", config=config, notes="testing")