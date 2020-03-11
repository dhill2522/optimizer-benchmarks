from gekko import GEKKO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./data/ERCOT_load_2019.csv")
nhrs = 50  # len(data['HourEnding'].values)
n = len(data['HourEnding'].values[0:nhrs])  # hours for whole year

######################Set up energy arrays for renewable and consumption
#######This section can be switched out with known or predicted energy arrays
t = np.linspace(0, n-1, n)  # time in hours
#####################    Initiate Gekko Model
m = GEKKO()
m.time = t

load = m.Param(value=data['WEST'].values[0:nhrs])
Cp = m.Param(value=1530)  # Heat Capacity in J/kg K

################### Normal nuclear reactor capacities
nuclear_capacity = 1280 * 2 # MW, capacity of South Texas Nuclear Generating station
nuclear_ramprate = .01  # normal ramp rate for a reactor

################# Initiate nuclear variable
mass = m.FV(value=6e6, lb=0)
mass.STATUS = 0
gen_nuclear = m.MV(value=1282.5, lb=0)
gen_nuclear.STATUS = 1
gen_nuclear.DMAX = nuclear_capacity * nuclear_ramprate

########## Thermal Storage in kg This does not take into account phase change
T = m.Var(value=300, lb=260, ub=550)

cost_nuclear = m.Param(value=.021)  # dollars per KWh
cost_salt = m.Param(value=10.98)  # dollars per kg
total_cost = m.Intermediate(gen_nuclear*cost_nuclear*1000 + mass*cost_salt)

m.Equation(T.dt() == 3.6e9*(gen_nuclear - load)/(mass*Cp))
m.Obj(total_cost)

######### Solve Model
m.options.IMODE = 5
m.options.SOLVER = 1
m.solve()

######## Plot Model
plt.subplot(3, 1, 1)
plt.plot(t, load, 'r-', label='gen need')
plt.plot(t, gen_nuclear.value, 'b--', label='Nuclear power')
plt.ylabel('Energy')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, T.value, '--', color='orange',
         label=f'mass of molten salt {np.round(mass.value[-1],1):.2e}kg')
plt.ylabel('Temperature')
plt.legend(loc='lower right')

plt.subplot(3, 1, 3)
#plt.plot(t,gen_solar,'--', label = 'solar power')
#plt.plot(t,gen_wind, '--', label = 'wind power')
plt.plot(t, load.value, 'r-', label='consumption')
plt.ylabel('Energy')
plt.xlabel('Time(s) For 2 days')
plt.legend()
plt.show()
