#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:56:02 2020

@author: nicholascooper

type <pyomo help --solvers> in a shell to see the supported list of solvers

https://towardsdatascience.com/modeling-and-optimization-of-a-weekly-workforce-with-python-and-pyomo-29484ba065bb
https://www.ima.umn.edu/materials/2017-2018.2/W8.21-25.17/26326/3_PyomoFundamentals.pdf

To include differential equations you want to use pyomo.DAE: https://pyomo.readthedocs.io/en/stable/modeling_extensions/dae.html#declaring-differential-equations
"""

from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%%

# def J(var):
#     x = var[0]
#     y = var[1]
#     return (1 - x)**2 + 100*(y - x**2)**2

# # Constraints
# def f1(var):
#     x = var[0]
#     y = var[1]
#     return x**2 + y**2
# def con1(var):
#     return 1-f1(var)

# def f2(var):
#     x = var[0]
#     y = var[1]
#     return x + 3*y
# def con2(var):
#     return 5-f2(var)

# #%%
    
# solvers = ['glpk', 'apopt.py', 'cbc']

# # A little test model - these solvers aren't built for this equaiton, or I'm 
# # doing it wrong
# m = ConcreteModel()

# m.x = Var(initialize=1.5)
# m.y = Var(initialize=1.5)

# def J(m):
#     return (1 - m.x)**2 + 100*(m.y - m.x**2)**2

# m.obj = Objective(rule=J)

# opt = SolverFactory('apopt.py')

# opt.solve(m)
# #%% Tutorial example
# model = ConcreteModel()

# # declare decision variables
# model.x = Var(domain=NonNegativeReals)

# # declare objective
# model.profit = Objective(
#     expr = 40*model.x,
#     sense = maximize)

# # declare constraints
# model.demand = Constraint(expr = model.x <= 40)
# model.laborA = Constraint(expr = model.x <= 80)
# model.laborB = Constraint(expr = 2*model.x <= 100)

# # solve
# SolverFactory('glpk').solve(model)


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

def model(gen, time, load):
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
    mass_salt = 6e8  # kg of salt for thermal energy storage
    cost_nuclear = 0.021  # $/KWh
    cost_salt = 10.98    # $/kg
    T_next = 350  # K
    Cp = 1530  # J/kg K, heat capacity of the salt
    T_hist = []

    cost_total = 0#cost_salt*mass_salt

    for i in range(len(time)):
        # Get next temperature by integrating difference between 
        # generation and demand

        step = odeint(thermal_storage, T_next, [0, 1],
                      args=(gen[i], load[i], mass_salt, Cp))
        T_next = step[1][0]

        T_hist.append(T_next)

    cost_total += np.sum(gen*cost_nuclear)

    return cost_total, T_hist

def gen_cost_constraints(gen, time, load, verbose=False):
    '''Lets us compute the cost if constraints are violated
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

    cost_total = 0#cost_salt*mass_salt

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
            cost_total += cost_blackout*(tes_min_t-T_next)
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

def con_max_temp(X):
    
    T = model(X, time, load)[1]
    
    return T

def con_min_temp(X):

    T = model(X, time, load)[1]
    return T

def con_max_ramp(X):
    '''Max ramp up or down does not exceed 2000 MW/hr'''
    dEdt = []
    for i in range(len(X)-1):
        dEdt.append(abs(X[i+1] - X[i]))
    
    return dEdt

def main():

    global time, load
    my_date = "2019-10-04"
    # day1 = "2019-10-04"
    # day2 = "2019-10-05"

    # data1 = get_data(day1)
    # data2 = get_data(day2)

    # data = data1.append(data2, ignore_index=True)
    data = get_data(my_date)
    
    time = data['HourEnding']
    load = data['ERCOT']
    
    nuclear_capacity = 54000
    mass_salt = 6e8  # kg of salt for thermal energy storage
    cost_salt = 10.98    # $/kg
    base_cost = mass_salt * cost_salt
    # Optimize generation to minimize cost
    # change gen_factor to change the initial starting point
    gen_factor = 0.97
    guess = np.ones(len(time))*nuclear_capacity*gen_factor
    
    TES_max_T = 700
    TES_min_T = 300


    m = ConcreteModel()

    # Variables
    m.GenSet = Set(initialize=guess)
    m.Gen = Var(m.GenSet)

    # Parameters
    m.TimeSet = Set(initialize=time)
    m.Time = Param(m.TimeSet)
    m.LoadSet = Set(initialize=load)
    m.Load = Param(m.LoadSet)

    # Objective
    def pyRule(m):
        return model(m.Gen, m.Time, m.Load)[0]

    # m.obj = Objective(rule=pyRule)

    # # Constraints
    # def pyTempCon(m):
    #     T = model(m.Gen, m.Time, m.Load)[1]
    #     b = all([(temp < TES_max_T and temp > TES_min_T) for temp in T])
    #     return b

    # # or
    # def pyTempConTup(m, i):
    #     T = model(m.Gen, m.Time, m.Load)[1]
    #     return (TES_min_T, T[i], TES_max_T)

    # m.TES = Constraint(rule=pyTempCon)


    # opt = SolverFactory('apopt.py')

    # opt.solve(m)

    # print(opt)
    print(m.Load.extract_values())
    #%% Tutorial example
    # model = ConcreteModel()

    # # declare decision variables
    # model.x = Var(domain=NonNegativeReals)

    # # declare objective
    # model.profit = Objective(
    #     expr = 40*model.x,
    #     sense = maximize)

    # # declare constraints
    # model.demand = Constraint(expr = model.x <= 40)
    # model.laborA = Constraint(expr = model.x <= 80)
    # model.laborB = Constraint(expr = 2*model.x <= 100)

    # # solve
    # SolverFactory('glpk').solve(model)

#def test():

global time, load
my_date = "2019-10-04"
# day1 = "2019-10-04"
# day2 = "2019-10-05"

# data1 = get_data(day1)
# data2 = get_data(day2)

# data = data1.append(data2, ignore_index=True)
data = get_data(my_date)

time = np.linspace(0, len(data['HourEnding'])-1, len(data['HourEnding']))
load = data['ERCOT']
mass_salt = 6e8  # kg of salt for thermal energy storage
cost_nuclear = 0.021  # $/KWh
cost_salt = 10.98    # $/kg
T_next = 350  # K
Cp = 1530  # J/kg K, heat capacity of the salt
T_hist = []

cost_total = 0#cost_salt*mass_salt

TES_max_T = 700
TES_min_T = 300

nuclear_capacity = 54000
gen_factor = 0.95
guess = np.ones(len(time))*nuclear_capacity*gen_factor

#%% Using odeint
m = ConcreteModel()

m.cost_nuclear = Param(initialize=cost_nuclear)
m.mass_salt = Param(initialize=mass_salt)
m.Cp = Param(initialize=Cp)

m.g = Set(initialize=range(len(data["HourEnding"])))
m.t = ContinuousSet(initialize=np.linspace(0, len(time)-1, len(time)))

m.T0 = Param(initialize=350)

def gen(m, i):
    return guess[int(i)]
m.Gen = Var(m.t, bounds=(0, 1e10), initialize=gen)

def load_rule(m, i):
    return np.array(load)[int(i)]
m.Load = Param(m.t, initialize=load_rule)

def Trule(m, i):
    return 500

m.T_next = Param(initialize = 350, mutable=True)

#def Trule(m, i):
#    
#    step = odeint(thermal_storage, m.T_next.value, [0, 1],
#                  args=(m.Gen[i].value, load[i], mass_salt, Cp))
#    
#    m.T_next = step[1][0]
#    m.T[i] = step[1][0]
#    
#    return m.T[i]

m.T = Param(m.t, initialize=350, mutable=True)#Trule)


def thermal_rule(m, i):
#for i in range(len(time)):
    # Get next temperature by integrating difference between 
    # generation and demand

#    step = odeint(thermal_storage, m.T_next.value, [0, 1],
#                  args=(m.Gen[i].value, load[i], mass_salt, Cp))
#    
#    m.T_next = step[1][0]
#    m.T[i] = step[1][0]
    
    T0 = m.T_next.value
    T0 = m.T[i]
    
    T_new = 3.6e9*(m.Gen[i].value-load[i])/mass_salt/Cp + T0
    print(T0, T_new)
    
    
    gen_min = (TES_min_T-T0)*mass_salt*Cp/3.6e9 + load[i]
    gen_max = (TES_max_T-T0)*mass_salt*Cp/3.6e9 + load[i]
    
    m.T_next = T_new
    if i < len(m.T)-1:
        m.T[i+1] = T_new
    
    print(gen_min)
    
    return (gen_min, m.Gen[i], gen_max)

m.thermal=Constraint(m.t, rule=thermal_rule)

def objrule(m):
    return summation(m.Gen)*cost_nuclear

m.Cost = Objective(rule=objrule, sense=1)


opt = SolverFactory('apopt.py')
sol = opt.solve(m, tee='true')

m.display()
print(sol)
#%%
xopt = list(m.Gen.extract_values().values())
plt.plot(time, xopt)
#plt.plot(time, load)
#plt.plot(time, guess)
#%%
T_hist = []
my_T = []
my_min = []
T_next=350
xopt = guess
for i in range(len(xopt)):
    step = odeint(thermal_storage, T_next, [0, 1],
                  args=(xopt[i], load[i], mass_salt, Cp))
    my_T.append(3.6e9*(xopt[i] - load[i])/mass_salt/Cp + T_next)
    my_min.append((TES_min_T-T_next)*mass_salt*Cp/3.6e9 + load[i])
    T_next = step[1][0]
    T_hist.append(T_next)
    
    

T_hist, my_T, my_min

#%% using pyomo.dae
m = ConcreteModel()

m.cost_nuclear = Param(initialize=cost_nuclear)
m.mass_salt = Param(initialize=mass_salt)
m.Cp = Param(initialize=Cp)

m.g = Set(initialize=range(len(data["HourEnding"])))
m.t = ContinuousSet(initialize=np.linspace(0, len(time)-1, len(time)))

m.T0 = Param(initialize=350)

def gen(m, i):
    return guess[int(i)]
m.Gen = Var(m.t, bounds=(0, 1e10), initialize=gen)
m.Gen[0].fix(guess[0])

def load_rule(m, i):
    return np.array(load)[int(i)]
m.Load = Param(m.t, initialize=load_rule)

m.T = Var(m.t, initialize=m.T0, bounds=(TES_min_T, TES_max_T))
m.dT = DerivativeVar(m.T, wrt=m.t)

def thermal_rule(m, t):
    if t == 0:
        return Constraint.Skip
    else:
        return m.dT[t] == 3.6e9*(m.Gen[t] - m.Load[t])/(m.mass_salt*m.Cp)

m.thermal = Constraint(m.t, rule=thermal_rule)

def thermal_con(m, i):
    
#sim = Simulator(m, package='scipy')
#tsim, profiles = sim.simulate(numpoints=24)

discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m, nfe=len(time)-1)

def objrule(m):
    return summation(m.Gen)*m.cost_nuclear

m.Cost = Objective(rule=objrule, sense=1)


opt = SolverFactory('apopt.py')
sol = opt.solve(m, tee='true')

m.display()
print(sol)

#%%
xopt = list(m.Gen.extract_values().values())
plt.plot(time, xopt)
plt.plot(time, load)
plt.plot(time, guess)
#%%
T_hist = []
T_next=350
for i in range(len(xopt)):
    step = odeint(thermal_storage, T_next, [0, 1],
                  args=(xopt[i], load[i], mass_salt, Cp))
    T_next = step[1][0]
    T_hist.append(T_next)

T_hist

dT = 3.6e9*(xopt[0]-load[0])/mass_salt/Cp + 350

def dT_exact(Tmin, Tmax, i):
    gen_min = Tmin*mass_salt*Cp/3.6e9 + load[i]
    gen_max = Tmax*mass_salt*Cp/3.6e9 + load[i]
    
    return gen_min, gen_max

dT_exact(300, 700, 1)

#%% Just penalty model no constraints
m = ConcreteModel()

m.cost_nuclear = Param(initialize=cost_nuclear)
m.mass_salt = Param(initialize=mass_salt)
m.Cp = Param(initialize=Cp)
m.T_next = Param(initialize=350, mutable=True)

m.g = Set(initialize=range(len(data["HourEnding"])))

def gen(m, i):
    return guess[int(i)]
m.Gen = Var(m.g, bounds=(0, 1e10), initialize=gen)

def load_rule(m, i):
    return np.array(load)[int(i)]
m.Load = Param(m.g, initialize=load_rule)

def objrule(m):
    global T_hist
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

    cost_total = 0
    
    for i in range(len(time)):
        # Get next temperature by integrating difference between 
        # generation and demand
        genval = m.Gen[i].value
        loadval = m.Load[i]
        
        T_next = m.T_next.value
        step = odeint(thermal_storage, m.T_next.value, [0, 1],
                      args=(genval, loadval, mass_salt, Cp))
        T_next = step[1][0]

        # Constraints - consider constrained optimization?
        if T_next < tes_min_t:
#            if verbose:
#                print('Warning: TES too cold.')
            cost_total += cost_blackout*(tes_min_t-T_next)
            T_next = tes_min_t

        if T_next > tes_max_t:
#            if verbose:
#                print('Warning: TES too hot.')
#                print(T_next)
            cost_total += cost_oversupply*(T_next-tes_max_t)
            T_next = tes_max_t
            
        m.T_next = T_next
        
        T_hist.append(T_next)

    cost_total += np.sum(m.Gen[i]*cost_nuclear)

    return cost_total

m.Cost = Objective(rule=objrule, sense=1)

opt = SolverFactory('apopt.py')
sol = opt.solve(m, tee='true')

m.display()
print(sol)

#%%



#cost_total += np.sum(gen*cost_nuclear)

#if __name__ == "__main__":
#    # main()
#    test()
