%% Initialize and import data
clc;clear;
nuclear_capacity  = 54000;
percent_operation = 0.95;
load('load.mat');
loads = ERCOT;
time  = linspace(0,23,24);  % have not figure out how to import time into matlab

%% Start optimization
A   = [];
b   = [];
Aeq = [];
beq = [];
lb  = [];
ub  = [];
%nonlcon = @(gen)non_linear_constraints(gen,time,loads);
nonlcon = [];
guess   = nuclear_capacity * percent_operation * ones(length(time),1);
options = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...  % choose one of: 'interior-point', 'sqp', 'active-set', 'trust-region-reflective'
        'Display', 'iter-detailed', ...  % display more information
        'MaxIterations', 800, ...  % maximum number of iterations
        'MaxFunctionEvaluations', 1e6, ...  % maximum number of function calls
        'StepTolerance',1e-16, ...
        'OptimalityTolerance', 1e-6, ...  % convergence tolerance on first order optimality
        'ConstraintTolerance', 1e-6, ...  % convergence tolerance on constraints
        'FiniteDifferenceType', 'forward', ...  % if finite differencing, can also use central
        'Diagnostics', 'on'); % display diagnotic information
[gen_opt, cost_opt, exitflag, ~] = fmincon(@(gen)obj(gen,time,loads),guess,A,b,Aeq,beq,lb,ub,nonlcon,options);

%% Plotting
load('python_optimized.mat');
[cost_python,T_python] = model(Python_opt, time, loads);
[cost_compare, T_compare] = model(guess, time, loads);
[cost_opt, T_opt] = model(gen_opt, time, loads);

figure;
box on;
hold on;
plot(time,T_compare,'r-','LineWidth',1.5);
plot(time,T_opt,'b-','LineWidth',1.5);
plot(time,T_python,'g-', 'LineWidth', 1.5)
legend(gca,'TES Comparison', 'Matlab optimized','Python optimized');

figure;
box on;
hold on;
plot(time,guess,'r-','LineWidth',1.5);
plot(time,gen_opt,'b-','LineWidth',1.5);
plot(time,e04,'y-','LineWidth',1.5);
plot(time,loads,'g-','LineWidth',1.5);
legend(gca,'Nuclear Comparison','Nuclear optimized Matlab','Nuclear optimized Python','Load');
