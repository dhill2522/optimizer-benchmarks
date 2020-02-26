%% Initialize and import data
clc;clear;
nuclear_capacity  = 54000;
percent_operation = 0.96;
load('load.mat');
loads = ERCOT;
time  = linspace(0,23,24);  % have not figure out how to import time into matlab

%% Start optimization
guess   = nuclear_capacity * percent_operation * ones(length(time),1);
options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...  % choose one of: 'interior-point', 'sqp', 'active-set', 'trust-region-reflective'
        'Display', 'iter-detailed', ...  % display more information
        'MaxIterations', 1000, ...  % maximum number of iterations
        'MaxFunctionEvaluations', 1e8, ...  % maximum number of function calls
        'StepTolerance',1e-20, ...
        'OptimalityTolerance', 1e-20, ...  % convergence tolerance on first order optimality
        'Diagnostics', 'on'); % display diagnotic information
[gen_opt,cost_opt] = fminunc(@(gen)obj(gen,time,loads),guess,options);

%% Plotting
[cost_compare, T_compare] = model(guess, time, loads);
[cost_opt, T_opt] = model(gen_opt, time, loads);

figure;
box on;
hold on;
plot(time,T_compare,'r-','LineWidth',1.5);
plot(time,T_opt,'b-','LineWidth',1.5);
legend(gca,'TES Comparison', 'TES optimized');

figure;
box on;
hold on;
plot(time,guess,'r-','LineWidth',1.5);
plot(time,gen_opt,'b-','LineWidth',1.5);
plot(time,loads,'g-','LineWidth',1.5);
legend(gca,'Nuclear Comparison','Nuclear optimized','Load');
