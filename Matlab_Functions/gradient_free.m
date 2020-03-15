%% Initialize and import data
clc;clear;
nuclear_capacity  = 54000;
percent_operation = 0.95;
time  = linspace(0,23,24);  % have not figure out how to import time into matlab
load('load.mat');
load('solar.mat');
load('wind.mat');
demand = loads;
net_load = demand - solar - wind;

%% Start optimization
A   = [];
b   = [];
Aeq = [];
beq = [];
lb  = zeros(24,1);
ub  = nuclear_capacity*ones(24,1);
% lb = [];
% ub = [];
nonlcon = @(gen)T_constraints(gen,time,net_load);
nvars   = length(guess);
options = optimoptions('ga', ...
        'Display', 'iter', ...  % display more information
        'InitialPopulationRange', [30000;50000],...
        'PlotFcn',@gaplotbestf); % display diagnotic information
[gen_opt, cost_opt] = ga(@obj,nvars,A,b,Aeq,beq,lb,ub,nonlcon,options);

%%
A   = [];
b   = [];
Aeq = [];
beq = [];
lb  = [];
ub  = nuclear_capacity*ones(24,1);
% lb = [];
% ub = [];
nonlcon = @(gen)T_constraints(gen,time,net_load);
guess   = nuclear_capacity*percent_operation*ones(length(time),1);

options = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...  % choose one of: 'interior-point', 'sqp', 'active-set', 'trust-region-reflective'
        'Display', 'iter-detailed', ...  % display more information
        'MaxIterations', 300, ...  % maximum number of iterations
        'MaxFunctionEvaluations', 1e6, ...  % maximum number of function calls
        'StepTolerance',1e-16, ...
        'OptimalityTolerance', 1e-8, ...  % convergence tolerance on first order optimality'ConstraintTolerance', 1e-16, ...  % convergence tolerance on constraints
        'FiniteDifferenceType', 'forward', ...  % if finite differencing, can also use central 'ScaleProblem', true, ...
        'Diagnostics', 'on'); % display diagnotic information
[gen_opt_fmincon, cost_opt_fmincon] = fmincon(@obj,guess,A,b,Aeq,beq,lb,ub,nonlcon,options);
%% Plotting
cost_compare = obj(guess);
cost_opt     = obj(gen_opt);
cost_opt_fmincon = obj(gen_opt_fmincon);
T_compare    = get_T(guess, time, net_load);
T_opt        = get_T(gen_opt, time, net_load);
T_opt_fmincon = get_T(gen_opt_fmincon, time, net_load);

figure;
box on;
hold on;
%plot(time,T_compare,'r-','LineWidth',1.5);
plot(time,T_opt,'b-','LineWidth',1.5);
plot(time,T_opt_fmincon,'g-', 'LineWidth', 1.5);
legend(gca,'GA optimized', 'Matlab optimized','Location','Best');

figure;
box on;
hold on;
plot(time,gen_opt,'r-','LineWidth',1.5);
plot(time,gen_opt_fmincon,'b-','LineWidth',1.5);
%plot(time,Python_opt,'y-','LineWidth',1.5);
plot(time,net_load,'g-','LineWidth',1.5);
legend(gca,'GA optimized nuclear generation','Fmincon optimized nuclear generation','Net generation demand','Location','Best')
