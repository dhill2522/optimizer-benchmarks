%% Initialize and import data
clc;clear;
nuclear_capacity  = 54000;
percent_operation = 0.95;
load('load.mat');
load('solar.mat');
load('wind.mat');
net_load = loads - solar - wind;
time  = linspace(1,24,24);  % have not figure out how to import time into matlab

%% Start optimization
A   = [];
b   = [];
Aeq = [];
beq = [];
lb  = [];
ub  = [];
nonlcon = @(gen)non_linear_constraints(gen,time,net_load);
%nonlcon = [];
guess   = nuclear_capacity*percent_operation*ones(length(time),1);
options = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...  % choose one of: 'interior-point', 'sqp', 'active-set', 'trust-region-reflective'
        'Display', 'iter-detailed', ...  % display more information
        'MaxIterations', 10000, ...  % maximum number of iterations
        'MaxFunctionEvaluations', 1e6, ...  % maximum number of function calls
        'StepTolerance',1e-9, ...
        'OptimalityTolerance', 1e-9, ...  % convergence tolerance on first order optimality'ConstraintTolerance', 1e-16, ...  % convergence tolerance on constraints
        'FiniteDifferenceType', 'forward', ...  % if finite differencing, can also use central
        'ScaleProblem', true, ...
        'Diagnostics', 'on'); % display diagnotic information
[gen_opt, cost_opt, exitflag, ~] = fmincon(@obj_constrained,guess,A,b,Aeq,beq,lb,ub,nonlcon,options);

%% Calculate temperature and rate
rate = zeros(length(gen_opt)-1, 1);
for i = 1:length(gen_opt)-1
   rate(i) = abs(gen_opt(i+1) - gen_opt(i)); 
end
tes_min_t = 300;
tes_max_t = 700;
T_hist_init = get_T(guess,time,net_load);
T_hist_opt = get_T(gen_opt,time,net_load);
lower_b   = tes_min_t * ones(length(T_hist_init),1);
upper_b   = tes_max_t * ones(length(T_hist_init),1);

% figure;hold on;box on;
% plot(time,net_load,'LineWidth',2);
% plot(time,guess,'LineWidth',2);
% plot(time,gen_opt,'LineWidth',2);
% legend('Net load','Nuclear initial', 'Nuclear optimized','Location','best')
% xlabel('Time');
% ylabel('Energy (MW)')
% 
% figure;hold on;box on;
% plot(time, T_hist_init,'LineWidth',2);
% plot(time, T_hist_opt,'LineWidth',2);
% plot(time, upper_b, 'k--','LineWidth',1);
% plot(time, lower_b, 'k--','LineWidth',1);
% legend('TES initial','TES optimized', 'Temperature Bound', 'Location','best')
% xlabel('Time');
% ylabel('Temperature (K)')

figure;
x0=10;
y0=10;
width=550;
height=550;
set(gcf,'units','points','position',[x0,y0,width,height])
x1 = subplot(2,1,1);hold on;box on;
time = time';
plot(time,net_load,'LineWidth',2);
plot(time,guess,'LineWidth',2);
plot(time,gen_opt,'LineWidth',2);
hold off;
set(gca,'XTickLabel',[]);
%xlim(x1,[0 24]);
legend('Net load','Nuclear initial', 'Nuclear optimized','Location','best')
ylabel('Energy (MW)')
x2 = subplot(2,1,2);hold on; box on;
plot(time, T_hist_init,'LineWidth',2);
plot(time, T_hist_opt,'LineWidth',2);
plot(time, upper_b, 'k--','LineWidth',1);
plot(time, lower_b, 'k--','LineWidth',1);
hold off;
%xlim(x2,[0 24]);
legend('TES initial','TES optimized', 'Temperature Bound', 'Location','best')
xlabel('Time');
ylabel('Temperature (K)')
p1 = get(x1, 'Position');
p2 = get(x2, 'Position');
p1(2) = p2(2)+p2(4)+ 0.02;
set(x1, 'pos', p1);
