%clc;clear;

nuclear_capacity  = 54000;
percent_operation = 0.95;
load('load.mat');
loads = ERCOT;
time  = linspace(0,23,24);  % have not figure out how to import time into matlab
gen   = nuclear_capacity * percent_operation * ones(length(time),1);

[cost_total, T_hist] = model(gen, time, loads);
load('python_T_hist.mat')

figure;
hold on;
box on;
plot(time, T_hist,'Linewidth',1.5);
plot(time,python,'Linewidth',1.5);
legend(gca,'Matlab','Python');
