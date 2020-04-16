function [c,ceq] = non_linear_constraints(gen, time, loads)
% Obtain the temperature profile
T_hist = get_T(gen, time, loads);
% Set up upper bound and lower bound
tes_min_t = 300;
tes_max_t = 700;
num_var   = length(T_hist);
lower_b   = tes_min_t * ones(num_var,1);
upper_b   = tes_max_t * ones(num_var,1);
max_rate  = 2000;
% Set up temperature inequality constraints
c1  = T_hist - upper_b;
c2  = lower_b - T_hist;
% Set up ramping speed constraints
rate = zeros(length(gen)-1, 1);
for i = 1:length(gen)-1
   rate(i) = abs(gen(i+1) - gen(i)); 
end
upper_rate = max_rate * ones(length(gen)-1, 1);
%lower_rate = -max_rate * ones(length(gen)-1 ,1);
%c3  = lower_rate - rate;
c4  = rate - upper_rate;
% Combine constraints together
c   = [c1;c2;c4];
ceq = [];
end