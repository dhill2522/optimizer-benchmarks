function [c,ceq] = T_constraints(gen, time, loads)
    % Obtain the temperature profile
    T_hist = get_T(gen, time, loads);
    % Set up upper bound and lower bound
    tes_min_t = 300;
    tes_max_t = 700;
    num_var   = length(T_hist);
    lower_b   = tes_min_t * ones(num_var,1);
    upper_b   = tes_max_t * ones(num_var,1);
    % Set up inequality constraints
    c1  = T_hist - upper_b;
    c2  = lower_b - T_hist;
    c   = [c1;c2];
    ceq = [];
end