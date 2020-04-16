function [cost_total] = obj_constrained(gen)
    %{
    Model the total cost of the system based on energy demand,
    time interval, and how much energy is generated
    Parameters:
    ----------
    gen   : 1D column array, energy generated at each time point
    time  : 1D column array, time interval
    loads : 1D column array, energy demand at each time point

    Outputs:
    ----------
    cost_total : integer, total cost of running the system
    T_hist : 1D array, Temperature of reactor at each time point
    %}
    cost_nuclear = 0.021; % $/kWh
    cost_ramp    = 1;
    cost_total   = 0;
    % include cost for ramping reactor
    for j = 1:length(gen)-1
        cost_total = cost_total + abs(gen(j+1) - gen(j))*cost_ramp;
    end
    cost_total = cost_total + sum(gen * cost_nuclear);
    
end