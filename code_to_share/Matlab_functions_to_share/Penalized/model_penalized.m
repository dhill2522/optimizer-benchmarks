function [cost_total, T_hist] = model_penalized(gen, time, loads)
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
    global funcall;
    cost_nuclear    = 0.021; % $/kWh
    cost_blackout   = 1E10;
    cost_oversupply = 1E10;
    T_next = 350;
    Cp     = 1530;
    mass_salt = 6e8;
    T_hist = [];
    tes_min_t  = 300;
    tes_max_t  = 700;
    cost_total = 0;
    cost_ramp  = 1;
    t_span = linspace(0,1,3);
    for i = 1:length(time)
       % Get next temperature by integrating difference between 
       % generation and demand 
       [~, T_new] = ode45(@(t,T)thermal_storage(t, T, gen(i), loads(i), mass_salt, Cp), t_span, T_next);
       T_next = T_new(3);
       if T_next < tes_min_t
           cost_total = cost_total + cost_blackout*(tes_min_t - T_next);
           T_next     = tes_min_t;
       end
       
       if T_next > tes_max_t
           cost_total = cost_total + cost_oversupply*(T_next - tes_max_t);
           T_next     = tes_max_t;
       end
       T_hist = [T_hist;T_next];
       
    end
    % include cost for ramping reactor
    for j = 1:length(gen)-1
        cost_total = cost_total + abs(gen(j+1) - gen(j))*cost_ramp;
    end
    cost_total = cost_total + sum(gen * cost_nuclear);
    funcall = funcall + 1;
end