function T_hist = get_T(gen, time, loads)
    mass_salt       = 6E8;   % kg of salt for thermal energy storage
    T_next = 350;
    Cp     = 1530;
    T_hist = [];
    t_span = linspace(0,1,3);
    for i = 1:length(time)
       % Get next temperature by integrating difference between 
       % generation and demand 
       [~, T_new] = ode45(@(t,T)thermal_storage(t, T, gen(i), loads(i), mass_salt, Cp), t_span, T_next);
       T_next = T_new(3);
       T_hist = [T_hist;T_next];
    end
end