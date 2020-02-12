
function ODE = thermal_storage(t, T, gen, load, mass_salt, Cp)
%{
    This needs to go in an ODE solver to get the T_next
    t    : time interval
    gen  : Energy generation
    load : energy demand
    gen, load, mass_salt and Cp was used as extra argument in odeint
    Need to figure out how it works in ode23 or ode45
%}
    ODE = 3.6E9 * (gen - load) / (mass_salt * Cp);
end