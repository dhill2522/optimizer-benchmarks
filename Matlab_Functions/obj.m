% Objective Function
function total_cost = obj(gen)
    cost_nuclear = 0.021; % $/kWh
    total_cost   = sum(gen * cost_nuclear);
end