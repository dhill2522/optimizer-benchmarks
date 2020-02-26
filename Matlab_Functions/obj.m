% Objective Function
function total_cost = obj(gen,time,loads)
    [total_cost, ~] = model(gen, time, loads);
end