% Objective Function
function total_cost = obj(gen,time,load)
    [total_cost, ~] = model(gen, time, load)
end