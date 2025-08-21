function error = fuzzy_rule_optimization(weights, fis, input_data, target_data)
    % Update rule weights in FIS
    for i = 1:length(fis.rule)
        fis.rule(i).weight = weights(i);
    end
    
    % Evaluate FIS predictions
    predicted_output = evalfis(fis, input_data);
    
    % Compute Root Mean Squared Error (RMSE)
    error = sqrt(mean((predicted_output - target_data).^2));
end

