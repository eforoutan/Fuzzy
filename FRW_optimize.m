clc
clear
format long g
tic
rng(0)

%% Reading Dataset

data_Austin = readmatrix('Data_Austin-Round Rock, TX.csv');

X = data_Austin(:, 1:7);  % Inputs (Features)
Y = data_Austin(:, 8);    % Output (Target)

%% Load FIS

fis = readfis('Austin_exp_tuned_PS_allData.fis');
% fis = readfis('Austin_Exp_NonTuned.fis');

% Extract initial rule weights

num_rules = length(fis.rule);
initial_weights = arrayfun(@(r) r.weight, fis.rule);

% Define lower and upper bounds for rule weights
lb = zeros(1, num_rules);  % Min weight = 0
ub = ones(1, num_rules);   % Max weight = 1

%% Define PSO Optimization Options with Parallel Processing

pso_options = optimoptions('particleswarm', ...
    'SwarmSize', 150, ...
    'MaxIterations', 150, ...
    'Display', 'iter', ...
    'PlotFcn', 'pswplotbestf', ...
    'UseParallel', true); % Enable parallel processing
        
%% Run PSO Optimization
[optimized_weights, final_error] = particleswarm(@(w) fuzzy_rule_optimization(w, fis, X, Y), ...
                                                 num_rules, lb, ub, pso_options);

%% Save the PSO Optimization Plot as .fig

saveas(gcf, 'PSO_Optimization_Exp_tuned_PS_PSO_allData.fig');

%% Update FIS with optimized weights
for i = 1:num_rules
     fis.rule(i).weight = optimized_weights(i);
end

%% Save the optimized FIS

writeFIS(fis, 'optimized_Austin_exp_tuned_PS_PSO_allData.fis');
% writeFIS(fis, 'optimized_Austin_Exp_NonTuned_allData.fis');

%% Evaluation of the Tuned Expert Rule

predicted_lr_tn_op = evalfis(fis, X);
RMSE_lr_tn_op = sqrt(mean((predicted_lr_tn_op - Y).^2));
MAE_lr_tn_op = mean(abs(predicted_lr_tn_op - Y));

% Store results
results = [RMSE_lr_tn_op, MAE_lr_tn_op];


%% Convert results to table
results_table = array2table(results, 'VariableNames', {'RMSE', 'MAE'});

%% Write results to CSV
writetable(results_table, 'optimized_Austin_exp_tuned_PS_PSO_Evaluation_allData.csv');
% writetable(results_table, 'optimized_Austin_Exp_NonTuned_Evaluation_allData.csv'); 
tt = toc;