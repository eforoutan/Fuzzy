clc
clear
format long g
tic
rng(0)

num_folds = 5;
results = zeros(num_folds, 5);

%% Reading Dataset

for f = 1:num_folds

    disp(f)
    % Construct file name
    filename_trn = sprintf("train_fold_%d.csv", f);
    filename_vld = sprintf("test_fold_%d.csv", f);
    
    % Read dataset
    data_trn = readmatrix(filename_trn);
    data_vld = readmatrix(filename_vld);
    
    % Split training data into input and output
    trnX = data_trn(:, 1:7);  % Inputs (Features)
    trnY = data_trn(:, 8);    % Output (Target)
    
    % Split training data into input and output
    vldX = data_vld(:, 1:7);  % Inputs (Features)
    vldY = data_vld(:, 8);    % Output (Target)

   %% Load FIS
    filename_fis = sprintf("Austin_exp_tuned_fold_%d.fis", f);
    
    fis = readfis(filename_fis);

    % Extract initial rule weights
    num_rules = length(fis.rule);
    initial_weights = arrayfun(@(r) r.weight, fis.rule);

    % Define lower and upper bounds for rule weights
    lb = zeros(1, num_rules);  % Min weight = 0
    ub = ones(1, num_rules);   % Max weight = 1

%% Define PSO Optimization Options

    pso_options = optimoptions('particleswarm', ...
        'SwarmSize', 150, ...       % Increased number of particles
        'MaxIterations', 100, ...   % Increased number of iterations for convergence
        'Display', 'iter', ...      % Show progress during optimization
        'PlotFcn', 'pswplotbestf', ... % Keep visualization, but remove OutputFcn
        'UseParallel', true); 
        
%% Run PSO Optimization
    [optimized_weights, final_error] = particleswarm(@(w) fuzzy_rule_optimization(w, fis, trnX, trnY), ...
                                                 num_rules, lb, ub, pso_options);

%% Save the PSO Optimization Plot as .fig

    % saveas(gcf, sprintf('PSO_Optimization_MFR_Fold_%d.fig', f));

%% Update FIS with optimized weights
    for i = 1:num_rules
        fis.rule(i).weight = optimized_weights(i);
    end

%% Save the optimized FIS
    writeFIS(fis, sprintf('Optimized_Austin_exp_tuned_fold_%d.fis', f));

    %% Train Evaluation of the Tuned Expert Rule

    trn_predicted_exp_tn_op = evalfis(fis, trnX);
    trn_RMSE_exp_tn_op = sqrt(mean((trn_predicted_exp_tn_op - trnY).^2));
    trn_MAE_exp_tn_op = mean(abs(trn_predicted_exp_tn_op - trnY));


     %% Test (Validation) Evaluation of the Tuned Expert Rule

    vld_predicted_exp_tn_op = evalfis(fis, vldX);
    vld_RMSE_exp_tn_op = sqrt(mean((vld_predicted_exp_tn_op - vldY).^2));
    vld_MAE_exp_tn_op = mean(abs(vld_predicted_exp_tn_op - vldY));

     % Store results
    results(f, :) = [f, trn_RMSE_exp_tn_op, vld_RMSE_exp_tn_op, trn_MAE_exp_tn_op, vld_MAE_exp_tn_op];



end


%% Convert results to table
results_table = array2table(results, ...
    'VariableNames', {'Fold', 'RMSE_Train', 'RMSE_Test', 'MAE_Train', 'MAE_Test'});

%% Write results to CSV
writetable(results_table, 'Optimized_Austin_EXP_Tuning_Evaluation_PS_PSO.csv'); 
tt = toc