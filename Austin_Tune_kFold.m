clc
clear
format long g
tic
rng(0)

%% Reading Dataset

num_folds = 5;
results = zeros(num_folds, 9);
data_Austin = readmatrix('Data_Austin-Round Rock, TX.csv');

%% Reading and Processing Rules
filename = 'fuzzy_rules_19.txt';
fid = fopen(filename, 'r');
rules = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
rules = rules{1};
expRules = string(rules);

%
expRules1 = replace(expRules  , "Number=ExtremelyLow", "Number=mf1");
expRules1 = replace(expRules1 , "Number=VeryLow", "Number=mf2");
expRules1 = replace(expRules1 , "Number=Low", "Number=mf3");
expRules1 = replace(expRules1 , "Number=Medium", "Number=mf4");
expRules1 = replace(expRules1 , "Number=High", "Number=mf5");
expRules1 = replace(expRules1 , "Number=VeryHigh", "Number=mf6");
expRules1 = replace(expRules1 , "Number=ExtremelyHigh", "Number=mf7");

expRules1 = replace(expRules1 , "Low", "mf1");
expRules1 = replace(expRules1 , "Medium", "mf2");
expRules1 = replace(expRules1 , "High", "mf3");

%% Loop through test folds

for i = 1:num_folds

    filename_trn = sprintf("train_fold_%d.csv", i);
    filename_vld = sprintf("test_fold_%d.csv", i);
    
    % Read dataset
    data_trn = readmatrix(filename_trn);
    data_vld = readmatrix(filename_vld);
    
    % Split training data into input and output
    trnX = data_trn(:, 1:7);  % Inputs (Features)
    trnY = data_trn(:, 8);    % Output (Target)
    
    % Split training data into input and output
    vldX = data_vld(:, 1:7);  % Inputs (Features)
    vldY = data_vld(:, 8);    % Output (Target)

    %% ##################
    fisin = mamfis;

    % Define variable names
    name = ["THEME1", "THEME2", "THEME3", "THEME4", "RH", "LST", "PopDens", "Number"];
        
    dataRange = [min(data_Austin)' max(data_Austin)'];
        
    for ii = 1:7
        
        fisin = addInput(fisin, dataRange(ii,:),'Name',name(ii),'NumMFs',3, 'MFType', 'gaussmf'); % 'trimf'
    end


    fisin = addOutput(fisin, dataRange(8,:),'Name',name(8),'NumMFs', 7, 'MFType', 'gaussmf');  % 'trimf'

    fisin = addRule(fisin,expRules1);
    
    %% Tuning the FIS (fisin) [above fis]

    [in,out,~] = getTunableSettings(fisin); % input and output membership function parameters
    
    % options_tn = tunefisOptions('Method', 'particleswarm', ...
    % 'OptimizationType', 'tuning');

    options_tn = tunefisOptions('Method', 'patternsearch', ...
    'OptimizationType', 'tuning');

    options_tn.UseParallel = true;
    options_tn.MethodOptions.MaxIterations = 60;

    fisin_tn = tunefis(fisin, [in;out], trnX, trnY, options_tn);
    
    %% Train Evaluation of the Fuzzy with Expert rule and MF (Without Tuning)

    trn_predicted_exp = evalfis(fisin, trnX);
    trn_RMSE_exp = sqrt(mean((trn_predicted_exp - trnY).^2));
    trn_MAE_exp = mean(abs(trn_predicted_exp - trnY));

    %% Test (Validation) Evaluation of the Fuzzy with Expert rule and MF (Without Tuning)

    vld_predicted_exp = evalfis(fisin, vldX);
    vld_RMSE_exp = sqrt(mean((vld_predicted_exp - vldY).^2));
    vld_MAE_exp = mean(abs(vld_predicted_exp - vldY));

    %% Train Evaluation of the Fuzzy with Expert rule and "Tunned MF"

    trn_predicted_exp_tn = evalfis(fisin_tn, trnX);
    trn_RMSE_exp_tn = sqrt(mean((trn_predicted_exp_tn - trnY).^2));
    trn_MAE_exp_tn = mean(abs(trn_predicted_exp_tn - trnY));
    
    %% Test (Validation) Evaluation of the Fuzzy with Expert rule and "Tunned MF"

    vld_predicted_exp_tn = evalfis(fisin_tn, vldX);
    vld_RMSE_exp_tn = sqrt(mean((vld_predicted_exp_tn - vldY).^2));
    vld_MAE_exp_tn = mean(abs(vld_predicted_exp_tn - vldY));
    
    %% Store results

    results(i, :) = [i, trn_RMSE_exp, vld_RMSE_exp, trn_MAE_exp, vld_MAE_exp, ...
                        trn_RMSE_exp_tn, vld_RMSE_exp_tn, trn_MAE_exp_tn, vld_MAE_exp_tn];


    %% Save the Tuned FIS with fold number
    writeFIS(fisin_tn, sprintf('Austin_exp_tuned_fold_%d.fis', i));

fprintf('Completed fold %d of %d.\n', i, num_folds);

end

%% Convert results to table
results_table = array2table(results, ...
    'VariableNames', {'Fold', 'RMSE_Train_exp', 'RMSE_Test_exp', 'MAE_Train_exp', 'MAE_Test_exp' , ...
                              'RMSE_Train_exp_tn', 'RMSE_Test_exp_tn', 'MAE_Train_exp_tn', 'MAE_Test_exp_tn'});

%% Write results to CSV
writetable(results_table, 'Austin_Exp_tuned_PS_Evaluation.csv');

% Display results
disp(results_table);
tt = toc;
fprintf('Total execution time: %.2f seconds.\n', tt);