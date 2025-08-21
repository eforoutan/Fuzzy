clc
clear
format long g
tic
rng(0)

%% Read dataset

data_Dallas = readmatrix('Data_Dallas-Fort Worth-Arlington, TX.csv');

%% Reading Rules
filename = 'fuzzy_rules_19.txt';
fid = fopen(filename, 'r');
rules = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
rules = rules{1};
expRules = string(rules);

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


%%
X = data_Dallas(:, 1:7);  % Inputs
Y = data_Dallas(:, 8);    % Output

%% ##################

fisin = mamfis;

% Define variable names
name = ["THEME1", "THEME2", "THEME3", "THEME4", "RH", "LST", "PopDens", "Number"];
        
dataRange = [min(data_Dallas)' max(data_Dallas)'];

        
for ii = 1:7
        
    fisin = addInput(fisin, dataRange(ii,:),'Name',name(ii),'NumMFs',3, 'MFType', 'gaussmf'); % 'trimf'
end


fisin = addOutput(fisin, dataRange(8,:),'Name',name(8),'NumMFs', 7, 'MFType', 'gaussmf');  % 'trimf'

fisin = addRule(fisin,expRules1);

 %% Tuning the FIS (fis) [above]

[in,out,~] = getTunableSettings(fisin);

% options_tn = tunefisOptions('Method', 'particleswarm', ...
% 'OptimizationType', 'tuning');

options_tn = tunefisOptions('Method', 'patternsearch', ...
'OptimizationType', 'tuning');

options_tn.UseParallel = true;
options_tn.MethodOptions.MaxIterations = 60;
fisin_tn = tunefis(fisin, [in;out], X, Y, options_tn);

%% Train Evaluation of the Exprert (before tuning)

trn_predicted_exp = evalfis(fisin, X);
trn_RMSE_exp = sqrt(mean((trn_predicted_exp - Y).^2));
trn_MAE_exp = mean(abs(trn_predicted_exp - Y));

%% Train Evaluation of the Tuned FIS

trn_predicted_exp_tn = evalfis(fisin_tn, X);
trn_RMSE_exp_tn = sqrt(mean((trn_predicted_exp_tn - Y).^2));
trn_MAE_exp_tn = mean(abs(trn_predicted_exp_tn - Y));

%% Test (Validation) Evaluation of the Tuned and Learned FIS
% No test at this step

%% Store results
results = [trn_RMSE_exp, trn_MAE_exp, ...
                    trn_RMSE_exp_tn, trn_MAE_exp_tn];

%% Save the Tuned FIS

writeFIS(fisin_tn, 'Dallas_exp_tuned_PS_allData.fis');

%% Save the Original (non-tunned) FIS

writeFIS(fisin, 'Dallas_Exp_NonTuned.fis');

%% Convert results to table

results_table = array2table(results, ...
    'VariableNames', {'RMSE_Train_exp', 'MAE_Train_exp', ...
                              'RMSE_Train_exp_tn', 'MAE_Train_exp_tn'});

%% Write results to CSV
writetable(results_table, 'Dallas_Exp_tuned_PS_Evaluation_allData.csv'); 

% Display result
disp(results_table);
tt = toc;
fprintf('Total execution time: %.2f seconds.\n', tt);