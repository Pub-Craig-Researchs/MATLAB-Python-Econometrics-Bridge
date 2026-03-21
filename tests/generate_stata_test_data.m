%% generate_stata_test_data.m
% Generate synthetic test data for cross-validation between MATLAB/Python and Stata.
% This script creates datasets for: OLS, Panel, Logit, Probit, MLogit, 
% Ordered Logit/Probit, Poisson, Negative Binomial, and IV models.
% Output: stata_test_data.csv in the tests/ directory.

%% Clear workspace and set seed
clear;
rng(42);
n = 500;

%% Panel structure
nEntity = 50; 
nTime = 10;
entityId = repelem((1:nEntity)', nTime);  % 50 entities, each 10 periods
timeId = repmat((1:nTime)', nEntity, 1);  % 10 time periods repeated

%% Explanatory variables
X1 = randn(n, 1);               % Continuous variable
X2 = randn(n, 1);               % Continuous variable
X3 = double(rand(n, 1) > 0.5);  % Binary dummy variable
clusterId = randi(20, n, 1);    % Cluster identifier (20 clusters)
weights = 0.5 + rand(n, 1);     % WLS weights (0.5 ~ 1.5)

%% Entity fixed effects
alpha_entity = randn(nEntity, 1) * 0.5;
alpha_i = alpha_entity(entityId);  % Fixed effect for each entity

%% Continuous dependent variable (OLS, Panel)
beta_ols = [1.0; 0.5; -0.3; 0.8];  % const, X1, X2, X3
epsilon_ols = randn(n, 1);
y_continuous = beta_ols(1) + beta_ols(2)*X1 + beta_ols(3)*X2 + beta_ols(4)*X3 + alpha_i + epsilon_ols;
% Version without fixed effects (for cross-sectional OLS)
y_ols = beta_ols(1) + beta_ols(2)*X1 + beta_ols(3)*X2 + beta_ols(4)*X3 + epsilon_ols;

%% Binary dependent variable (Logit, Probit)
beta_logit = [0.5; 1.0; -0.5; 0.8];
linear_pred = beta_logit(1) + beta_logit(2)*X1 + beta_logit(3)*X2 + beta_logit(4)*X3;
prob_binary = 1 ./ (1 + exp(-linear_pred));
y_binary = double(rand(n, 1) < prob_binary);

%% Multinomial dependent variable (MLogit, 3 categories: 0, 1, 2)
beta_cat1 = [0.3; 0.8; -0.3; 0.5];   % vs base category 0
beta_cat2 = [0.1; 1.5; -0.8; 1.0];
V0 = zeros(n, 1);
V1 = beta_cat1(1) + beta_cat1(2)*X1 + beta_cat1(3)*X2 + beta_cat1(4)*X3;
V2 = beta_cat2(1) + beta_cat2(2)*X1 + beta_cat2(3)*X2 + beta_cat2(4)*X3;
exp_V = [exp(V0), exp(V1), exp(V2)];
prob_multi = exp_V ./ sum(exp_V, 2);
u_multi = rand(n, 1);
y_multi = zeros(n, 1);
y_multi(u_multi > prob_multi(:,1)) = 1;
y_multi(u_multi > prob_multi(:,1) + prob_multi(:,2)) = 2;

%% Ordered dependent variable (Ordered Logit/Probit, 4 levels: 1,2,3,4)
beta_ordered = [0.6; -0.4; 0.5];  % X1, X2, X3 (no constant, thresholds instead)
latent = beta_ordered(1)*X1 + beta_ordered(2)*X2 + beta_ordered(3)*X3 + randn(n,1)*1.2;
thresholds = [-1.0, 0.5, 1.5];
y_ordered = ones(n, 1);
y_ordered(latent > thresholds(1)) = 2;
y_ordered(latent > thresholds(2)) = 3;
y_ordered(latent > thresholds(3)) = 4;

%% Count dependent variable (Poisson, Negative Binomial)
beta_count = [0.5; 0.3; -0.2; 0.4];
lambda = exp(beta_count(1) + beta_count(2)*X1 + beta_count(3)*X2 + beta_count(4)*X3);
y_count = poissrnd(lambda);

%% IV setup (Instrumental Variables)
Z1 = randn(n, 1);    % Instrument 1
Z2 = randn(n, 1);    % Instrument 2
error_endog = randn(n, 1);
X_endog = 0.5*Z1 + 0.3*Z2 + 0.7*X3 + error_endog;  % Endogenous variable
epsilon_iv = randn(n, 1) + 0.5*error_endog;        % Correlated with endogeneity
y_iv = 1.0 + 0.8*X_endog + 0.5*X3 + epsilon_iv;

%% Export to CSV
T = table(entityId, timeId, X1, X2, X3, clusterId, weights, ...
    y_ols, y_continuous, y_binary, y_multi, y_ordered, y_count, ...
    X_endog, Z1, Z2, y_iv);

outputPath = fullfile(fileparts(mfilename('fullpath')), 'stata_test_data.csv');
writetable(T, outputPath);

%% Print data summary
fprintf('\n========================================\n');
fprintf('  Stata Test Data Generation Summary\n');
fprintf('========================================\n\n');
fprintf('Output file: %s\n\n', outputPath);
fprintf('Total observations: %d\n', n);
fprintf('Panel structure: %d entities x %d time periods\n\n', nEntity, nTime);

fprintf('--- Explanatory Variables ---\n');
fprintf('X1 (continuous): mean=%.3f, std=%.3f\n', mean(X1), std(X1));
fprintf('X2 (continuous): mean=%.3f, std=%.3f\n', mean(X2), std(X2));
fprintf('X3 (binary): mean=%.3f (proportion of 1s)\n', mean(X3));
fprintf('clusterId: %d unique clusters\n', numel(unique(clusterId)));
fprintf('weights: range [%.3f, %.3f]\n\n', min(weights), max(weights));

fprintf('--- Dependent Variables ---\n');
fprintf('y_ols (continuous): mean=%.3f, std=%.3f\n', mean(y_ols), std(y_ols));
fprintf('y_continuous (with FE): mean=%.3f, std=%.3f\n', mean(y_continuous), std(y_continuous));
fprintf('y_binary: 0=%d (%.1f%%), 1=%d (%.1f%%)\n', ...
    sum(y_binary==0), 100*mean(y_binary==0), ...
    sum(y_binary==1), 100*mean(y_binary==1));
fprintf('y_multi: 0=%d (%.1f%%), 1=%d (%.1f%%), 2=%d (%.1f%%)\n', ...
    sum(y_multi==0), 100*mean(y_multi==0), ...
    sum(y_multi==1), 100*mean(y_multi==1), ...
    sum(y_multi==2), 100*mean(y_multi==2));
fprintf('y_ordered: 1=%d, 2=%d, 3=%d, 4=%d\n', ...
    sum(y_ordered==1), sum(y_ordered==2), sum(y_ordered==3), sum(y_ordered==4));
fprintf('y_count (Poisson): mean=%.3f, var=%.3f, max=%d\n', ...
    mean(y_count), var(y_count), max(y_count));

fprintf('\n--- IV Variables ---\n');
fprintf('X_endog: mean=%.3f, std=%.3f\n', mean(X_endog), std(X_endog));
fprintf('Z1: mean=%.3f, std=%.3f\n', mean(Z1), std(Z1));
fprintf('Z2: mean=%.3f, std=%.3f\n', mean(Z2), std(Z2));
fprintf('y_iv: mean=%.3f, std=%.3f\n', mean(y_iv), std(y_iv));

fprintf('\n========================================\n');
fprintf('  Data generation complete!\n');
fprintf('========================================\n');
