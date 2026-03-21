%% logit_mlogit_demo.m
% Demonstration of Logit, MLogit Models with HAC Standard Errors,
% Marginal Effects, and RegressionTable Output
%
% This example demonstrates:
%   1. Binary Logit regression
%   2. Multinomial Logit (MLogit) regression  
%   3. HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
%   4. Marginal effects calculation (AME, MEM, elasticity forms)
%   5. Professional regression table output using RegressionTable
%
% Requirements:
%   - Python with statsmodels package installed
%   - pyBridge library in MATLAB path
%

%% Section 1: Data Generation
% Generate simulated data for demonstrating Logit and MLogit models

% Set random seed for reproducibility
rng(42);

% Sample size
n = 500;

% Generate independent variables
% X1: Continuous variable (e.g., income level)
X1 = randn(n, 1);

% X2: Continuous variable (e.g., age, standardized)
X2 = randn(n, 1);

% X3: Dummy variable (e.g., gender: 0=female, 1=male)
X3 = double(rand(n, 1) > 0.5);

% Combine into design matrix (without constant - will be added by the functions)
X = [X1, X2, X3];

% Generate cluster IDs for clustered standard errors demonstration
% 20 clusters (e.g., firms, regions)
clusterIds = randi(20, n, 1);

%% Generate Binary Dependent Variable (y_binary)
% Use logistic probability model: P(y=1) = 1 / (1 + exp(-Xb))
% True coefficients: beta0=0.5, beta1=1.0, beta2=-0.5, beta3=0.8

beta_true = [0.5; 1.0; -0.5; 0.8];  % [const, X1, X2, X3]
linear_predictor = beta_true(1) + X * beta_true(2:end);

% Logistic probability
prob_binary = 1 ./ (1 + exp(-linear_predictor));

% Generate binary outcome based on probability
y_binary = double(rand(n, 1) < prob_binary);

fprintf('Binary outcome distribution:\n');
fprintf('  y=0: %d (%.1f%%)\n', sum(y_binary==0), 100*mean(y_binary==0));
fprintf('  y=1: %d (%.1f%%)\n', sum(y_binary==1), 100*mean(y_binary==1));

%% Generate Multi-class Dependent Variable (y_multi)
% Three categories: 0=Low, 1=Medium, 2=High
% Using multinomial logit probability model

% True coefficients for Medium (category 1) vs Low (reference)
beta_medium = [0.3; 0.8; -0.3; 0.5];

% True coefficients for High (category 2) vs Low (reference)
beta_high = [0.1; 1.5; -0.8; 1.0];

% Linear predictors (vs reference category 0)
V_low = zeros(n, 1);  % Reference category
V_medium = beta_medium(1) + X * beta_medium(2:end);
V_high = beta_high(1) + X * beta_high(2:end);

% Multinomial logit probabilities
exp_V = [exp(V_low), exp(V_medium), exp(V_high)];
prob_multi = exp_V ./ sum(exp_V, 2);

% Generate multi-class outcome based on cumulative probability
u = rand(n, 1);
y_multi = zeros(n, 1);
y_multi(u > prob_multi(:,1)) = 1;
y_multi(u > prob_multi(:,1) + prob_multi(:,2)) = 2;

fprintf('\nMulti-class outcome distribution:\n');
fprintf('  y=0 (Low):    %d (%.1f%%)\n', sum(y_multi==0), 100*mean(y_multi==0));
fprintf('  y=1 (Medium): %d (%.1f%%)\n', sum(y_multi==1), 100*mean(y_multi==1));
fprintf('  y=2 (High):   %d (%.1f%%)\n', sum(y_multi==2), 100*mean(y_multi==2));

%% Section 2: Binary Logit Regression
% Estimate binary logit model using pyBridge.StatsmodelsWrapper.logistic()

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('Section 2: Binary Logit Regression\n');
fprintf('%s\n', repmat('=', 1, 70));

% Basic Logit regression (classical standard errors)
result_logit = pyBridge.StatsmodelsWrapper.logistic(y_binary, X);

fprintf('\nLogit Model Results:\n');
fprintf('  Number of observations: %d\n', result_logit.nObs);
fprintf('  Log-Likelihood: %.4f\n', result_logit.logLikelihood);
fprintf('  AIC: %.4f\n', result_logit.aic);
fprintf('  BIC: %.4f\n', result_logit.bic);

% Display coefficients
varNames = {'const', 'X1', 'X2', 'X3'};
fprintf('\n  Coefficients:\n');
fprintf('  %-10s %10s %10s %10s %10s\n', 'Variable', 'Coef', 'Std.Err', 'z-stat', 'p-value');
for i = 1:length(result_logit.params)
    fprintf('  %-10s %10.4f %10.4f %10.4f %10.4f\n', ...
        varNames{i}, result_logit.params(i), result_logit.stdErrors(i), ...
        result_logit.zStatistics(i), result_logit.pValues(i));
end

%% Section 3: HAC Standard Error Correction
% Demonstrate HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
% HAC is typically used with time series or panel data to account for serial correlation

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('Section 3: HAC Standard Error Correction\n');
fprintf('%s\n', repmat('=', 1, 70));

% Calculate automatic lag selection using Newey-West rule
% Formula: lag = floor(4 * ((T/100)^(2/9)))
T = size(X, 1);  % Number of observations
autoLag = floor(4 * ((T/100)^(2/9)));
fprintf('\nAutomatic lag selection (Newey-West rule):\n');
fprintf('  T = %d, autoLag = floor(4 * ((%d/100)^(2/9))) = %d\n', T, T, autoLag);

%% 3.1 HAC Standard Errors for OLS Regression
% Note: HAC is more commonly used with OLS for continuous dependent variables
% Here we use y_binary for demonstration, treating it as a linear probability model

fprintf('\n--- HAC Standard Errors for OLS (Linear Probability Model) ---\n');

% Newey-West kernel with automatic lag selection
% Note: "newey-west", "nw", and "bartlett" are all equivalent aliases
result_hac_bartlett = pyBridge.StatsmodelsWrapper.ols(y_binary, X, ...
    covType="hac", lag=autoLag, kernel="newey-west");

fprintf('\nOLS with HAC (Newey-West kernel, lag=%d):\n', autoLag);
fprintf('  %-10s %10s %10s\n', 'Variable', 'Coef', 'HAC SE');
for i = 1:length(result_hac_bartlett.params)
    fprintf('  %-10s %10.4f %10.4f\n', ...
        varNames{i}, result_hac_bartlett.params(i), result_hac_bartlett.stdErrors(i));
end

% Parzen kernel
result_hac_parzen = pyBridge.StatsmodelsWrapper.ols(y_binary, X, ...
    covType="hac", lag=autoLag, kernel="parzen");

fprintf('\nOLS with HAC (Parzen kernel, lag=%d):\n', autoLag);
fprintf('  %-10s %10s %10s\n', 'Variable', 'Coef', 'HAC SE');
for i = 1:length(result_hac_parzen.params)
    fprintf('  %-10s %10.4f %10.4f\n', ...
        varNames{i}, result_hac_parzen.params(i), result_hac_parzen.stdErrors(i));
end

% Quadratic Spectral kernel with fixed lag
result_hac_qs = pyBridge.StatsmodelsWrapper.ols(y_binary, X, ...
    covType="hac", lag=6, kernel="qs");

fprintf('\nOLS with HAC (Quadratic Spectral kernel, lag=6):\n');
fprintf('  %-10s %10s %10s\n', 'Variable', 'Coef', 'HAC SE');
for i = 1:length(result_hac_qs.params)
    fprintf('  %-10s %10.4f %10.4f\n', ...
        varNames{i}, result_hac_qs.params(i), result_hac_qs.stdErrors(i));
end

%% 3.2 HAC Standard Errors for Logistic Regression
% Logistic regression with HAC standard errors for panel/time series data

fprintf('\n--- HAC Standard Errors for Logistic Regression ---\n');

result_logit_hac = pyBridge.StatsmodelsWrapper.logistic(y_binary, X, ...
    covType="HAC", lag=autoLag);

fprintf('\nLogit with HAC (Bartlett kernel, lag=%d):\n', autoLag);
fprintf('  %-10s %10s %10s %10s\n', 'Variable', 'Coef', 'HAC SE', 'p-value');
for i = 1:length(result_logit_hac.params)
    fprintf('  %-10s %10.4f %10.4f %10.4f\n', ...
        varNames{i}, result_logit_hac.params(i), result_logit_hac.stdErrors(i), ...
        result_logit_hac.pValues(i));
end

%% Section 4: Multinomial Logit (MLogit) Regression
% Estimate multinomial logit model for multi-class outcomes

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('Section 4: Multinomial Logit (MLogit) Regression\n');
fprintf('%s\n', repmat('=', 1, 70));

%% 4.1 Basic MLogit Model
result_mlogit = pyBridge.StatsmodelsWrapper.multinomialLogit(y_multi, X);

fprintf('\nMultinomial Logit Results:\n');
fprintf('  Number of observations: %d\n', result_mlogit.nObs);
fprintf('  Number of categories: %d\n', result_mlogit.nCategories);
fprintf('  Log-Likelihood: %.4f\n', result_mlogit.logLikelihood);
fprintf('  AIC: %.4f\n', result_mlogit.aic);

% Display coefficients (MLogit has K-1 sets of coefficients)
% Reference category is 0 (Low)
fprintf('\n  Coefficients (Reference: Low):\n');
fprintf('  Parameters shape: %s\n', mat2str(size(result_mlogit.params)));

%% 4.2 MLogit with Custom Reference Level
% By default, the smallest category (0=Low) is the reference
% Here we set "High" (category 2) as the reference level
result_mlogit_ref2 = pyBridge.StatsmodelsWrapper.multinomialLogit(y_multi, X, ...
    referenceLevel=2);

fprintf('\nMLogit with Reference Level = 2 (High):\n');
fprintf('  Reference category: %d\n', result_mlogit_ref2.referenceLevel);
fprintf('  Non-reference categories: %s\n', mat2str(result_mlogit_ref2.categoryOrder));

%% 4.3 MLogit with Clustered Standard Errors
result_mlogit_cluster = pyBridge.StatsmodelsWrapper.multinomialLogit(y_multi, X, ...
    covType="cluster", clusterIds=clusterIds);

fprintf('\nMLogit with Clustered Standard Errors:\n');
fprintf('  Number of clusters: %d\n', result_mlogit_cluster.nClusters);
fprintf('  Covariance type: %s\n', result_mlogit_cluster.covType);

%% Section 5: Marginal Effects Calculation
% Calculate various types of marginal effects for the Logit model

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('Section 5: Marginal Effects Calculation\n');
fprintf('%s\n', repmat('=', 1, 70));

%% 5.1 Average Marginal Effects (AME)
% AME: Average of marginal effects computed at each observation
% margeffAt="overall" computes average marginal effects over all observations

fprintf('\n--- Average Marginal Effects (AME) ---\n');
fprintf('AME computes marginal effects at each observation and averages them.\n');

result_ame = pyBridge.StatsmodelsWrapper.logistic(y_binary, X, ...
    margeffMethod="dydx", margeffAt="overall");

fprintf('\nAverage Marginal Effects (AME):\n');
fprintf('  %-10s %10s %10s %10s %10s\n', 'Variable', 'Effect', 'Std.Err', 'z-stat', 'p-value');

% Note: marginalEffects excludes constant term
meVarNames = {'X1', 'X2', 'X3'};
for i = 1:length(result_ame.marginalEffects)
    fprintf('  %-10s %10.4f %10.4f %10.4f %10.4f\n', ...
        meVarNames{i}, ...
        result_ame.marginalEffects(i), ...
        result_ame.marginalEffectsSE(i), ...
        result_ame.marginalEffectsT(i), ...
        result_ame.marginalEffectsP(i));
end

%% 5.2 Marginal Effects at Mean (MEM)
% MEM: Marginal effects computed at the mean values of X

fprintf('\n--- Marginal Effects at Mean (MEM) ---\n');
fprintf('MEM computes marginal effects at the mean values of covariates.\n');

result_mem = pyBridge.StatsmodelsWrapper.logistic(y_binary, X, ...
    margeffMethod="dydx", margeffAt="mean");

fprintf('\nMarginal Effects at Mean (MEM):\n');
fprintf('  %-10s %10s %10s %10s %10s\n', 'Variable', 'Effect', 'Std.Err', 'z-stat', 'p-value');
for i = 1:length(result_mem.marginalEffects)
    fprintf('  %-10s %10.4f %10.4f %10.4f %10.4f\n', ...
        meVarNames{i}, ...
        result_mem.marginalEffects(i), ...
        result_mem.marginalEffectsSE(i), ...
        result_mem.marginalEffectsT(i), ...
        result_mem.marginalEffectsP(i));
end

%% 5.3 Elasticity Forms
% eyex: Elasticity (percentage change in y for percentage change in x)
% dyex: Semi-elasticity (change in y for percentage change in x)
% eydx: Semi-elasticity (percentage change in y for unit change in x)

fprintf('\n--- Elasticity: eyex (%%dy/%%dx) ---\n');
result_eyex = pyBridge.StatsmodelsWrapper.logistic(y_binary, X, ...
    margeffMethod="eyex", margeffAt="mean");

fprintf('  %-10s %10s\n', 'Variable', 'Elasticity');
for i = 1:length(result_eyex.marginalEffects)
    fprintf('  %-10s %10.4f\n', meVarNames{i}, result_eyex.marginalEffects(i));
end

fprintf('\n--- Semi-elasticity: dyex (dy/%%dx) ---\n');
result_dyex = pyBridge.StatsmodelsWrapper.logistic(y_binary, X, ...
    margeffMethod="dyex", margeffAt="mean");

fprintf('  %-10s %10s\n', 'Variable', 'Effect');
for i = 1:length(result_dyex.marginalEffects)
    fprintf('  %-10s %10.4f\n', meVarNames{i}, result_dyex.marginalEffects(i));
end

fprintf('\n--- Semi-elasticity: eydx (%%dy/dx) ---\n');
result_eydx = pyBridge.StatsmodelsWrapper.logistic(y_binary, X, ...
    margeffMethod="eydx", margeffAt="mean");

fprintf('  %-10s %10s\n', 'Variable', 'Effect');
for i = 1:length(result_eydx.marginalEffects)
    fprintf('  %-10s %10.4f\n', meVarNames{i}, result_eydx.marginalEffects(i));
end

%% Section 6: Regression Table Output
% Create professional regression tables using RegressionTable class

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('Section 6: Regression Table Output\n');
fprintf('%s\n', repmat('=', 1, 70));

% Create RegressionTable instance
tbl = pyBridge.RegressionTable();

% Add Logit basic model
tbl = tbl.addModel("Logit", result_logit, ...
    varNames={"const", "X1", "X2", "X3"}, ...
    covType="Classical");

% Add Logit with HAC standard errors
tbl = tbl.addModel("Logit_HAC", result_logit_hac, ...
    varNames={"const", "X1", "X2", "X3"}, ...
    covType="HAC");

% Add MLogit model (multi-column: one column per category)
% Note: MLogit has K-1 categories (excluding reference)
tbl = tbl.addMLogit("MLogit", result_mlogit, ...
    varNames={"X1", "X2", "X3"}, ...
    categoryNames={"Medium", "High"}, ...
    referenceName="Low");

% Add MLogit with custom reference level (High as reference)
tbl = tbl.addMLogit("MLogit_ref2", result_mlogit_ref2, ...
    varNames={"X1", "X2", "X3"}, ...
    categoryNames={"Low", "Medium"}, ...
    referenceName="High");

% Set variable display order
tbl = tbl.setVarOrder({"const", "X1", "X2", "X3"});

% Display the regression table
fprintf('\n--- Combined Regression Results Table ---\n');
tbl.display();

%% Section 7: Marginal Effects Table
% Create a separate table to display marginal effects from different methods

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('Section 7: Marginal Effects Comparison Table\n');
fprintf('%s\n', repmat('=', 1, 70));

% Create a new RegressionTable for marginal effects
tbl_me = pyBridge.RegressionTable();

% Build result structs compatible with addModel for marginal effects display
% AME result
me_ame = struct();
me_ame.params = result_ame.marginalEffects;
me_ame.stdErrors = result_ame.marginalEffectsSE;
me_ame.pValues = result_ame.marginalEffectsP;
me_ame.nObs = result_ame.nObs;
me_ame.modelType = "AME";

% MEM result
me_mem = struct();
me_mem.params = result_mem.marginalEffects;
me_mem.stdErrors = result_mem.marginalEffectsSE;
me_mem.pValues = result_mem.marginalEffectsP;
me_mem.nObs = result_mem.nObs;
me_mem.modelType = "MEM";

% eyex result
me_eyex = struct();
me_eyex.params = result_eyex.marginalEffects;
me_eyex.stdErrors = result_eyex.marginalEffectsSE;
me_eyex.pValues = result_eyex.marginalEffectsP;
me_eyex.nObs = result_eyex.nObs;
me_eyex.modelType = "eyex";

% dyex result
me_dyex = struct();
me_dyex.params = result_dyex.marginalEffects;
me_dyex.stdErrors = result_dyex.marginalEffectsSE;
me_dyex.pValues = result_dyex.marginalEffectsP;
me_dyex.nObs = result_dyex.nObs;
me_dyex.modelType = "dyex";

% eydx result
me_eydx = struct();
me_eydx.params = result_eydx.marginalEffects;
me_eydx.stdErrors = result_eydx.marginalEffectsSE;
me_eydx.pValues = result_eydx.marginalEffectsP;
me_eydx.nObs = result_eydx.nObs;
me_eydx.modelType = "eydx";

% Variable names for marginal effects (excludes constant)
meVarNamesCell = {"X1", "X2", "X3"};

% Add all marginal effects methods to the table
tbl_me = tbl_me.addModel("AME", me_ame, varNames=meVarNamesCell, ...
    note="Average ME");
tbl_me = tbl_me.addModel("MEM", me_mem, varNames=meVarNamesCell, ...
    note="ME at Mean");
tbl_me = tbl_me.addModel("eyex", me_eyex, varNames=meVarNamesCell, ...
    note="Elasticity");
tbl_me = tbl_me.addModel("dyex", me_dyex, varNames=meVarNamesCell, ...
    note="Semi-elast.");
tbl_me = tbl_me.addModel("eydx", me_eydx, varNames=meVarNamesCell, ...
    note="Semi-elast.");

% Set variable order
tbl_me = tbl_me.setVarOrder({"X1", "X2", "X3"});

% Display marginal effects comparison table
fprintf('\n--- Marginal Effects Comparison ---\n');
fprintf("AME: Average Marginal Effect (at=""overall"")\n");
fprintf("MEM: Marginal Effect at Mean (at=""mean"")\n");
fprintf("eyex: Elasticity (%%dy/%%dx)\n");
fprintf("dyex: Semi-elasticity (dy/%%dx)\n");
fprintf("eydx: Semi-elasticity (%%dy/dx)\n\n");
tbl_me.display();
tbl_me.toExcel("outputs/regression_results.xlsx");

%% Summary
fprintf('\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('\nThis demonstration covered:\n');
fprintf('  1. Data generation for binary and multi-class outcomes\n');
fprintf('  2. Binary Logit regression with classical standard errors\n');
fprintf('  3. HAC standard errors for OLS and Logistic regression\n');
fprintf('     - Newey-West (bartlett/nw), Parzen, and Quadratic Spectral kernels\n');
fprintf('     - Automatic lag selection: lag = floor(4*((T/100)^(2/9)))\n');
fprintf('  4. Multinomial Logit for multi-class outcomes\n');
fprintf('     - Basic MLogit and MLogit with clustered standard errors\n');
fprintf('  5. Marginal effects calculation:\n');
fprintf('     - AME: Average Marginal Effect (averaged over all observations)\n');
fprintf('     - MEM: Marginal Effect at Mean\n');
fprintf('     - Elasticity forms: eyex, dyex, eydx\n');
fprintf('  6. RegressionTable for publication-ready output\n');
fprintf('     - addModel() for standard models\n');
fprintf('     - addMLogit() for multinomial logit (multi-column display)\n');
fprintf('%s\n', repmat('=', 1, 70));

%% End of demonstration
