%% Advanced Econometric Methods Example
% Demonstrates Multinomial Logit, HAC Standard Errors, Multi-way Clustering


%% Clear Workspace
clear; clc;

%% Initialize Environment
startup();

%% 1. Multinomial Logit Regression
fprintf("\n=== 1. Multinomial Logit Regression Example ===\n");

% Generate sample data: Occupation choice (3 categories)
rng(42, "twister");
nObs = 500;

% Independent variables: education, experience, age
education = randn(nObs, 1) * 2 + 12; % Years of education
experience = randn(nObs, 1) * 5 + 10; % Work experience
age = education + experience + randn(nObs, 1) * 3 + 18; % Age
X = [education, experience, age];

% Dependent variable: Occupation choice (0=Blue-collar, 1=White-collar, 2=Professional)
prob1 = 1 ./ (1 + exp(-0.5*education + 0.2*experience));
prob2 = 1 ./ (1 + exp(-0.3*education - 0.1*experience + 0.05*age));
randDraw = rand(nObs, 1);
y = zeros(nObs, 1);
y(randDraw > prob1 & randDraw < prob1 + prob2*0.5) = 1;
y(randDraw > prob1 + prob2*0.5) = 2;

% Fit multinomial logit model
result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);

fprintf("Model Type: %s\n", result.modelType);
fprintf("Number of Categories: %d\n", result.nCategories);
fprintf("Coefficient Matrix (First 3 Rows):\n");
disp(result.params(1:3, :));
fprintf("Marginal Effects (First 3 Rows):\n");
disp(result.marginalEffects(1:3, :));

%% 2. Ordered Logit Regression
fprintf("\n=== 2. Ordered Logit Regression Example ===\n");

% Dependent variable: Satisfaction rating (0=Unsatisfied, 1=Neutral, 2=Satisfied, 3=Very Satisfied)
yOrdered = discretize(rand(nObs, 1), [0, 0.25, 0.5, 0.75, 1]) - 1;

% Note: addConstant defaults to false (OrderedModel uses thresholds as intercepts)
% Use addConstant=true only if you explicitly need a constant column
result = pyBridge.StatsmodelsWrapper.orderedLogit(yOrdered, X);

fprintf("Model Type: %s\n", result.modelType);
fprintf("Threshold Parameters:\n");
disp(result.thresholds);
fprintf("Coefficients (Education Effect):\n");
disp(result.params(1));

%% 3. Ordered Probit Regression
fprintf("\n=== 3. Ordered Probit Regression Example ===\n");

% Note: addConstant defaults to false (OrderedModel uses thresholds as intercepts)
% Use addConstant=true only if you explicitly need a constant column
result = pyBridge.StatsmodelsWrapper.orderedProbit(yOrdered, X);

fprintf("Model Type: %s\n", result.modelType);
fprintf("AIC: %.2f\n", result.aic);
fprintf("BIC: %.2f\n", result.bic);

%% 4. Heteroskedasticity-Robust Standard Errors (HC Series)
fprintf("\n=== 4. Heteroskedasticity-Robust SE Example ===\n");

% Generate data with heteroskedasticity
X = randn(nObs, 3);
epsilon = randn(nObs, 1) .* (1 + 0.5*X(:,1)); % Heteroskedastic errors
y = 2 + 3*X(:,1) + 1.5*X(:,2) - 0.5*X(:,3) + epsilon;

% Compare different SE types
fprintf("\nClassical Standard Errors:\n");
resultClassic = pyBridge.StatsmodelsWrapper.ols(y, X, covType="nonrobust");
fprintf("  Coefficients: [%.3f, %.3f, %.3f]\n", resultClassic.params(2:4));
fprintf("  Std Errors: [%.3f, %.3f, %.3f]\n", resultClassic.stdErrors(2:4));

fprintf("\nHC1 Robust SE (Stata Default):\n");
resultHC1 = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC1");
fprintf("  Coefficients: [%.3f, %.3f, %.3f]\n", resultHC1.params(2:4));
fprintf("  Std Errors: [%.3f, %.3f, %.3f]\n", resultHC1.stdErrors(2:4));

fprintf("\nHC3 Robust SE (Most Conservative):\n");
resultHC3 = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC3");
fprintf("  Coefficients: [%.3f, %.3f, %.3f]\n", resultHC3.params(2:4));
fprintf("  Std Errors: [%.3f, %.3f, %.3f]\n", resultHC3.stdErrors(2:4));

%% 5. HAC Standard Errors (Newey-West)
fprintf("\n=== 5. HAC Standard Errors Example ===\n");

% Generate time series data (with autocorrelation and heteroskedasticity)
nTime = 200;
X = randn(nTime, 2);
epsilon = zeros(nTime, 1);
epsilon(1) = randn();
for t = 2:nTime
    epsilon(t) = 0.5*epsilon(t-1) + randn(); % AR(1) process
end
y = 1 + 2*X(:,1) + 1.5*X(:,2) + epsilon;

% HAC standard errors
% Note: "newey-west" is internally mapped to "bartlett"
resultHAC = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="hac", lag=4, kernel="newey-west");

fprintf("HAC SE (Newey-West, 4 Lags):\n");
fprintf("  Coefficients: [%.3f, %.3f]\n", resultHAC.params(2:3));
fprintf("  Std Errors: [%.3f, %.3f]\n", resultHAC.stdErrors(2:3));
fprintf("  Kernel: %s\n", resultHAC.kernel);

% Compare different lag orders
fprintf("\nStandard Errors by Lag Order:\n");
for lag = 0:2:8
    if lag == 0
        resultLag = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC0");
        fprintf("  Lag=%d (HC0): [%.3f, %.3f]\n", lag, resultLag.stdErrors(2:3));
    else
        resultLag = pyBridge.StatsmodelsWrapper.ols(y, X, ...
            covType="hac", lag=lag);
        fprintf("  Lag=%d: [%.3f, %.3f]\n", lag, resultLag.stdErrors(2:3));
    end
end

%% 6. One-Way Clustered Standard Errors
fprintf("\n=== 6. One-Way Clustered SE Example ===\n");

% Generate panel data: 50 firms, 20 periods each
nFirms = 50;
nPeriods = 20;
nObs = nFirms * nPeriods;

firmId = repelem(1:nFirms, nPeriods);
yearId = repmat(1:nPeriods, 1, nFirms);

X = randn(nObs, 2);
firmFE = randn(nFirms, 1);
firmFE = repelem(firmFE, nPeriods);
y = 1 + 2*X(:,1) + 1.5*X(:,2) + firmFE + randn(nObs, 1);

% Cluster by firm
resultCluster = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="cluster", clusterIds=firmId);

fprintf("Clustered SE (By Firm):\n");
fprintf("  Coefficients: [%.3f, %.3f]\n", resultCluster.params(2:3));
fprintf("  Std Errors: [%.3f, %.3f]\n", resultCluster.stdErrors(2:3));
fprintf("  Number of Clusters: %d\n", resultCluster.nClusters);

%% 7. Two-Way Clustered Standard Errors
fprintf("\n=== 7. Two-Way Clustered SE Example ===\n");

% Two-way clustering: Firm x Year
resultMultiway = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="multiway", clusterGroups={firmId, yearId});

fprintf("Two-Way Clustered SE (Firm x Year):\n");
fprintf("  Coefficients: [%.3f, %.3f]\n", resultMultiway.params(2:3));
fprintf("  Std Errors: [%.3f, %.3f]\n", resultMultiway.stdErrors(2:3));
fprintf("  Clustering Dimensions: %d\n", resultMultiway.nDimensions);

% Compare three SE types
fprintf("\nStandard Error Comparison:\n");
fprintf("  Classical: [%.3f, %.3f]\n", resultClassic.stdErrors(2:3));
fprintf("  Firm Clustered: [%.3f, %.3f]\n", resultCluster.stdErrors(2:3));
fprintf("  Two-Way Clustered: [%.3f, %.3f]\n", resultMultiway.stdErrors(2:3));

%% 8. Panel Data Models + Clustered SE
fprintf("\n=== 8. Panel Data Models Example ===\n");

% Panel data models using linearmodels
resultPanel = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmId, yearId, ...
    entityEffects=true, timeEffects=false, covType="clustered_entity");

fprintf("Fixed Effects Model + Entity Clustered SE:\n");
fprintf("  Coefficients: [%.3f, %.3f]\n", resultPanel.params);
fprintf("  Std Errors: [%.3f, %.3f]\n", resultPanel.stdErrors);
fprintf("  R-squared: %.4f\n", resultPanel.rsquared);

% Two-way fixed effects + two-way clustering
resultPanel2 = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmId, yearId, ...
    entityEffects=true, timeEffects=true, covType="clustered_both");

fprintf("\nTwo-Way Fixed Effects + Two-Way Clustered SE:\n");
fprintf("  Coefficients: [%.3f, %.3f]\n", resultPanel2.params);
fprintf("  Std Errors: [%.3f, %.3f]\n", resultPanel2.stdErrors);
fprintf("  R-squared: %.4f\n", resultPanel2.rsquared);

%% 9. Poisson and Negative Binomial Regression
fprintf("\n=== 9. Count Data Models Example ===\n");

% Generate count data
lambda = exp(1 + 0.5*X(:,1) + 0.3*X(:,2));
yCount = poissrnd(lambda);

% Poisson regression
resultPoisson = pyBridge.StatsmodelsWrapper.poisson(yCount, X);
fprintf("Poisson Regression:\n");
fprintf("  Coefficients: [%.3f, %.3f]\n", resultPoisson.params(2:3));
fprintf("  Overdispersion Ratio: %.3f\n", resultPoisson.overdispersionRatio);
fprintf("  Overdispersion Test p-value: %.4f (%s)\n", ...
    resultPoisson.overdispersionPValue, ...
    ternary(resultPoisson.hasOverdispersion, "Significant", "Not Significant"));

% Negative binomial regression (handles overdispersion)
resultNegBin = pyBridge.StatsmodelsWrapper.negativeBinomial(yCount, X);
fprintf("\nNegative Binomial Regression:\n");
fprintf("  Coefficients: [%.3f, %.3f]\n", resultNegBin.params(2:3));
fprintf("  Alpha Parameter: %.3f (Overdispersion Parameter)\n", resultNegBin.alpha);

%% Complete
fprintf("\n=== All Examples Complete! ===\n");
