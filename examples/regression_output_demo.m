%% regression_output_demo.m
% Regression Output Formatting Example
% Demonstrates Outreg2-style table output, Zero-Inflated Negative Binomial, MLogit robust SE
%

clear; close all; clc;

%% 1. Generate Simulated Data
rng(0, "twister");
N = 500;

% Independent variables
X1 = randn(N, 1);
X2 = randn(N, 1) + 0.5 * X1;
X3 = randn(N, 1);
X = [X1, X2, X3];

% Firm IDs (for clustering)
firmIds = repmat((1:50)', 10, 1);
firmIds = firmIds(randperm(N));

% Dependent variable (continuous)
yCont = 2 + 0.5*X1 - 0.3*X2 + 0.1*X3 + randn(N, 1)*0.5;

% Dependent variable (binary)
yBinary = yCont > median(yCont);
yBinary = double(yBinary);

% Dependent variable (multi-class)
yMulti = zeros(N, 1);
yMulti(yCont < quantile(yCont, 0.33)) = 0;
yMulti(yCont >= quantile(yCont, 0.33) & yCont < quantile(yCont, 0.67)) = 1;
yMulti(yCont >= quantile(yCont, 0.67)) = 2;

% Dependent variable (count, for ZINB)
yCount = round(max(0, yCont * 2 + randn(N, 1)));
% Add extra zeros (simulate zero-inflation)
zeroIdx = rand(N, 1) < 0.3;
yCount(zeroIdx) = 0;

fprintf("Data Generation Complete:\n");
fprintf("  - Sample Size: %d\n", N);
fprintf("  - Independent Variables: X1, X2, X3\n");
fprintf("  - Number of Firm Clusters: %d\n", length(unique(firmIds)));
fprintf("  - Zero Proportion in Count Data: %.1f%%\n\n", 100*mean(yCount==0));

%% 2. Outreg2-Style Table Output Demo

fprintf("="*60 + "\n");
fprintf("Feature 1: Outreg2-Style Regression Table Output\n");
fprintf("="*60 + "\n\n");

% Create output table
tbl = pyBridge.RegressionTable();

% Model 1: OLS
result1 = pyBridge.StatsmodelsWrapper.ols(yCont, X, varNames={"X1", "X2", "X3"});
tbl.addModel("OLS", result1, varNames={"const", "X1", "X2", "X3"}, note="None");

% Model 2: OLS with HC1
result2 = pyBridge.StatsmodelsWrapper.ols(yCont, X, covType="HC1", varNames={"X1", "X2", "X3"});
tbl.addModel("OLS-HC1", result2, varNames={"const", "X1", "X2", "X3"}, covType="HC1");

% Model 3: OLS with clustered SE
result3 = pyBridge.StatsmodelsWrapper.ols(yCont, X, covType="cluster", clusterIds=firmIds, varNames={"X1", "X2", "X3"});
tbl.addModel("OLS-Cluster", result3, varNames={"const", "X1", "X2", "X3"}, covType="Firm Clustered");

% Display console table
fprintf("Console Output:\n");
tbl.display();

% Export to CSV
csvPath = "regression_results.csv";
tbl.toCSV(csvPath);

% Export to LaTeX
latexPath = "regression_results.tex";
tbl.toLaTeX(latexPath, caption="Regression Results Comparison", label="tab:regression");

% Export to Excel
excelPath = "regression_results.xlsx";
tbl.toExcel(excelPath);

%% 3. Zero-Inflated Negative Binomial Demo

fprintf("\n" + "="*60 + "\n");
fprintf("Feature 2: Zero-Inflated Negative Binomial (ZINB)\n");
fprintf("="*60 + "\n\n");

fprintf("Count Data Characteristics:\n");
fprintf("  - Zero Count: %d (%.1f%%)\n", sum(yCount==0), 100*mean(yCount==0));
fprintf("  - Mean: %.2f\n", mean(yCount));
fprintf("  - Variance: %.2f\n", var(yCount));
fprintf("  - Overdispersion Ratio (Var/Mean): %.2f\n\n", var(yCount)/mean(yCount));

% Standard Negative Binomial regression
resultNB = pyBridge.StatsmodelsWrapper.negativeBinomial(yCount, X, varNames={"X1", "X2", "X3"});
fprintf("Negative Binomial Regression Results:\n");
fprintf("  - Alpha (Overdispersion Parameter): %.4f\n", resultNB.alpha);
fprintf("  - AIC: %.2f\n", resultNB.aic);
fprintf("  - BIC: %.2f\n\n", resultNB.bic);

% Zero-Inflated Negative Binomial regression
resultZINB = pyBridge.StatsmodelsWrapper.zeroInflatedNB(yCount, X, varNames={"X1", "X2", "X3"});

fprintf("Zero-Inflated Negative Binomial Results:\n");
fprintf("  - Count Part Coefficients:\n");
fprintf("    const: %.4f, X1: %.4f, X2: %.4f, X3: %.4f\n", ...
    resultZINB.paramsCount);
fprintf("  - Zero-Inflation Part Coefficients:\n");
fprintf("    %.4f\n", resultZINB.paramsInflate);
fprintf("  - Alpha: %.4f\n", resultZINB.alpha);
fprintf("  - AIC: %.2f\n", resultZINB.aic);
fprintf("  - BIC: %.2f\n", resultZINB.bic);

if isfield(resultZINB, 'vuongTest')
    fprintf("\nVuong Test (ZINB vs NB):\n");
    fprintf("  - Statistic: %.4f\n", resultZINB.vuongTest.statistic);
    fprintf("  - p-value: %.4f\n", resultZINB.vuongTest.pValue);
    if resultZINB.vuongTest.preferZINB
        fprintf("  - Conclusion: ZINB model preferred\n");
    else
        fprintf("  - Conclusion: ZINB model not preferred\n");
    end
end

%% 4. MLogit Robust Standard Errors Demo

fprintf("\n" + "="*60 + "\n");
fprintf("Feature 3: MLogit Robust Standard Errors\n");
fprintf("="*60 + "\n\n");

fprintf("Multi-Class Data Distribution:\n");
fprintf("  - Category 0: %d (%.1f%%)\n", sum(yMulti==0), 100*mean(yMulti==0));
fprintf("  - Category 1: %d (%.1f%%)\n", sum(yMulti==1), 100*mean(yMulti==1));
fprintf("  - Category 2: %d (%.1f%%)\n\n", sum(yMulti==2), 100*mean(yMulti==2));

% Standard MLogit
resultMLogit1 = pyBridge.StatsmodelsWrapper.multinomialLogit(yMulti, X, varNames={"X1", "X2", "X3"});

fprintf("MLogit Classical SE:\n");
fprintf("  - Model Fit Successful\n");
fprintf("  - Number of Categories: %d\n", resultMLogit1.nCategories);
fprintf("  - AIC: %.2f\n\n", resultMLogit1.aic);

% MLogit with clustered SE
resultMLogit2 = pyBridge.StatsmodelsWrapper.multinomialLogit(yMulti, X, ...
    covType="cluster", clusterIds=firmIds, varNames={"X1", "X2", "X3"});

fprintf("MLogit Clustered SE:\n");
fprintf("  - Number of Clusters: %d\n", resultMLogit2.nClusters);
fprintf("  - Covariance Type: %s\n\n", resultMLogit2.covType);

% Compare standard errors
fprintf("Standard Error Comparison (Selected Coefficients):\n");
fprintf("%-20s %12s %12s\n", "Parameter", "Classical SE", "Clustered SE");
fprintf("%-20s %12.4f %12.4f\n", "X1(Category 1)", resultMLogit1.stdErrors(1), resultMLogit2.stdErrors(1));
fprintf("%-20s %12.4f %12.4f\n", "X1(Category 2)", resultMLogit1.stdErrors(4), resultMLogit2.stdErrors(4));

%% 5. Comprehensive Output Table

fprintf("\n" + "="*60 + "\n");
fprintf("Comprehensive Output: Integrate All Models into One Table\n");
fprintf("="*60 + "\n\n");

% Create comprehensive table
finalTbl = pyBridge.RegressionTable();
finalTbl.setVarOrder(["const", "X1", "X2", "X3"]);

% Add various models
finalTbl.addModel("OLS", result1, varNames={"const", "X1", "X2", "X3"});
finalTbl.addModel("Logit", ...
    pyBridge.StatsmodelsWrapper.logistic(yBinary, X, varNames={"X1", "X2", "X3"}), ...
    varNames={"const", "X1", "X2", "X3"});
finalTbl.addModel("NB", resultNB, varNames={"const", "X1", "X2", "X3", "alpha"});

% Export final table
finalTbl.toExcel("final_regression_table.xlsx");

%% 6. MLogit Multi-Column Output Demo (Outreg2 Style)

fprintf("\n" + "="*60 + "\n");
fprintf("Feature 4: MLogit Multi-Column Output (Outreg2 Style, One Column per Category)\n");
fprintf("="*60 + "\n\n");

% Run MLogit
resultMLogit = pyBridge.StatsmodelsWrapper.multinomialLogit(yMulti, X, varNames={"X1", "X2", "X3"});

% Create MLogit-specific table
mlogitTbl = pyBridge.RegressionTable();
mlogitTbl.addMLogit("MLogit", resultMLogit, ...
    varNames={"X1", "X2", "X3"}, ...
    categoryNames={"Category 1", "Category 2"}, ...  % Excluding reference level
    referenceName="Baseline");

fprintf("MLogit Output (One Column per Category):\n");
mlogitTbl.display();

% Export
mlogitTbl.toExcel("mlogit_results.xlsx");

%% 7. ZINB Two-Part Output Demo

fprintf("\n" + "="*60 + "\n");
fprintf("Feature 5: ZINB Zero-Inflation Part Output\n");
fprintf("="*60 + "\n\n");

% Create ZINB table
zinbTbl = pyBridge.RegressionTable();
zinbTbl.addZINB("ZINB", resultZINB, ...
    varNames={"X1", "X2", "X3"}, ...
    note="Zero-Inflated Model");

zinbTbl.display();
zinbTbl.toExcel("zinb_results.xlsx");

%% 8. MLogit HAC Standard Errors Demo

fprintf("\n" + "="*60 + "\n");
fprintf("Feature 6: MLogit HAC Standard Errors (Newey-West)\n");
fprintf("="*60 + "\n\n");

% Generate time-series correlated data
T = 300;
X_ts = randn(T, 3);
y_ts = zeros(T, 1);
for t = 2:T
    X_ts(t, :) = 0.7 * X_ts(t-1, :) + randn(1, 3) * 0.5;
end
yCont_ts = 1 + 0.5*X_ts(:,1) - 0.3*X_ts(:,2) + 0.2*X_ts(:,3) + randn(T, 1)*0.3;
yMulti_ts = discretize(yCont_ts, [min(yCont_ts), quantile(yCont_ts, 0.33), quantile(yCont_ts, 0.67), max(yCont_ts)+0.1]) - 1;

% Classical SE
resultMLogitClassic = pyBridge.StatsmodelsWrapper.multinomialLogit(yMulti_ts, X_ts, varNames={"X1", "X2", "X3"});

% HAC SE
% Note: "newey-west" is internally mapped to "bartlett"
resultMLogitHAC = pyBridge.StatsmodelsWrapper.multinomialLogit(yMulti_ts, X_ts, ...
    covType="HAC", lag=4, varNames={"X1", "X2", "X3"});

fprintf("MLogit HAC SE Results:\n");
fprintf("  - Covariance Type: %s\n", resultMLogitHAC.covType);
fprintf("  - Lag: %d\n", resultMLogitHAC.lag);
fprintf("  - Kernel: %s\n", resultMLogitHAC.kernel);
fprintf("  - AIC: %.2f\n\n", resultMLogitHAC.aic);

% Compare standard errors
fprintf("Standard Error Comparison (Category 1-X1):\n");
fprintf("  Classical SE: %.4f\n", resultMLogitClassic.stdErrors(1));
fprintf("  HAC SE:       %.4f\n", resultMLogitHAC.stdErrors(1));
fprintf("  Difference: %.1f%%\n\n", 100*abs(resultMLogitHAC.stdErrors(1)-resultMLogitClassic.stdErrors(1))/resultMLogitClassic.stdErrors(1));

%% 9. Marginal Effects and Significance Demo

fprintf("\n" + "="*60 + "\n");
fprintf("Feature 7: Discrete Model Marginal Effects and Significance Tests\n");
fprintf("="*60 + "\n\n");

% Logit model marginal effects
logitResult = pyBridge.StatsmodelsWrapper.logistic(yBinary, X, varNames={"X1", "X2", "X3"});

fprintf("Logit Model Marginal Effects (At Mean):\n");
pyBridge.StatsmodelsWrapper.printMarginalEffects(logitResult);

% Compute marginal effects and get results
margEff = pyBridge.StatsmodelsWrapper.marginalEffects(logitResult);
fprintf("\nMarginal Effects Detailed Results:\n");
fprintf("  Number of Variables: %d\n", length(margEff.effects));
fprintf("  Computation Method: %s\n", margEff.method);
fprintf("  Evaluation Point: %s\n", margEff.at);

% Compute marginal effects at median
fprintf("\nMarginal Effects at Median:\n");
margEffMedian = pyBridge.StatsmodelsWrapper.marginalEffects(logitResult, at="median");
for i = 1:length(margEffMedian.effects)
    fprintf("  %s: %.4f (SE: %.4f, p: %.4f)\n", ...
        margEffMedian.varNames{i}, margEffMedian.effects(i), ...
        margEffMedian.stdErrors(i), margEffMedian.pValues(i));
end

% MLogit marginal effects
fprintf("\nMLogit Model Marginal Effects:\n");
mlogitMargEff = pyBridge.StatsmodelsWrapper.marginalEffects(resultMLogit1);
fprintf("  Marginal Effects Matrix Dimensions: %s\n", mat2str(size(mlogitMargEff.effects)));

%% 10. Marginal Effects Table Output

fprintf("\n" + "="*60 + "\n");
fprintf("Feature 8: Marginal Effects Table Output\n");
fprintf("="*60 + "\n\n");

% Create marginal effects comparison table
margTbl = pyBridge.RegressionTable();
margTbl.setVarOrder(["const", "X1", "X2", "X3"]);

% Add coefficient results
margTbl.addModel("Coefficients", logitResult, varNames={"const", "X1", "X2", "X3"});

% Add marginal effects results (as new model)
margEffResult = struct();
margEffResult.params = margEff.effects;
margEffResult.stdErrors = margEff.stdErrors;
margEffResult.pValues = margEff.pValues;
margEffResult.nObs = logitResult.nObs;
margEffResult.modelType = "MarginalEffects";
margTbl.addModel("Marg.Effects", margEffResult, varNames=margEff.varNames);

margTbl.display();
margTbl.toExcel("marginal_effects_table.xlsx");

fprintf("\n" + "="*60 + "\n");
fprintf("Complete! All Output Files Generated:\n");
fprintf("="*60 + "\n");
fprintf("  - regression_results.csv\n");
fprintf("  - regression_results.tex\n");
fprintf("  - regression_results.xlsx\n");
fprintf("  - final_regression_table.xlsx\n");
fprintf("  - mlogit_results.xlsx (MLogit Multi-Column)\n");
fprintf("  - zinb_results.xlsx (ZINB Two-Part)\n");
fprintf("  - marginal_effects_table.xlsx (Marginal Effects)\n");
