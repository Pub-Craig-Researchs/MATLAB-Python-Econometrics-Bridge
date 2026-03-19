%% PyBridge Quick Start Example
% This script demonstrates the basic usage of the PyBridge toolbox
% Author: WorkBuddy
% Date: 2026-03-18

%% 1. Environment Setup
fprintf("=== 1. Environment Setup ===\n");

% Add toolbox path
addpath('..');  % Assuming script is in examples subdirectory

% Create configuration instance
config = pyBridge.PyBridgeConfig();

% Initialize Python environment
config.initialize();

% Verify all required libraries
results = config.verifyAll();

fprintf("\n");

%% 2. scipy Statistical Functions
fprintf("=== 2. scipy Statistical Functions ===\n");

% Create statistics utility instance
stats = pyBridge.internal.ScipyStats();

% Generate sample data
rng(0, "twister");
data1 = randn(100, 1);
data2 = randn(100, 1) + 0.5;

% Normal distribution functions
fprintf("\nNormal Distribution:\n");
x = linspace(-3, 3, 5);
pdf = stats.normPDF(x, 0, 1);
fprintf("  x: "); fprintf("%.2f ", x); fprintf("\n");
fprintf("PDF: "); fprintf("%.4f ", pdf); fprintf("\n");

% t-test
fprintf("\nt-test:\n");
tResult = stats.tTest(data1, data2);
fprintf("  t-statistic: %.4f\n", tResult.tStatistic);
fprintf("  p-value: %.4f\n", tResult.pValue);
fprintf("  Significant: %s\n", ternary(tResult.significant, 'Yes', 'No'));

% Normality test
fprintf("\nNormality Test (Shapiro-Wilk):\n");
normResult = stats.normalityTest(data1, 'shapiro');
fprintf("  Statistic: %.4f\n", normResult.statistic);
fprintf("  p-value: %.4f\n", normResult.pValue);
fprintf("  Is Normal: %s\n", ternary(normResult.isNormal, 'Yes', 'No'));

% Descriptive statistics
fprintf("\nDescriptive Statistics:\n");
desc = stats.describe(data1);
fprintf("  Sample Size: %d\n", desc.nObs);
fprintf("  Mean: %.4f\n", desc.mean);
fprintf("  Std Dev: %.4f\n", sqrt(desc.variance));
fprintf("  Min: %.4f\n", desc.min);
fprintf("  Max: %.4f\n", desc.max);

fprintf("\n");

%% 3. scipy Optimization Functions
fprintf("=== 3. scipy Optimization Functions ===\n");

% Create optimization utility instance
opt = pyBridge.internal.ScipyOptimize();

% Example 1: Minimize Rosenbrock function
fprintf("\nMinimize Rosenbrock Function:\n");
rosenbrock = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
result = opt.minimize(rosenbrock, [0; 0]);
fprintf("  Minimum: %.6f\n", result.fun);
fprintf("  Optimal Solution: [%.6f, %.6f]\n", result.x);
fprintf("  Iterations: %d\n", result.nIter);

% Example 2: Curve fitting
fprintf("\nCurve Fitting (Linear):\n");
xData = linspace(0, 10, 50);
yTrue = 2.5 * xData + 1.5;
yNoisy = yTrue + randn(size(xData)) * 0.5;

linearFun = @(x, a, b) a * x + b;
fitResult = opt.curveFit(linearFun, xData, yNoisy);
fprintf("  True Parameters: a=%.4f, b=%.4f\n", 2.5, 1.5);
fprintf("  Fitted Parameters: a=%.4f, b=%.4f\n", fitResult.parameters);
fprintf("  Std Errors: [%.4f, %.4f]\n", fitResult.stdErrors);

fprintf("\n");

%% 4. scipy Signal Processing
fprintf("=== 4. scipy Signal Processing ===\n");

% Create signal processing utility instance
sig = pyBridge.internal.ScipySignal();

% Design filter
fprintf("\nButterworth Lowpass Filter:\n");
fs = 1000;  % Sampling rate
cutoff = 100;  % Cutoff frequency
[b, a] = sig.butter(4, cutoff/(fs/2), 'low');
fprintf("  Sampling Rate: %d Hz\n", fs);
fprintf("  Cutoff Frequency: %d Hz\n", cutoff);
fprintf("  Filter Order: 4\n");

% Generate test signal
t = 0:1/fs:0.5;
signal = sin(2*pi*50*t) + 0.5*sin(2*pi*200*t); % 50Hz + 200Hz
signalNoisy = signal + 0.3*randn(size(t));

% Filter
signalFiltered = sig.filtfilt(b, a, signalNoisy);

% Power spectral density
psd = sig.welch(signalNoisy, fs, nperseg=256);
fprintf("  PSD Estimation Complete, Frequency Points: %d\n", length(psd.frequencies));

fprintf("\n");

%% 5. statsmodels Regression Analysis
fprintf("=== 5. statsmodels Regression Analysis ===\n");

% OLS regression
fprintf("\nOLS Regression:\n");
n = 100;
X = randn(n, 2);
y = 1.5 + 2*X(:,1) - 3*X(:,2) + randn(n, 1)*0.5;

olsResult = pyBridge.StatsmodelsWrapper.ols(y, X);
fprintf("  R-squared: %.4f\n", olsResult.rSquared);
fprintf("  Adj R-squared: %.4f\n", olsResult.adjRSquared);
fprintf("  Parameter Estimates:\n");
fprintf("    Intercept: %.4f (True: 1.5)\n", olsResult.params(1));
fprintf("    X1: %.4f (True: 2.0)\n", olsResult.params(2));
fprintf("    X2: %.4f (True: -3.0)\n", olsResult.params(3));
fprintf("  AIC: %.2f\n", olsResult.aic);

% Heteroskedasticity test
fprintf("\nHeteroskedasticity Test (Breusch-Pagan):\n");
bpResult = pyBridge.StatsmodelsWrapper.breuschPagan(y, X);
fprintf("  LM Statistic: %.4f\n", bpResult.lmStatistic);
fprintf("  p-value: %.4f\n", bpResult.lmPValue);
fprintf("  Heteroskedasticity Present: %s\n", ternary(bpResult.hasHeteroskedasticity, 'Yes', 'No'));

fprintf("\n");

%% 6. linearmodels Panel Data
fprintf("=== 6. linearmodels Panel Data ===\n");

% Generate panel data
nEntities = 20;
nTimes = 10;
nObs = nEntities * nTimes;

yPanel = randn(nObs, 1);
XPanel = randn(nObs, 2);
entityIds = repelem((1:nEntities)', nTimes);
timeIds = repmat((1:nTimes)', nEntities, 1);

% Fixed effects model
fprintf("\nFixed Effects Model:\n");
feResult = pyBridge.LinearmodelsWrapper.panelOLS(yPanel, XPanel, entityIds, timeIds);
fprintf("  R-squared (Overall): %.4f\n", feResult.rSquared);
fprintf("  R-squared (Within): %.4f\n", feResult.rSquaredWithin);
fprintf("  Observations: %d\n", feResult.nObs);
fprintf("  Entities: %d\n", feResult.nEntities);

% Random effects model
fprintf("\nRandom Effects Model:\n");
reResult = pyBridge.LinearmodelsWrapper.randomEffects(yPanel, XPanel, entityIds, timeIds);

% Hausman test
fprintf("\nHausman Test:\n");
hausman = pyBridge.LinearmodelsWrapper.hausmanTest(feResult, reResult);
fprintf("  Statistic: %.4f\n", hausman.hausmanStatistic);
fprintf("  p-value: %.4f\n", hausman.pValue);
fprintf("  Conclusion: %s\n", hausman.conclusion);

fprintf("\n");

%% 7. econml Causal Inference
fprintf("=== 7. econml Causal Inference ===\n");

% Generate causal data
fprintf("\nDouble Machine Learning:\n");
nCausal = 500;
XCausal = randn(nCausal, 3);
TCausal = 0.5*XCausal(:,1) + randn(nCausal, 1)*0.5;
YCausal = 2*TCausal + 1.5*XCausal(:,1) - 0.8*XCausal(:,2) + randn(nCausal, 1)*0.5;

% DML estimation
dmlResult = pyBridge.EconmlWrapper.dml(YCausal, TCausal, XCausal);
fprintf("  True ATE: 2.0\n");
fprintf("  Estimated ATE: %.4f\n", dmlResult.ate);
fprintf("  95%% CI: [%.4f, %.4f]\n", ...
    dmlResult.ateConfInt.lower, dmlResult.ateConfInt.upper);
fprintf("  p-value: %.4f\n", dmlResult.atePValue);

fprintf("\n");

%% 8. Data Conversion Example
fprintf("=== 8. Data Conversion Example ===\n");

% MATLAB array to NumPy
fprintf("\nMATLAB Array -> NumPy:\n");
mlArray = magic(3);
pyArray = pyBridge.DataConverter.toPython(mlArray);
fprintf("  MATLAB Array Type: %s\n", class(mlArray));
fprintf("  Python Object Type: %s\n", char(py.getattr(py.type(pyArray), '__name__')));

% MATLAB table to Pandas DataFrame
fprintf("\nMATLAB Table -> DataFrame:\n");
mlTable = table(randn(10,1), randn(10,1), VariableNames=["A", "B"]);
pyDf = pyBridge.DataConverter.table2Df(mlTable);
fprintf("  MATLAB Table Size: %dx%d\n", height(mlTable), width(mlTable));
fprintf("  DataFrame Size: %dx%d\n", pyDf.shape{1}, pyDf.shape{2});

% Python dict to MATLAB struct
fprintf("\nPython Dict -> MATLAB Struct:\n");
pyDict = py.dict(a=1, b=2, c="test");
mlStruct = pyBridge.DataConverter.dict2Struct(pyDict);
fprintf("  Struct Fields: %s\n", strjoin(fieldnames(mlStruct), ", "));

fprintf("\n");

%% Complete
fprintf("=== Quick Start Example Complete! ===\n\n");
fprintf("More examples available at:\n");
fprintf("  - examples/scipy_examples.m\n");
fprintf("  - examples/statsmodels_examples.m\n");
fprintf("  - examples/linearmodels_examples.m\n");
fprintf("  - examples/econml_examples.m\n");
fprintf("  - README.md\n");

%% Helper Functions
function result = ternary(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end
