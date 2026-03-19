# PyBridge - MATLAB Python Scientific Computing Toolbox

**PyBridge** is a MATLAB toolbox that provides a clean interface to Python scientific computing libraries (scipy, statsmodels, linearmodels, econml), enabling users to call Python functions via the `py.` syntax while automatically handling data conversion and result parsing.

## Features

- **Environment Management**: Automatic Python environment detection and configuration
- **Data Conversion**: Bidirectional MATLAB-Python data conversion
- **Result Parsing**: Convert Python return objects to MATLAB-friendly formats
- **Error Handling**: Unified exception handling with MATLAB-style error messages
- **Complete Wrappers**:
  - `StatsmodelsWrapper` - OLS, WLS, GLM, Logistic, Probit, MNLogit, Ordered, NegBin, ZINB, VAR, ARIMA, and more
  - `LinearmodelsWrapper` - PanelOLS, RandomEffects, BetweenOLS, PooledOLS, FirstDifference, IV2SLS/LIML/GMM, Hausman test
  - `EconmlWrapper` - DML, DR Learner, S/T/X Learner, Causal Forest, sensitivity analysis
  - `ScipyWrapper` - stats, optimize, signal, linalg, integrate
- **Supporting Components**:
  - `CovarianceTypes` - HAC, HC0-HC3, Clustered, Multi-way Clustered, MLE Robust
  - `ResultParser` - Automatic parsing of Python results to MATLAB structures
  - `DataConverter` - Bidirectional MATLAB-Python data conversion with py.None handling
  - `ErrorHandler` - Unified exception handling with MATLAB-style error messages
  - `PyBridgeConfig` - Environment configuration and package verification

### Advanced Econometrics Support

- **Discrete Choice Models**: Multinomial Logit, Ordered Logit/Probit, Poisson, Negative Binomial, Zero-Inflated Negative Binomial (ZINB)
- **Robust Standard Errors**: HC0-HC3 heteroskedasticity-robust standard errors
- **HAC Standard Errors**: Newey-West standard errors (handling autocorrelation and heteroskedasticity)
- **Clustered Standard Errors**: Single-way clustering, multi-way clustering (e.g., firm x year two-way clustering)
- **Panel Data**: Fixed effects, random effects, two-way fixed effects with clustered standard errors
- **Time Series**: VAR models, ARIMA, Durbin-Watson test, unit root tests
- **Generalized Models**: GLM (Gaussian, Binomial, Poisson, Gamma, InverseGaussian families)

## Test Status

**137 tests, 135 passing, 0 failures, 2 skipped** (6 test suites)

| Test Suite | Description | Status |
|------------|-------------|--------|
| AcademicCorrectionsTest | Academic notation and statistical corrections | ✓ Pass |
| MLogitHACSpecializedTest | Multinomial Logit with HAC standard errors | ✓ Pass |
| PyBridgeConsistencyTest | Core functionality and API consistency | ✓ Pass |
| PanelCausalTest | Panel data analysis and causal inference (EconmlWrapper + LinearmodelsWrapper) | ✓ Pass |
| StatsParserCovTest | Statistical functions, result parsing and covariance types (ScipyStats + ResultParser + CovarianceTypes) | ✓ Pass |
| StatsmodelsExtendedTest | Extended econometric models (WLS, GLM, NegBin, ZINB, VAR, Durbin-Watson, etc.) | ✓ Pass |

## Quick Start

### 1. Requirements

- MATLAB R2020b or higher
- Python 3.7+ (miniconda/anaconda recommended)
  - **Python 3.13 Compatibility**: Fully tested with MATLAB R2025b, all features working correctly (though not officially supported by MathWorks yet)
  - **linearmodels v7.0 Compatibility**: Updated for new PanelOLS API constructor syntax
- Required Python packages:
  ```bash
  pip install scipy statsmodels linearmodels econml numpy pandas scikit-learn
  ```

### 2. Configure Environment

```matlab
% Add toolbox to path (use the actual path where you cloned this repo)
addpath(genpath('path/to/MATLAB-Python-Econometrics-Bridge'));  % Or use startup.m

% Initialize environment
config = pyBridge.PyBridgeConfig();
config.initialize();
config.verifyAll();  % Verify all packages are installed
```

### 3. Basic Usage

#### scipy Statistics

```matlab
% Create statistics instance
stats = pyBridge.internal.ScipyStats();

% Normal distribution
x = linspace(-3, 3, 100);
pdf = stats.normPDF(x, 0, 1);
cdf = stats.normCDF(x, 0, 1);

% t-test
data1 = randn(100, 1);
data2 = randn(100, 1) + 0.5;
result = stats.tTest(data1, data2);
fprintf('t-statistic: %.4f, p-value: %.4f\n', result.tStatistic, result.pValue);

% Normality test
normality = stats.normalityTest(data1, 'shapiro');
fprintf('Is normal: %s\n', ternary(normality.isNormal, 'Yes', 'No'));
```

#### scipy Optimization

```matlab
% Create optimization instance
opt = pyBridge.internal.ScipyOptimize();

% Minimize function
fun = @(x) x(1)^2 + x(2)^2;
result = opt.minimize(fun, [1; 1]);
fprintf('Minimum: %.6f, Location: [%.6f, %.6f]\n', ...
    result.fun, result.x(1), result.x(2));

% Curve fitting
xData = linspace(0, 10, 50);
yData = 2.5 * xData + 1.5 + randn(size(xData)) * 0.5;

fitFun = @(x, a, b) a * x + b;
fitResult = opt.curveFit(fitFun, xData, yData);
fprintf('Fitted parameters: a=%.4f, b=%.4f\n', fitResult.parameters);
```

#### scipy Signal Processing

```matlab
% Create signal processing instance
sig = pyBridge.internal.ScipySignal();

% Design Butterworth filter
[b, a] = sig.butter(4, 0.1, 'low');

% Filter signal
fs = 1000; % Sampling rate
t = 0:1/fs:1;
x = sin(2*pi*50*t) + 0.5*randn(size(t)); % 50Hz signal + noise
y = sig.filtfilt(b, a, x);

% Power spectral density
psd = sig.welch(x, fs);
figure;
plot(psd.frequencies, psd.powerSpectralDensity);
xlabel('Frequency (Hz)'); ylabel('Power Spectral Density');
```

#### statsmodels Regression

```matlab
% OLS regression
n = 100;
X = randn(n, 2);
y = 1.5 + 2*X(:,1) - 3*X(:,2) + randn(n, 1)*0.5;

result = pyBridge.StatsmodelsWrapper.ols(y, X);
pyBridge.ResultParser.printResult(result);

% Logistic regression
yBinary = double(y > median(y));
logitResult = pyBridge.StatsmodelsWrapper.logistic(yBinary, X);
fprintf('AIC: %.2f, Confusion Matrix:\n', logitResult.aic);
disp(logitResult.confusionMatrix);

% Time series
y = cumsum(randn(200, 1)); % Random walk
arimaResult = pyBridge.StatsmodelsWrapper.arima(y, [1, 0, 1]);

% Unit root test
adfResult = pyBridge.StatsmodelsWrapper.adfuller(y);
fprintf('ADF test p-value: %.4f, Stationary: %s\n', ...
    adfResult.pValue, ternary(adfResult.isStationary, 'Yes', 'No'));
```

#### linearmodels Panel Data

```matlab
% Panel data
nEntities = 50;
nTimes = 10;
nObs = nEntities * nTimes;

y = randn(nObs, 1);
X = randn(nObs, 2);
entityIds = repmat((1:nEntities)', nTimes, 1);
timeIds = repmat(1:nTimes, nEntities, 1);

% Fixed effects model
feResult = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds);
fprintf('R-squared: %.4f\n', feResult.rSquared);

% Random effects model
reResult = pyBridge.LinearmodelsWrapper.randomEffects(y, X, entityIds, timeIds);

% Hausman test
hausman = pyBridge.LinearmodelsWrapper.hausmanTest(feResult, reResult);
fprintf('%s\n', hausman.conclusion);
```

### Advanced Econometrics

#### Discrete Choice Models

```matlab
%% Multinomial Logit (Multinomial Classification)
y = [0; 1; 2; 0; 1; ...]; % Multinomial dependent variable (0,1,2,...)
X = randn(500, 3);
result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);
fprintf('Number of categories: %d\n', result.nCategories);
fprintf('Marginal effects:\n');
disp(result.marginalEffects);

%% Ordered Logit (Ordinal Choice)
yOrdered = [0; 1; 2; 3; 1; ...]; % Ordinal dependent variable (0,1,2,3,...)
result = pyBridge.StatsmodelsWrapper.orderedLogit(yOrdered, X);
fprintf('Threshold parameters:\n');
disp(result.thresholds);

%% Count Data Models
yCount = poissrnd(lambda); % Poisson-distributed count data
resultPoisson = pyBridge.StatsmodelsWrapper.poisson(yCount, X);
fprintf('Overdispersion ratio: %.3f\n', resultPoisson.overdispersionRatio);
fprintf('Overdispersion test p-value: %.4f (%s)\n', ...
    resultPoisson.overdispersionPValue, ...
    ternary(resultPoisson.hasOverdispersion, 'Significant', 'Not significant'));

% Negative binomial regression (for overdispersed data)
resultNegBin = pyBridge.StatsmodelsWrapper.negativeBinomial(yCount, X);
fprintf('Alpha parameter: %.3f\n', resultNegBin.alpha);
```

#### Robust Standard Errors

```matlab
%% Heteroskedasticity-robust standard errors
resultHC0 = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC0"); % White
resultHC1 = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC1"); % Stata default
resultHC3 = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC3"); % Most conservative

%% HAC standard errors (Newey-West)
% Handle autocorrelation and heteroskedasticity
resultHAC = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="hac", maxLags=4, kernel="newey-west");
fprintf('HAC standard errors: [%.3f, %.3f]\n', resultHAC.stdErrors(2:3));

%% Clustered standard errors
% Single-way clustering (by firm)
resultCluster = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="cluster", clusterIds=firmIds);
fprintf('Number of clusters: %d\n', resultCluster.nClusters);

% Multi-way clustering (firm x year)
resultMultiway = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="multiway", clusterGroups={firmIds, yearIds});
fprintf('Standard errors: [%.3f, %.3f]\n', resultMultiway.stdErrors(2:3));
```

#### Panel Data + Clustered Standard Errors

```matlab
%% Fixed effects + firm-clustered standard errors
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
    entityEffects=true, covType="clustered_entity");

%% Two-way fixed effects + two-way clustered standard errors
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
    entityEffects=true, timeEffects=true, covType="clustered_both");
fprintf('Coefficients: [%.3f, %.3f]\n', result.params);
fprintf('Standard errors: [%.3f, %.3f]\n', result.stdErrors);
```

#### econml Causal Inference

```matlab
% Double Machine Learning
n = 1000;
X = randn(n, 5); % Covariates
T = 0.5*X(:,1) + randn(n, 1); % Treatment variable
Y = 2*T + 1.5*X(:,1) - 0.8*X(:,2) + randn(n, 1); % Outcome variable

dmlResult = pyBridge.EconmlWrapper.dml(Y, T, X);
fprintf('Average Treatment Effect (ATE): %.4f\n', dmlResult.ate);
fprintf('95%% Confidence Interval: [%.4f, %.4f]\n', ...
    dmlResult.ateConfInt.lower, dmlResult.ateConfInt.upper);

% Causal Forest
cfResult = pyBridge.EconmlWrapper.causalForest(Y, T, X);
fprintf('Mean CATE: %.4f\n', mean(cfResult.cate));
```

## Documentation

### Core Components

#### 1. PyBridgeConfig - Environment Configuration

```matlab
config = pyBridge.PyBridgeConfig();
config.initialize();           % Initialize Python environment
config.verifyAll();            % Verify all packages
config.printInfo();            % Print environment info
config.checkLibrary('scipy');  % Check a single library
```

#### 2. DataConverter - Data Conversion

```matlab
% MATLAB -> Python
pyArray = pyBridge.DataConverter.toPython(magic(3));
pyDf = pyBridge.DataConverter.table2Df(tableData);
pyDict = pyBridge.DataConverter.struct2Dict(structData);

% Python -> MATLAB
mlArray = pyBridge.DataConverter.toMatlab(pyArray);
mlTable = pyBridge.DataConverter.df2Table(pyDf);
mlStruct = pyBridge.DataConverter.dict2Struct(pyDict);
```

#### 3. ResultParser - Result Parsing

```matlab
% Auto-parse
result = pyBridge.ResultParser.parse(pyObject);

% Type-specific parsing
statsResult = pyBridge.ResultParser.parseStatsmodels(modelResult);
econmlResult = pyBridge.ResultParser.parseEconML(ateResult);

% Print results
pyBridge.ResultParser.printResult(result);
```

#### 4. ErrorHandler - Error Handling

```matlab
% Wrap calls
result = pyBridge.ErrorHandler.wrapCall(@() pyFunction());

% Manual handling
try
    result = pyFunction();
catch ME
    pyBridge.ErrorHandler.handlePyError(ME);
end

% Safe call (returns default on failure)
result = pyBridge.ErrorHandler.safeCall(@() riskyFunction(), defaultValue);
```

### Advanced Usage

#### Direct Access to Python Objects

All wrappers retain access to the original Python object:

```matlab
% Get original Python object
result = pyBridge.StatsmodelsWrapper.ols(y, X);
pyObj = result.originalObject;  % If original object was saved

% Call Python methods directly
pyObj.summary();
```

#### Custom Callback Functions

Use MATLAB functions in optimization and fitting:

```matlab
% Custom objective function
objective = @(x) (x(1)-1)^2 + (x(2)-2.5)^2;
result = pyBridge.internal.ScipyOptimize().minimize(objective, [0, 0]);

% Custom fitting function
fitFun = @(x, a, b, c) a*exp(-b*x) + c;
result = pyBridge.internal.ScipyOptimize().curveFit(fitFun, xData, yData);
```

#### Batch Operations

```matlab
% Batch hypothesis testing
stats = pyBridge.internal.ScipyStats();
tests = {'shapiro', 'normaltest', 'kstest'};
for i = 1:length(tests)
    result = stats.normalityTest(data, tests{i});
    fprintf('%s: p=%.4f\n', tests{i}, result.pValue);
end
```

## FAQ

### Q: How to change Python path?

```matlab
config = pyBridge.PyBridgeConfig('path/to/python');
config.initialize();
```

### Q: How to install missing Python packages?

```matlab
% Check for missing packages
config = pyBridge.PyBridgeConfig.getInstance();
config.verifyAll();

% Install via pip (in system terminal)
% pip install scipy statsmodels linearmodels econml
```

### Q: What data types are supported for conversion?

- **MATLAB -> Python**:

  - Numeric arrays -> NumPy arrays
  - Tables -> Pandas DataFrames
  - Structs -> dicts
  - Cell arrays -> lists
- **Python -> MATLAB**:

  - NumPy arrays -> double arrays
  - Pandas DataFrames -> tables
  - dicts -> structs
  - lists -> cells

### Q: How to handle large datasets?

```matlab
% Use chunked processing
nChunks = 10;
for i = 1:nChunks
    chunk = data((i-1)*chunkSize+1 : i*chunkSize, :);
    chunkPy = pyBridge.DataConverter.toPython(chunk);
    % Process...
end
```

## Notes

1. **Memory Management**: Large data conversions may consume significant memory; consider chunked processing
2. **Type Conversion**: Python and MATLAB types don't map 1-to-1; watch for precision loss
3. **Function Signatures**: Refer to each library's official documentation for function parameters
4. **Performance**: Python-MATLAB interaction has overhead; for performance-critical scenarios, consider pure Python implementation


## License

GPL3.0 License

## References

- [scipy Documentation](https://docs.scipy.org/doc/scipy/)
- [statsmodels Documentation](https://www.statsmodels.org/)
- [linearmodels Documentation](https://bashtage.github.io/linearmodels/)
- [econml Documentation](https://econml.azurewebsites.net/)
- [MATLAB Python Interface](https://www.mathworks.com/help/matlab/call-python-libraries.html)

---

**Assistant by WorkBuddy & Qoder**
