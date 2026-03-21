# PyBridge API Reference

## Table of Contents

1. [Core Components](#core-components)
2. [scipy Wrapper](#scipy-wrapper)
3. [statsmodels Wrapper](#statsmodels-wrapper)
4. [linearmodels Wrapper](#linearmodels-wrapper)
5. [econml Wrapper](#econml-wrapper)
6. [Advanced Econometric Methods](#advanced-econometric-methods)

---

## Core Components

### PyBridgeConfig

Environment configuration and library verification class.

#### Constructor

```matlab
config = pyBridge.PyBridgeConfig(pythonPath)
```

**Parameters:**
- `pythonPath` (optional): Python executable path, automatically detects current Python environment by default

#### Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Initialize Python environment |
| `checkLibrary(libName)` | Check if specified library is installed |
| `getLibVersion(libName)` | Get library version information |
| `verifyAll()` | Verify all required libraries, returns table |
| `printInfo()` | Print environment information |

#### Static Methods

| Method | Description |
|--------|-------------|
| `getInstance()` | Get singleton instance |
| `quickCheck()` | Quick check if environment is ready |

---

### DataConverter

MATLAB-Python bidirectional data conversion tool.

---

## Advanced Econometric Methods

### CovarianceTypes - Covariance Matrix Computation

Provides multiple robust standard error calculation methods, located in `pyBridge.internal.CovarianceTypes`.

#### HAC Standard Errors (Newey-West)

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.hac(residuals, X, options)
```

**Parameters:**
- `residuals`: Regression residuals
- `X`: Design matrix
- `options.maxLags`: Maximum lag order (default: auto-selected)
- `options.kernel`: Kernel function type
  - `"bartlett"` - Bartlett kernel (default, equivalent to Newey-West)
  - `"newey-west"` - Newey-West kernel (internally mapped to bartlett)
  - `"parzen"` - Parzen kernel
  - `"qs"` - Quadratic spectral kernel

**Notes:**
- `newey-west` is internally mapped to `bartlett`, both are equivalent
- `parzen` and `qs` kernels are not supported in discrete choice models (MNLogit/Logit/Probit), will automatically fall back to `bartlett` with a warning

**Returns:**
- `covMatrix`: HAC covariance matrix

**Example:**
```matlab
% First fit OLS
result = pyBridge.StatsmodelsWrapper.ols(y, X);
residuals = result.residuals;

% Compute HAC standard errors (default uses bartlett kernel)
covHAC = pyBridge.internal.CovarianceTypes.hac(residuals, X, ...
    maxLags=4, kernel="bartlett");
stdErrorsHAC = sqrt(diag(covHAC));
```

#### Clustered Standard Errors

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.clustered(residuals, X, clusterIds)
```

**Parameters:**
- `residuals`: Regression residuals
- `X`: Design matrix
- `clusterIds`: Cluster identifier vector
- `options.useCorrection`: Whether to use small sample correction (default: true)

**Example:**
```matlab
% Cluster by firm
firmIds = [1,1,1,2,2,2,...]; % Firm IDs
covCluster = pyBridge.internal.CovarianceTypes.clustered(...
    residuals, X, firmIds);
```

#### Multi-way Clustered Standard Errors

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.multiwayClustered(...
    residuals, X, clusterGroups)
```

**Parameters:**
- `clusterGroups`: Cell array containing multiple clustering dimensions
  - e.g., `{firmIds, yearIds}` for firm × year two-way clustering

**Example:**
```matlab
% Two-way clustering: firm × year
covMultiway = pyBridge.internal.CovarianceTypes.multiwayClustered(...
    residuals, X, {firmIds, yearIds});

% Three-way clustering: firm × year × industry
covMultiway3 = pyBridge.internal.CovarianceTypes.multiwayClustered(...
    residuals, X, {firmIds, yearIds, industryIds});
```

#### Heteroskedasticity-Robust Standard Errors (HC Series)

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.heteroskedastic(residuals, X, type)
```

**Parameters:**
- `type`: HC type
  - `"HC0"` - White standard errors
  - `"HC1"` - Stata small sample correction (recommended)
  - `"HC2"` - Suitable for small samples
  - `"HC3"` - Most conservative, for severe heteroskedasticity

---

### StatsmodelsWrapper - Advanced Regression Methods

#### Multinomial Logit

```matlab
result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, options)
```

**Parameters:**
- `y`: Multi-category dependent variable (integer encoded: 0, 1, 2, ..., K-1)
- `X`: Independent variable matrix
- `options.addConstant`: Whether to add constant term (default: true)
- `options.maxIter`: Maximum iterations (default: 100)
- `options.covType`: Covariance matrix type (optional)
  - `"nonrobust"` - Classical standard errors (default)
  - `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"` - Heteroskedasticity-robust standard errors
  - `"hac"` - HAC standard errors (only bartlett kernel supported)
- `options.covKwds`: HAC parameters (used when covType="hac")
  - `covKwds.maxlags`: Maximum lag order
  - `covKwds.kernel`: Kernel function (only "bartlett" supported)

**Return Fields:**
- `params`: Coefficient matrix ((K-1) × p)
- `stdErrors`: Standard errors
- `tStatistics`: t-statistics
- `pValues`: p-values
- `probabilities`: Predicted probabilities (n × K)
- `marginalEffects`: Marginal effects
- `marginalEffectsSE`: Marginal effects standard errors
- `marginalEffectsP`: Marginal effects p-values
- `marginalEffectsT`: Marginal effects t-statistics
- `nCategories`: Number of categories

**HAC Standard Errors Support Notes:**
- MNLogit supports HAC standard errors, but only with `bartlett` kernel
- `parzen` and `qs` kernels are not supported, will automatically fall back to `bartlett` with a warning

**Example:**
```matlab
% Occupational choice model (blue-collar=0, white-collar=1, professional=2)
y = [0; 1; 2; 0; 1; ...];
X = [education, experience, age];
result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);

% Using HAC standard errors
resultHAC = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
    covType="hac", covKwds=struct('maxlags', 4, 'kernel', 'bartlett'));
```

#### Ordered Logit/Probit

```matlab
result = pyBridge.StatsmodelsWrapper.orderedLogit(y, X, options)
result = pyBridge.StatsmodelsWrapper.orderedProbit(y, X, options)
```

**Parameters:**
- `y`: Ordered dependent variable (integer encoded: 0, 1, 2, ..., K-1)
- `X`: Independent variable matrix
- `options.addConstant` (logical, default: **false**): Whether to add constant term. Note: OrderedModel does not allow constant column in X (threshold parameters serve as intercepts)
- `options.maxIter` (double, default: **1000**): Maximum iterations for MLE optimization
- `options.covType` (string, default: "nonrobust"): Covariance matrix type: "nonrobust", "HC0", "HC1", "HC2", "HC3"

**Return Fields:**
- `coefficients`: Regression coefficients (excluding thresholds)
- `stdErrors`: Coefficient standard errors
- `thresholds`: Actual cutpoints (converted to Stata convention format)
- `rawThresholds`: Original statsmodels parameterization format (log-diff format)
- `thresholdStdErrors`: Threshold standard errors
- `params`: Coefficient estimates (same as coefficients)
- `tStatistics`: t-statistics
- `pValues`: p-values

**Example:**
```matlab
% Satisfaction rating (dissatisfied=0, neutral=1, satisfied=2, very satisfied=3)
y = [0; 1; 2; 3; 1; ...];
% Note: addConstant=false by default (OrderedModel uses thresholds as intercepts)
result = pyBridge.StatsmodelsWrapper.orderedLogit(y, X);
fprintf('Thresholds: ');
disp(result.thresholds);

% Using robust standard errors
result = pyBridge.StatsmodelsWrapper.orderedLogit(y, X, covType="HC1");
```

#### Poisson Regression

```matlab
result = pyBridge.StatsmodelsWrapper.poisson(y, X, options)
```

**Parameters:**
- `y`: Count data
- `X`: Independent variable matrix
- `options.addConstant` (logical, default: **true**): Whether to add constant term
- `options.exposure`: Exposure variable (optional)
- `options.covType` (string, default: "nonrobust"): Covariance matrix type: "nonrobust", "HC0", "HC1", "HC2", "HC3"

**Return Fields:**
- `params`: Coefficient estimates
- `stdErrors`: Standard errors
- `tStatistics`: t-statistics
- `pValues`: p-values
- `overdispersionTest`: Overdispersion test statistic
- `hasOverdispersion`: Whether overdispersion exists (>1.5)

**Example:**
```matlab
% Hospital visit counts
y = [0; 2; 5; 1; 3; ...];
result = pyBridge.StatsmodelsWrapper.poisson(y, X);

if result.hasOverdispersion
    fprintf('Overdispersion detected, consider using Negative Binomial regression\n');
end

% Using robust standard errors
result = pyBridge.StatsmodelsWrapper.poisson(y, X, covType="HC1");
```

#### Negative Binomial Regression

```matlab
result = pyBridge.StatsmodelsWrapper.negativeBinomial(y, X)
```

**Return Fields:**
- `alpha`: Overdispersion parameter
- Other fields same as Poisson regression

---

### StatsmodelsWrapper.ols - Advanced Standard Error Options

OLS method supports multiple standard error types:

```matlab
result = pyBridge.StatsmodelsWrapper.ols(y, X, options)
```

**Standard Error Types (`options.covType`):**

| Type | Description | Additional Parameters |
|------|-------------|----------------------|
| `"nonrobust"` | Classical standard errors | - |
| `"HC0"` | White standard errors | - |
| `"HC1"` | Stata robust standard errors | - |
| `"HC2"` | Small sample robust | - |
| `"HC3"` | Most conservative robust | - |
| `"hac"` | Newey-West HAC | `maxLags`, `kernel` |
| `"cluster"` | Single-way clustering | `clusterIds` |
| `"multiway"` | Multi-way clustering | `clusterGroups` |

**Complete Example:**

```matlab
%% Heteroskedasticity-robust standard errors
result = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC1");
fprintf('HC1 standard errors: %.3f\n', result.stdErrors(2));

%% HAC standard errors
result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="hac", maxLags=4, kernel="newey-west");
fprintf('HAC standard errors: %.3f\n', result.stdErrors(2));

%% Single-way clustered standard errors
result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="cluster", clusterIds=firmIds);
fprintf('Number of clusters: %d\n', result.nClusters);

%% Multi-way clustered standard errors
result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="multiway", clusterGroups={firmIds, yearIds});
fprintf('Two-way clustered standard errors: %.3f\n', result.stdErrors(2));
```

---

### LinearmodelsWrapper.panelOLS - Clustered Standard Error Options

Panel data models support multiple clustered standard errors:

```matlab
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, options)
```

**Standard Error Types (`options.covType`):**

| Type | Description |
|------|-------------|
| `"unadjusted"` | Classical standard errors |
| `"robust"` | Heteroskedasticity-robust (default) |
| `"clustered"` | Clustered standard errors |
| `"clustered_entity"` | Cluster by entity |
| `"clustered_time"` | Cluster by time |
| `"clustered_both"` | Two-way clustering (entity × time) |

**Example:**

```matlab
%% Fixed effects + firm clustering
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
    entityEffects=true, covType="clustered_entity");

%% Two-way fixed effects + two-way clustering
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
    entityEffects=true, timeEffects=true, covType="clustered_both");
```

#### Static Methods

##### toPython

```matlab
pyObj = pyBridge.DataConverter.toPython(mlData, options)
```

**Parameters:**
- `mlData`: MATLAB data (array, table, struct, cell, etc.)
- `options.dtype`: Specify NumPy data type (optional)
- `options.copy`: Whether to create a copy (optional, default: false)

**Returns:** Python object (NumPy array, Pandas DataFrame, or dict)

##### toMatlab

```matlab
mlData = pyBridge.DataConverter.toMatlab(pyObj, options)
```

**Parameters:**
- `pyObj`: Python object
- `options.type`: Specify output type ('array', 'table', 'struct')
- `options.rowNames`: Whether to preserve row names when converting DataFrame

**Returns:** MATLAB data

##### Specialized Conversion Methods

| Method | Description |
|--------|-------------|
| `array2Numpy(mlArray)` | MATLAB array → NumPy array |
| `numpy2Array(pyArray)` | NumPy array → MATLAB array |
| `table2Df(mlTable)` | MATLAB table → Pandas DataFrame |
| `df2Table(pyDf)` | Pandas DataFrame → MATLAB table |
| `struct2Dict(mlStruct)` | MATLAB struct → Python dict |
| `dict2Struct(pyDict)` | Python dict → MATLAB struct |
| `cell2List(mlCell)` | MATLAB cell → Python list |
| `list2Cell(pyList)` | Python list → MATLAB cell |

---

### ResultParser

Python return result parser.

#### Static Methods

##### parse

```matlab
result = pyBridge.ResultParser.parse(pyObj, parseType)
```

**Parameters:**
- `pyObj`: Python return object
- `parseType` (optional): Parse type ('auto', 'statsmodels', 'econml', 'linearmodels', 'array', 'dataframe')

**Returns:** MATLAB struct or array

##### Specialized Parsing Methods

| Method | Description |
|--------|-------------|
| `parseStatsmodels(pyObj)` | Parse statsmodels regression results |
| `parseEconML(pyResult)` | Parse econml causal inference results |
| `parseLinearmodels(pyObj)` | Parse linearmodels panel data results |
| `parseSummary(pyObj)` | Parse statistical summary |
| `parseArray(pyObj)` | Parse NumPy array |
| `extractAttributes(pyObj, attrNames)` | Extract specified attributes |

##### Utility Methods

| Method | Description |
|--------|-------------|
| `printResult(result, maxDepth)` | Print parsed results |

---

### ErrorHandler

Unified error handling module.

#### Static Methods

| Method | Description |
|--------|-------------|
| `wrapCall(func, varargin)` | Wrap function call, automatically catch exceptions |
| `handlePyError(ME, options)` | Handle Python exceptions |
| `getPyTraceback()` | Get Python error stack trace |
| `formatError(errorInfo, options)` | Format error messages |
| `safeCall(func, defaultResult, varargin)` | Safe call, return default value on failure |
| `validateInputs(varargin)` | Validate input parameters |
| `printPyError(ME)` | Print Python error details |
| `tryImport(moduleName)` | Try to import Python module |
| `assertPyAvailable(libName)` | Ensure Python library is available |

---

## scipy Wrapper

### ScipyStats

Statistical functionality wrapper class.

#### Constructor

```matlab
stats = pyBridge.internal.ScipyStats();
```

#### Probability Distribution Methods

| Method | Description |
|--------|-------------|
| `normPDF(x, mu, sigma)` | Normal distribution PDF |
| `normCDF(x, mu, sigma)` | Normal distribution CDF |
| `normPPF(p, mu, sigma)` | Normal distribution quantile |
| `tPDF(x, df)` | t-distribution PDF |
| `tCDF(x, df)` | t-distribution CDF |

#### Hypothesis Testing Methods

| Method | Description |
|--------|-------------|
| `tTest(data1, data2, options)` | t-test (independent/one-sample) |
| `tTestPaired(data1, data2)` | Paired t-test |
| `chi2Test(observed, expected)` | Chi-square test |
| `fTest(data1, data2)` | F-test (ANOVA) |
| `kruskalWallis(varargin)` | Kruskal-Wallis H test |
| `mannWhitneyU(data1, data2, options)` | Mann-Whitney U test |
| `normalityTest(data, testName)` | Normality test |

#### Descriptive Statistics Methods

| Method | Description |
|--------|-------------|
| `describe(data, options)` | Descriptive statistics |
| `correlation(data1, data2, method)` | Correlation analysis |
| `fitDistribution(data, distName)` | Distribution fitting |
| `percentile(data, percentiles)` | Percentiles |
| `quantile(data, quantiles)` | Quantiles |

---

### ScipyOptimize

Optimization functionality wrapper class.

#### Constructor

```matlab
opt = pyBridge.internal.ScipyOptimize();
```

#### Minimization Methods

| Method | Description |
|--------|-------------|
| `minimize(fun, x0, options)` | Multivariate function minimization |
| `minimizeScalar(fun, bounds, options)` | Univariate function minimization |
| `linprog(c, A, b, Aeq, beq, bounds, options)` | Linear programming |

#### Root Finding Methods

| Method | Description |
|--------|-------------|
| `root(fun, x0, options)` | Equation root finding |
| `fsolve(fun, x0, options)` | Solve nonlinear equations (MATLAB style) |

#### Fitting Methods

| Method | Description |
|--------|-------------|
| `curveFit(fun, xData, yData, p0, bounds)` | Curve fitting |
| `leastSquares(fun, x0, bounds, options)` | Nonlinear least squares |

#### Global Optimization Methods

| Method | Description |
|--------|-------------|
| `differentialEvolution(fun, bounds, options)` | Differential evolution algorithm |
| `basinhopping(fun, x0, options)` | Basin hopping algorithm |

---

### ScipySignal

Signal processing functionality wrapper class.

#### Constructor

```matlab
sig = pyBridge.internal.ScipySignal();
```

#### Filter Design Methods

| Method | Description |
|--------|-------------|
| `butter(order, wn, filterType)` | Butterworth filter |
| `cheby1(order, rp, wn, filterType)` | Chebyshev Type I filter |
| `ellip(order, rp, rs, wn, filterType)` | Elliptic filter |

#### Filtering Methods

| Method | Description |
|--------|-------------|
| `filter(b, a, x)` | IIR/FIR filtering |
| `filtfilt(b, a, x)` | Zero-phase filtering |
| `resample(x, num, options)` | Resampling |

#### Spectral Analysis Methods

| Method | Description |
|--------|-------------|
| `welch(x, fs, options)` | Welch power spectral density estimation |
| `periodogram(x, fs, options)` | Periodogram power spectral estimation |
| `spectrogram(x, fs, options)` | Short-time Fourier transform |
| `stft(x, fs, options)` | STFT |

#### Other Methods

| Method | Description |
|--------|-------------|
| `correlate(x, y, mode)` | Cross-correlation |
| `convolve(x, y, mode)` | Convolution |
| `findPeaks(x, options)` | Peak detection |
| `getWindow(windowType, n)` | Generate window function |
| `hilbert(x)` | Hilbert transform |

---

## statsmodels Wrapper

### StatsmodelsWrapper

Regression analysis, time series, and hypothesis testing wrapper class.

#### Regression Analysis Methods

| Method | Description |
|--------|-------------|
| `ols(y, X, options)` | Ordinary least squares regression |
| `wls(y, X, weights, options)` | Weighted least squares regression |
| `glm(y, X, options)` | Generalized linear model |
| `logistic(y, X, options)` | Logistic regression |
| `probit(y, X, options)` | Probit regression |

#### Time Series Methods

| Method | Description |
|--------|-------------|
| `arima(y, order, options)` | ARIMA time series model |
| `varModel(data, maxLags, options)` | VAR vector autoregression model |
| `adfuller(y, options)` | ADF unit root test |
| `kpss(y, options)` | KPSS unit root test |

#### Diagnostic Testing Methods

| Method | Description |
|--------|-------------|
| `breuschPagan(y, X, options)` | Breusch-Pagan heteroskedasticity test |
| `whiteTest(y, X, options)` | White heteroskedasticity test |
| `durbinWatson(residuals)` | Durbin-Watson autocorrelation test |
| `vif(X, options)` | Variance inflation factor (multicollinearity) |

---

## linearmodels Wrapper

### LinearmodelsWrapper

Panel data analysis and instrumental variable regression wrapper class.

#### Panel Data Methods

| Method | Description |
|--------|-------------|
| `panelOLS(y, X, entityIds, timeIds, options)` | Fixed effects panel model |
| `randomEffects(y, X, entityIds, timeIds, options)` | Random effects model |
| `betweenOLS(y, X, entityIds, timeIds)` | Between estimator |
| `pooledOLS(y, X, entityIds, timeIds, options)` | Pooled OLS |
| `firstDifference(y, X, entityIds, timeIds)` | First difference model |

#### Instrumental Variable Methods

| Method | Description |
|--------|-------------|
| `iv2SLS(y, endogVars, exogVars, instruments, options)` | Two-stage least squares |
| `ivLIML(y, endogVars, exogVars, instruments, options)` | Limited information maximum likelihood |
| `ivGMM(y, endogVars, exogVars, instruments, options)` | Generalized method of moments |

**Parameter Description:**
- `y`: Dependent variable
- `endogVars`: Endogenous regressors (variables subject to endogeneity that need to be instrumented)
- `exogVars`: Exogenous control variables (control variables that do not need to be instrumented), pass `[]` when there are no exogenous controls
- `instruments`: Instrumental variables (exclusion restrictions, used to identify endogenous variables, count must be >= number of endogenous variables)
- `options.addConstant` (logical, default: **true**): Whether to add constant term to exogenous variables
- `options.covType` (string, default: "unadjusted"): Covariance matrix type: "unadjusted", "robust", "kernel", "clustered"
- `options.weights`: Observation weights (optional)

**Example:**
```matlab
% Wage equation: education is endogenous, use parental education as instrument
% wage = f(education, experience), where education is endogenous
result = pyBridge.LinearmodelsWrapper.iv2SLS(wage, education, experience, parents_edu);

% Case with no exogenous control variables
result = pyBridge.LinearmodelsWrapper.iv2SLS(y, endogX, [], instruments);
```

#### Testing Methods

| Method | Description |
|--------|-------------|
| `hausmanTest(fixedEffects, randomEffects)` | Hausman test |
| `panelUnitTest(y, entityIds, timeIds, testName)` | Panel unit root test |

---

## econml Wrapper

### EconmlWrapper

Causal inference and treatment effect estimation wrapper class.

#### Causal Inference Methods

| Method | Description |
|--------|-------------|
| `dml(Y, T, X, W, options)` | Double Machine Learning |
| `drLearner(Y, T, X, W, options)` | Doubly Robust Learner |
| `causalForest(Y, T, X, W, options)` | Causal Forest |

#### Meta-learner Methods

| Method | Description |
|--------|-------------|
| `sLearner(Y, T, X, options)` | S-Learner |
| `tLearner(Y, T, X, options)` | T-Learner |
| `xLearner(Y, T, X, options)` | X-Learner |

#### Interpretation and Prediction Methods

| Method | Description |
|--------|-------------|
| `interpret(result, X, featureNames)` | Interpret causal effects |
| `predictEffect(fittedModel, Xnew)` | Predict treatment effects |
| `predictInterval(fittedModel, Xnew, alpha)` | Predict confidence intervals |
| `sensitivityAnalysis(fittedModel, Y, T, X)` | Sensitivity analysis |

---

## Result Structures

### OLS Regression Results

```matlab
result.nObs           % Number of observations
result.dfModel        % Model degrees of freedom
result.dfResiduals    % Residual degrees of freedom
result.rSquared       % R²
result.adjRSquared    % Adjusted R²
result.aic            % AIC
result.bic            % BIC
result.params         % Parameter estimates
result.stdErrors      % Standard errors
result.tStatistics    % t-statistics
result.pValues        % p-values
result.confInt        % Confidence intervals (struct)
  .lower              % Lower bound
  .upper              % Upper bound
result.paramNames     % Parameter names
result.residuals      % Residuals
result.fittedValues   % Fitted values
result.fStatistic     % F-statistic
result.fPValue        % F-test p-value
```

### DML Causal Inference Results

```matlab
result.modelType      % Model type
result.ate            % Average treatment effect
result.ateConfInt     % ATE confidence interval (struct)
  .lower              % Lower bound
  .upper              % Upper bound
result.atePValue      % ATE p-value
result.cate           % Conditional average treatment effect
result.fittedModel    % Fitted model object
```

### Panel Data Results

```matlab
result.nObs           % Number of observations
result.nEntities      % Number of entities
result.nTimes         % Number of time periods
result.rSquared       % R² (overall)
result.rSquaredWithin % R² (within)
result.rSquaredBetween % R² (between)
result.params         % Parameter estimates
result.stdErrors      % Standard errors
result.tStats         % t-statistics
result.pValues        % p-values
result.paramNames     % Parameter names
result.fStatistic     % F-statistic
result.fPValue        % F-test p-value
```

---

## Common Option Parameters

### Hypothesis Testing Options

```matlab
options.equalVar      % Assume equal variances (default: true)
options.alternative   % Alternative hypothesis ('two-sided', 'less', 'greater')
```

### Optimization Options

```matlab
options.method        % Optimization method
options.maxIter       % Maximum iterations
options.tol           % Convergence tolerance
options.display       % Whether to display output
```

### Regression Options

```matlab
options.addConstant   % Whether to add constant term (default: true)
options.covType       % Covariance matrix type
```

### Panel Data Options

```matlab
options.entityEffects % Include entity fixed effects
options.timeEffects   % Include time fixed effects
options.weights       % Weights
```

### Causal Inference Options

```matlab
options.modelY        % Model for Y ('linear', 'lasso', 'ridge', 'forest')
options.modelT        % Model for T
options.discreteTreatment % Whether T is discrete
options.randomState   % Random seed
```

---

**Document Version**: 1.0.1  
**Last Updated**: 2026-03-20

---

## Python 3.13 Compatibility Notes

This toolbox has been tested with Python 3.13 + MATLAB R2025b environment, all features work correctly.

**Notes:**
- MathWorks has not officially announced support for Python 3.13, but all features work correctly in actual testing
- Verified with 80 test cases, including:
  - AcademicCorrectionsTest: 10/10 ✓
  - MLogitHACSpecializedTest: 29/29 ✓
  - PyBridgeConsistencyTest: 41/41 ✓
- If you encounter compatibility issues, consider falling back to Python 3.10 or 3.11
