%% scipy Detailed Examples
% This script demonstrates detailed usage of the PyBridge toolbox for scipy library

%% Initialization
addpath('..');
config = pyBridge.PyBridgeConfig.getInstance();

%% ==================== Statistical Functions ====================
fprintf("\n========== scipy.stats Statistical Functions ==========\n");

stats = pyBridge.internal.ScipyStats();

%% 1. Probability Distributions
fprintf("\n--- 1. Probability Distributions ---\n");

% Normal distribution
x = linspace(-4, 4, 100);
pdfNorm = stats.normPDF(x, 0, 1);
cdfNorm = stats.normCDF(x, 0, 1);
ppfNorm = stats.normPPF([0.025, 0.5, 0.975], 0, 1);

fprintf("Normal Distribution N(0,1):\n");
fprintf("  PDF at x=0: %.4f\n", stats.normPDF(0, 0, 1));
fprintf("  CDF at x=0: %.4f\n", stats.normCDF(0, 0, 1));
fprintf("  2.5%% Quantile: %.4f\n", ppfNorm(1));
fprintf("  Median: %.4f\n", ppfNorm(2));
fprintf("  97.5%% Quantile: %.4f\n", ppfNorm(3));

% t distribution
df = 10;
pdfT = stats.tPDF(x, df);
cdfT = stats.tCDF(x, df);

fprintf("\nt Distribution (df=10):\n");
fprintf("  PDF at x=0: %.4f\n", stats.tPDF(0, df));
fprintf("  CDF at x=0: %.4f\n", stats.tCDF(0, df));

%% 2. Hypothesis Testing
fprintf("\n--- 2. Hypothesis Testing ---\n");

rng(0, "twister");

% Independent samples t-test
group1 = randn(50, 1);
group2 = randn(50, 1) + 0.5;

fprintf("Independent Samples t-test:\n");
tTestResult = stats.tTest(group1, group2);
fprintf("  t-statistic: %.4f\n", tTestResult.tStatistic);
fprintf("  p-value: %.4f\n", tTestResult.pValue);
fprintf("  Significant: %s\n", ternary(tTestResult.significant, 'Yes', 'No'));

% Paired t-test
before = randn(30, 1);
after = before + 0.3 + randn(30, 1)*0.5;

fprintf("\nPaired t-test:\n");
pairedResult = stats.tTestPaired(before, after);
fprintf("  t-statistic: %.4f\n", pairedResult.tStatistic);
fprintf("  p-value: %.4f\n", pairedResult.pValue);

% Chi-square test
observed = [16, 18, 16, 14, 12, 12];
expected = [16, 16, 16, 16, 16, 8] / 4 * sum(observed);

fprintf("\nChi-Square Test:\n");
chi2Result = stats.chi2Test(observed, expected);
fprintf("  Chi-Square Statistic: %.4f\n", chi2Result.chi2Statistic);
fprintf("  p-value: %.4f\n", chi2Result.pValue);
fprintf("  Degrees of Freedom: %d\n", chi2Result.dof);

% F-test (variance homogeneity)
fprintf("\nF-test (ANOVA):\n");
groupA = randn(30, 1);
groupB = randn(30, 1) + 0.8;
groupC = randn(30, 1) - 0.5;
fResult = stats.fTest(groupA, groupB);
fprintf("  F-statistic: %.4f\n", fResult.fStatistic);
fprintf("  p-value: %.4f\n", fResult.pValue);

% Kruskal-Wallis test
fprintf("\nKruskal-Wallis H Test:\n");
kwResult = stats.kruskalWallis(groupA, groupB, groupC);
fprintf("  H-statistic: %.4f\n", kwResult.hStatistic);
fprintf("  p-value: %.4f\n", kwResult.pValue);

% Mann-Whitney U test
fprintf("\nMann-Whitney U Test:\n");
mwResult = stats.mannWhitneyU(groupA, groupB);
fprintf("  U-statistic: %.4f\n", mwResult.uStatistic);
fprintf("  p-value: %.4f\n", mwResult.pValue);

% Normality tests
fprintf("\nNormality Tests:\n");
normalData = randn(100, 1);
for testName = ["shapiro", "normaltest", "kstest"]
    normResult = stats.normalityTest(normalData, testName);
    fprintf("  %s: Statistic=%.4f, p-value=%.4f, Normal=%s\n", ...
        testName, normResult.statistic, normResult.pValue, ...
        ternary(normResult.isNormal, 'Yes', 'No'));
end

%% 3. Descriptive Statistics
fprintf("\n--- 3. Descriptive Statistics ---\n");

data = randn(100, 1) * 2 + 5;

descResult = stats.describe(data);
fprintf("Sample Size: %d\n", descResult.nObs);
fprintf("Mean: %.4f\n", descResult.mean);
fprintf("Variance: %.4f\n", descResult.variance);
fprintf("Std Dev: %.4f\n", sqrt(descResult.variance));
fprintf("Skewness: %.4f\n", descResult.skewness);
fprintf("Kurtosis: %.4f\n", descResult.kurtosis);
fprintf("Min: %.4f\n", descResult.min);
fprintf("Max: %.4f\n", descResult.max);

%% 4. Correlation Analysis
fprintf("\n--- 4. Correlation Analysis ---\n");

x1 = randn(100, 1);
x2 = 0.8*x1 + randn(100, 1)*0.6;
x3 = randn(100, 1);

% Two-variable correlation
fprintf("Two-Variable Correlation Analysis:\n");
for method = ["pearson", "spearman", "kendall"]
    corrResult = stats.correlation(x1, x2, method);
    fprintf("  %s Correlation: r=%.4f, p=%.4f\n", ...
        method, corrResult.correlation, corrResult.pValue);
end

% Correlation matrix
fprintf("\nCorrelation Matrix:\n");
dataMatrix = [x1, x2, x3];
corrMatrix = stats.correlation(dataMatrix, [], "pearson");
fprintf("  Correlation Coefficient Matrix:\n");
disp(corrMatrix.correlationMatrix);

%% 5. Distribution Fitting
fprintf("\n--- 5. Distribution Fitting ---\n");

% Generate normal distribution data
trueMu = 3;
trueSigma = 1.5;
fitData = trueMu + trueSigma * randn(500, 1);

fprintf("Fit Normal Distribution (True Parameters: mu=%.2f, sigma=%.2f):\n", trueMu, trueSigma);
fitResult = stats.fitDistribution(fitData, "norm");
fprintf("  Estimated mu: %.4f\n", fitResult.mu);
fprintf("  Estimated sigma: %.4f\n", fitResult.sigma);

%% 6. Percentiles
fprintf("\n--- 6. Percentiles ---\n");

percentiles = [25, 50, 75];
q = stats.percentile(data, percentiles);
fprintf("Percentiles:\n");
for i = 1:length(percentiles)
    fprintf("  P%d: %.4f\n", percentiles(i), q(i));
end

quartiles = stats.quantile(data, [0, 0.25, 0.5, 0.75, 1]);
fprintf("\nQuantiles:\n");
fprintf("  Min: %.4f\n", quartiles(1));
fprintf("  Q1: %.4f\n", quartiles(2));
fprintf("  Median: %.4f\n", quartiles(3));
fprintf("  Q3: %.4f\n", quartiles(4));
fprintf("  Max: %.4f\n", quartiles(5));

%% ==================== Optimization Functions ====================
fprintf("\n\n========== scipy.optimize Optimization Functions ==========\n");

opt = pyBridge.internal.ScipyOptimize();

%% 1. Multivariate Function Minimization
fprintf("\n--- 1. Multivariate Function Minimization ---\n");

% Rosenbrock function
rosenbrock = @(x) 100*(x(2)-x(1)^2)^2 + (1-x(1))^2;

fprintf("Rosenbrock Function Minimization:\n");
minResult = opt.minimize(rosenbrock, [-1; 2]);
fprintf("  Minimum Value: %.6e\n", minResult.fun);
fprintf("  Optimal Solution: [%.6f, %.6f]\n", minResult.x);
fprintf("  Iterations: %d\n", minResult.nIter);
fprintf("  Success: %s\n", ternary(minResult.success, 'Yes', 'No'));

% Beale function
beale = @(x) (1.5 - x(1) + x(1)*x(2))^2 + ...
             (2.25 - x(1) + x(1)*x(2)^2)^2 + ...
             (2.625 - x(1) + x(1)*x(2)^3)^2;

fprintf("\nBeale Function Minimization:\n");
minResult2 = opt.minimize(beale, [0; 0], "method", "BFGS");
fprintf("  Minimum Value: %.6e\n", minResult2.fun);
fprintf("  Optimal Solution: [%.6f, %.6f]\n", minResult2.x);

%% 2. Univariate Function Minimization
fprintf("\n--- 2. Univariate Function Minimization ---\n");

f1 = @(x) (x - 2)^2;

fprintf("Univariate Function Minimization:\n");
scalarMin = opt.minimizeScalar(f1);
fprintf("  Minimum Value: %.6f\n", scalarMin.fun);
fprintf("  Optimal Solution: %.6f\n", scalarMin.x);

% Bounded optimization
fprintf("\nBounded Optimization ([0, 1]):\n");
boundedMin = opt.minimizeScalar(f1, [0, 1]);
fprintf("  Minimum Value: %.6f\n", boundedMin.fun);
fprintf("  Optimal Solution: %.6f\n", boundedMin.x);

%% 3. Linear Programming
fprintf("\n--- 3. Linear Programming ---\n");

% min: -x - 2y
% s.t.: x + y <= 4, x >= 0, y >= 0

c = [-1; -2];
A = [1, 1];
b = 4;

fprintf("Linear Programming:\n");
fprintf("  Objective: min -x - 2y\n");
fprintf("  Constraints: x + y <= 4, x,y >= 0\n");

lpResult = opt.linprog(c, A, b, [], [], [0; 0; 0]);
fprintf("  Optimal Value: %.4f\n", -lpResult.fun);
fprintf("  Optimal Solution: x=%.4f, y=%.4f\n", lpResult.x);

%% 4. Root Finding
fprintf("\n--- 4. Root Finding ---\n");

% System of equations: x^2 + y^2 - 5 = 0, x*y - 2 = 0
equations = @(x) [x(1)^2 + x(2)^2 - 5; x(1)*x(2) - 2];

fprintf("Nonlinear System of Equations:\n");
fprintf("  x^2 + y^2 = 5\n");
fprintf("  x*y = 2\n");

rootResult = opt.root(equations, [1; 1]);
fprintf("  Solution: [%.4f, %.4f]\n", rootResult.x);
fprintf("  Residuals: [%.6f, %.6f]\n", rootResult.fun);

%% 5. Curve Fitting
fprintf("\n--- 5. Curve Fitting ---\n");

% Polynomial fitting
xData = linspace(-3, 3, 50);
yTrue = 2*xData.^2 - 3*xData + 1;
yNoisy = yTrue + randn(size(xData))*2;

polyFun = @(x, a, b, c) a*x.^2 + b*x + c;

fprintf("Quadratic Polynomial Fitting:\n");
fprintf("  True Parameters: a=2, b=-3, c=1\n");

fitResult = opt.curveFit(polyFun, xData, yNoisy, [2; -3; 1]);
fprintf("  Fitted Parameters: a=%.4f, b=%.4f, c=%.4f\n", fitResult.parameters);
fprintf("  Std Errors: [%.4f, %.4f, %.4f]\n", fitResult.stdErrors);

% Exponential fitting
expFun = @(x, a, b, c) a*exp(-b*x) + c;
xExp = linspace(0, 5, 50);
yExp = 3*exp(-0.5*xExp) + 1 + randn(size(xExp))*0.2;

fprintf("\nExponential Function Fitting:\n");
fprintf("  True Parameters: a=3, b=0.5, c=1\n");

expFit = opt.curveFit(expFun, xExp, yExp, [3; 0.5; 1]);
fprintf("  Fitted Parameters: a=%.4f, b=%.4f, c=%.4f\n", expFit.parameters);

%% 6. Least Squares
fprintf("\n--- 6. Nonlinear Least Squares ---\n");

% Minimize residual sum of squares
residualFun = @(x) [x(1) + x(2) - 3; x(1) - 2*x(2) + 1; 2*x(1) + x(2) - 4];

fprintf("Least Squares Problem:\n");
lsqResult = opt.leastSquares(residualFun, [0; 0]);
fprintf("  Optimal Solution: [%.4f, %.4f]\n", lsqResult.x);
fprintf("  Residual Cost: %.6f\n", lsqResult.cost);

%% 7. Global Optimization
fprintf("\n--- 7. Global Optimization ---\n");

% Rastrigin function (multimodal)
rastrigin = @(x) 20 + x(1)^2 + x(2)^2 - 10*(cos(2*pi*x(1)) + cos(2*pi*x(2)));

fprintf("Rastrigin Function (Multimodal Optimization):\n");
deResult = opt.differentialEvolution(rastrigin, [-5 5; -5 5]);
fprintf("  Minimum Value: %.6f\n", deResult.fun);
fprintf("  Optimal Solution: [%.6f, %.6f]\n", deResult.x);

%% ==================== Signal Processing Functions ====================
fprintf("\n\n========== scipy.signal Signal Processing ==========\n");

sig = pyBridge.internal.ScipySignal();

%% 1. Filter Design
fprintf("\n--- 1. Filter Design ---\n");

fs = 1000; % Sampling rate

% Butterworth
[b_butter, a_butter] = sig.butter(4, 100/(fs/2), 'low');
fprintf("Butterworth Lowpass Filter (4th order, 100Hz):\n");
fprintf("  Numerator Coefficient Length: %d\n", length(b_butter));

% Chebyshev Type I
[b_cheby, a_cheby] = sig.cheby1(4, 1, 100/(fs/2), 'low');
fprintf("\nChebyshev Type I Lowpass (4th order, 1dB passband ripple):\n");
fprintf("  Numerator Coefficient Length: %d\n", length(b_cheby));

% Elliptic filter
[b_ellip, a_ellip] = sig.ellip(4, 1, 40, 100/(fs/2), 'low');
fprintf("\nElliptic Filter (4th order, 1dB passband ripple, 40dB stopband attenuation):\n");
fprintf("  Numerator Coefficient Length: %d\n", length(b_ellip));

%% 2. Signal Filtering
fprintf("\n--- 2. Signal Filtering ---\n");

t = 0:1/fs:1;
signal = sin(2*pi*50*t) + 0.5*sin(2*pi*200*t) + 0.3*randn(size(t));

fprintf("Pre-Filtering Signal Statistics:\n");
fprintf("  Mean: %.4f\n", mean(signal));
fprintf("  Std Dev: %.4f\n", std(signal));

% Zero-phase filtering
filtered = sig.filtfilt(b_butter, a_butter, signal);
fprintf("\nPost-Filtering Signal Statistics:\n");
fprintf("  Mean: %.4f\n", mean(filtered));
fprintf("  Std Dev: %.4f\n", std(filtered));

%% 3. Spectral Analysis
fprintf("\n--- 3. Spectral Analysis ---\n");

% Welch power spectrum
psd = sig.welch(signal, fs, nperseg=256);
fprintf("Welch Power Spectral Density:\n");
fprintf("  Number of Frequency Points: %d\n", length(psd.frequencies));
fprintf("  Frequency Range: [%.1f, %.1f] Hz\n", psd.frequencies(1), psd.frequencies(end));

% Find dominant frequency
[~, idx] = max(psd.powerSpectralDensity);
fprintf("  Dominant Frequency: %.1f Hz\n", psd.frequencies(idx));

% Periodogram
periodogramResult = sig.periodogram(signal, fs);
fprintf("\nPeriodogram:\n");
fprintf("  Number of Frequency Points: %d\n", length(periodogramResult.frequencies));

%% 4. Short-Time Fourier Transform
fprintf("\n--- 4. Short-Time Fourier Transform (STFT) ---\n");

stftResult = sig.stft(signal, fs, nperseg=128);
fprintf("STFT Results:\n");
fprintf("  Number of Time Frames: %d\n", length(stftResult.times));
fprintf("  Number of Frequency Points: %d\n", length(stftResult.frequencies));

%% 5. Peak Detection
fprintf("\n--- 5. Peak Detection ---\n");

% Create signal with peaks
peakSignal = sin(2*pi*10*t);
peaks = sig.findPeaks(peakSignal, distance=50);

fprintf("Detected Peaks:\n");
fprintf("  Number of Peaks: %d\n", length(peaks.indices));
if ~isempty(peaks.indices)
    fprintf("  First Peak Location: t=%.3f seconds\n", t(peaks.indices(1)));
end

%% 6. Hilbert Transform
fprintf("\n--- 6. Hilbert Transform ---\n");

hilbertResult = sig.hilbert(signal);
fprintf("Hilbert Transform:\n");
fprintf("  Analytic Signal Length: %d\n", length(hilbertResult.analyticSignal));
fprintf("  Instantaneous Amplitude Range: [%.4f, %.4f]\n", ...
    min(hilbertResult.amplitude), max(hilbertResult.amplitude));

%% 7. Correlation and Convolution
fprintf("\n--- 7. Correlation and Convolution ---\n");

x = [1, 2, 3, 4, 5];
h = [1, 1, 1];

% Convolution
convResult = sig.convolve(x, h, 'full');
fprintf("Convolution Result:\n");
fprintf("  Input: [%s]\n", num2str(x));
fprintf("  Filter: [%s]\n", num2str(h));
fprintf("  Output: [%s]\n", num2str(convResult));

% Autocorrelation
autocorr = sig.correlate(x, [], 'full');
fprintf("\nAutocorrelation:\n");
fprintf("  Result Length: %d\n", length(autocorr));

%% Complete
fprintf("\n\n========== scipy Examples Complete! ==========\n");

%% Helper Functions
function result = ternary(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end
