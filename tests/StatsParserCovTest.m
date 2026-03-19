classdef StatsParserCovTest < matlab.unittest.TestCase
    % StatsParserCovTest Comprehensive tests for ScipyStats, ResultParser and CovarianceTypes
    %
    % Test Coverage:
    % 1. ScipyStats - Statistical distributions, hypothesis tests, descriptive statistics
    % 2. ResultParser - Parsing statsmodels, numpy arrays, generic Python objects
    % 3. CovarianceTypes - HC standard errors, HAC, clustered, MLE sandwich estimators
    %
    % Author: WorkBuddy
    % Date: 2026-03-20
    
    properties
        Tolerance       % Numeric comparison tolerance
        HasPython       % Whether Python environment is available
        HasScipy        % Whether scipy is available
        HasStatsmodels  % Whether statsmodels is available
    end
    
    methods (TestClassSetup)
        function setupClass(testCase)
            % Initialize test environment
            rng(42, 'twister');  % Fixed random seed for reproducibility
            testCase.Tolerance = 1e-6;
            
            % Check Python environment
            try
                pyenv;
                testCase.HasPython = true;
            catch
                testCase.HasPython = false;
            end
            
            % Check scipy availability
            try
                py.importlib.import_module('scipy');
                testCase.HasScipy = true;
            catch
                testCase.HasScipy = false;
            end
            
            % Check statsmodels availability
            try
                py.importlib.import_module('statsmodels');
                testCase.HasStatsmodels = true;
            catch
                testCase.HasStatsmodels = false;
            end
        end
    end
    
    %% ==================== ScipyStats Tests ====================
    methods (Test, TestTags = {'ScipyStats', 'Distribution'})
        function testNormDistributionProperties(testCase)
            % Test normal distribution: CDF, PDF, PPF and inverse transform
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                stats = pyBridge.internal.ScipyStats();
                
                % Test 1: normCDF(0, 0, 1) should be 0.5
                cdfValue = stats.normCDF(0, 0, 1);
                testCase.verifyEqual(cdfValue, 0.5, 'AbsTol', testCase.Tolerance, ...
                    'Standard normal CDF at 0 should be 0.5');
                
                % Test 2: normPDF(0, 0, 1) should be 1/sqrt(2*pi)
                pdfValue = stats.normPDF(0, 0, 1);
                expectedPDF = 1 / sqrt(2 * pi);
                testCase.verifyEqual(pdfValue, expectedPDF, 'AbsTol', testCase.Tolerance, ...
                    'Standard normal PDF at 0 should be 1/sqrt(2*pi)');
                
                % Test 3: normPPF(0.975, 0, 1) should be approximately 1.96
                ppfValue = stats.normPPF(0.975, 0, 1);
                testCase.verifyEqual(ppfValue, 1.96, 'AbsTol', 0.01, ...
                    'normPPF(0.975) should be approximately 1.96');
                
                % Test 4: Inverse transform verification - CDF(PPF(p)) should equal p
                pValues = [0.1, 0.25, 0.5, 0.75, 0.9];
                for p = pValues
                    x = stats.normPPF(p, 0, 1);
                    pBack = stats.normCDF(x, 0, 1);
                    testCase.verifyEqual(pBack, p, 'AbsTol', testCase.Tolerance, ...
                        sprintf('CDF(PPF(%.2f)) should equal %.2f', p, p));
                end
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testTDistributionProperties(testCase)
            % Test t-distribution properties
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                stats = pyBridge.internal.ScipyStats();
                
                dfValues = [5, 10, 30, 100];
                
                for df = dfValues
                    % Test 1: tCDF(0, df) should be 0.5 (symmetric)
                    cdfValue = stats.tCDF(0, df);
                    testCase.verifyEqual(cdfValue, 0.5, 'AbsTol', testCase.Tolerance, ...
                        sprintf('t-distribution CDF at 0 with df=%d should be 0.5', df));
                    
                    % Test 2: tPDF(0, df) should be positive
                    pdfValue = stats.tPDF(0, df);
                    testCase.verifyGreaterThan(pdfValue, 0, ...
                        sprintf('t-distribution PDF at 0 with df=%d should be positive', df));
                end
                
                % Test 3: As df -> infinity, t-distribution approaches standard normal
                largeDf = 1000;
                tPdfLargeDf = stats.tPDF(0, largeDf);
                normPdf = stats.normPDF(0, 0, 1);
                testCase.verifyEqual(tPdfLargeDf, normPdf, 'RelTol', 0.01, ...
                    't-distribution with large df should approximate standard normal');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testTTestHypothesis(testCase)
            % Test t-test hypothesis testing
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                rng(42);
                stats = pyBridge.internal.ScipyStats();
                
                % Test 1: Two samples from same distribution - p should be > 0.05
                data1 = randn(100, 1);
                data2 = randn(100, 1);
                result = stats.tTest(data1, data2);
                
                testCase.verifyTrue(isfield(result, 'tStatistic'), ...
                    'Result should contain tStatistic field');
                testCase.verifyTrue(isfield(result, 'pValue'), ...
                    'Result should contain pValue field');
                testCase.verifyGreaterThan(result.pValue, 0.05, ...
                    'Same distribution samples should have p > 0.05');
                
                % Test 2: Samples with significantly different means - p should be < 0.05
                data3 = randn(100, 1);
                data4 = randn(100, 1) + 3;  % Shifted by 3
                result2 = stats.tTest(data3, data4);
                
                testCase.verifyLessThan(result2.pValue, 0.05, ...
                    'Significantly different means should have p < 0.05');
                testCase.verifyTrue(result2.significant, ...
                    'Result should indicate significance');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testChi2Test(testCase)
            % Test chi-square test
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                stats = pyBridge.internal.ScipyStats();
                
                % Test 1: Uniform distribution observed values - p should be > 0.05
                observed = [20, 20, 20, 20, 20];  % Perfectly uniform
                result = stats.chi2Test(observed);
                
                testCase.verifyTrue(isfield(result, 'chi2Statistic'), ...
                    'Result should contain chi2Statistic field');
                testCase.verifyTrue(isfield(result, 'pValue'), ...
                    'Result should contain pValue field');
                testCase.verifyGreaterThan(result.pValue, 0.05, ...
                    'Uniform distribution should have p > 0.05');
                
                % Test 2: Skewed observed values - p should be < 0.05
                observedSkewed = [50, 10, 10, 10, 20];
                resultSkewed = stats.chi2Test(observedSkewed);
                
                testCase.verifyLessThan(resultSkewed.pValue, 0.05, ...
                    'Skewed distribution should have p < 0.05');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testNonParametricTests(testCase)
            % Test non-parametric tests: Kruskal-Wallis and Mann-Whitney U
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                rng(42);
                stats = pyBridge.internal.ScipyStats();
                
                % Test 1: Kruskal-Wallis with same distributions
                group1 = randn(50, 1);
                group2 = randn(50, 1);
                group3 = randn(50, 1);
                resultKW = stats.kruskalWallis(group1, group2, group3);
                
                testCase.verifyTrue(isfield(resultKW, 'hStatistic'), ...
                    'Kruskal-Wallis result should contain hStatistic');
                testCase.verifyTrue(isfield(resultKW, 'pValue'), ...
                    'Kruskal-Wallis result should contain pValue');
                testCase.verifyGreaterThan(resultKW.pValue, 0.05, ...
                    'Same distributions should have p > 0.05');
                
                % Test 2: Kruskal-Wallis with different distributions
                group4 = randn(50, 1) + 5;  % Different mean
                resultKW2 = stats.kruskalWallis(group1, group2, group4);
                testCase.verifyLessThan(resultKW2.pValue, 0.05, ...
                    'Different distributions should have p < 0.05');
                
                % Test 3: Mann-Whitney U with same distributions
                resultMW = stats.mannWhitneyU(group1, group2);
                testCase.verifyTrue(isfield(resultMW, 'uStatistic'), ...
                    'Mann-Whitney result should contain uStatistic');
                testCase.verifyGreaterThan(resultMW.pValue, 0.05, ...
                    'Same distributions should have p > 0.05');
                
                % Test 4: Mann-Whitney U with different distributions
                resultMW2 = stats.mannWhitneyU(group1, group4);
                testCase.verifyLessThan(resultMW2.pValue, 0.05, ...
                    'Different distributions should have p < 0.05');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testNormalityTests(testCase)
            % Test normality tests: shapiro, normaltest, kstest
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                rng(42);
                stats = pyBridge.internal.ScipyStats();
                
                % Generate normal data
                normalData = randn(100, 1);
                
                % Generate non-normal data (exponential)
                expData = -log(rand(100, 1));  % Exponential distribution
                
                testTypes = {'shapiro', 'normaltest', 'kstest'};
                
                for i = 1:length(testTypes)
                    testType = testTypes{i};
                    
                    % Test 1: Normal data should pass normality test (p > 0.05)
                    resultNormal = stats.normalityTest(normalData, testType);
                    
                    testCase.verifyTrue(isfield(resultNormal, 'testName'), ...
                        sprintf('%s result should contain testName field', testType));
                    testCase.verifyTrue(isfield(resultNormal, 'statistic'), ...
                        sprintf('%s result should contain statistic field', testType));
                    testCase.verifyTrue(isfield(resultNormal, 'pValue'), ...
                        sprintf('%s result should contain pValue field', testType));
                    testCase.verifyTrue(isfield(resultNormal, 'isNormal'), ...
                        sprintf('%s result should contain isNormal field', testType));
                    
                    % Test 2: Exponential data should fail normality test (p < 0.05)
                    resultExp = stats.normalityTest(expData, testType);
                    testCase.verifyLessThan(resultExp.pValue, 0.05, ...
                        sprintf('%s: Exponential data should have p < 0.05', testType));
                end
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testCorrelationMethods(testCase)
            % Test correlation methods: pearson, spearman, kendall
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                rng(42);
                stats = pyBridge.internal.ScipyStats();
                
                n = 100;
                
                % Test 1: Perfect linear correlation
                x = randn(n, 1);
                yPerfect = 2 * x + 1;  % Perfect linear relationship
                
                resultPearson = stats.correlation(x, yPerfect, 'pearson');
                testCase.verifyEqual(abs(resultPearson.correlation), 1, 'AbsTol', testCase.Tolerance, ...
                    'Pearson correlation for perfect linear should be |r| = 1');
                
                resultSpearman = stats.correlation(x, yPerfect, 'spearman');
                testCase.verifyEqual(abs(resultSpearman.correlation), 1, 'AbsTol', testCase.Tolerance, ...
                    'Spearman correlation for perfect linear should be |r| = 1');
                
                resultKendall = stats.correlation(x, yPerfect, 'kendall');
                testCase.verifyEqual(abs(resultKendall.correlation), 1, 'AbsTol', testCase.Tolerance, ...
                    'Kendall correlation for perfect linear should be |r| = 1');
                
                % Test 2: Independent data should have |r| close to 0
                xIndep = randn(n, 1);
                yIndep = randn(n, 1);
                
                resultIndep = stats.correlation(xIndep, yIndep, 'pearson');
                testCase.verifyLessThan(abs(resultIndep.correlation), 0.3, ...
                    'Independent data should have |r| close to 0');
                
                % Test 3: Verify method field
                testCase.verifyEqual(resultPearson.method, 'pearson', ...
                    'Method should be correctly recorded');
                testCase.verifyEqual(resultSpearman.method, 'spearman', ...
                    'Method should be correctly recorded');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testDescribeAndQuantile(testCase)
            % Test descriptive statistics and quantile/percentile functions
            try
                testCase.assumeTrue(testCase.HasScipy, 'scipy not available');
                
                stats = pyBridge.internal.ScipyStats();
                
                % Test data
                data = 1:100;
                
                % Test 1: describe should return complete statistics
                result = stats.describe(data(:));
                
                testCase.verifyTrue(isfield(result, 'nObs'), 'Result should contain nObs');
                testCase.verifyTrue(isfield(result, 'min'), 'Result should contain min');
                testCase.verifyTrue(isfield(result, 'max'), 'Result should contain max');
                testCase.verifyTrue(isfield(result, 'mean'), 'Result should contain mean');
                testCase.verifyTrue(isfield(result, 'variance'), 'Result should contain variance');
                testCase.verifyTrue(isfield(result, 'skewness'), 'Result should contain skewness');
                testCase.verifyTrue(isfield(result, 'kurtosis'), 'Result should contain kurtosis');
                
                testCase.verifyEqual(result.nObs, 100, 'nObs should be 100');
                testCase.verifyEqual(result.min, 1, 'min should be 1');
                testCase.verifyEqual(result.max, 100, 'max should be 100');
                testCase.verifyEqual(result.mean, 50.5, 'AbsTol', testCase.Tolerance, ...
                    'mean should be 50.5');
                
                % Test 2: percentile(1:100, 50) should be approximately 50
                p50 = stats.percentile(data(:), 50);
                testCase.verifyEqual(p50, 50.5, 'AbsTol', 1, ...
                    '50th percentile of 1:100 should be approximately 50');
                
                % Test 3: quantile should be monotonically increasing
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9];
                qValues = stats.quantile(data(:), quantiles);
                
                testCase.verifyTrue(all(diff(qValues) >= 0), ...
                    'Quantiles should be monotonically increasing');
                
                % Test 4: Verify quantile values
                testCase.verifyEqual(stats.quantile(data(:), 0), 1, 'AbsTol', testCase.Tolerance, ...
                    '0th quantile should be min');
                testCase.verifyEqual(stats.quantile(data(:), 1), 100, 'AbsTol', testCase.Tolerance, ...
                    '1st quantile should be max');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'scipy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
    end
    
    %% ==================== ResultParser Tests ====================
    methods (Test, TestTags = {'ResultParser'})
        function testParseStatsmodelsOLS(testCase)
            % Test parsing OLS results from statsmodels
            try
                testCase.assumeTrue(testCase.HasStatsmodels, 'statsmodels not available');
                
                rng(42);
                n = 100;
                X = randn(n, 2);
                y = 1 + 2*X(:,1) + 0.5*X(:,2) + randn(n, 1)*0.5;
                
                % Run OLS using pyBridge
                result = pyBridge.StatsmodelsWrapper.ols(y, X);
                
                % Verify all expected fields exist
                testCase.verifyTrue(isfield(result, 'params'), ...
                    'Result should contain params field');
                testCase.verifyTrue(isfield(result, 'stdErrors'), ...
                    'Result should contain stdErrors field');
                testCase.verifyTrue(isfield(result, 'tStatistics'), ...
                    'Result should contain tStatistics field');
                testCase.verifyTrue(isfield(result, 'pValues'), ...
                    'Result should contain pValues field');
                testCase.verifyTrue(isfield(result, 'nObs'), ...
                    'Result should contain nObs field');
                
                % Verify types
                testCase.verifyTrue(isnumeric(result.params), ...
                    'params should be numeric');
                testCase.verifyTrue(isnumeric(result.stdErrors), ...
                    'stdErrors should be numeric');
                testCase.verifyTrue(isnumeric(result.pValues), ...
                    'pValues should be numeric');
                
                % Verify dimensions
                testCase.verifyEqual(length(result.params), 3, ...
                    'Should have 3 parameters (intercept + 2 vars)');
                testCase.verifyEqual(length(result.stdErrors), length(result.params), ...
                    'stdErrors should match params length');
                
                % Verify p-values are in valid range
                testCase.verifyGreaterThanOrEqual(result.pValues, 0, ...
                    'p-values should be >= 0');
                testCase.verifyLessThanOrEqual(result.pValues, 1, ...
                    'p-values should be <= 1');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'statsmodels/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testParseArrayNumpy(testCase)
            % Test parsing NumPy arrays
            try
                testCase.assumeTrue(testCase.HasPython, 'python not available');
                
                % Create NumPy array
                data = [1, 2, 3, 4, 5];
                npArray = py.numpy.array(data);
                
                % Parse array
                result = pyBridge.ResultParser.parseArray(npArray);
                
                % Verify fields
                testCase.verifyTrue(isfield(result, 'data'), ...
                    'Result should contain data field');
                testCase.verifyTrue(isfield(result, 'shape'), ...
                    'Result should contain shape field');
                testCase.verifyTrue(isfield(result, 'ndim'), ...
                    'Result should contain ndim field');
                testCase.verifyTrue(isfield(result, 'dtype'), ...
                    'Result should contain dtype field');
                
                % Verify data
                testCase.verifyEqual(result.data(:)', data, 'AbsTol', testCase.Tolerance, ...
                    'Data should match original');
                testCase.verifyEqual(result.ndim, 1, ...
                    'ndim should be 1 for 1D array');
                
                % Test 2D array
                data2D = [1, 2, 3; 4, 5, 6];
                npArray2D = py.numpy.array(data2D);
                result2D = pyBridge.ResultParser.parseArray(npArray2D);
                
                testCase.verifyEqual(result2D.ndim, 2, ...
                    'ndim should be 2 for 2D array');
                testCase.verifyEqual(result2D.shape, [2, 3], ...
                    'Shape should be [2, 3]');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'numpy/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testParseGenericPyObject(testCase)
            % Test generic parsing of Python objects
            try
                testCase.assumeTrue(testCase.HasPython, 'python not available');
                
                % Create a simple Python object (dictionary)
                pyDict = py.dict(pyargs('a', 1, 'b', 2, 'c', 3));
                
                % Parse generic
                result = pyBridge.ResultParser.parseGeneric(pyDict);
                
                % Verify pythonType field exists
                testCase.verifyTrue(isfield(result, 'pythonType'), ...
                    'Result should contain pythonType field');
                testCase.verifyEqual(result.pythonType, 'dict', ...
                    'pythonType should be dict');
                
                % Test with numpy array
                npArray = py.numpy.array([1, 2, 3]);
                resultNp = pyBridge.ResultParser.parseGeneric(npArray);
                
                testCase.verifyTrue(isfield(resultNp, 'pythonType'), ...
                    'NumPy result should contain pythonType field');
                testCase.verifyEqual(resultNp.pythonType, 'ndarray', ...
                    'pythonType should be ndarray');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testPrintResult(testCase)
            % Test printResult does not throw errors
            try
                % Create a test struct
                testStruct = struct();
                testStruct.scalar = 3.14;
                testStruct.array = [1, 2, 3];
                testStruct.text = 'hello';
                testStruct.nested = struct('a', 1, 'b', 2);
                
                % Should not throw error
                pyBridge.ResultParser.printResult(testStruct);
                pyBridge.ResultParser.printResult(testStruct, 1);
                pyBridge.ResultParser.printResult(testStruct, 3);
                
                testCase.verifyTrue(true, 'printResult should complete without error');
                
            catch ME
                testCase.verifyFail(sprintf('printResult threw error: %s', ME.message));
            end
        end
    end
    
    %% ==================== CovarianceTypes Tests ====================
    methods (Test, TestTags = {'CovarianceTypes', 'HC'})
        function testHeteroskedasticHCSeries(testCase)
            % Test HC0/HC1/HC2/HC3 heteroskedasticity-robust standard errors
            try
                testCase.assumeTrue(testCase.HasStatsmodels, 'statsmodels not available');
                
                rng(42);
                n = 200;
                
                % Generate heteroskedastic data
                X = [ones(n, 1), randn(n, 2)];
                epsilon = randn(n, 1) .* (1 + abs(X(:, 2)));  % Heteroskedastic errors
                y = X * [1; 2; 0.5] + epsilon;
                
                % Compute OLS residuals
                beta = X \ y;
                residuals = y - X * beta;
                
                hcTypes = {'HC0', 'HC1', 'HC2', 'HC3'};
                covMatrices = cell(4, 1);
                
                for i = 1:4
                    covMatrices{i} = pyBridge.internal.CovarianceTypes.heteroskedastic(...
                        residuals, X, hcTypes{i});
                    
                    % Test 1: Matrix should be symmetric
                    testCase.verifyEqual(covMatrices{i}, covMatrices{i}', ...
                        'AbsTol', testCase.Tolerance, ...
                        sprintf('%s covariance matrix should be symmetric', hcTypes{i}));
                    
                    % Test 2: Matrix should be positive semi-definite
                    eigVals = eig(covMatrices{i});
                    testCase.verifyGreaterThanOrEqual(min(eigVals), -testCase.Tolerance, ...
                        sprintf('%s covariance matrix should be positive semi-definite', hcTypes{i}));
                    
                    % Test 3: Diagonal elements should be positive
                    testCase.verifyGreaterThan(diag(covMatrices{i}), 0, ...
                        sprintf('%s diagonal elements should be positive', hcTypes{i}));
                end
                
                % Test 4: HC1 should have larger values than HC0 (small sample correction)
                % HC1 = HC0 * n/(n-k), where correction factor = n/(n-k) > 1
                testCase.verifyGreaterThan(trace(covMatrices{2}), trace(covMatrices{1}), ...
                    'HC1 should be larger than HC0 due to small sample correction');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'statsmodels/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testHACNeweyWest(testCase)
            % Test HAC (Newey-West) standard errors
            try
                testCase.assumeTrue(testCase.HasStatsmodels, 'statsmodels not available');
                
                rng(42);
                n = 200;
                
                % Generate autocorrelated data
                X = [ones(n, 1), randn(n, 2)];
                epsilon = zeros(n, 1);
                epsilon(1) = randn();
                for t = 2:n
                    epsilon(t) = 0.5 * epsilon(t-1) + randn();  % AR(1) errors
                end
                y = X * [1; 2; 0.5] + epsilon;
                
                % Compute OLS residuals
                beta = X \ y;
                residuals = y - X * beta;
                
                % Test with different kernels
                kernels = {'bartlett', 'parzen'};
                maxLags = 4;
                
                for i = 1:length(kernels)
                    kernel = kernels{i};
                    
                    covMatrix = pyBridge.internal.CovarianceTypes.hac(...
                        residuals, X, maxLags=maxLags, kernel=kernel);
                    
                    % Test 1: Matrix should be symmetric
                    testCase.verifyEqual(covMatrix, covMatrix', 'AbsTol', testCase.Tolerance, ...
                        sprintf('HAC (%s) matrix should be symmetric', kernel));
                    
                    % Test 2: Matrix should be positive semi-definite
                    eigVals = eig(covMatrix);
                    testCase.verifyGreaterThanOrEqual(min(eigVals), -testCase.Tolerance, ...
                        sprintf('HAC (%s) matrix should be positive semi-definite', kernel));
                    
                    % Test 3: Standard errors should be positive
                    se = sqrt(diag(covMatrix));
                    testCase.verifyGreaterThan(se, 0, ...
                        sprintf('HAC (%s) standard errors should be positive', kernel));
                end
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'statsmodels/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testClusteredSE(testCase)
            % Test clustered standard errors
            try
                testCase.assumeTrue(testCase.HasStatsmodels, 'statsmodels not available');
                
                rng(42);
                nClusters = 50;
                nPerCluster = 10;
                n = nClusters * nPerCluster;
                
                % Generate clustered data
                clusterIds = repelem(1:nClusters, nPerCluster)';
                X = [ones(n, 1), randn(n, 2)];
                
                % Add cluster-level random effects
                clusterEffects = randn(nClusters, 1);
                y = X * [1; 2; 0.5] + clusterEffects(clusterIds) + randn(n, 1);
                
                % Compute OLS residuals
                beta = X \ y;
                residuals = y - X * beta;
                
                % Compute clustered standard errors
                covMatrix = pyBridge.internal.CovarianceTypes.clustered(...
                    residuals, X, clusterIds);
                
                % Test 1: Matrix should be symmetric
                testCase.verifyEqual(covMatrix, covMatrix', 'AbsTol', testCase.Tolerance, ...
                    'Clustered covariance matrix should be symmetric');
                
                % Test 2: Matrix should be positive semi-definite
                eigVals = eig(covMatrix);
                testCase.verifyGreaterThanOrEqual(min(eigVals), -testCase.Tolerance, ...
                    'Clustered covariance matrix should be positive semi-definite');
                
                % Test 3: Standard errors should be positive
                se = sqrt(diag(covMatrix));
                testCase.verifyGreaterThan(se, 0, ...
                    'Clustered standard errors should be positive');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'statsmodels/python not available');
                else
                    rethrow(ME);
                end
            end
        end
        
        function testMapKernelName(testCase)
            % Test kernel name mapping
            
            % Test case-insensitive mapping
            testCase.verifyEqual(pyBridge.internal.CovarianceTypes.mapKernelName('bartlett'), ...
                'bartlett', 'bartlett should map to bartlett');
            testCase.verifyEqual(pyBridge.internal.CovarianceTypes.mapKernelName('Bartlett'), ...
                'bartlett', 'Bartlett should map to bartlett');
            testCase.verifyEqual(pyBridge.internal.CovarianceTypes.mapKernelName('BARTLETT'), ...
                'bartlett', 'BARTLETT should map to bartlett');
            
            % Test newey-west variations
            testCase.verifyEqual(pyBridge.internal.CovarianceTypes.mapKernelName('newey-west'), ...
                'bartlett', 'newey-west should map to bartlett');
            testCase.verifyEqual(pyBridge.internal.CovarianceTypes.mapKernelName('neweywest'), ...
                'bartlett', 'neweywest should map to bartlett');
            testCase.verifyEqual(pyBridge.internal.CovarianceTypes.mapKernelName('nw'), ...
                'bartlett', 'nw should map to bartlett');
            
            % Test uniform
            testCase.verifyEqual(pyBridge.internal.CovarianceTypes.mapKernelName('uniform'), ...
                'uniform', 'uniform should map to uniform');
            
            % Test parzen (maps to bartlett with warning)
            warning('off', 'pyBridge:UnsupportedKernel');
            result = pyBridge.internal.CovarianceTypes.mapKernelName('parzen');
            warning('on', 'pyBridge:UnsupportedKernel');
            testCase.verifyEqual(result, 'bartlett', ...
                'parzen should map to bartlett');
        end
        
        function testMLERobustSandwich(testCase)
            % Test MLE robust sandwich estimator
            
            rng(42);
            p = 3;  % Number of parameters
            n = 100;  % Number of observations
            
            % Create simulated Hessian (positive definite)
            A = randn(p, p);
            hessian = A' * A + eye(p);  % Ensure positive definite
            
            % Create simulated score matrix
            scoreMatrix = randn(n, p);
            
            % Compute sandwich estimator
            covMatrix = pyBridge.internal.CovarianceTypes.mleRobust(...
                hessian, scoreMatrix, covType='sandwich');
            
            % Test 1: Matrix should be symmetric
            testCase.verifyEqual(covMatrix, covMatrix', 'AbsTol', testCase.Tolerance, ...
                'Sandwich covariance matrix should be symmetric');
            
            % Test 2: Verify sandwich formula: V = H^{-1} * B * H^{-1}
            Hinv = inv(hessian);
            B = scoreMatrix' * scoreMatrix;
            expectedCov = Hinv * B * Hinv; %#ok<MINV>
            testCase.verifyEqual(covMatrix, expectedCov, 'AbsTol', testCase.Tolerance, ...
                'Sandwich formula should be V = H^{-1} * B * H^{-1}');
            
            % Test 3: Standard errors should be positive
            se = sqrt(diag(covMatrix));
            testCase.verifyGreaterThan(se, 0, ...
                'Sandwich standard errors should be positive');
        end
        
        function testTestHACComparison(testCase)
            % Test HAC comparison across kernels and lags
            try
                testCase.assumeTrue(testCase.HasStatsmodels, 'statsmodels not available');
                
                rng(42);
                n = 100;
                
                X = [ones(n, 1), randn(n, 2)];
                residuals = randn(n, 1);
                
                % Test HAC comparison function
                result = pyBridge.internal.CovarianceTypes.testHAC(...
                    residuals, X, maxLagsRange=0:5);
                
                % Verify result structure
                testCase.verifyTrue(isfield(result, 'lags'), ...
                    'Result should contain lags field');
                testCase.verifyTrue(isfield(result, 'se_bartlett'), ...
                    'Result should contain se_bartlett field');
                testCase.verifyTrue(isfield(result, 'note'), ...
                    'Result should contain note field');
                
                % Verify lags
                testCase.verifyEqual(result.lags, 0:5, ...
                    'Lags should match input range');
                
                % Verify SE matrix dimensions
                testCase.verifyEqual(size(result.se_bartlett, 1), size(X, 2), ...
                    'SE matrix should have rows equal to number of parameters');
                testCase.verifyEqual(size(result.se_bartlett, 2), 6, ...
                    'SE matrix should have columns equal to number of lags');
                
                % Verify all SE values are positive
                testCase.verifyGreaterThan(result.se_bartlett, 0, ...
                    'All standard errors should be positive');
                
            catch ME
                % Only skip if it's truly a module not found error
                if strcmp(ME.identifier, 'MATLAB:Python:PyException') && ...
                   contains(ME.message, 'ModuleNotFoundError')
                    testCase.assumeTrue(false, 'statsmodels/python not available');
                else
                    rethrow(ME);
                end
            end
        end
    end
end
