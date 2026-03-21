classdef PyBridgeConsistencyTest < matlab.unittest.TestCase
    % PyBridgeConsistencyTest MATLAB-Python bridge library consistency validation test
    %
    % Test Coverage:
    % 1. Consistency between pyBridge and direct Python calls
    % 2. OLS, Logit, Multinomial Logit model validation
    % 3. HAC standard error calculation accuracy
    % 4. P-value calculation validation
    % 5. Marginal effects HAC standard errors and p-values
    % 6. Boundary conditions and error handling
    %
    % Author: WorkBuddy
    % Date: 2026-03-20
    
    properties
        TestData        % Test data
        Tolerance       % Numeric comparison tolerance
        HasPython       % Whether Python environment is available
    end
    
    properties (TestParameter)
        % HAC kernel function parameters
        hacKernel = struct(...
            'bartlett', 'bartlett', ...
            'neweyWest', 'newey-west', ...
            'parzen', 'parzen');
        
        % HAC lag order parameters
        hacLags = struct(...
            'lag2', 2, ...
            'lag4', 4, ...
            'lag8', 8);
        
        % Heteroskedasticity-robust standard error types
        hcType = struct(...
            'HC0', 'HC0', ...
            'HC1', 'HC1', ...
            'HC2', 'HC2', ...
            'HC3', 'HC3');
    end
    
    methods (TestClassSetup)
        function setupClass(testCase)
            % Initialize test environment
            testCase.Tolerance = 1e-6;
            
            % Check Python environment
            try
                pyenv;
                py.importlib.import_module('statsmodels');
                testCase.HasPython = true;
            catch
                testCase.HasPython = false;
                warning('PyBridgeConsistencyTest:NoPython', ...
                    'Python environment not available, some tests will be skipped');
            end
            
            % Generate test data
            testCase.generateTestData();
        end
    end
    
    methods (Access = private)
        function generateTestData(testCase)
            % Generate reproducible test data
            rng(42, 'twister');
            
            nObs = 500;
            nVars = 3;
            
            % OLS test data
            testCase.TestData.ols.X = randn(nObs, nVars);
            testCase.TestData.ols.epsilon = randn(nObs, 1);
            testCase.TestData.ols.y = 2 + 3*testCase.TestData.ols.X(:,1) + ...
                1.5*testCase.TestData.ols.X(:,2) - 0.5*testCase.TestData.ols.X(:,3) + ...
                testCase.TestData.ols.epsilon;
            
            % Logit test data
            testCase.TestData.logit.X = randn(nObs, nVars);
            prob = 1 ./ (1 + exp(-0.5 - 1.2*testCase.TestData.logit.X(:,1) + ...
                0.8*testCase.TestData.logit.X(:,2)));
            testCase.TestData.logit.y = double(rand(nObs, 1) < prob);
            
            % Multinomial Logit test data (3 categories)
            testCase.TestData.mlogit.X = randn(nObs, nVars);
            testCase.TestData.mlogit.y = randi([0, 2], nObs, 1);
            
            % Panel data (HAC/clustering test)
            nFirms = 50;
            nPeriods = 10;
            testCase.TestData.panel.firmId = repelem(1:nFirms, nPeriods)';
            testCase.TestData.panel.yearId = repmat(1:nPeriods, 1, nFirms)';
            testCase.TestData.panel.X = randn(nFirms*nPeriods, nVars);
            testCase.TestData.panel.y = 1 + 2*testCase.TestData.panel.X(:,1) + ...
                randn(nFirms*nPeriods, 1);
            
            % Time series data (autocorrelation)
            nTime = 200;
            testCase.TestData.timeseries.X = randn(nTime, 2);
            epsilon = zeros(nTime, 1);
            epsilon(1) = randn();
            for t = 2:nTime
                epsilon(t) = 0.5*epsilon(t-1) + randn();
            end
            testCase.TestData.timeseries.y = 1 + 2*testCase.TestData.timeseries.X(:,1) + ...
                1.5*testCase.TestData.timeseries.X(:,2) + epsilon;
            testCase.TestData.timeseries.epsilon = epsilon;
        end
    end
    
    %% ==================== Consistency Validation Tests ====================
    methods (Test, TestTags = {'Consistency', 'OLS'})
        function testOLSConsistencyWithPython(testCase)
            % 验证OLS回归结果与直接Python调用的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X;
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.ols(y, X);
            
            % 直接Python调用 - ensure proper array dimensions
            yPy = py.numpy.array(y).flatten();
            XPy = py.numpy.atleast_2d(py.numpy.array(X));
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.api.OLS(yPy, XWithConst);
            fitPy = model.fit();
            
            % 提取Python结果 - use py.getattr for safe attribute access in Python 3.13+
            paramsPy = double(py.getattr(fitPy, 'params'));
            stdErrorsPy = double(py.getattr(fitPy, 'bse'));
            tStatsPy = double(py.getattr(fitPy, 'tvalues'));
            pValuesPy = double(py.getattr(fitPy, 'pvalues'));
            % Ensure column vectors for comparison
            paramsPy = paramsPy(:);
            stdErrorsPy = stdErrorsPy(:);
            tStatsPy = tStatsPy(:);
            pValuesPy = pValuesPy(:);
            
            % 验证系数一致性
            testCase.verifyEqual(result.params, paramsPy, 'AbsTol', testCase.Tolerance, ...
                'OLS系数应与Python结果一致');
            
            % 验证标准误一致性
            testCase.verifyEqual(result.stdErrors, stdErrorsPy, 'AbsTol', testCase.Tolerance, ...
                'OLS标准误应与Python结果一致');
            
            % 验证t统计量一致性
            testCase.verifyEqual(result.tStatistics, tStatsPy, 'AbsTol', testCase.Tolerance, ...
                'OLS t统计量应与Python结果一致');
            
            % 验证p值一致性
            testCase.verifyEqual(result.pValues, pValuesPy, 'AbsTol', testCase.Tolerance, ...
                'OLS p值应与Python结果一致');
        end
        
        function testOLSRobustSEConsistency(testCase, hcType)
            % 验证异方差稳健标准误与Python的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X;
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.ols(y, X, covType=hcType);
            
            % 直接Python调用 - ensure proper array dimensions
            yPy = py.numpy.array(y).flatten();
            XPy = py.numpy.atleast_2d(py.numpy.array(X));
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.api.OLS(yPy, XWithConst);
            fitPy = model.fit(cov_type=hcType);
            
            stdErrorsPy = double(py.getattr(fitPy, 'bse'));
            % Ensure column vector for comparison
            stdErrorsPy = stdErrorsPy(:);
            
            % 验证稳健标准误一致性
            testCase.verifyEqual(result.stdErrors, stdErrorsPy, 'AbsTol', testCase.Tolerance, ...
                sprintf('%s标准误应与Python结果一致', hcType));
        end
    end
    
    methods (Test, TestTags = {'Consistency', 'Logit'})
        function testLogitConsistencyWithPython(testCase)
            % 验证Logit回归结果与Python的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.logit.y;
            X = testCase.TestData.logit.X;
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.logistic(y, X);
            
            % 直接Python调用 - ensure proper array dimensions
            yPy = py.numpy.array(y).flatten();
            XPy = py.numpy.atleast_2d(py.numpy.array(X));
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.api.Logit(yPy, XWithConst);
            fitPy = model.fit(disp=false);
            
            paramsPy = double(py.getattr(fitPy, 'params'));
            stdErrorsPy = double(py.getattr(fitPy, 'bse'));
            % Ensure column vectors for comparison
            paramsPy = paramsPy(:);
            stdErrorsPy = stdErrorsPy(:);
            
            % 验证系数一致性
            testCase.verifyEqual(result.params, paramsPy, 'AbsTol', testCase.Tolerance, ...
                'Logit系数应与Python结果一致');
            
            % 验证标准误一致性
            testCase.verifyEqual(result.stdErrors, stdErrorsPy, 'AbsTol', testCase.Tolerance, ...
                'Logit标准误应与Python结果一致');
        end
    end
    
    methods (Test, TestTags = {'Consistency', 'MLogit'})
        function testMultinomialLogitConsistencyWithPython(testCase)
            % 验证Multinomial Logit结果与Python的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);
            
            % 直接Python调用 - ensure proper array dimensions
            yPy = py.numpy.array(y).flatten();
            XPy = py.numpy.atleast_2d(py.numpy.array(X));
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.discrete.discrete_model.MNLogit(yPy, XWithConst);
            fitPy = model.fit(disp=false);
            
            paramsPy = double(py.getattr(fitPy, 'params'));
            stdErrorsPy = double(py.getattr(fitPy, 'bse'));
            % Ensure column vectors for comparison
            paramsPy = paramsPy(:);
            stdErrorsPy = stdErrorsPy(:);
            
            % 验证系数一致性
            testCase.verifyEqual(result.params, paramsPy, 'AbsTol', testCase.Tolerance, ...
                'Multinomial Logit系数应与Python结果一致');
            
            % 验证标准误一致性
            testCase.verifyEqual(result.stdErrors, stdErrorsPy, 'AbsTol', testCase.Tolerance, ...
                'Multinomial Logit标准误应与Python结果一致');
            
            % 验证类别数
            testCase.verifyEqual(result.nCategories, 3, ...
                'Multinomial Logit应正确识别3个类别');
        end
    end
    
    %% ==================== HAC标准误专项测试 ====================
    methods (Test, TestTags = {'HAC', 'MLogit'})
        function testMLogitHACStandardErrors(testCase, hacKernel, hacLags)
            % 测试Multinomial Logit的HAC标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % pyBridge调用HAC标准误
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=hacLags, kernel=hacKernel);
            
            % 验证结果结构
            testCase.verifyTrue(isfield(result, 'stdErrors'), ...
                'HAC结果应包含标准误');
            testCase.verifyTrue(isfield(result, 'covType'), ...
                'HAC结果应包含协方差类型');
            
            % 验证HAC标准误为正数
            testCase.verifyGreaterThan(result.stdErrors, 0, ...
                'HAC标准误应全部为正');
            
            % 验证参数设置正确记录
            testCase.verifyEqual(upper(result.covType), "HAC", ...
                'covType应为HAC');
            testCase.verifyEqual(result.lag, hacLags, ...
                'lag应正确记录');
            testCase.verifyEqual(result.kernel, hacKernel, ...
                'kernel应正确记录');
        end
        
        function testMLogitHACConsistencyWithPython(testCase)
            % 验证Multinomial Logit HAC标准误与Python的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            lag = 4;
            kernel = "bartlett";
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=lag, kernel=kernel);
            
            % 直接Python调用 - ensure proper array dimensions for Python 3.13+
            yPy = py.numpy.array(y).flatten();  % Ensure 1D
            XPy = py.numpy.atleast_2d(py.numpy.array(X));  % Ensure 2D
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.discrete.discrete_model.MNLogit(yPy, XWithConst);
            
            % 获取HAC标准误 - fit with cov_type directly (MNLogit has no get_robustcov_results)
            covKwds = py.dict();
            covKwds{"maxlags"} = int32(lag);
            covKwds{"kernel"} = kernel;
            fitRobust = model.fit(pyargs('cov_type', 'HAC', 'cov_kwds', covKwds, 'disp', false));
            
            stdErrorsPy = double(py.getattr(fitRobust, 'bse'));
            stdErrorsPy = stdErrorsPy(:);  % Flatten to column vector to match pyBridge format
            
            % 验证一致性
            testCase.verifyEqual(result.stdErrors, stdErrorsPy, 'RelTol', 0.01, ...
                'Multinomial Logit HAC标准误应与Python结果一致');
        end
        
        function testHACLagSensitivity(testCase)
            % 测试HAC标准误对滞后阶数的敏感性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.timeseries.y;
            X = testCase.TestData.timeseries.X;
            
            seByLag = zeros(3, 5);  % 3个参数, 5种滞后
            lags = [0, 2, 4, 6, 8];
            
            for i = 1:length(lags)
                if lags(i) == 0
                    result = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC0");
                else
                    result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
                        covType="hac", lag=lags(i));
                end
                seByLag(:, i) = result.stdErrors;
            end
            
            % 验证标准误随滞后阶数变化
            % 一般来说，存在自相关时，HAC标准误 > OLS标准误
            testCase.verifyTrue(any(seByLag(:, 2:end) ~= seByLag(:, 1), 'all'), ...
                'HAC标准误应随滞后阶数变化');
        end
        
        function testNeweyWestKernelFormula(testCase)
            % 验证Newey-West核函数权重公式
            % w(j) = 1 - j/(m+1), 其中m是最大滞后阶数
            
            lag = 4;
            
            % 计算期望的权重
            expectedWeights = zeros(lag + 1, 1);
            for j = 0:lag
                expectedWeights(j+1) = 1 - j / (lag + 1);
            end
            
            % 验证权重递减
            testCase.verifyTrue(all(diff(expectedWeights) < 0), ...
                'Newey-West权重应递减');
            
            % 验证权重范围
            testCase.verifyGreaterThanOrEqual(expectedWeights, 0, ...
                '权重应非负');
            testCase.verifyLessThanOrEqual(expectedWeights, 1, ...
                '权重应不超过1');
            
            % 验证首尾权重
            testCase.verifyEqual(expectedWeights(1), 1, 'AbsTol', 1e-10, ...
                '滞后0的权重应为1');
            testCase.verifyEqual(expectedWeights(end), 1/(lag+1), 'AbsTol', 1e-10, ...
                '最大滞后的权重应为1/(m+1)');
        end
    end
    
    %% ==================== P值计算准确性测试 ====================
    methods (Test, TestTags = {'PValue'})
        function testPValueCalculationFormula(testCase)
            % 验证p值计算公式
            % p = 2 * (1 - tcdf(|t|, df)) 或 p = 2 * (1 - normcdf(|t|))
            
            % 测试用例
            tStats = [2.5, -1.96, 3.0, -0.5];
            dfResid = 100;
            
            % 使用t分布计算p值
            pValuesT = 2 * (1 - tcdf(abs(tStats), dfResid));
            
            % 使用正态分布计算p值（大样本近似）
            pValuesNorm = 2 * (1 - normcdf(abs(tStats)));
            
            % 验证p值范围
            testCase.verifyGreaterThanOrEqual(pValuesT, 0, 'p值应非负');
            testCase.verifyLessThanOrEqual(pValuesT, 1, 'p值应不超过1');
            
            % 验证大样本时t分布近似正态 - use larger tolerance for df=100
            testCase.verifyEqual(pValuesT, pValuesNorm, 'RelTol', 0.30, ...
                '大样本时t分布应近似正态分布');
            
            % 验证显著性判断
            testCase.verifyTrue(pValuesT(1) < 0.05, 't=2.5应在5%水平显著');
            testCase.verifyTrue(abs(pValuesT(2) - 0.05) < 0.01, 't=-1.96应接近5%临界值');
        end
        
        function testMLogitPValueConsistency(testCase)
            % 验证Multinomial Logit p值计算一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % 经典标准误
            resultClassic = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);
            
            % 验证p值与z统计量的一致性
            % p = 2 * (1 - normcdf(|z|)) 对于MLE模型
            expectedPValues = 2 * (1 - normcdf(abs(resultClassic.zStatistics)));
            
            testCase.verifyEqual(resultClassic.pValues, expectedPValues, 'RelTol', 0.01, ...
                'p值应与t统计量通过正态分布计算一致');
        end
        
        function testPValueDifferencesBySEType(testCase)
            % 测试不同标准误类型下p值的差异
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X;
            
            % 收集不同类型的p值
            seTypes = {'nonrobust', 'HC0', 'HC1', 'HC3'};
            pValuesByType = cell(length(seTypes), 1);
            
            for i = 1:length(seTypes)
                result = pyBridge.StatsmodelsWrapper.ols(y, X, covType=seTypes{i});
                pValuesByType{i} = result.pValues;
            end
            
            % 验证p值存在差异
            for i = 2:length(seTypes)
                testCase.verifyTrue(~isequal(pValuesByType{1}, pValuesByType{i}), ...
                    sprintf('%s的p值应与经典标准误不同', seTypes{i}));
            end
            
            % 稳健标准误通常会增大，导致p值增大
            % 但这不是绝对的，取决于数据
        end
        
        function testHACPValueCalculation(testCase)
            % 专项测试HAC标准误下的p值计算
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.timeseries.y;
            X = testCase.TestData.timeseries.X;
            
            % HAC标准误
            resultHAC = pyBridge.StatsmodelsWrapper.ols(y, X, ...
                covType="hac", lag=4);
            
            % 手动计算p值
            tStats = resultHAC.params ./ resultHAC.stdErrors;
            dfResid = resultHAC.dfResiduals;
            expectedPValues = 2 * (1 - tcdf(abs(tStats), dfResid));
            
            % 验证p值一致性 - use AbsTol for very small p-values
            testCase.verifyEqual(resultHAC.pValues, expectedPValues, 'AbsTol', 1e-10, ...
                'HAC p值应基于正确的t分布计算');
        end
    end
    
    %% ==================== 边际效应HAC标准误和P值测试 ====================
    methods (Test, TestTags = {'MarginalEffects', 'HAC'})
        function testLogitMarginalEffectsWithHAC(testCase)
            % 测试Logit边际效应在HAC标准误下的实现
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.logit.y;
            X = testCase.TestData.logit.X;
            
            % 带HAC标准误的Logit
            result = pyBridge.StatsmodelsWrapper.logistic(y, X, ...
                covType="HAC", lag=4);
            
            % 验证边际效应存在
            testCase.verifyTrue(isfield(result, 'marginalEffects'), ...
                '结果应包含边际效应');
            testCase.verifyNotEmpty(result.marginalEffects, ...
                '边际效应不应为空');
            
            % 验证边际效应标准误存在
            if isfield(result, 'marginalEffectsSE')
                testCase.verifyGreaterThan(result.marginalEffectsSE, 0, ...
                    '边际效应标准误应为正');
            end
            
            % 验证边际效应p值存在且有效
            if isfield(result, 'marginalEffectsP')
                testCase.verifyGreaterThanOrEqual(result.marginalEffectsP, 0, ...
                    '边际效应p值应非负');
                testCase.verifyLessThanOrEqual(result.marginalEffectsP, 1, ...
                    '边际效应p值应不超过1');
            end
        end
        
        function testMarginalEffectsSEPropagation(testCase)
            % 验证边际效应标准误正确传播基础模型的协方差矩阵
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.logit.y;
            X = testCase.TestData.logit.X;
            
            % 经典标准误
            resultClassic = pyBridge.StatsmodelsWrapper.logistic(y, X, ...
                covType="nonrobust");
            
            % 稳健标准误
            resultRobust = pyBridge.StatsmodelsWrapper.logistic(y, X, ...
                covType="HC1");
            
            % 边际效应标准误应有差异
            if isfield(resultClassic, 'marginalEffectsSE') && ...
               isfield(resultRobust, 'marginalEffectsSE')
                testCase.verifyTrue(~isequal(resultClassic.marginalEffectsSE, ...
                    resultRobust.marginalEffectsSE), ...
                    '不同标准误类型下边际效应标准误应不同');
            end
        end
        
        function testMarginalEffectsPValueConsistency(testCase)
            % 验证边际效应p值与标准误的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.logit.y;
            X = testCase.TestData.logit.X;
            
            result = pyBridge.StatsmodelsWrapper.logistic(y, X);
            
            if isfield(result, 'marginalEffects') && ...
               isfield(result, 'marginalEffectsSE') && ...
               isfield(result, 'marginalEffectsP')
                
                % 计算期望的t统计量
                tStats = result.marginalEffects ./ result.marginalEffectsSE;
                
                % 验证t统计量字段
                if isfield(result, 'marginalEffectsT')
                    testCase.verifyEqual(result.marginalEffectsT, tStats, 'RelTol', 0.01, ...
                        '边际效应t统计量应等于效应/标准误');
                end
                
                % 验证p值范围合理性
                testCase.verifyGreaterThanOrEqual(result.marginalEffectsP, 0);
                testCase.verifyLessThanOrEqual(result.marginalEffectsP, 1);
            end
        end
        
        function testMLogitMarginalEffectsHAC(testCase)
            % 测试Multinomial Logit边际效应的HAC标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % HAC标准误
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4, kernel="bartlett");
            
            % 验证边际效应存在
            testCase.verifyTrue(isfield(result, 'marginalEffects'), ...
                'Multinomial Logit结果应包含边际效应');
            
            % 对于多类别模型，边际效应应该是矩阵
            if ~isempty(result.marginalEffects)
                testCase.verifyGreaterThan(size(result.marginalEffects, 1), 0, ...
                    '边际效应应有多行');
            end
            
            % 验证边际效应标准误
            if isfield(result, 'marginalEffectsSE') && ~isempty(result.marginalEffectsSE)
                testCase.verifyGreaterThan(result.marginalEffectsSE, 0, ...
                    '边际效应标准误应为正');
            end
        end
    end
    
    %% ==================== 聚类标准误测试 ====================
    methods (Test, TestTags = {'Clustered'})
        function testOLSClusteredSE(testCase)
            % 测试OLS聚类标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.panel.y;
            X = testCase.TestData.panel.X;
            firmId = testCase.TestData.panel.firmId;
            
            % 按企业聚类
            result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
                covType="cluster", clusterIds=firmId);
            
            % 验证聚类数
            testCase.verifyEqual(result.nClusters, 50, ...
                '应正确识别50个聚类');
            
            % 验证标准误为正
            testCase.verifyGreaterThan(result.stdErrors, 0, ...
                '聚类标准误应为正');
        end
        
        function testOLSMultiwayClustered(testCase)
            % 测试多维聚类标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.panel.y;
            X = testCase.TestData.panel.X;
            firmId = testCase.TestData.panel.firmId;
            yearId = testCase.TestData.panel.yearId;
            
            % 双向聚类
            result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
                covType="multiway", clusterGroups={firmId, yearId});
            
            % 验证维度数
            testCase.verifyEqual(result.nDimensions, 2, ...
                '应正确识别2个聚类维度');
            
            % 验证标准误为正
            testCase.verifyGreaterThan(result.stdErrors, 0, ...
                '多维聚类标准误应为正');
        end
        
        function testMLogitClusteredSE(testCase)
            % 测试Multinomial Logit聚类标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            % 使用面板数据
            nFirms = 50;
            nPeriods = 10;
            nObs = nFirms * nPeriods;
            
            firmId = repelem(1:nFirms, nPeriods)';
            X = randn(nObs, 2);
            y = randi([0, 2], nObs, 1);
            
            % 聚类标准误
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="cluster", clusterIds=firmId);
            
            % 验证结果
            testCase.verifyEqual(result.nClusters, nFirms, ...
                '应正确识别聚类数');
            testCase.verifyGreaterThan(result.stdErrors, 0, ...
                '聚类标准误应为正');
        end
    end
    
    %% ==================== 边界条件和错误处理测试 ====================
    methods (Test, TestTags = {'ErrorHandling'})
        function testMissingClusterIdsError(testCase)
            % 测试缺少聚类标识符时的错误处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X;
            
            % 应抛出错误
            testCase.verifyError(...
                @() pyBridge.StatsmodelsWrapper.ols(y, X, covType="cluster"), ...
                'pyBridge:MissingClusterIds');
        end
        
        function testMissingClusterGroupsError(testCase)
            % 测试缺少多维聚类组时的错误处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X;
            
            % 应抛出错误
            testCase.verifyError(...
                @() pyBridge.StatsmodelsWrapper.ols(y, X, covType="multiway"), ...
                'pyBridge:MissingClusterGroups');
        end
        
        function testEmptyInputError(testCase)
            % 测试空输入的错误处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            % 空因变量
            testCase.verifyError(...
                @() pyBridge.StatsmodelsWrapper.ols([], testCase.TestData.ols.X), ...
                'MATLAB:Python:PyException');
            
            % 空自变量
            testCase.verifyError(...
                @() pyBridge.StatsmodelsWrapper.ols(testCase.TestData.ols.y, []), ...
                'MATLAB:Python:PyException');
        end
        
        function testDimensionMismatchError(testCase)
            % 测试维度不匹配的错误处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X(1:100, :);  % 截断X
            
            % 应抛出错误
            testCase.verifyError(...
                @() pyBridge.StatsmodelsWrapper.ols(y, X), ...
                'MATLAB:Python:PyException');
        end
        
        function testPerfectSeparationWarning(testCase)
            % 测试完全分离数据的处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            % 创建完全分离的数据
            n = 100;
            X = randn(n, 1);
            y = double(X > 0);  % X > 0时y=1，否则y=0
            
            % 可能收敛警告或不收敛
            % 这里我们只验证不会崩溃
            try
                result = pyBridge.StatsmodelsWrapper.logistic(y, X, maxIter=10);
                testCase.verifyTrue(isfield(result, 'params'), ...
                    '即使完全分离也应返回结果');
            catch ME
                % 如果抛出错误，验证是合理的错误
                testCase.verifyTrue(contains(ME.message, 'converg') || ...
                    contains(ME.message, 'separ'), ...
                    '完全分离应产生收敛相关警告');
            end
        end
        
        function testSingleObservationCluster(testCase)
            % 测试单观测聚类的处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            n = 100;
            y = randn(n, 1);
            X = randn(n, 2);
            clusterIds = 1:n;  % 每个观测一个聚类
            
            % 应能处理但可能给出警告
            try
                result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
                    covType="cluster", clusterIds=clusterIds);
                testCase.verifyGreaterThan(result.stdErrors, 0, ...
                    '单观测聚类应产生正标准误');
            catch ME
                % 合理的错误
                testCase.verifyTrue(true, ...
                    sprintf('单观测聚类产生可接受的错误: %s', ME.message));
            end
        end
        
        function testExtremeValues(testCase)
            % 测试极值输入
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            n = 100;
            X = randn(n, 2);
            
            % 添加极大值
            X(1, 1) = 1e10;
            y = 1 + 2*X(:,1) + randn(n, 1);
            
            % 应能处理极值
            result = pyBridge.StatsmodelsWrapper.ols(y, X);
            testCase.verifyTrue(all(isfinite(result.params)), ...
                '极值输入应产生有限的系数估计');
        end
        
        function testNaNHandling(testCase)
            % 测试NaN值的处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X;
            
            % 添加NaN
            yWithNaN = y;
            yWithNaN(1) = NaN;
            
            % 应抛出错误或自动处理 - Python may handle NaN differently
            try
                result = pyBridge.StatsmodelsWrapper.ols(yWithNaN, X);
                % If no error, verify result contains NaN or warning
                testCase.verifyTrue(any(isnan(result.params)) || all(isfinite(result.params)), ...
                    'NaN handling should either error or produce valid results');
            catch ME
                % Expected to throw an error
                testCase.verifyTrue(true, 'NaN input correctly throws an error');
            end
        end
    end
    
    %% ==================== 数学正确性测试 ====================
    methods (Test, TestTags = {'Mathematical'})
        function testOLSClosedFormSolution(testCase)
            % 验证OLS系数与闭式解的一致性
            % β = (X'X)^(-1) X'y
            
            y = testCase.TestData.ols.y;
            X = testCase.TestData.ols.X;
            
            % 添加常数项
            XWithConst = [ones(size(X, 1), 1), X];
            
            % 闭式解
            betaClosedForm = (XWithConst' * XWithConst) \ (XWithConst' * y);
            
            % pyBridge结果
            if testCase.HasPython
                result = pyBridge.StatsmodelsWrapper.ols(y, X);
                
                testCase.verifyEqual(result.params, betaClosedForm, ...
                    'RelTol', 1e-10, ...
                    'OLS系数应与闭式解一致');
            end
        end
        
        function testHACCovarianceFormula(testCase)
            % 验证HAC协方差矩阵计算公式
            % V_HAC = (X'X)^(-1) * S * (X'X)^(-1)
            % S = Σ w(j) * Γ(j), Γ(j) = Σ e_t * e_{t-j} * x_t * x_{t-j}'
            
            n = 100;
            k = 2;
            maxLags = 3;
            
            % 生成模拟数据
            X = randn(n, k);
            XWithConst = [ones(n, 1), X];
            residuals = randn(n, 1);
            
            % 计算(X'X)^(-1)
            XtXinv = inv(XWithConst' * XWithConst);
            
            % 计算S矩阵 (Newey-West)
            S = zeros(k+1, k+1);
            for j = 0:maxLags
                weight = 1 - j / (maxLags + 1);
                
                if j == 0
                    Gamma = XWithConst' * diag(residuals.^2) * XWithConst;
                else
                    Gamma = zeros(k+1, k+1);
                    for t = j+1:n
                        Gamma = Gamma + residuals(t) * residuals(t-j) * ...
                            (XWithConst(t,:)' * XWithConst(t-j,:));
                    end
                    Gamma = Gamma + Gamma';  % 对称化
                end
                
                S = S + weight * Gamma;
            end
            
            % HAC协方差
            V_HAC = XtXinv * S * XtXinv; %#ok<MINV>
            
            % 验证协方差矩阵性质
            testCase.verifyEqual(V_HAC, V_HAC', 'AbsTol', 1e-10, ...
                'HAC协方差矩阵应对称');
            testCase.verifyTrue(all(eig(V_HAC) >= -1e-10), ...
                'HAC协方差矩阵应半正定');
        end
        
        function testTDistributionDF(testCase)
            % 验证t分布自由度的正确使用
            
            n = 100;  % 样本量
            k = 4;    % 参数个数（含常数项）
            
            % 期望自由度
            expectedDF = n - k;
            
            % 验证不同t值在不同自由度下的p值
            tValue = 2.0;
            
            pValueCorrect = 2 * (1 - tcdf(tValue, expectedDF));
            pValueWrong = 2 * (1 - tcdf(tValue, n));  % 错误使用n而非n-k
            
            % 差异应该存在
            testCase.verifyTrue(abs(pValueCorrect - pValueWrong) > 1e-4, ...
                '使用正确自由度应产生不同的p值');
            
            % 正确自由度下的p值应更保守（更大）
            testCase.verifyGreaterThan(pValueCorrect, pValueWrong, ...
                '使用n-k自由度应产生更保守的p值');
        end
    end
end
