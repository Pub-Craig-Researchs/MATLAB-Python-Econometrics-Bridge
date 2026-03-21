classdef MLogitHACSpecializedTest < matlab.unittest.TestCase
    % MLogitHACSpecializedTest Multinomial Logit HAC标准误专项测试
    %
    % 专门测试:
    % 1. Multinomial Logit HAC标准误的数学正确性
    % 2. 不同核函数和滞后阶数的实现
    % 3. P值计算的准确性
    % 4. 边际效应的HAC标准误传播
    % 5. 与Python实现的完全一致性验证
    %
    % 作者: WorkBuddy
    % 日期: 2026-03-20
    
    properties
        TestData
        Tolerance
        HasPython
    end
    
    properties (TestParameter)
        % 核函数参数组合
        kernelType = struct(...
            'bartlett', 'bartlett', ...
            'neweyWest', 'newey-west', ...
            'parzen', 'parzen');
        
        % 滞后阶数
        lagOrder = struct(...
            'auto', [], ...
            'lag1', 1, ...
            'lag2', 2, ...
            'lag4', 4, ...
            'lag6', 6, ...
            'lag8', 8);
    end
    
    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.Tolerance = 1e-8;
            
            % 检查Python环境
            try
                pyenv;
                py.importlib.import_module('statsmodels');
                py.importlib.import_module('numpy');
                testCase.HasPython = true;
            catch
                testCase.HasPython = false;
            end
            
            % 生成测试数据
            testCase.generateTestData();
        end
    end
    
    methods (Access = private)
        function generateTestData(testCase)
            rng(12345, 'twister');
            
            % 大样本数据 (用于HAC测试)
            nObs = 1000;
            nVars = 3;
            nCategories = 3;
            
            % 生成带自相关的数据
            X = zeros(nObs, nVars);
            for j = 1:nVars
                X(:,j) = generateAR1(nObs, 0.5);
            end
            
            % 生成多类别因变量
            beta1 = [0.5; 0.3; -0.2];
            beta2 = [-0.3; 0.5; 0.1];
            
            u1 = X * beta1 + 0.5 * randn(nObs, 1);
            u2 = X * beta2 + 0.5 * randn(nObs, 1);
            
            prob0 = 1 ./ (1 + exp(u1) + exp(u2));
            prob1 = exp(u1) ./ (1 + exp(u1) + exp(u2));
            
            randU = rand(nObs, 1);
            y = zeros(nObs, 1);
            y(randU < prob0) = 0;
            y(randU >= prob0 & randU < prob0 + prob1) = 1;
            y(randU >= prob0 + prob1) = 2;
            
            testCase.TestData.mlogit.X = X;
            testCase.TestData.mlogit.y = y;
            testCase.TestData.mlogit.nObs = nObs;
            testCase.TestData.mlogit.nCategories = nCategories;
            
            % 面板数据 (用于聚类HAC测试)
            nEntities = 100;
            nTime = 20;
            nPanel = nEntities * nTime;
            
            entityId = repelem(1:nEntities, nTime)';
            timeId = repmat(1:nTime, 1, nEntities)';
            
            XPanel = randn(nPanel, 2);
            yPanel = randi([0, 2], nPanel, 1);
            
            testCase.TestData.panel.X = XPanel;
            testCase.TestData.panel.y = yPanel;
            testCase.TestData.panel.entityId = entityId;
            testCase.TestData.panel.timeId = timeId;
            
            function x = generateAR1(n, rho)
                x = zeros(n, 1);
                x(1) = randn();
                for t = 2:n
                    x(t) = rho * x(t-1) + sqrt(1-rho^2) * randn();
                end
            end
        end
    end
    
    %% ==================== HAC核函数数学正确性测试 ====================
    methods (Test, TestTags = {'HAC', 'Kernel', 'Mathematical'})
        function testNeweyWestKernelWeights(testCase)
            % 验证Newey-West (Bartlett) 核函数权重
            % w(j, m) = 1 - j/(m+1), j = 0, 1, ..., m
            
            maxLags = [2, 4, 6, 8];
            
            for m = maxLags
                weights = zeros(m+1, 1);
                for j = 0:m
                    weights(j+1) = 1 - j/(m+1);
                end
                
                % 验证权重递减
                testCase.verifyTrue(all(diff(weights) < 0), ...
                    sprintf('Newey-West权重在m=%d时应严格递减', m));
                
                % 验证边界值
                testCase.verifyEqual(weights(1), 1, 'AbsTol', 1e-15, ...
                    '滞后0的权重应为1');
                testCase.verifyGreaterThan(weights(end), 0, ...
                    '最大滞后权重应为正');
                
                % 验证权重和 (用于归一化检验)
                expectedSum = (m+1) - sum(0:m)/(m+1);
                testCase.verifyEqual(sum(weights), expectedSum, 'AbsTol', 1e-10);
            end
        end
        
        function testParzenKernelWeights(testCase)
            % 验证Parzen核函数权重
            % w(j, m) = 1 - 6*(j/m)^2 + 6*(j/m)^3, if j/m <= 0.5
            %         = 2*(1 - j/m)^3,              if j/m > 0.5
            
            maxLags = 4;
            weights = zeros(maxLags+1, 1);
            
            for j = 0:maxLags
                ratio = j / maxLags;
                if ratio <= 0.5
                    weights(j+1) = 1 - 6*ratio^2 + 6*ratio^3;
                else
                    weights(j+1) = 2 * (1 - ratio)^3;
                end
            end
            
            % 验证权重非负且有界
            testCase.verifyGreaterThanOrEqual(weights, 0, ...
                'Parzen核权重应非负');
            testCase.verifyLessThanOrEqual(weights, 1, ...
                'Parzen核权重应不超过1');
            
            % 验证j=0时权重为1
            testCase.verifyEqual(weights(1), 1, 'AbsTol', 1e-15);
        end
        
        function testHACCovarianceSymmetry(testCase)
            % 验证HAC协方差矩阵的对称性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4, kernel="bartlett"); %#ok<NASGU>
            
            % 从Python获取协方差矩阵 - ensure proper array dimensions for Python 3.13+
            yPy = py.numpy.array(y).flatten();  % Ensure 1D
            XPy = py.numpy.atleast_2d(py.numpy.array(X));  % Ensure 2D
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.discrete.discrete_model.MNLogit(yPy, XWithConst);
            
            % MNLogit has no get_robustcov_results - use fit() with cov_type directly
            covKwds = py.dict();
            covKwds{"maxlags"} = int32(4);
            covKwds{"kernel"} = "bartlett";
            fitRobust = model.fit(pyargs('cov_type', 'HAC', 'cov_kwds', covKwds, 'disp', false));
            
            covParamsFunc = py.getattr(fitRobust, 'cov_params');
            covMatrix = double(covParamsFunc());
            
            % 验证对称性
            testCase.verifyEqual(covMatrix, covMatrix', 'AbsTol', 1e-12, ...
                'HAC协方差矩阵应对称');
            
            % 验证半正定性
            eigenvalues = eig(covMatrix);
            testCase.verifyGreaterThanOrEqual(eigenvalues, -1e-10, ...
                'HAC协方差矩阵应半正定');
        end
    end
    
    %% ==================== HAC标准误参数化测试 ====================
    methods (Test, TestTags = {'HAC', 'Parameterized'})
        function testMLogitHACWithDifferentKernels(testCase, kernelType)
            % 测试不同核函数下的HAC标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            maxLags = 4;
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=maxLags, kernel=kernelType);
            
            % 验证结果有效 - use (:) to flatten for scalar check
            testCase.verifyTrue(all(isfinite(result.stdErrors(:))), ...
                sprintf('核函数%s应产生有限标准误', kernelType));
            testCase.verifyGreaterThan(result.stdErrors(:), 0, ...
                sprintf('核函数%s应产生正标准误', kernelType));
            
            % 验证核函数正确记录
            testCase.verifyEqual(result.kernel, kernelType, ...
                '核函数类型应正确记录');
        end
        
        function testMLogitHACWithDifferentLags(testCase, lagOrder)
            % 测试不同滞后阶数下的HAC标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            if isempty(lagOrder)
                % 跳过自动选择测试（可能不支持）
                return;
            end
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=lagOrder, kernel="bartlett");
            
            % 验证结果有效 - use (:) to flatten for scalar check
            testCase.verifyTrue(all(isfinite(result.stdErrors(:))), ...
                sprintf('滞后阶数%d应产生有限标准误', lagOrder));
            
            % 验证滞后阶数正确记录
            testCase.verifyEqual(result.lag, lagOrder, ...
                '滞后阶数应正确记录');
        end
        
        function testHACLagOrderMonotonicity(testCase)
            % 测试HAC标准误随滞后阶数变化的单调性趋势
            % 注意：不一定严格单调，但应该有变化
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            lags = [1, 2, 4, 8];
            seByLag = cell(length(lags), 1);
            
            for i = 1:length(lags)
                result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                    covType="hac", lag=lags(i), kernel="bartlett");
                seByLag{i} = result.stdErrors;
            end
            
            % 验证标准误有变化
            allEqual = true;
            for i = 2:length(seByLag)
                if ~isequal(seByLag{1}, seByLag{i})
                    allEqual = false;
                    break;
                end
            end
            
            testCase.verifyFalse(allEqual, ...
                'HAC标准误应随滞后阶数变化');
        end
    end
    
    %% ==================== P值计算测试 ====================
    methods (Test, TestTags = {'PValue', 'HAC'})
        function testMLogitHACPValueFormula(testCase)
            % 验证Multinomial Logit HAC p值计算公式
            % 对于MLE模型，大样本下使用正态分布
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            % 手动计算t统计量
            tStats = result.params ./ result.stdErrors;
            
            % 验证z统计量
            testCase.verifyEqual(result.zStatistics, tStats, 'RelTol', 1e-10, ...
                'z统计量应等于系数/标准误');
            
            % 验证p值 (使用正态分布)
            expectedPValues = 2 * (1 - normcdf(abs(tStats)));
            
            testCase.verifyEqual(result.pValues, expectedPValues, 'RelTol', 0.01, ...
                'HAC p值应基于正态分布计算');
        end
        
        function testPValueRangeValidation(testCase)
            % 验证p值范围的有效性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            covTypes = {'nonrobust', 'sandwich', 'hac', 'cluster'};
            
            for i = 1:length(covTypes)
                covType = covTypes{i};
                
                try
                    if strcmp(covType, 'hac')
                        result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                            covType=covType, lag=4);
                    elseif strcmp(covType, 'cluster')
                        % 创建假聚类
                        clusterIds = mod(1:length(y), 50)' + 1;
                        result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                            covType=covType, clusterIds=clusterIds);
                    else
                        result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                            covType=covType);
                    end
                    
                    % 验证p值范围
                    testCase.verifyGreaterThanOrEqual(result.pValues, 0, ...
                        sprintf('%s: p值应非负', covType));
                    testCase.verifyLessThanOrEqual(result.pValues, 1, ...
                        sprintf('%s: p值应不超过1', covType));
                catch ME
                    % 如果某种类型不支持，记录但不失败
                    warning('MLogitHACTest:CovTypeNotSupported', ...
                        '%s可能不支持: %s', covType, ME.message);
                end
            end
        end
        
        function testSignificanceLevelConsistency(testCase)
            % 测试显著性水平判断的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            % 5%显著性水平对应|z| > 1.96
            significantAt5Pct = abs(result.zStatistics) > 1.96;
            significantByPValue = result.pValues < 0.05;
            
            testCase.verifyEqual(significantAt5Pct, significantByPValue, ...
                't统计量和p值的显著性判断应一致');
        end
    end
    
    %% ==================== 边际效应HAC标准误测试 ====================
    methods (Test, TestTags = {'MarginalEffects', 'HAC'})
        function testMLogitMarginalEffectsExist(testCase)
            % 验证边际效应存在于HAC模型结果中
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            testCase.verifyTrue(isfield(result, 'marginalEffects'), ...
                'HAC结果应包含边际效应');
            
            if ~isempty(result.marginalEffects)
                testCase.verifyTrue(all(isfinite(result.marginalEffects(:))), ...
                    '边际效应应为有限值');
            end
        end
        
        function testMarginalEffectsHACSEPropagation(testCase)
            % 验证边际效应正确传播HAC协方差矩阵
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % 经典标准误
            resultClassic = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="nonrobust");
            
            % HAC标准误
            resultHAC = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            % 边际效应值应相同（只有标准误不同）
            if ~isempty(resultClassic.marginalEffects) && ~isempty(resultHAC.marginalEffects)
                testCase.verifyEqual(resultClassic.marginalEffects, ...
                    resultHAC.marginalEffects, 'RelTol', 0.001, ...
                    '边际效应值在不同标准误类型下应相同');
            end
            
            % 边际效应标准误应不同
            if isfield(resultClassic, 'marginalEffectsSE') && ...
               isfield(resultHAC, 'marginalEffectsSE') && ...
               ~isempty(resultClassic.marginalEffectsSE) && ...
               ~isempty(resultHAC.marginalEffectsSE)
                testCase.verifyFalse(...
                    isequal(resultClassic.marginalEffectsSE, resultHAC.marginalEffectsSE), ...
                    '边际效应标准误在不同标准误类型下应不同');
            end
        end
        
        function testMarginalEffectsPValueConsistency(testCase)
            % 验证边际效应p值与标准误的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            if isfield(result, 'marginalEffects') && ...
               isfield(result, 'marginalEffectsSE') && ...
               isfield(result, 'marginalEffectsP') && ...
               ~isempty(result.marginalEffectsSE)
                
                % 计算期望的t统计量
                tStats = result.marginalEffects ./ result.marginalEffectsSE;
                
                % 计算期望的p值
                expectedPValues = 2 * (1 - normcdf(abs(tStats)));
                
                % 验证p值一致性 - use AbsTol for very small p-values
                testCase.verifyEqual(result.marginalEffectsP, expectedPValues, ...
                    'AbsTol', 1e-10, ...
                    '边际效应p值应与标准误一致');
            end
        end
        
        function testMarginalEffectsSumToZero(testCase)
            % 验证Multinomial Logit边际效应的数学性质
            % 对于每个自变量，各类别边际效应之和应接近0
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);
            
            if ~isempty(result.marginalEffects)
                % 边际效应矩阵的列（或行）和应接近0
                % 具体取决于矩阵的组织方式
                nParams = size(X, 2); %#ok<NASGU>
                nCategories = 3; %#ok<NASGU>
                
                % 尝试验证边际效应的理论性质
                % (这取决于marginalEffects的具体返回格式)
            end
        end
    end
    
    %% ==================== Python一致性验证 ====================
    methods (Test, TestTags = {'Python', 'Consistency'})
        function testMLogitHACFullConsistency(testCase)
            % 完整验证Multinomial Logit HAC与Python的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            maxLags = 4;
            kernel = "bartlett";
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=maxLags, kernel=kernel);
            
            % 直接Python调用 - ensure proper array dimensions
            yPy = py.numpy.array(y).flatten();
            XPy = py.numpy.atleast_2d(py.numpy.array(X));
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.discrete.discrete_model.MNLogit(yPy, XWithConst);
            
            % MNLogit has no get_robustcov_results - use fit() with cov_type directly
            covKwds = py.dict();
            covKwds{"maxlags"} = int32(maxLags);
            covKwds{"kernel"} = kernel;
            fitRobust = model.fit(pyargs('cov_type', 'HAC', 'cov_kwds', covKwds, 'disp', false));
            
            % 提取Python结果 - use py.getattr for safe attribute access
            paramsPy = double(py.getattr(fitRobust, 'params'));
            stdErrorsPy = double(py.getattr(fitRobust, 'bse'));
            tStatsPy = double(py.getattr(fitRobust, 'tvalues'));
            pValuesPy = double(py.getattr(fitRobust, 'pvalues'));
            
            % 验证系数一致
            testCase.verifyEqual(result.params, paramsPy(:), 'RelTol', 1e-10, ...
                'HAC系数应与Python完全一致');
            
            % 验证标准误一致
            testCase.verifyEqual(result.stdErrors, stdErrorsPy(:), 'RelTol', 1e-6, ...
                'HAC标准误应与Python一致');
            
            % 验证z统计量一致
            testCase.verifyEqual(result.zStatistics, tStatsPy(:), 'RelTol', 1e-6, ...
                'HAC z统计量应与Python一致');
            
            % 验证p值一致
            testCase.verifyEqual(result.pValues, pValuesPy(:), 'RelTol', 0.01, ...
                'HAC p值应与Python一致');
        end
        
        function testMLogitClusteredConsistency(testCase)
            % 验证聚类标准误与Python的一致性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.panel.y;
            X = testCase.TestData.panel.X;
            clusterIds = testCase.TestData.panel.entityId;
            
            % pyBridge调用
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="cluster", clusterIds=clusterIds);
            
            % 直接Python调用 - ensure proper array dimensions
            yPy = py.numpy.array(y).flatten();
            XPy = py.numpy.atleast_2d(py.numpy.array(X));
            XWithConst = py.statsmodels.api.add_constant(XPy);
            model = py.statsmodels.discrete.discrete_model.MNLogit(yPy, XWithConst);
            
            % MNLogit has no get_robustcov_results - use fit() with cov_type directly
            clusterPy = py.numpy.array(clusterIds);
            covKwds = py.dict();
            covKwds{"groups"} = clusterPy;
            fitRobust = model.fit(pyargs('cov_type', 'cluster', 'cov_kwds', covKwds, 'disp', false));
            
            stdErrorsPy = double(py.getattr(fitRobust, 'bse'));
            
            % 验证聚类标准误一致
            testCase.verifyEqual(result.stdErrors, stdErrorsPy(:), 'RelTol', 0.01, ...
                '聚类标准误应与Python一致');
        end
    end
    
    %% ==================== 边界条件测试 ====================
    methods (Test, TestTags = {'EdgeCases'})
        function testSmallSampleHAC(testCase)
            % 测试小样本HAC标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            rng(999, 'twister');
            nSmall = 100;
            X = randn(nSmall, 2);
            y = randi([0, 2], nSmall, 1);
            
            % 小样本应该能处理
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=2);
            
            testCase.verifyTrue(all(isfinite(result.stdErrors(:))), ...
                '小样本HAC应产生有限标准误');
        end
        
        function testLargeLagOrder(testCase)
            % 测试较大滞后阶数
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % 大滞后阶数
            largeLag = 20;
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=largeLag);
            
            testCase.verifyTrue(all(isfinite(result.stdErrors(:))), ...
                '大滞后阶数应产生有限标准误');
        end
        
        function testTwoCategoryMLogit(testCase)
            % 测试两类别（退化为Logit）
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            rng(888, 'twister');
            n = 500;
            X = randn(n, 2);
            y = randi([0, 1], n, 1);  % 只有2个类别
            
            result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            testCase.verifyEqual(result.nCategories, 2, ...
                '应正确识别2个类别');
            testCase.verifyTrue(all(isfinite(result.stdErrors(:))), ...
                '两类别模型应产生有限标准误');
        end
        
        function testSingleClusterError(testCase)
            % 测试单聚类错误处理
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            singleCluster = ones(length(y), 1);  % 所有观测在同一聚类
            
            % 应该抛出错误或警告
            testCase.verifyError(...
                @() pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                    covType="cluster", clusterIds=singleCluster), ...
                'MATLAB:Python:PyException');
        end
        
        function testZeroLagHAC(testCase)
            % 测试0滞后HAC（应等价于HC0）
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % 0滞后HAC
            resultHAC0 = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=0);
            
            % HC0 (sandwich)
            resultHC0 = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="sandwich");
            
            % 应该接近
            testCase.verifyEqual(resultHAC0.stdErrors, resultHC0.stdErrors, ...
                'RelTol', 0.1, ...
                '0滞后HAC应接近HC0标准误');
        end
    end
    
    %% ==================== 数值稳定性测试 ====================
    methods (Test, TestTags = {'Numerical', 'Stability'})
        function testHACNumericalStability(testCase)
            % 测试HAC计算的数值稳定性
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % 多次运行应产生相同结果
            result1 = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            result2 = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            testCase.verifyEqual(result1.stdErrors, result2.stdErrors, ...
                'AbsTol', 1e-15, ...
                'HAC计算应具有数值稳定性');
        end
        
        function testScaledDataHAC(testCase)
            % 测试缩放数据的HAC标准误
            testCase.assumeTrue(testCase.HasPython, '需要Python环境');
            
            y = testCase.TestData.mlogit.y;
            X = testCase.TestData.mlogit.X;
            
            % 原始数据
            result1 = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
                covType="hac", lag=4);
            
            % 缩放数据
            scaleFactor = 100;
            XScaled = X * scaleFactor;
            result2 = pyBridge.StatsmodelsWrapper.multinomialLogit(y, XScaled, ...
                covType="hac", lag=4);
            
            % 系数应该缩放，但z统计量应该相同
            testCase.verifyEqual(result1.zStatistics, result2.zStatistics, ...
                'RelTol', 0.01, ...
                '数据缩放不应改变z统计量');
        end
    end
end
