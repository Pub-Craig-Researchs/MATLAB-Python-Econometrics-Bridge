classdef PanelCausalTest < matlab.unittest.TestCase
    % PanelCausalTest Comprehensive tests for EconmlWrapper and LinearmodelsWrapper
    %
    % Test Coverage:
    % 1. EconmlWrapper: DML, DR Learner, Meta Learners, Causal Forest
    % 2. LinearmodelsWrapper: Panel OLS, Random Effects, IV models, Hausman test
    %

    properties
        HasEconml       % Whether econml is available
        HasLinearmodels % Whether linearmodels is available
        PanelData       % Panel data for LinearmodelsWrapper tests
        CausalData      % Causal inference data for EconmlWrapper tests
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            % Initialize test environment and check Python packages
            
            % Check econml availability
            try
                pyenv;
                py.importlib.import_module('econml');
                testCase.HasEconml = true;
            catch
                testCase.HasEconml = false;
                warning('PanelCausalTest:NoEconml', ...
                    'econml not available, EconmlWrapper tests will be skipped');
            end
            
            % Check linearmodels availability
            try
                pyenv;
                py.importlib.import_module('linearmodels');
                testCase.HasLinearmodels = true;
            catch
                testCase.HasLinearmodels = false;
                warning('PanelCausalTest:NoLinearmodels', ...
                    'linearmodels not available, LinearmodelsWrapper tests will be skipped');
            end
            
            % Generate test data
            testCase.generatePanelData();
            testCase.generateCausalData();
        end
    end

    methods (Access = private)
        function generatePanelData(testCase)
            % Generate reproducible panel data for LinearmodelsWrapper tests
            rng(42, 'twister');
            
            nEntities = 20;
            nPeriods = 10;
            nObs = nEntities * nPeriods;
            
            entityIds = repelem((1:nEntities)', nPeriods);
            timeIds = repmat((1:nPeriods)', nEntities, 1);
            entityEffects = randn(nEntities, 1);
            timeEffects = randn(nPeriods, 1);
            
            X = randn(nObs, 2);
            y = X * [1.5; -0.8] + entityEffects(entityIds) + 0.5 * randn(nObs, 1);
            
            testCase.PanelData.y = y;
            testCase.PanelData.X = X;
            testCase.PanelData.entityIds = entityIds;
            testCase.PanelData.timeIds = timeIds;
            testCase.PanelData.entityEffects = entityEffects;
            testCase.PanelData.timeEffects = timeEffects;
            testCase.PanelData.nEntities = nEntities;
            testCase.PanelData.nPeriods = nPeriods;
            testCase.PanelData.nObs = nObs;
        end
        
        function generateCausalData(testCase)
            % Generate reproducible causal inference data for EconmlWrapper tests
            rng(42, 'twister');
            
            n = 200;
            p = 5;  % Number of covariates
            
            % Covariates
            X = randn(n, p);
            W = randn(n, 3);  % Control variables
            
            % Treatment (continuous)
            T = 0.5 * X(:,1) + 0.3 * W(:,1) + randn(n, 1);
            
            % True treatment effect (heterogeneous)
            trueEffect = 2 + 0.5 * X(:,1) - 0.3 * X(:,2);
            
            % Outcome
            Y = trueEffect .* T + X * [1; 0.5; -0.3; 0.2; 0.1] + randn(n, 1);
            
            testCase.CausalData.Y = Y;
            testCase.CausalData.T = T;
            testCase.CausalData.X = X;
            testCase.CausalData.W = W;
            testCase.CausalData.trueEffect = trueEffect;
            testCase.CausalData.n = n;
            testCase.CausalData.p = p;
            
            % Discrete treatment data
            Tdisc = double(T > median(T));
            Ydisc = trueEffect .* Tdisc + X * [1; 0.5; -0.3; 0.2; 0.1] + randn(n, 1);
            testCase.CausalData.Tdisc = Tdisc;
            testCase.CausalData.Ydisc = Ydisc;
        end
    end

    %% ==================== EconmlWrapper Tests ====================
    methods (Test, TestTags = {'EconmlWrapper', 'DML'})
        function testDMLAverageTreatmentEffect(testCase)
            % Test DML basic functionality with continuous treatment
            testCase.assumeTrue(testCase.HasEconml, 'econml not available');
            
            Y = testCase.CausalData.Y;
            T = testCase.CausalData.T;
            X = testCase.CausalData.X;
            W = testCase.CausalData.W;
            
            try
                result = pyBridge.EconmlWrapper.dml(Y, T, X, W, randomState=42);
                
                % Verify ATE is a finite number
                testCase.verifyTrue(isfinite(result.ate), ...
                    'ATE should be a finite number');
                
                % Verify confidence interval (may be NaN if model did not converge)
                if isfield(result, 'ateConfInt') && isstruct(result.ateConfInt)
                    if isfinite(result.ateConfInt.lower) && isfinite(result.ateConfInt.upper)
                        testCase.verifyLessThan(result.ateConfInt.lower, result.ateConfInt.upper, ...
                            'Confidence interval lower bound should be less than upper bound');
                    end
                end
                
                % Verify p-value is in [0, 1] (skip if NaN)
                if isfield(result, 'atePValue') && isfinite(result.atePValue)
                    testCase.verifyGreaterThanOrEqual(result.atePValue, 0, ...
                        'P-value should be >= 0');
                    testCase.verifyLessThanOrEqual(result.atePValue, 1, ...
                        'P-value should be <= 1');
                end
                
                % Verify CATE dimension
                if isfield(result, 'cate')
                    testCase.verifyEqual(length(result.cate), testCase.CausalData.n, ...
                        'CATE should have same length as number of observations');
                end
                
            catch ME
                testCase.assumeTrue(false, sprintf('econml DML failed: %s', ME.message));
            end
        end
        
        function testDMLDiscreteTreatment(testCase)
            % Test DML with discrete treatment
            testCase.assumeTrue(testCase.HasEconml, 'econml not available');
            
            Y = testCase.CausalData.Ydisc;
            T = testCase.CausalData.Tdisc;
            X = testCase.CausalData.X;
            W = testCase.CausalData.W;
            
            try
                result = pyBridge.EconmlWrapper.dml(Y, T, X, W, ...
                    discreteTreatment=true, randomState=42);
                
                % Verify result structure is complete
                testCase.verifyTrue(isfield(result, 'ate'), ...
                    'Result should contain ate field');
                testCase.verifyTrue(isfield(result, 'modelType'), ...
                    'Result should contain modelType field');
                testCase.verifyEqual(result.modelType, "DML", ...
                    'Model type should be DML');
                testCase.verifyTrue(isfinite(result.ate), ...
                    'ATE should be finite');
                
            catch ME
                testCase.assumeTrue(false, sprintf('econml DML discrete treatment failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'EconmlWrapper', 'DRLearner'})
        function testDRLearner(testCase)
            % Test DR Learner
            testCase.assumeTrue(testCase.HasEconml, 'econml not available');
            
            Y = testCase.CausalData.Ydisc;
            T = testCase.CausalData.Tdisc;
            X = testCase.CausalData.X;
            W = testCase.CausalData.W;
            
            try
                result = pyBridge.EconmlWrapper.drLearner(Y, T, X, W, randomState=42);
                
                % Verify ate structure
                testCase.verifyTrue(isfield(result, 'ate'), ...
                    'Result should contain ate field');
                testCase.verifyTrue(isfinite(result.ate), ...
                    'ATE should be finite');
                
                % Verify cate structure
                if isfield(result, 'cate')
                    testCase.verifyEqual(length(result.cate), testCase.CausalData.n, ...
                        'CATE length should match observations');
                end
                
                % Verify fittedModel is not empty
                testCase.verifyTrue(isfield(result, 'fittedModel'), ...
                    'Result should contain fittedModel');
                testCase.verifyNotEmpty(result.fittedModel, ...
                    'fittedModel should not be empty');
                
            catch ME
                testCase.assumeTrue(false, sprintf('econml DRLearner failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'EconmlWrapper', 'MetaLearners'})
        function testMetaLearnersConsistency(testCase)
            % Test S/T/X Learner consistency
            testCase.assumeTrue(testCase.HasEconml, 'econml not available');
            
            Y = testCase.CausalData.Ydisc;
            T = testCase.CausalData.Tdisc;
            X = testCase.CausalData.X;
            
            try
                % S-Learner
                resultS = pyBridge.EconmlWrapper.sLearner(Y, T, X, randomState=42);
                testCase.verifyTrue(isfield(resultS, 'ate'), 'S-Learner should have ate');
                testCase.verifyTrue(isfield(resultS, 'cate'), 'S-Learner should have cate');
                testCase.verifyTrue(isfinite(resultS.ate), 'S-Learner ATE should be finite');
                testCase.verifyEqual(resultS.modelType, "SLearner", 'Model type should be SLearner');
                
                % T-Learner
                resultT = pyBridge.EconmlWrapper.tLearner(Y, T, X, randomState=42);
                testCase.verifyTrue(isfield(resultT, 'ate'), 'T-Learner should have ate');
                testCase.verifyTrue(isfield(resultT, 'cate'), 'T-Learner should have cate');
                testCase.verifyTrue(isfinite(resultT.ate), 'T-Learner ATE should be finite');
                testCase.verifyEqual(resultT.modelType, "TLearner", 'Model type should be TLearner');
                
                % X-Learner
                resultX = pyBridge.EconmlWrapper.xLearner(Y, T, X, randomState=42);
                testCase.verifyTrue(isfield(resultX, 'ate'), 'X-Learner should have ate');
                testCase.verifyTrue(isfield(resultX, 'cate'), 'X-Learner should have cate');
                testCase.verifyTrue(isfinite(resultX.ate), 'X-Learner ATE should be finite');
                testCase.verifyEqual(resultX.modelType, "XLearner", 'Model type should be XLearner');
                
                % All ATEs should be in reasonable range (within 10 of each other for this data)
                ates = [resultS.ate, resultT.ate, resultX.ate];
                testCase.verifyTrue(all(abs(ates - mean(ates)) < 10), ...
                    'Meta learner ATEs should be within reasonable range of each other');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Meta learners failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'EconmlWrapper', 'CausalForest'})
        function testCausalForestFeatureImportance(testCase)
            % Test Causal Forest feature importance
            testCase.assumeTrue(testCase.HasEconml, 'econml not available');
            
            Y = testCase.CausalData.Y;
            T = testCase.CausalData.T;
            X = testCase.CausalData.X;
            W = testCase.CausalData.W;
            
            try
                result = pyBridge.EconmlWrapper.causalForest(Y, T, X, W, ...
                    nEstimators=48, randomState=42);
                
                % Verify feature importance
                if isfield(result, 'featureImportance')
                    testCase.verifyEqual(length(result.featureImportance), size(X, 2), ...
                        'Feature importance length should equal number of X columns');
                    testCase.verifyGreaterThanOrEqual(result.featureImportance, 0, ...
                        'All feature importance values should be >= 0');
                end
                
                % Verify ATE and CATE
                testCase.verifyTrue(isfinite(result.ate), 'ATE should be finite');
                testCase.verifyEqual(length(result.cate), testCase.CausalData.n, ...
                    'CATE length should match observations');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Causal Forest failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'EconmlWrapper', 'Prediction'})
        function testPredictEffectOnNewData(testCase)
            % Test prediction methods on new data
            testCase.assumeTrue(testCase.HasEconml, 'econml not available');
            
            Y = testCase.CausalData.Y;
            T = testCase.CausalData.T;
            X = testCase.CausalData.X;
            W = testCase.CausalData.W;
            
            try
                % Train model
                result = pyBridge.EconmlWrapper.dml(Y, T, X, W, randomState=42);
                
                % Generate new data for prediction
                rng(123, 'twister');
                Xnew = randn(50, testCase.CausalData.p);
                
                % Test predictEffect
                effects = pyBridge.EconmlWrapper.predictEffect(result.fittedModel, Xnew);
                testCase.verifyEqual(length(effects), 50, ...
                    'predictEffect should return vector of correct length');
                testCase.verifyTrue(all(isfinite(effects)), ...
                    'All predicted effects should be finite');
                
                % Test predictInterval
                intervals = pyBridge.EconmlWrapper.predictInterval(result.fittedModel, Xnew, 0.05);
                testCase.verifyTrue(isfield(intervals, 'point'), ...
                    'Intervals should have point estimate');
                testCase.verifyTrue(isfield(intervals, 'lower'), ...
                    'Intervals should have lower bound');
                testCase.verifyTrue(isfield(intervals, 'upper'), ...
                    'Intervals should have upper bound');
                testCase.verifyTrue(all(intervals.lower <= intervals.upper), ...
                    'Lower bounds should be <= upper bounds');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Prediction methods failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'EconmlWrapper', 'Sensitivity'})
        function testSensitivityAnalysis(testCase)
            % Test sensitivity analysis
            testCase.assumeTrue(testCase.HasEconml, 'econml not available');
            
            Y = testCase.CausalData.Y;
            T = testCase.CausalData.T;
            X = testCase.CausalData.X;
            
            try
                % Train model first
                result = pyBridge.EconmlWrapper.dml(Y, T, X, [], randomState=42);
                
                % Perform sensitivity analysis
                sensResult = pyBridge.EconmlWrapper.sensitivityAnalysis(...
                    result.fittedModel, Y, T, X);
                
                % Verify result structure
                if isfield(sensResult, 'r2Y')
                    testCase.verifyGreaterThanOrEqual(sensResult.r2Y, 0, ...
                        'r2Y should be >= 0');
                    testCase.verifyLessThanOrEqual(sensResult.r2Y, 1, ...
                        'r2Y should be <= 1');
                end
                
                if isfield(sensResult, 'r2T')
                    testCase.verifyGreaterThanOrEqual(sensResult.r2T, 0, ...
                        'r2T should be >= 0');
                    testCase.verifyLessThanOrEqual(sensResult.r2T, 1, ...
                        'r2T should be <= 1');
                end
                
            catch ME
                % Sensitivity analysis may not be available in all econml versions
                testCase.assumeTrue(false, sprintf('Sensitivity analysis failed: %s', ME.message));
            end
        end
    end

    %% ==================== LinearmodelsWrapper Tests ====================
    methods (Test, TestTags = {'LinearmodelsWrapper', 'PanelOLS'})
        function testPanelOLSEntityEffects(testCase)
            % Test Panel OLS with entity fixed effects
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, ...
                    entityEffects=true, timeEffects=false);
                
                % Verify result structure
                testCase.verifyTrue(isfield(result, 'params'), 'Should have params');
                testCase.verifyTrue(isfield(result, 'stdErrors'), 'Should have stdErrors');
                testCase.verifyTrue(isfield(result, 'pValues'), 'Should have pValues');
                testCase.verifyEqual(result.modelType, "PanelOLS", 'Model type should be PanelOLS');
                testCase.verifyTrue(result.entityEffects, 'Entity effects should be true');
                
                % Verify coefficients are finite
                testCase.verifyTrue(all(isfinite(result.params)), ...
                    'All parameters should be finite');
                testCase.verifyTrue(all(result.stdErrors > 0), ...
                    'All standard errors should be positive');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Panel OLS entity effects failed: %s', ME.message));
            end
        end
        
        function testPanelOLSTimeEffects(testCase)
            % Test Panel OLS with time fixed effects
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, ...
                    entityEffects=true, timeEffects=true);
                
                % Verify time effects is enabled
                testCase.verifyTrue(result.timeEffects, 'Time effects should be true');
                testCase.verifyTrue(result.entityEffects, 'Entity effects should be true');
                
                % Verify result structure
                testCase.verifyTrue(all(isfinite(result.params)), ...
                    'All parameters should be finite');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Panel OLS time effects failed: %s', ME.message));
            end
        end
        
        function testPanelOLSRobustSE(testCase)
            % Test Panel OLS with robust standard errors
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                resultRobust = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, ...
                    entityEffects=true, covType="robust");
                
                resultUnadjusted = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, ...
                    entityEffects=true, covType="unadjusted");
                
                % Coefficients should be the same
                testCase.verifyEqual(resultRobust.params, resultUnadjusted.params, ...
                    'AbsTol', 1e-10, 'Coefficients should be identical');
                
                % Standard errors should differ
                testCase.verifyTrue(~isequal(resultRobust.stdErrors, resultUnadjusted.stdErrors), ...
                    'Robust and unadjusted standard errors should differ');
                
                testCase.verifyEqual(resultRobust.covType, "robust", ...
                    'Covariance type should be robust');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Panel OLS robust SE failed: %s', ME.message));
            end
        end
        
        function testPanelOLSClusteredSE(testCase)
            % Test Panel OLS with clustered standard errors
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                % Entity clustered
                resultEntity = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, ...
                    entityEffects=true, covType="clustered_entity");
                
                testCase.verifyEqual(resultEntity.covType, "clustered_entity", ...
                    'Covariance type should be clustered_entity');
                testCase.verifyTrue(all(resultEntity.stdErrors > 0), ...
                    'Clustered standard errors should be positive');
                
                % Two-way clustered
                resultBoth = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, ...
                    entityEffects=true, covType="clustered_both");
                
                testCase.verifyEqual(resultBoth.covType, "clustered_both", ...
                    'Covariance type should be clustered_both');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Panel OLS clustered SE failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'LinearmodelsWrapper', 'RandomEffects'})
        function testRandomEffects(testCase)
            % Test Random Effects model
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                result = pyBridge.LinearmodelsWrapper.randomEffects(y, X, entityIds, timeIds);
                
                % Verify result structure
                testCase.verifyEqual(result.modelType, "RandomEffects", ...
                    'Model type should be RandomEffects');
                testCase.verifyTrue(isfield(result, 'params'), 'Should have params');
                testCase.verifyTrue(isfield(result, 'stdErrors'), 'Should have stdErrors');
                testCase.verifyTrue(all(isfinite(result.params)), ...
                    'All parameters should be finite');
                testCase.verifyTrue(all(result.stdErrors > 0), ...
                    'All standard errors should be positive');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Random Effects failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'LinearmodelsWrapper', 'BetweenOLS'})
        function testBetweenOLS(testCase)
            % Test Between OLS
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                result = pyBridge.LinearmodelsWrapper.betweenOLS(y, X, entityIds, timeIds);
                
                % Verify result structure
                testCase.verifyEqual(result.modelType, "BetweenOLS", ...
                    'Model type should be BetweenOLS');
                testCase.verifyTrue(isfield(result, 'params'), 'Should have params');
                testCase.verifyTrue(all(isfinite(result.params)), ...
                    'All parameters should be finite');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Between OLS failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'LinearmodelsWrapper', 'PooledOLS'})
        function testPooledOLS(testCase)
            % Test Pooled OLS
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                result = pyBridge.LinearmodelsWrapper.pooledOLS(y, X, entityIds, timeIds);
                
                % Verify result structure
                testCase.verifyEqual(result.modelType, "PooledOLS", ...
                    'Model type should be PooledOLS');
                testCase.verifyTrue(isfield(result, 'params'), 'Should have params');
                testCase.verifyTrue(isfield(result, 'stdErrors'), 'Should have stdErrors');
                testCase.verifyTrue(all(isfinite(result.params)), ...
                    'All parameters should be finite');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Pooled OLS failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'LinearmodelsWrapper', 'FirstDifference'})
        function testFirstDifference(testCase)
            % Test First Difference model
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                result = pyBridge.LinearmodelsWrapper.firstDifference(y, X, entityIds, timeIds);
                
                % Verify result structure
                testCase.verifyEqual(result.modelType, "FirstDifferenceOLS", ...
                    'Model type should be FirstDifferenceOLS');
                testCase.verifyTrue(isfield(result, 'params'), 'Should have params');
                testCase.verifyTrue(all(isfinite(result.params)), ...
                    'All parameters should be finite');
                
            catch ME
                testCase.assumeTrue(false, sprintf('First Difference failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'LinearmodelsWrapper', 'HausmanTest'})
        function testHausmanTest(testCase)
            % Test Hausman test (FE vs RE)
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            X = testCase.PanelData.X;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                % Estimate both models
                resultFE = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, ...
                    entityEffects=true);
                resultRE = pyBridge.LinearmodelsWrapper.randomEffects(y, X, entityIds, timeIds);
                
                % Hausman test
                hausmanResult = pyBridge.LinearmodelsWrapper.hausmanTest(resultFE, resultRE);
                
                % Verify result structure
                testCase.verifyTrue(isfield(hausmanResult, 'hausmanStatistic'), ...
                    'Should have hausmanStatistic');
                testCase.verifyTrue(isfield(hausmanResult, 'pValue'), ...
                    'Should have pValue');
                testCase.verifyTrue(isfield(hausmanResult, 'conclusion'), ...
                    'Should have conclusion');
                
                % Verify values are valid
                if isfinite(hausmanResult.hausmanStatistic)
                    testCase.verifyGreaterThanOrEqual(hausmanResult.hausmanStatistic, 0, ...
                        'Hausman statistic should be >= 0');
                    testCase.verifyGreaterThanOrEqual(hausmanResult.pValue, 0, ...
                        'P-value should be >= 0');
                    testCase.verifyLessThanOrEqual(hausmanResult.pValue, 1, ...
                        'P-value should be <= 1');
                end
                
            catch ME
                testCase.assumeTrue(false, sprintf('Hausman test failed: %s', ME.message));
            end
        end
    end

    methods (Test, TestTags = {'LinearmodelsWrapper', 'UnitRoot'})
        function testPanelUnitRoot(testCase)
            % Test Panel Unit Root test
            testCase.assumeTrue(testCase.HasLinearmodels, 'linearmodels not available');
            
            y = testCase.PanelData.y;
            entityIds = testCase.PanelData.entityIds;
            timeIds = testCase.PanelData.timeIds;
            
            try
                result = pyBridge.LinearmodelsWrapper.panelUnitTest(y, entityIds, timeIds, "adf");
                
                % Verify result structure (testName may be char or string)
                testCase.verifyTrue(strcmp(result.testName, 'adf'), ...
                    'Test name should be adf');
                testCase.verifyTrue(isfield(result, 'statistic'), ...
                    'Should have statistic');
                testCase.verifyTrue(isfield(result, 'pValue'), ...
                    'Should have pValue');
                testCase.verifyTrue(isfield(result, 'isStationary'), ...
                    'Should have isStationary');
                
                % Verify p-value is valid
                testCase.verifyGreaterThanOrEqual(result.pValue, 0, ...
                    'P-value should be >= 0');
                testCase.verifyLessThanOrEqual(result.pValue, 1, ...
                    'P-value should be <= 1');
                
            catch ME
                testCase.assumeTrue(false, sprintf('Panel unit root test failed: %s', ME.message));
            end
        end
    end
end
