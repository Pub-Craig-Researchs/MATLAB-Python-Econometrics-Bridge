classdef EconmlWrapper
    % ECONMLWRAPPER Wrapper for econml library
    %   Provides causal inference and treatment effect estimation
    %
    % Static Methods:
    %   dml - Double Machine Learning
    %   drLearner - Doubly Robust Learner
    %   metaLearners - Meta learning methods (S/T/X/R Learner)
    %   interpret - Causal effect interpretation
    %
    % Example:
    %   result = pyBridge.EconmlWrapper.dml(Y, T, X, W);
    %   ate = result.ate;
    %
    
    methods(Static)
        %% Double Machine Learning
        function result = dml(Y, T, X, W, options)
            % DML Double Machine Learning for causal effect estimation
            %
            % Parameters:
            %   Y - Outcome variable (n x 1)
            %   T - Treatment variable (n x 1)
            %   X - Covariates (n x p, for estimating heterogeneous effects)
            %   W - Control variables (n x q, for controlling confounders)
            %   options.modelY - Model for Y estimation
            %   options.modelT - Model for T estimation
            %   options.discreteTreatment - Whether T is discrete
            
            arguments
                Y double
                T double
                X double = []
                W double = []
                options.modelY char = "linear"
                options.modelT char = "linear"
                options.discreteTreatment logical = false
                options.randomState double = 42
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            % Convert data
            YPy = pyBridge.DataConverter.toPython(Y(:));
            TPy = pyBridge.DataConverter.toPython(T(:));
            
            if isempty(X)
                XPy = py.None;
            else
                XPy = pyBridge.DataConverter.toPython(X);
            end
            
            if isempty(W)
                WPy = py.None;
            else
                WPy = pyBridge.DataConverter.toPython(W);
            end
            
            % Select model
            switch lower(options.modelY)
                case "linear"
                    modelY = py.sklearn.linear_model.LinearRegression();
                case "lasso"
                    modelY = py.sklearn.linear_model.LassoCV();
                case "ridge"
                    modelY = py.sklearn.linear_model.RidgeCV();
                case "forest"
                    modelY = py.sklearn.ensemble.RandomForestRegressor(...
                        n_estimators=int32(100), random_state=int32(options.randomState));
                otherwise
                    error("pyBridge:InvalidModel", "Unknown Y model: %s", options.modelY);
            end
            
            switch lower(options.modelT)
                case "linear"
                    if options.discreteTreatment
                        modelT = py.sklearn.linear_model.LogisticRegression();
                    else
                        modelT = py.sklearn.linear_model.LinearRegression();
                    end
                case "lasso"
                    if options.discreteTreatment
                        modelT = py.sklearn.linear_model.LogisticRegressionCV();
                    else
                        modelT = py.sklearn.linear_model.LassoCV();
                    end
                case "forest"
                    if options.discreteTreatment
                        modelT = py.sklearn.ensemble.RandomForestClassifier(...
                            n_estimators=int32(100), random_state=int32(options.randomState));
                    else
                        modelT = py.sklearn.ensemble.RandomForestRegressor(...
                            n_estimators=int32(100), random_state=int32(options.randomState));
                    end
                otherwise
                    error("pyBridge:InvalidModel", "Unknown T model: %s", options.modelT);
            end
            
            % Create DML model (academic standard: Chernozhukov et al. 2018)
            % Discrete treatment should use LinearDML+discrete_treatment=True, not DML
            est = py.econml.dml.LinearDML(...
                model_y=modelY, ...
                model_t=modelT, ...
                discrete_treatment=options.discreteTreatment, ...
                random_state=int32(options.randomState));
            
            % Fit model
            est.fit(YPy, TPy, X=XPy, W=WPy);
            
            % Extract results
            result = struct();
            result.modelType = "DML";
            result.modelY = options.modelY;
            result.modelT = options.modelT;
            
            % Average Treatment Effect (ATE)
            ateResult = est.ate(XPy);
            result.ate = double(ateResult);
            
            % ATE confidence interval
            result.ateConfInt = struct('lower', NaN, 'upper', NaN);
            result.atePValue = NaN;
            try
                ateInference = est.ate_inference(XPy);
                confIntResult = ateInference.conf_int();
                result.ateConfInt.lower = double(confIntResult{1});
                result.ateConfInt.upper = double(confIntResult{2});
                result.atePValue = double(ateInference.pvalue());
            catch
                % Some models may not support inference, keep default NaN values
            end
            
            % Conditional Average Treatment Effect (CATE)
            if ~isempty(X)
                cateResult = est.effect(XPy);
                result.cate = double(cateResult);
                
                % Save model for subsequent predictions
                result.fittedModel = est;
            end
        end
        
        %% Doubly Robust Learner
        function result = drLearner(Y, T, X, W, options)
            % DRLEARNER Doubly Robust Learner
            
            arguments
                Y double
                T double
                X double = []
                W double = []
                options.modelY char = "linear"
                options.modelT char = "linear"
                options.randomState double = 42
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            % Convert data
            YPy = pyBridge.DataConverter.toPython(Y(:));
            TPy = pyBridge.DataConverter.toPython(T(:));
            
            if isempty(X)
                XPy = py.None;
            else
                XPy = pyBridge.DataConverter.toPython(X);
            end
            
            if isempty(W)
                WPy = py.None;
            else
                WPy = pyBridge.DataConverter.toPython(W);
            end
            
            % Create DR Learner
            est = py.econml.dr.DRLearner(random_state=int32(options.randomState));
            
            % Fit
            est.fit(YPy, TPy, X=XPy, W=WPy);
            
            % Extract results
            result = struct();
            result.modelType = "DRLearner";
            
            % ATE
            result.ate = double(est.ate(XPy));
            
            % CATE
            if ~isempty(X)
                result.cate = double(est.effect(XPy));
            end
            
            result.fittedModel = est;
        end
        
        %% Meta Learners
        function result = sLearner(Y, T, X, options)
            % SLEARNER S-Learner (Single Learner)
            
            arguments
                Y double
                T double
                X double
                options.baseModel char = "forest"
                options.randomState double = 42
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            YPy = pyBridge.DataConverter.toPython(Y(:));
            TPy = pyBridge.DataConverter.toPython(T(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            % Select base model
            switch lower(options.baseModel)
                case "forest"
                    baseModel = py.sklearn.ensemble.RandomForestRegressor(...
                        n_estimators=int32(100), random_state=int32(options.randomState));
                case "linear"
                    baseModel = py.sklearn.linear_model.LinearRegression();
                otherwise
                    error("pyBridge:InvalidModel", "Unknown base model: %s", options.baseModel);
            end
            
            % Create S-Learner
            est = py.econml.metalearners.SLearner(overall_model=baseModel);
            
            est.fit(YPy, TPy, X=XPy);
            
            result = struct();
            result.modelType = "SLearner";
            result.ate = double(est.ate(XPy));
            result.cate = double(est.effect(XPy));
            result.fittedModel = est;
        end
        
        function result = tLearner(Y, T, X, options)
            % TLEARNER T-Learner (Two Learners)
            
            arguments
                Y double
                T double
                X double
                options.baseModel char = "forest"
                options.randomState double = 42
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            YPy = pyBridge.DataConverter.toPython(Y(:));
            TPy = pyBridge.DataConverter.toPython(T(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            switch lower(options.baseModel)
                case "forest"
                    baseModel = py.sklearn.ensemble.RandomForestRegressor(...
                        n_estimators=int32(100), random_state=int32(options.randomState));
                case "linear"
                    baseModel = py.sklearn.linear_model.LinearRegression();
                otherwise
                    error("pyBridge:InvalidModel", "Unknown base model: %s", options.baseModel);
            end
            
            est = py.econml.metalearners.TLearner(models=baseModel);
            est.fit(YPy, TPy, X=XPy);
            
            result = struct();
            result.modelType = "TLearner";
            result.ate = double(est.ate(XPy));
            result.cate = double(est.effect(XPy));
            result.fittedModel = est;
        end
        
        function result = xLearner(Y, T, X, options)
            % XLEARNER X-Learner
            
            arguments
                Y double
                T double
                X double
                options.baseModel char = "forest"
                options.randomState double = 42
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            YPy = pyBridge.DataConverter.toPython(Y(:));
            TPy = pyBridge.DataConverter.toPython(T(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            switch lower(options.baseModel)
                case "forest"
                    baseModel = py.sklearn.ensemble.RandomForestRegressor(...
                        n_estimators=int32(100), random_state=int32(options.randomState));
                case "linear"
                    baseModel = py.sklearn.linear_model.LinearRegression();
                otherwise
                    error("pyBridge:InvalidModel", "Unknown base model: %s", options.baseModel);
            end
            
            est = py.econml.metalearners.XLearner(models=baseModel);
            est.fit(YPy, TPy, X=XPy);
            
            result = struct();
            result.modelType = "XLearner";
            result.ate = double(est.ate(XPy));
            result.cate = double(est.effect(XPy));
            result.fittedModel = est;
        end
        
        %% Causal Forest
        function result = causalForest(Y, T, X, W, options)
            % CAUSALFOREST Causal Forest
            
            arguments
                Y double
                T double
                X double
                W double = []
                options.nEstimators double = 100
                options.randomState double = 42
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            YPy = pyBridge.DataConverter.toPython(Y(:));
            TPy = pyBridge.DataConverter.toPython(T(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if isempty(W)
                WPy = py.None;
            else
                WPy = pyBridge.DataConverter.toPython(W);
            end
            
            % Create causal forest
            est = py.econml.dml.CausalForestDML(...
                n_estimators=int32(options.nEstimators), ...
                random_state=int32(options.randomState));
            
            est.fit(YPy, TPy, X=XPy, W=WPy);
            
            result = struct();
            result.modelType = "CausalForest";
            result.nEstimators = options.nEstimators;
            
            % ATE
            result.ate = double(est.ate(XPy));
            
            % CATE
            result.cate = double(est.effect(XPy));
            
            % Feature importance
            try
                result.featureImportance = double(est.feature_importances_);
            catch
                % Ignore
            end
            
            result.fittedModel = est;
        end
        
        %% Interpretation and Visualization
        function result = interpret(result, X, featureNames)
            % INTERPRET Interpret causal effects
            
            arguments
                result struct % Result from dml and other functions
                X double
                featureNames cell = {}
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            if ~isfield(result, 'fittedModel')
                error("pyBridge:NoModel", "Fitted model not found in result");
            end
            
            est = result.fittedModel;
            XPy = pyBridge.DataConverter.toPython(X);
            
            interpretResult = struct();
            
            % Feature importance
            if py_builtin.hasattr(est, 'feature_importances_')
                importance = double(est.feature_importances_);
                
                if isempty(featureNames)
                    featureNames = strcat("X", string(1:length(importance)));
                end
                
                interpretResult.featureImportance = table(featureNames, importance, ...
                    VariableNames=["Feature", "Importance"]);
            end
            
            % SHAP values (with version compatibility)
            try
                if py_builtin.hasattr(est, 'shap_values')
                    shapValues = est.shap_values(XPy);
                    interpretResult.shapValues = double(shapValues);
                elseif py_builtin.hasattr(est, 'interpret')
                    % econml >= 0.14 new API
                    interpretResult.interpretationText = char(est.interpret(XPy));
                end
            catch ME
                interpretResult.shapWarning = "SHAP values unavailable (may need econml>=0.14)";
            end
            
            result.interpretation = interpretResult;
        end
        
        %% Prediction
        function effects = predictEffect(fittedModel, Xnew)
            % PREDICTEFFECT Predict treatment effects for new samples
            
            arguments
                fittedModel % Fitted Python model object
                Xnew double
            end
            
            XnewPy = pyBridge.DataConverter.toPython(Xnew);
            
            effects = double(fittedModel.effect(XnewPy));
        end
        
        function effects = predictInterval(fittedModel, Xnew, alpha)
            % PREDICTINTERVAL Predict treatment effect confidence intervals
            
            arguments
                fittedModel
                Xnew double
                alpha double = 0.05
            end
            
            XnewPy = pyBridge.DataConverter.toPython(Xnew);
            
            inference = fittedModel.effect_inference(XnewPy);
            
            effects = struct();
            effects.point = double(inference.point_estimate);
            confIntResult = inference.conf_int(pyargs('alpha', alpha));
            effects.lower = double(confIntResult{1});
            effects.upper = double(confIntResult{2});
        end
        
        %% Sensitivity Analysis
        function result = sensitivityAnalysis(fittedModel, Y, T, X)
            % SENSITIVITYANALYSIS Sensitivity analysis
            
            arguments
                fittedModel
                Y double
                T double
                X double
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("econml");
            
            YPy = pyBridge.DataConverter.toPython(Y(:));
            TPy = pyBridge.DataConverter.toPython(T(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            % Hidden confounding analysis
            try
                sensitivity = py.econml.sensitivity.SensitivityAnalysis(fittedModel);
                sensitivity.fit(YPy, TPy, X=XPy);
                            
                result = struct();
                result.r2Y = double(sensitivity.r2_y);
                result.r2T = double(sensitivity.r2_t);
                result.bias = double(sensitivity.bias);
            catch
                result = struct();
                result.message = "Sensitivity analysis unavailable";
            end
        end
    end
end
