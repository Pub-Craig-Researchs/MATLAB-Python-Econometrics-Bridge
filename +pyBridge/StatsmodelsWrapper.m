classdef StatsmodelsWrapper
    % STATSMODELSWRAPPER Wrapper for statsmodels library
    %   Provides regression analysis, time series, hypothesis testing, etc.
    %
    % Static Methods:
    %   ols - Ordinary Least Squares regression
    %   glm - Generalized Linear Model
    %   arima - ARIMA time series model
    %   var - VAR Vector Autoregression
    %
    % Example:
    %   result = pyBridge.StatsmodelsWrapper.ols(y, X);
    %   pyBridge.ResultParser.printResult(result);
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    methods(Static)
        %% Regression Analysis
        function result = ols(y, X, options)
            % OLS Ordinary Least Squares regression (supports multiple standard error types)
            %
            % Parameters:
            %   y - Dependent variable
            %   X - Independent variable matrix
            %   options.addConstant - Whether to add constant term
            %   options.covType - Standard error type:
            %       'nonrobust' - Classical standard errors
            %       'HC0', 'HC1', 'HC2', 'HC3' - Heteroskedasticity-robust standard errors
            %       'hac' - HAC (Newey-West) standard errors (requires options.maxLags)
            %       'cluster' - Clustered standard errors (requires options.clusterIds)
            %       'multiway' - Multiway clustering (requires options.clusterGroups)
            %   options.maxLags - Maximum lags for HAC
            %   options.kernel - HAC kernel ('bartlett', 'parzen', 'qs')
            %   options.clusterIds - Cluster identifiers (single dimension)
            %   options.clusterGroups - Multiway cluster identifiers (cell array)
            %
            % Example:
            %   % Heteroskedasticity-robust standard errors
            %   result = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC1");
            %
            %   % HAC standard errors
            %   result = pyBridge.StatsmodelsWrapper.ols(y, X, covType="hac", maxLags=4);
            %
            %   % Clustered standard errors
            %   result = pyBridge.StatsmodelsWrapper.ols(y, X, covType="cluster", clusterIds=firmIds);
            %
            %   % Multiway clustering (firm x year)
            %   result = pyBridge.StatsmodelsWrapper.ols(y, X, covType="multiway", ...
            %       clusterGroups={firmIds, yearIds});
            
            arguments
                y double
                X double
                options.addConstant logical = true
                options.covType char = "nonrobust"
                options.maxLags double = []
                options.kernel char = "bartlett"
                options.clusterIds double = []
                options.clusterGroups cell = {}
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Convert data
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            % Add constant term
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            % Fit model
            model = py.statsmodels.api.OLS(yPy, XPy);
            
            % Select fitting method based on standard error type
            covTypeLower = lower(options.covType);
            
            if strcmp(covTypeLower, "hac")
                % HAC standard errors: use get_robustcov_results for correct API
                fitResult = model.fit();
                
                % Map kernel name to statsmodels-accepted name
                mappedKernel = pyBridge.internal.CovarianceTypes.mapKernelName(options.kernel);
                
                % Get robust covariance results - use py.getattr for method call
                % Pass maxlags and kernel directly as keyword arguments
                getRobustCovFunc = py.getattr(fitResult, 'get_robustcov_results');
                if ~isempty(options.maxLags)
                    fitRobust = getRobustCovFunc(pyargs('cov_type', 'HAC', 'maxlags', int64(options.maxLags), 'kernel', mappedKernel));
                else
                    fitRobust = getRobustCovFunc(pyargs('cov_type', 'HAC', 'kernel', mappedKernel));
                end
                
                % Parse result from robust fit
                result = pyBridge.ResultParser.parseStatsmodels(fitRobust);
                result.covType = "HAC";
                result.kernel = options.kernel;
                result.maxLags = options.maxLags;
                
            elseif strcmp(covTypeLower, "cluster")
                % Single-dimensional clustered standard errors
                fitResult = model.fit();
                
                if isempty(options.clusterIds)
                    error("pyBridge:MissingClusterIds", ...
                        "Clustered standard errors require clusterIds parameter");
                end
                
                % Prepare cluster groups
                clusterPy = pyBridge.DataConverter.toPython(options.clusterIds(:));
                clusterPy = py.numpy.asarray(clusterPy).flatten();
                
                % Get robust covariance results - pass groups directly as keyword arg
                getRobustCovFunc = py.getattr(fitResult, 'get_robustcov_results');
                fitRobust = getRobustCovFunc(pyargs('cov_type', 'cluster', 'groups', clusterPy));
                
                % Parse result from robust fit
                result = pyBridge.ResultParser.parseStatsmodels(fitRobust);
                result.covType = "Clustered";
                result.nClusters = length(unique(options.clusterIds));
                
            elseif strcmp(covTypeLower, "multiway")
                % Multiway clustered standard errors
                fitResult = model.fit();
                
                if isempty(options.clusterGroups)
                    error("pyBridge:MissingClusterGroups", ...
                        "Multiway clustered standard errors require clusterGroups parameter (cell array)");
                end
                
                % Use CGM (2011) formula via get_robustcov_results
                covMatrix = pyBridge.internal.CovarianceTypes.multiwayClustered(...
                    fitResult, options.clusterGroups);
                
                stdErrors = sqrt(diag(covMatrix));
                fitParams = double(py.getattr(fitResult, 'params'));
                % Ensure column vectors
                stdErrors = stdErrors(:);
                fitParams = fitParams(:);
                tStats = fitParams ./ stdErrors;
                
                result = pyBridge.ResultParser.parseStatsmodels(fitResult);
                result.stdErrors = stdErrors;
                result.tStatistics = tStats;
                result.pValues = 2 * (1 - tcdf(abs(tStats), result.dfResiduals));
                result.confidenceIntervals = [fitParams - 1.96*stdErrors, ...
                                              fitParams + 1.96*stdErrors];
                result.covType = "MultiwayClustered";
                result.nDimensions = length(options.clusterGroups);
                result.clusterDimensions = options.clusterGroups;
                
            else
                % Standard fitting (including HC series)
                fitResult = model.fit(cov_type=options.covType);
                result = pyBridge.ResultParser.parseStatsmodels(fitResult);
                result.covType = string(options.covType);
            end
        end
        
        function result = wls(y, X, weights, options)
            % WLS Weighted Least Squares regression
            
            arguments
                y double
                X double
                weights double
                options.addConstant logical = true
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            weightsPy = pyBridge.DataConverter.toPython(weights(:));
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            model = py.statsmodels.api.WLS(yPy, XPy, weights=weightsPy);
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
        end
        
        function result = glm(y, X, options)
            % GLM Generalized Linear Model
            
            arguments
                y double
                X double
                options.family char = "gaussian" % 'gaussian', 'binomial', 'poisson', 'gamma'
                options.addConstant logical = true
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            % Select distribution family
            switch lower(options.family)
                case "gaussian"
                    family = py.statsmodels.genmod.families.family.Gaussian();
                case "binomial"
                    family = py.statsmodels.genmod.families.family.Binomial();
                case "poisson"
                    family = py.statsmodels.genmod.families.family.Poisson();
                case "gamma"
                    family = py.statsmodels.genmod.families.family.Gamma();
                otherwise
                    error("pyBridge:InvalidFamily", "Unknown distribution family: %s", options.family);
            end
            
            model = py.statsmodels.api.GLM(yPy, XPy, family=family);
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.family = string(options.family);
        end
        
        function result = logistic(y, X, options)
            % LOGISTIC Logistic regression (binary classification)
            %   Supports robust standard errors and marginal effects significance tests
            %
            % Parameters:
            %   y - Dependent variable (0/1)
            %   X - Independent variable matrix
            %   options.addConstant - Whether to add constant term
            %   options.covType - Standard error type:
            %       'nonrobust' - Classical standard errors (default)
            %       'HC0', 'HC1', 'HC2', 'HC3' - Heteroskedasticity-robust
            %       'cluster' - Clustered standard errors (requires clusterIds)
            %       'HAC' - Newey-West HAC standard errors (requires maxLags)
            %   options.clusterIds - Cluster identifiers
            %   options.maxLags - Maximum lags for HAC
            %   options.kernel - HAC kernel (default 'bartlett')
            %   options.margeffMethod - Marginal effects calculation method:
            %       'dydx' - Derivative with respect to x (default)
            %       'eyex' - Elasticity (dy/dx * x/y)
            %       'dyex' - Semi-elasticity (dy/d(lnx))
            %       'eydx' - Semi-elasticity (d(lny)/dx)
            %   options.margeffAt - Location for marginal effects calculation:
            %       'mean' - At mean values (default)
            %       'median' - At median values
            %       'all' - Average over all observations
            %
            % Returns:
            %   result.params - Coefficients
            %   result.marginalEffects - Marginal effects
            %   result.marginalEffectsSE - Marginal effects standard errors
            %   result.marginalEffectsP - Marginal effects p-values
            %
            % Example:
            %   % Basic usage
            %   result = pyBridge.StatsmodelsWrapper.logistic(y, X);
            %
            %   % Clustered standard errors
            %   result = pyBridge.StatsmodelsWrapper.logistic(y, X, ...
            %       covType="cluster", clusterIds=firmIds);
            %
            %   % HAC standard errors
            %   result = pyBridge.StatsmodelsWrapper.logistic(y, X, ...
            %       covType="HAC", maxLags=4);
            
            arguments
                y double % 0/1
                X double
                options.addConstant logical = true
                options.maxIter double = 100
                options.covType char = "nonrobust"
                options.clusterIds double = []
                options.maxLags double = []
                options.kernel char = "bartlett"
                options.margeffMethod char = "dydx"
                options.margeffAt char = "mean"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Convert y to 1D array (n,) - Logit requires 1D y
            yPy = pyBridge.DataConverter.toPython(y(:));
            yPy = py.numpy.asarray(yPy).flatten();
            
            % Convert X to 2D array (n, k)
            XPy = pyBridge.DataConverter.toPython(X);
            % Ensure X is 2D - use reshape instead of atleast_2d for correct orientation
            XPy_ndim = double(py.getattr(XPy, 'ndim'));
            if XPy_ndim == 1
                XPy = py.numpy.reshape(XPy, py.tuple({int64(-1), int64(1)}));
            end
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            model = py.statsmodels.api.Logit(yPy, XPy);
            
            % Select fitting method based on standard error type
            % Note: Logit (BinaryResultsWrapper) does not support get_robustcov_results
            % Use cov_type and cov_kwds parameters directly in fit()
            covTypeLower = lower(options.covType);
            
            if strcmp(covTypeLower, "hac")
                % HAC standard errors - use fit() with cov_type
                covKwds = py.dict();
                if ~isempty(options.maxLags)
                    covKwds{"maxlags"} = int64(options.maxLags);
                end
                % Map kernel name to statsmodels-accepted name
                mappedKernel = pyBridge.internal.CovarianceTypes.mapKernelName(options.kernel);
                covKwds{"kernel"} = mappedKernel;
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', 'HAC', 'cov_kwds', covKwds));
                
            elseif strcmp(covTypeLower, "cluster")
                % Clustered standard errors
                if isempty(options.clusterIds)
                    error("pyBridge:MissingClusterIds", "Clustered standard errors require clusterIds parameter");
                end
                clusterPy = pyBridge.DataConverter.toPython(options.clusterIds(:));
                clusterPy = py.numpy.asarray(clusterPy).flatten();
                covKwds = py.dict();
                covKwds{"groups"} = clusterPy;
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', 'cluster', 'cov_kwds', covKwds));
                
            elseif startsWith(covTypeLower, "hc")
                % Heteroskedasticity-robust standard errors - use fit() with cov_type
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', upper(options.covType)));
                
            else
                % Classical standard errors
                fitResult = model.fit(maxiter=int32(options.maxIter), disp=false);
            end
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "Logistic";
            result.covType = string(options.covType);
            
            % Add logistic regression specific results
            try
                hasPredTable = py.builtins.hasattr(fitResult, 'pred_table');
                if hasPredTable
                    predTableFunc = py.getattr(fitResult, 'pred_table');
                    result.confusionMatrix = double(predTableFunc());
                end
            catch
                % pred_table may not be available
            end
            
            % Marginal effects calculation (with standard errors and significance)
            try
                margeffKwds = py.dict();
                margeffKwds{"method"} = options.margeffMethod;
                margeffKwds{"at"} = options.margeffAt;
                
                margeff = fitResult.get_margeff(kwargs=margeffKwds);
                
                result.marginalEffects = double(margeff.margeff);
                result.marginalEffectsMethod = options.margeffMethod;
                result.marginalEffectsAt = options.margeffAt;
                
                % Marginal effects standard errors and significance
                try
                    result.marginalEffectsSE = double(margeff.margeff_se);
                    result.marginalEffectsT = double(margeff.tvalues);
                    result.marginalEffectsP = double(margeff.pvalues);
                    
                    % Marginal effects confidence intervals
                    result.marginalEffectsCI = struct();
                    ciArray = double(margeff.conf_int());
                    result.marginalEffectsCI.lower = ciArray(:, 1);
                    result.marginalEffectsCI.upper = ciArray(:, 2);
                catch ME
                    result.marginalEffectsError = ME.message;
                end
            catch ME
                result.marginalEffectsError = ME.message;
            end
            
            % Predicted probabilities
            try
                predictFunc = py.getattr(fitResult, 'predict');
                result.probabilities = double(predictFunc());
            catch
                % predict may fail
            end
        end
        
        function result = probit(y, X, options)
            % PROBIT Probit regression
            %   Supports robust standard errors and marginal effects significance tests
            %
            % Parameters:
            %   y - Dependent variable (0/1)
            %   X - Independent variable matrix
            %   options.addConstant - Whether to add constant term
            %   options.covType - Standard error type
            %   options.clusterIds - Cluster identifiers
            %   options.maxLags - Maximum lags for HAC
            %   options.margeffMethod - Marginal effects calculation method
            %   options.margeffAt - Location for marginal effects calculation
            %
            % Returns:
            %   result.params - Coefficients
            %   result.marginalEffects - Marginal effects
            %   result.marginalEffectsSE - Marginal effects standard errors
            %   result.marginalEffectsP - Marginal effects p-values
            
            arguments
                y double
                X double
                options.addConstant logical = true
                options.maxIter double = 100
                options.covType char = "nonrobust"
                options.clusterIds double = []
                options.maxLags double = []
                options.kernel char = "bartlett"
                options.margeffMethod char = "dydx"
                options.margeffAt char = "mean"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Convert y to 1D array (n,) - Probit requires 1D y
            yPy = pyBridge.DataConverter.toPython(y(:));
            yPy = py.numpy.asarray(yPy).flatten();
            
            % Convert X to 2D array (n, k)
            XPy = pyBridge.DataConverter.toPython(X);
            % Ensure X is 2D - use reshape instead of atleast_2d for correct orientation
            XPy_ndim = double(py.getattr(XPy, 'ndim'));
            if XPy_ndim == 1
                XPy = py.numpy.reshape(XPy, py.tuple({int64(-1), int64(1)}));
            end
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            model = py.statsmodels.api.Probit(yPy, XPy);
            
            % Select fitting method based on standard error type
            % Note: Probit (BinaryResultsWrapper) does not support get_robustcov_results
            % Use cov_type and cov_kwds parameters directly in fit()
            covTypeLower = lower(options.covType);
            
            if strcmp(covTypeLower, "hac")
                % HAC standard errors - use fit() with cov_type
                covKwds = py.dict();
                if ~isempty(options.maxLags)
                    covKwds{"maxlags"} = int64(options.maxLags);
                end
                % Map kernel name to statsmodels-accepted name
                mappedKernel = pyBridge.internal.CovarianceTypes.mapKernelName(options.kernel);
                covKwds{"kernel"} = mappedKernel;
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', 'HAC', 'cov_kwds', covKwds));
                
            elseif strcmp(covTypeLower, "cluster")
                % Clustered standard errors
                if isempty(options.clusterIds)
                    error("pyBridge:MissingClusterIds", "Clustered standard errors require clusterIds parameter");
                end
                clusterPy = pyBridge.DataConverter.toPython(options.clusterIds(:));
                clusterPy = py.numpy.asarray(clusterPy).flatten();
                covKwds = py.dict();
                covKwds{"groups"} = clusterPy;
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', 'cluster', 'cov_kwds', covKwds));
                
            elseif startsWith(covTypeLower, "hc")
                % Heteroskedasticity-robust standard errors - use fit() with cov_type
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', upper(options.covType)));
                
            else
                % Classical standard errors
                fitResult = model.fit(maxiter=int32(options.maxIter), disp=false);
            end
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "Probit";
            result.covType = string(options.covType);
            
            % Marginal effects calculation
            try
                margeffKwds = py.dict();
                margeffKwds{"method"} = options.margeffMethod;
                margeffKwds{"at"} = options.margeffAt;
                
                margeff = fitResult.get_margeff(kwargs=margeffKwds);
                
                result.marginalEffects = double(margeff.margeff);
                result.marginalEffectsMethod = options.margeffMethod;
                result.marginalEffectsAt = options.margeffAt;
                
                try
                    result.marginalEffectsSE = double(margeff.margeff_se);
                    result.marginalEffectsT = double(margeff.tvalues);
                    result.marginalEffectsP = double(margeff.pvalues);
                    result.marginalEffectsCI = struct();
                    ciArray = double(margeff.conf_int());
                    result.marginalEffectsCI.lower = ciArray(:, 1);
                    result.marginalEffectsCI.upper = ciArray(:, 2);
                catch
                end
            catch
            end
            
            try
                predictFunc = py.getattr(fitResult, 'predict');
                result.probabilities = double(predictFunc());
            catch
                % predict may fail
            end
        end
        
        function result = multinomialLogit(y, X, options)
            % MULTINOMIALLOGIT Multinomial Logit regression (multi-class)
            %   Supports robust standard errors (clustered, HAC, sandwich)
            %
            % Parameters:
            %   y - Dependent variable (integer coded, 0,1,2,...,K-1)
            %   X - Independent variable matrix
            %   options.addConstant - Whether to add constant term
            %   options.covType - Standard error type:
            %       'nonrobust' - Classical standard errors (default)
            %       'sandwich' - Robust standard errors (Huber-White)
            %       'cluster' - Clustered standard errors (requires clusterIds)
            %       'HAC' - Newey-West HAC standard errors (requires maxLags)
            %       'hac-groupsum' - Driscoll-Kraay panel HAC (requires timeIds and maxLags)
            %   options.clusterIds - Cluster identifiers (for clustered standard errors)
            %   options.maxLags - Maximum lags for HAC (for HAC standard errors)
            %   options.timeIds - Time identifiers (for hac-groupsum)
            %   options.kernel - HAC kernel (default 'bartlett')
            %
            % Returns:
            %   result.params - Coefficient matrix (K-1 x p)
            %   result.probabilities - Predicted probabilities
            %   result.marginalEffects - Marginal effects
            %
            % Example:
            %   % Basic usage
            %   result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);
            %
            %   % Clustered standard errors
            %   result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
            %       covType="cluster", clusterIds=firmIds);
            %
            %   % HAC standard errors (Newey-West)
            %   result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
            %       covType="HAC", maxLags=4);
            %
            %   % Driscoll-Kraay panel HAC
            %   result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
            %       covType="hac-groupsum", timeIds=yearIds, maxLags=4);
            
            arguments
                y double % Multi-class dependent variable (0,1,2,...)
                X double
                options.addConstant logical = true
                options.maxIter double = 100
                options.covType char = "nonrobust"
                options.clusterIds double = []
                options.maxLags double = []
                options.timeIds double = []
                options.kernel char = "bartlett"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Convert y to 1D array (n,) - MNLogit requires 1D y
            yPy = pyBridge.DataConverter.toPython(y(:));
            % Ensure y is truly 1D (flatten in case it's still (n,1))
            yPy = py.numpy.asarray(yPy).flatten();
            
            % Convert X to 2D array (n, k)
            XPy = pyBridge.DataConverter.toPython(X);
            % Ensure X is 2D - use reshape instead of atleast_2d for correct orientation
            XPy_ndim = double(py.getattr(XPy, 'ndim'));
            if XPy_ndim == 1
                % Single column - reshape to (n, 1)
                XPy = py.numpy.reshape(XPy, py.tuple({int64(-1), int64(1)}));
            end
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            % Fit multinomial Logit model
            model = py.statsmodels.discrete.discrete_model.MNLogit(yPy, XPy);
            
            % Select fitting method based on standard error type
            % MNLogit uses cov_type and cov_kwds parameters directly in fit()
            covTypeLower = lower(options.covType);
            
            if strcmp(covTypeLower, "hac") || strcmp(covTypeLower, "hac-groupsum") || ...
               strcmp(covTypeLower, "hac-panel")
                % HAC standard errors: specify cov_type and cov_kwds in fit()
                covKwds = py.dict();
                if ~isempty(options.maxLags)
                    covKwds{"maxlags"} = int64(options.maxLags);
                end
                
                % Map kernel name to statsmodels-accepted name
                mappedKernel = pyBridge.internal.CovarianceTypes.mapKernelName(options.kernel);
                covKwds{"kernel"} = mappedKernel;
                
                if strcmp(covTypeLower, "hac-groupsum") || strcmp(covTypeLower, "hac-panel")
                    if isempty(options.timeIds)
                        error("pyBridge:MissingTimeIds", ...
                            "hac-groupsum/hac-panel requires timeIds parameter");
                    end
                    timePy = pyBridge.DataConverter.toPython(options.timeIds(:));
                    timePy = py.numpy.asarray(timePy).flatten();
                    covKwds{"time"} = timePy;
                end
                
                % Fit with HAC covariance
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', 'HAC', 'cov_kwds', covKwds));
                
                result = pyBridge.ResultParser.parseStatsmodels(fitResult);
                result.modelType = "MultinomialLogit";
                result.covType = string(upper(options.covType));
                result.maxLags = options.maxLags;
                result.kernel = options.kernel;  % Keep original for user reference
                
            elseif strcmp(covTypeLower, "cluster")
                % Clustered standard errors
                if isempty(options.clusterIds)
                    error("pyBridge:MissingClusterIds", ...
                        "Clustered standard errors require clusterIds parameter");
                end
                
                clusterPy = pyBridge.DataConverter.toPython(options.clusterIds(:));
                clusterPy = py.numpy.asarray(clusterPy).flatten();
                covKwds = py.dict();
                covKwds{"groups"} = clusterPy;
                
                % Fit with clustered covariance
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', 'cluster', 'cov_kwds', covKwds));
                
                result = pyBridge.ResultParser.parseStatsmodels(fitResult);
                result.modelType = "MultinomialLogit";
                result.covType = "Clustered";
                result.nClusters = length(unique(options.clusterIds));
                
            elseif strcmp(covTypeLower, "sandwich")
                % Sandwich = HC0 (Huber-White robust standard errors)
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', 'HC0'));
                
                result = pyBridge.ResultParser.parseStatsmodels(fitResult);
                result.modelType = "MultinomialLogit";
                result.covType = "HC0 (sandwich)";
                
            elseif strcmp(covTypeLower, "hc0") || ...
                   strcmp(covTypeLower, "hc1") || strcmp(covTypeLower, "hc2") || ...
                   strcmp(covTypeLower, "hc3")
                % Heteroskedasticity-robust standard errors
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false, ...
                    'cov_type', options.covType));
                
                result = pyBridge.ResultParser.parseStatsmodels(fitResult);
                result.modelType = "MultinomialLogit";
                result.covType = string(options.covType);
                
            else
                % Classical standard errors (nonrobust)
                fitResult = model.fit(pyargs('maxiter', int32(options.maxIter), 'disp', false));
                result = pyBridge.ResultParser.parseStatsmodels(fitResult);
                result.modelType = "MultinomialLogit";
            end
            
            % Get number of categories and handle dimension issues
            uniqueY = unique(y);
            nCategories = length(uniqueY);
            result.nCategories = nCategories;
            
            % Warning for binary case - recommend using logistic instead
            if nCategories == 2
                warning('pyBridge:BinaryMNLogit', ...
                    'Only 2 categories detected. Consider using logistic() for binary classification.');
            end
            
            % Marginal effects - handle dimension issues carefully
            try
                margeff = fitResult.get_margeff();
                margeffArr = py.numpy.asarray(margeff.margeff);
                % Ensure 2D array for marginal effects
                if double(py.getattr(margeffArr, 'ndim')) == 1
                    margeffArr = py.numpy.atleast_2d(margeffArr);
                end
                result.marginalEffects = double(margeffArr);
                
                % Marginal effects at means - also handle dimensions
                try
                    atMeansArr = py.numpy.asarray(margeff.margeff_at_means);
                    if double(py.getattr(atMeansArr, 'ndim')) == 1
                        atMeansArr = py.numpy.atleast_2d(atMeansArr);
                    end
                    result.marginalEffectsAtMeans = double(atMeansArr);
                catch
                    result.marginalEffectsAtMeans = [];
                end
                
                % Marginal effects standard errors and significance
                try
                    result.marginalEffectsSE = double(margeff.margeff_se);
                    result.marginalEffectsT = double(margeff.tvalues);
                    result.marginalEffectsP = double(margeff.pvalues);
                catch
                    % Standard errors may not be available in some cases
                end
            catch ME
                % Marginal effects calculation may fail in some cases
                result.marginalEffectsError = ME.message;
            end
            
            % Predicted probabilities
            try
                predictFunc = py.getattr(fitResult, 'predict');
                result.probabilities = double(predictFunc());
            catch
                % predict may fail in some cases
            end
        end
        
        function result = orderedLogit(y, X, options)
            % ORDEREDLOGIT Ordered Logit regression
            %
            % Parameters:
            %   y - Ordered dependent variable (integer coded, 0,1,2,...,K-1)
            %   X - Independent variable matrix
            %
            % Example:
            %   result = pyBridge.StatsmodelsWrapper.orderedLogit(y, X);
            
            arguments
                y double % Ordered dependent variable (0,1,2,...)
                X double
                options.addConstant logical = true
                options.maxIter double = 100
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            % Ordered Logit/Probit model (with version compatibility)
            try
                % statsmodels >= 0.14 uses new API
                model = py.statsmodels.discrete.ordered_model.OrderedModel(yPy, XPy, distr="logit");
            catch
                % Fallback for older versions
                model = py.statsmodels.miscmodels.ordinal_model.OrderedModel(yPy, XPy, distr="logit");
            end
            fitResult = model.fit(maxiter=int32(options.maxIter), disp=false);

            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "OrderedLogit";

            % Threshold parameters (k_constant needs to be converted to MATLAB scalar)
            kConst = double(fitResult.k_constant);
            result.thresholds = double(fitResult.params(1:kConst-1));
        end
        
        function result = orderedProbit(y, X, options)
            % ORDEREDPROBIT Ordered Probit regression
            %
            % Parameters:
            %   y - Ordered dependent variable (integer coded, 0,1,2,...,K-1)
            %   X - Independent variable matrix
            %
            % Example:
            %   result = pyBridge.StatsmodelsWrapper.orderedProbit(y, X);
            
            arguments
                y double % Ordered dependent variable (0,1,2,...)
                X double
                options.addConstant logical = true
                options.maxIter double = 100
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            % Ordered Probit model
            model = py.statsmodels.miscmodels.ordinal_model.OrderedModel(yPy, XPy, distr="probit");
            fitResult = model.fit(maxiter=int32(options.maxIter), disp=false);
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "OrderedProbit";
            result.thresholds = double(fitResult.params(1:fitResult.k_constant-1));
        end
        
        function result = poisson(y, X, options)
            % POISSON Poisson regression (count data)
            
            arguments
                y double % Count data
                X double
                options.addConstant logical = true
                options.exposure double = [] % Exposure variable
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            if isempty(options.exposure)
                model = py.statsmodels.discrete.discrete_model.Poisson(yPy, XPy);
            else
                exposurePy = pyBridge.DataConverter.toPython(options.exposure(:));
                model = py.statsmodels.discrete.discrete_model.Poisson(yPy, XPy, exposure=exposurePy);
            end
            
            fitResult = model.fit(disp=false);
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "Poisson";
            
            % Overdispersion test
            result.overdispersionTest = double(fitResult.pearson_chi2 / fitResult.df_resid);
            result.hasOverdispersion = result.overdispersionTest > 1.5;
        end
        
        function result = negativeBinomial(y, X, options)
            % NEGATIVEBINOMIAL Negative binomial regression (overdispersed count data)
            
            arguments
                y double
                X double
                options.addConstant logical = true
                options.exposure double = []
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            if isempty(options.exposure)
                model = py.statsmodels.discrete.discrete_model.NegativeBinomial(yPy, XPy);
            else
                exposurePy = pyBridge.DataConverter.toPython(options.exposure(:));
                model = py.statsmodels.discrete.discrete_model.NegativeBinomial(yPy, XPy, exposure=exposurePy);
            end
            
            fitResult = model.fit(disp=false);
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "NegativeBinomial";
            
            % Overdispersion parameter (alpha) is the last parameter
            params = double(py.getattr(fitResult, 'params'));
            result.alpha = params(end);
        end
        
        function result = zeroInflatedNB(y, X, options)
            % ZEROINFLATEDNB Zero-Inflated Negative Binomial regression
            %   Used for count data with excess zeros (e.g., patent applications, hospital visits)
            %
            % Parameters:
            %   y - Dependent variable (count data with excess zeros)
            %   X - Independent variable matrix (count part)
            %   options.addConstant - Whether to add constant term (default true)
            %   options.inflation - Zero-inflation model type:
            %       'logit' - Logit model (default)
            %       'probit' - Probit model
            %   options.exogInflation - Independent variables for zero-inflation part (default same as X)
            %   options.exposure - Exposure variable
            %
            % Returns:
            %   result.params - Full parameter vector
            %   result.paramsCount - Count part coefficients
            %   result.paramsInflate - Zero-inflation part coefficients
            %   result.alpha - Overdispersion parameter
            %   result.vuongTest - Vuong test statistic (comparing ZINB vs NB)
            %
            % Example:
            %   % Basic usage
            %   result = pyBridge.StatsmodelsWrapper.zeroInflatedNB(y, X);
            %
            %   % Specify zero-inflation model type
            %   result = pyBridge.StatsmodelsWrapper.zeroInflatedNB(y, X, inflation="probit");
            %
            %   % Use different zero-inflation independent variables
            %   result = pyBridge.StatsmodelsWrapper.zeroInflatedNB(y, X, exogInflation=X2);
            
            arguments
                y double
                X double
                options.addConstant logical = true
                options.inflation char = "logit" % 'logit' or 'probit'
                options.exogInflation double = []
                options.exposure double = []
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Convert data
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            % Add constant term
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            % Independent variables for zero-inflation part
            if isempty(options.exogInflation)
                % Default to using the same independent variables
                exogInflatePy = XPy;
            else
                exogInflatePy = pyBridge.DataConverter.toPython(options.exogInflation);
                if options.addConstant
                    exogInflatePy = py.statsmodels.api.add_constant(exogInflatePy);
                end
            end
            
            % Create model
            if isempty(options.exposure)
                model = py.statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(...
                    yPy, XPy, exog_infl=exogInflatePy, inflation=options.inflation);
            else
                exposurePy = pyBridge.DataConverter.toPython(options.exposure(:));
                model = py.statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP(...
                    yPy, XPy, exog_infl=exogInflatePy, exposure=exposurePy, inflation=options.inflation);
            end
            
            % Fit model
            fitResult = model.fit(disp=false);
            
            % Parse result
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "ZeroInflatedNB";
            
            % Separate count part and zero-inflation part coefficients
            nCountParams = size(X, 2) + options.addConstant;
            allParams = double(py.getattr(fitResult, 'params'));
            
            % Count part coefficients (including overdispersion parameter alpha)
            result.paramsCount = allParams(1:nCountParams);
            
            % Zero-inflation part coefficients
            result.paramsInflate = allParams(nCountParams+1:end-1);
            
            % Overdispersion parameter
            result.alpha = allParams(end);
            
            % Coefficient names
            if isfield(result, 'paramNames')
                result.countParamNames = result.paramNames(1:nCountParams);
                result.inflateParamNames = result.paramNames(nCountParams+1:end-1);
            end
            
            % Vuong test (comparing ZINB vs NB)
            try
                vuong = fitResult.vuong();
                result.vuongTest = struct();
                result.vuongTest.statistic = double(vuong(1));
                result.vuongTest.pValue = double(vuong(2));
                result.vuongTest.preferZINB = result.vuongTest.statistic > 1.96;
            catch
                % Vuong test may fail
            end
            
            % Predicted zero probability
            try
                result.probZero = double(fitResult.predict(which="prob-zero"));
            catch
                % Ignore
            end
            
            % Predicted mean
            try
                result.predictedMean = double(fitResult.predict(which="mean"));
            catch
                % Ignore
            end
        end
        
        %% Time Series
        function result = arima(y, order, options)
            % ARIMA ARIMA time series model
            
            arguments
                y double
                order double = [1, 0, 0] % [p, d, q]
                options.exog double = []
                options.trend char = "c"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            orderPy = pyBridge.DataConverter.toPython(order);
            
            if isempty(options.exog)
                exogPy = py.None;
            else
                exogPy = pyBridge.DataConverter.toPython(options.exog);
            end
            
            model = py.statsmodels.tsa.arima.model.ARIMA(yPy, order=orderPy, ...
                exog=exogPy, trend=options.trend);
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.order = order;
            result.modelType = "ARIMA";
            
            % Forecast
            result.residuals = double(fitResult.resid);
        end
        
        function result = varModel(data, maxLags, options)
            % VAR Vector Autoregression model
            
            arguments
                data double % T x N matrix
                maxLags double
                options.ic char = "aic" % 'aic', 'bic', 'hqic'
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            dataPy = pyBridge.DataConverter.toPython(data);
            
            model = py.statsmodels.tsa.api.VAR(dataPy);
            fitResult = model.fit(maxlags=int32(maxLags), ic=options.ic);
            
            result = struct();
            result.coefs = double(py.getattr(fitResult, 'coefs'));
            result.sigma = double(py.getattr(fitResult, 'sigma_u'));
            result.kAr = double(py.getattr(fitResult, 'k_ar'));
            result.nObs = double(py.getattr(fitResult, 'nobs'));
            result.aic = double(py.getattr(fitResult, 'aic'));
            result.bic = double(py.getattr(fitResult, 'bic'));
            result.modelType = "VAR";
        end
        
        function result = adfuller(y, options)
            % ADFULLER ADF unit root test
            
            arguments
                y double
                options.maxLags double = []
                options.regression char = "c" % 'c', 'ct', 'ctt', 'n'
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            
            if isempty(options.maxLags)
                [adfStat, pValue, usedLag, nObs, criticalValues, ~] = ...
                    py.statsmodels.tsa.stattools.adfuller(yPy, regression=options.regression);
            else
                [adfStat, pValue, usedLag, nObs, criticalValues, ~] = ...
                    py.statsmodels.tsa.stattools.adfuller(yPy, ...
                    maxlag=int32(options.maxLags), regression=options.regression);
            end
            
            result = struct();
            result.adfStatistic = double(adfStat);
            result.pValue = double(pValue);
            result.usedLag = double(usedLag);
            result.nObs = double(nObs);
            result.criticalValues = pyBridge.DataConverter.dict2Struct(criticalValues);
            result.isStationary = result.pValue < 0.05;
        end
        
        function result = kpss(y, options)
            % KPSS KPSS unit root test
            
            arguments
                y double
                options.regression char = "c" % 'c', 'ct'
                options.lags double = []
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            
            if isempty(options.lags)
                [kpssStat, pValue, lags, criticalValues] = ...
                    py.statsmodels.tsa.stattools.kpss(yPy, regression=options.regression);
            else
                [kpssStat, pValue, lags, criticalValues] = ...
                    py.statsmodels.tsa.stattools.kpss(yPy, ...
                    regression=options.regression, lags=int32(options.lags));
            end
            
            result = struct();
            result.kpssStatistic = double(kpssStat);
            result.pValue = double(pValue);
            result.lags = double(lags);
            result.criticalValues = pyBridge.DataConverter.dict2Struct(criticalValues);
            result.isStationary = result.pValue > 0.05; % KPSS null hypothesis is stationarity
        end
        
        %% Panel Data
        function result = panelOLS(y, X, entityIds, timeIds, options)
            % PANELOLS Panel data OLS (simplified version)
            %   Recommend using linearmodels library for full functionality
            
            arguments
                y double
                X double
                entityIds double %#ok<INUSA>
                timeIds double %#ok<INUSA>
                options.addConstant logical = true
                options.entityEffects logical = false
                options.timeEffects logical = false
            end
            
            % Suggest using linearmodels
            warning("pyBridge:Recommendation", ...
                "For panel data analysis, recommend using pyBridge.LinearmodelsWrapper for full functionality");
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            model = py.statsmodels.api.OLS(yPy, XPy);
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseStatsmodels(fitResult);
            result.modelType = "PooledOLS";
        end
        
        %% Diagnostic Tests
        function result = breuschPagan(y, X, options)
            % BREUSCHPAGAN Breusch-Pagan heteroskedasticity test
            
            arguments
                y double
                X double
                options.addConstant logical = true
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            % First do OLS regression
            model = py.statsmodels.api.OLS(yPy, XPy);
            fitResult = model.fit();
            
            % BP test
            [lmStat, lmPValue, fStat, fPValue] = ...
                py.statsmodels.stats.diagnostic.het_breuschpagan(fitResult.resid, XPy);
            
            result = struct();
            result.lmStatistic = double(lmStat);
            result.lmPValue = double(lmPValue);
            result.fStatistic = double(fStat);
            result.fPValue = double(fPValue);
            result.hasHeteroskedasticity = result.lmPValue < 0.05;
        end
        
        function result = whiteTest(y, X, options)
            % WHITETEST White heteroskedasticity test
            
            arguments
                y double
                X double
                options.addConstant logical = true
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            yPy = pyBridge.DataConverter.toPython(y(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            model = py.statsmodels.api.OLS(yPy, XPy);
            fitResult = model.fit();
            
            [lmStat, lmPValue, fStat, fPValue] = ...
                py.statsmodels.stats.diagnostic.het_white(fitResult.resid, XPy);
            
            result = struct();
            result.lmStatistic = double(lmStat);
            result.lmPValue = double(lmPValue);
            result.fStatistic = double(fStat);
            result.fPValue = double(fPValue);
            result.hasHeteroskedasticity = result.lmPValue < 0.05;
        end
        
        function result = durbinWatson(residuals)
            % DURBINWATSON Durbin-Watson autocorrelation test
            
            arguments
                residuals double
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            residPy = pyBridge.DataConverter.toPython(residuals(:));
            
            dw = py.statsmodels.stats.stattools.durbin_watson(residPy);
            
            result = struct();
            result.durbinWatson = double(dw);
            result.interpretation = "DW close to 2 indicates no autocorrelation, close to 0 indicates positive autocorrelation, close to 4 indicates negative autocorrelation";
        end
        
        function result = vif(X, options)
            % VIF Variance Inflation Factor (multicollinearity test)
            
            arguments
                X double
                options.addConstant logical = true
                options.varNames cell = {}
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            XPy = pyBridge.DataConverter.toPython(X);
            
            if options.addConstant
                XPy = py.statsmodels.api.add_constant(XPy);
            end
            
            nVars = size(X, 2) + options.addConstant;
            vifValues = zeros(nVars, 1);
            
            for i = 1:nVars
                vifValues(i) = double(py.statsmodels.stats.outliers_influence.variance_inflation_factor(XPy, i-1));
            end

            result = struct();
            result.vif = vifValues;

            if ~isempty(options.varNames)
                result.varNames = options.varNames;
            else
                result.varNames = strcat("X", string(1:nVars));
            end

            % Exclude constant term for multicollinearity check (VIF=1 for constant is normal)
            if options.addConstant && nVars > 1
                nonConstantVIF = vifValues(2:end);  % Exclude the first one (constant term)
                result.hasMulticollinearity = any(nonConstantVIF > 10);
                result.vifExConstant = nonConstantVIF;
            else
                result.hasMulticollinearity = any(vifValues > 10);
            end
            result.table = table(result.varNames, vifValues, ...
                VariableNames=["Variable", "VIF"]);
        end
        
        %% Prediction
        function predictions = predict(fitResult, newX, options)
            % PREDICT Make predictions using fitted results
            
            arguments
                fitResult struct % Result from ols and other functions
                newX double
                options.addConstant logical = true
                options.alpha double = 0.05
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            newXPy = pyBridge.DataConverter.toPython(newX);
            
            if options.addConstant
                newXPy = py.statsmodels.api.add_constant(newXPy);
            end
            
            % Get original model (needs to be saved)
            if isfield(fitResult, 'originalModel')
                pyModel = fitResult.originalModel;
                predictions = struct();
                predictions.predicted = double(pyModel.predict(newXPy));
            else
                error("pyBridge:ModelNotFound", "Original model not found in fitted result");
            end
        end
        
        %% Marginal Effects Calculation
        function result = marginalEffects(fitResult, options)
            % MARGINALEFFECTS Calculate marginal effects and significance for discrete models
            %
            % Parameters:
            %   fitResult - Discrete model fitted result (from logistic/probit/multinomialLogit etc.)
            %   options.method - Marginal effects calculation method:
            %       'dydx' - At mean values (default)
            %       'eyex' - Elasticity form
            %       'dyex' - Semi-elasticity
            %       'eydx' - Semi-elasticity
            %   options.at - Location for marginal effects calculation:
            %       'mean' - At mean values (default)
            %       'median' - At median values
            %       'zero' - At zero
            %       or provide specific value vector
            %   options.useRobustSE - Whether to use robust standard errors for marginal effects significance
            %
            % Returns:
            %   result.effects - Marginal effects values
            %   result.stdErrors - Marginal effects standard errors
            %   result.tValues - t statistics
            %   result.pValues - p values
            %   result.confInt - Confidence intervals
            %   result.varNames - Variable names
            %
            % Example:
            %   % Logit model marginal effects
            %   logitResult = pyBridge.StatsmodelsWrapper.logistic(y, X);
            %   margEff = pyBridge.StatsmodelsWrapper.marginalEffects(logitResult);
            %
            %   % Calculate at median
            %   margEff = pyBridge.StatsmodelsWrapper.marginalEffects(logitResult, at="median");
            %
            %   % Use robust standard errors
            %   margEff = pyBridge.StatsmodelsWrapper.marginalEffects(logitResult, useRobustSE=true);
            
            arguments
                fitResult struct
                options.method char = "dydx"
                options.at char = "mean"
                options.useRobustSE logical = false
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Check if original Python model exists
            if ~isfield(fitResult, 'originalModel') && ~isfield(fitResult, 'pyModel')
                error("pyBridge:MissingPyModel", ...
                    "Original Python model missing from fitted result, cannot calculate marginal effects");
            end
            
            % Get Python model
            if isfield(fitResult, 'originalModel')
                pyModel = fitResult.originalModel;
            else
                pyModel = fitResult.pyModel;
            end
            
            % Build parameters
            margeffKwds = py.dict();
            margeffKwds{"method"} = options.method;
            margeffKwds{"at"} = options.at;
            
            % Calculate marginal effects
            try
                margeff = pyModel.get_margeff(kwargs=margeffKwds);
            catch ME
                error("pyBridge:MargEffFailed", "Marginal effects calculation failed: %s", ME.message);
            end
            
            % Extract results
            result = struct();
            result.method = options.method;
            result.at = options.at;
            
            % Marginal effects values
            result.effects = double(margeff.margeff);
            
            % Marginal effects summary
            try
                summary = margeff.summary();
                result.summary = char(summary);
            catch
            end
            
            % Standard errors and significance
            try
                result.stdErrors = double(margeff.margeff_se);
                result.tValues = double(margeff.tvalues);
                result.pValues = double(margeff.pvalues);
                
                % Confidence intervals
                ciArray = double(margeff.conf_int());
                result.confInt = struct();
                result.confInt.lower = ciArray(:, 1);
                result.confInt.upper = ciArray(:, 2);
                
            catch
                % Standard errors may not be available in some cases
                result.stdErrors = [];
                result.tValues = [];
                result.pValues = [];
            end
            
            % Variable names
            try
                result.varNames = cell(margeff.names);
            catch
                if isfield(fitResult, 'paramNames')
                    result.varNames = fitResult.paramNames;
                else
                    result.varNames = strcat("X", string(1:length(result.effects)));
                end
            end
            
            % Create table output
            if ~isempty(result.stdErrors)
                nVars = length(result.effects);
                result.table = table(...
                    result.varNames, ...
                    result.effects, ...
                    result.stdErrors, ...
                    result.tValues, ...
                    result.pValues, ...
                    result.confInt.lower, ...
                    result.confInt.upper, ...
                    VariableNames=["Variable", "MarginalEffect", "StdError", "tValue", "pValue", "CI_lower", "CI_upper"]);
                
                % Add significance stars
                result.stars = strings(nVars, 1);
                for i = 1:nVars
                    if result.pValues(i) < 0.01
                        result.stars(i) = "***";
                    elseif result.pValues(i) < 0.05
                        result.stars(i) = "**";
                    elseif result.pValues(i) < 0.1
                        result.stars(i) = "*";
                    else
                        result.stars(i) = "";
                    end
                end
                result.table.Significance = result.stars;
            end
        end
        
        function printMarginalEffects(fitResult, options)
            % PRINTMARGINALEFFECTS Print marginal effects table
            %
            % Parameters:
            %   fitResult - Discrete model fitted result
            %   options.method - Marginal effects calculation method (default 'dydx')
            %   options.at - Calculation location (default 'mean')
            %   options.decimals - Decimal places (default 4)
            
            arguments
                fitResult struct
                options.method char = "dydx"
                options.at char = "mean"
                options.decimals double = 4
            end
            
            % Calculate marginal effects
            result = pyBridge.StatsmodelsWrapper.marginalEffects(fitResult, ...
                method=options.method, at=options.at);
            
            % Print header
            fprintf("\n");
            fprintf("="*70 + "\n");
            fprintf("Marginal Effects (Method: %s, At: %s)\n", options.method, options.at);
            fprintf("="*70 + "\n");
            fprintf("%-20s %12s %12s %12s %12s\n", ...
                "Variable", "Marg.Eff.", "Std.Error", "t-value", "p-value");
            fprintf("-"*70 + "\n");
            
            % Print each row
            nVars = length(result.effects);
            for i = 1:nVars
                varName = result.varNames{i};
                
                if ~isempty(result.stdErrors)
                    % Get significance stars
                    stars = "";
                    if result.pValues(i) < 0.01
                        stars = "***";
                    elseif result.pValues(i) < 0.05
                        stars = "**";
                    elseif result.pValues(i) < 0.1
                        stars = "*";
                    end
                    
                    fprintf("%-20s %12.*f%s %12.*f %12.*f %12.*f\n", ...
                        varName, ...
                        options.decimals, result.effects(i), stars, ...
                        options.decimals, result.stdErrors(i), ...
                        options.decimals, result.tValues(i), ...
                        options.decimals, result.pValues(i));
                else
                    fprintf("%-20s %12.*f\n", ...
                        varName, options.decimals, result.effects(i));
                end
            end
            
            fprintf("-"*70 + "\n");
            fprintf("Significance: *** p<0.01, ** p<0.05, * p<0.1\n");
            fprintf("="*70 + "\n");
        end
    end
end
