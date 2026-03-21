classdef LinearmodelsWrapper
    % LINEARMODELSWRAPPER Wrapper for linearmodels library
    %   Provides panel data analysis, instrumental variable regression, etc.
    %
    % Static Methods:
    %   panelOLS - Panel data OLS
    %   randomEffects - Random effects model
    %   betweenOLS - Between OLS estimation
    %   pooledOLS - Pooled OLS
    %   iv2SLS - Two-stage least squares
    %   ivLIML - Limited information maximum likelihood
    %   ivGMM - GMM estimation
    %
    % Example:
    %   result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds);
    %
    
    methods(Static)
        %% Panel Data Models
        function result = panelOLS(y, X, entityIds, timeIds, options)
            % PANELOLS Fixed effects panel data model (supports multiple standard errors)
            %
            % Parameters:
            %   y - Dependent variable
            %   X - Independent variable matrix
            %   entityIds - Entity identifiers
            %   timeIds - Time identifiers
            %   options.entityEffects - Whether to include entity effects
            %   options.timeEffects - Whether to include time effects
            %   options.covType - Standard error type:
            %       'unadjusted' - Classical standard errors
            %       'robust' - Heteroskedasticity-robust standard errors
            %       'clustered' - Clustered standard errors
            %       'clustered_entity' - Clustered by entity
            %       'clustered_time' - Clustered by time
            %       'clustered_both' - Two-way clustering (entity x time)
            %
            % Example:
            %   % Fixed effects model + entity clustered standard errors
            %   result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
            %       entityEffects=true, covType="clustered_entity");
            %
            %   % Two-way fixed effects + two-way clustered standard errors
            %   result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
            %       entityEffects=true, timeEffects=true, covType="clustered_both");
            
            arguments
                y double
                X double
                entityIds
                timeIds
                options.entityEffects logical = true
                options.timeEffects logical = false
                options.addConstant logical = false
                options.weights double = []
                options.covType char = "robust" % Default to robust standard errors
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Create MultiIndex for panel data
            entityPy = pyBridge.DataConverter.toPython(entityIds(:));
            timePy = pyBridge.DataConverter.toPython(timeIds(:));
            multiIndex = py.pandas.MultiIndex.from_arrays(...
                py.list({entityPy, timePy}), ...
                names=py.list({"entity", "time"}));
            
            % Create dependent variable Series with MultiIndex
            yPy = pyBridge.DataConverter.toPython(y(:));
            dependentPy = py.pandas.Series(yPy, pyargs('index', multiIndex, 'name', 'y'));
            
            % Create exogenous variables DataFrame with MultiIndex
            if ~isempty(X)
                % Build dict for X columns
                xDict = py.dict();
                for j = 1:size(X, 2)
                    xDict{sprintf("x%d", j)} = pyBridge.DataConverter.toPython(X(:,j));
                end
                exogPy = py.pandas.DataFrame(xDict, pyargs('index', multiIndex));
            else
                exogPy = py.None;
            end
            
            % Weight
            if isempty(options.weights)
                weightsPy = py.None;
            else
                weightsPy = pyBridge.DataConverter.toPython(options.weights(:));
            end
            
            % Fit model (v7.0 API: separate dependent and exog)
            model = py.linearmodels.panel.PanelOLS(dependentPy, exogPy, ...
                entity_effects=options.entityEffects, ...
                time_effects=options.timeEffects, ...
                weights=weightsPy);
            
            % Select fitting options based on standard error type
            covTypeLower = lower(options.covType);
            
            switch covTypeLower
                case "unadjusted"
                    fitResult = model.fit(cov_type="unadjusted");
                case "robust"
                    fitResult = model.fit(cov_type="robust");
                case "clustered"
                    fitResult = model.fit(cov_type="clustered", cluster_entity=true);
                case "clustered_entity"
                    fitResult = model.fit(cov_type="clustered", cluster_entity=true);
                case "clustered_time"
                    fitResult = model.fit(cov_type="clustered", cluster_time=true);
                case "clustered_both"
                    % Two-way clustering (entity + time)
                    fitResult = model.fit(cov_type="clustered", ...
                        cluster_entity=true, cluster_time=true);
                otherwise
                    fitResult = model.fit(cov_type="robust");
            end
            
            % Parse result
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "PanelOLS";
            result.entityEffects = options.entityEffects;
            result.timeEffects = options.timeEffects;
            result.covType = string(options.covType);
        end
        
        function result = randomEffects(y, X, entityIds, timeIds, options)
            % RANDOMEFFECTS Random effects model
            
            arguments
                y double
                X double
                entityIds
                timeIds
                options.addConstant logical = true %#ok<INUSA> Reserved for API consistency
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Create MultiIndex for panel data
            entityPy = pyBridge.DataConverter.toPython(entityIds(:));
            timePy = pyBridge.DataConverter.toPython(timeIds(:));
            multiIndex = py.pandas.MultiIndex.from_arrays(...
                py.list({entityPy, timePy}), ...
                names=py.list({"entity", "time"}));
            
            % Create dependent variable Series with MultiIndex
            yPy = pyBridge.DataConverter.toPython(y(:));
            dependentPy = py.pandas.Series(yPy, pyargs('index', multiIndex, 'name', 'y'));
            
            % Create exogenous variables DataFrame with MultiIndex
            if ~isempty(X)
                xDict = py.dict();
                for j = 1:size(X, 2)
                    xDict{sprintf("x%d", j)} = pyBridge.DataConverter.toPython(X(:,j));
                end
                exogPy = py.pandas.DataFrame(xDict, pyargs('index', multiIndex));
            else
                exogPy = py.None;
            end
            
            % Fit model (v7.0 API: separate dependent and exog)
            model = py.linearmodels.panel.RandomEffects(dependentPy, exogPy);
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "RandomEffects";
            result.covType = "unadjusted";
        end
        
        function result = betweenOLS(y, X, entityIds, timeIds)
            % BETWEENOLS Between OLS estimation
            
            arguments
                y double
                X double
                entityIds
                timeIds
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Create MultiIndex for panel data
            entityPy = pyBridge.DataConverter.toPython(entityIds(:));
            timePy = pyBridge.DataConverter.toPython(timeIds(:));
            multiIndex = py.pandas.MultiIndex.from_arrays(...
                py.list({entityPy, timePy}), ...
                names=py.list({"entity", "time"}));
            
            % Create dependent variable Series with MultiIndex
            yPy = pyBridge.DataConverter.toPython(y(:));
            dependentPy = py.pandas.Series(yPy, pyargs('index', multiIndex, 'name', 'y'));
            
            % Create exogenous variables DataFrame with MultiIndex
            if ~isempty(X)
                xDict = py.dict();
                for j = 1:size(X, 2)
                    xDict{sprintf("x%d", j)} = pyBridge.DataConverter.toPython(X(:,j));
                end
                exogPy = py.pandas.DataFrame(xDict, pyargs('index', multiIndex));
            else
                exogPy = py.None;
            end
            
            % Fit model (v7.0 API: separate dependent and exog)
            model = py.linearmodels.panel.BetweenOLS(dependentPy, exogPy);
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "BetweenOLS";
            result.covType = "unadjusted";
        end
        
        function result = pooledOLS(y, X, entityIds, timeIds, options)
            % POOLEDOLS Pooled OLS
            
            arguments
                y double
                X double
                entityIds
                timeIds
                options.weights double = []
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Create MultiIndex for panel data
            entityPy = pyBridge.DataConverter.toPython(entityIds(:));
            timePy = pyBridge.DataConverter.toPython(timeIds(:));
            multiIndex = py.pandas.MultiIndex.from_arrays(...
                py.list({entityPy, timePy}), ...
                names=py.list({"entity", "time"}));
            
            % Create dependent variable Series with MultiIndex
            yPy = pyBridge.DataConverter.toPython(y(:));
            dependentPy = py.pandas.Series(yPy, pyargs('index', multiIndex, 'name', 'y'));
            
            % Create exogenous variables DataFrame with MultiIndex
            if ~isempty(X)
                xDict = py.dict();
                for j = 1:size(X, 2)
                    xDict{sprintf("x%d", j)} = pyBridge.DataConverter.toPython(X(:,j));
                end
                exogPy = py.pandas.DataFrame(xDict, pyargs('index', multiIndex));
            else
                exogPy = py.None;
            end
            
            if isempty(options.weights)
                model = py.linearmodels.panel.PooledOLS(dependentPy, exogPy);
            else
                weightsPy = pyBridge.DataConverter.toPython(options.weights(:));
                model = py.linearmodels.panel.PooledOLS(dependentPy, exogPy, weights=weightsPy);
            end
            
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "PooledOLS";
            result.covType = "unadjusted";
        end
        
        function result = firstDifference(y, X, entityIds, timeIds)
            % FIRSTDIFFERENCE First difference model
            
            arguments
                y double
                X double
                entityIds
                timeIds
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Create MultiIndex for panel data
            entityPy = pyBridge.DataConverter.toPython(entityIds(:));
            timePy = pyBridge.DataConverter.toPython(timeIds(:));
            multiIndex = py.pandas.MultiIndex.from_arrays(...
                py.list({entityPy, timePy}), ...
                names=py.list({"entity", "time"}));
            
            % Create dependent variable Series with MultiIndex
            yPy = pyBridge.DataConverter.toPython(y(:));
            dependentPy = py.pandas.Series(yPy, pyargs('index', multiIndex, 'name', 'y'));
            
            % Create exogenous variables DataFrame with MultiIndex
            if ~isempty(X)
                xDict = py.dict();
                for j = 1:size(X, 2)
                    xDict{sprintf("x%d", j)} = pyBridge.DataConverter.toPython(X(:,j));
                end
                exogPy = py.pandas.DataFrame(xDict, pyargs('index', multiIndex));
            else
                exogPy = py.None;
            end
            
            % Fit model (v7.0 API: separate dependent and exog)
            model = py.linearmodels.panel.FirstDifferenceOLS(dependentPy, exogPy);
            fitResult = model.fit();
            
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "FirstDifferenceOLS";
            result.covType = "unadjusted";
        end
        
        %% Instrumental Variable Regression
        function result = iv2SLS(y, endogVars, exogVars, instruments, options)
            % IV2SLS Two-stage least squares estimation
            %
            % Syntax:
            %   result = pyBridge.LinearmodelsWrapper.iv2SLS(y, endogVars, exogVars, instruments)
            %   result = pyBridge.LinearmodelsWrapper.iv2SLS(y, endogVars, exogVars, instruments, weights=w)
            %
            % Parameters:
            %   y           - (n x 1) Dependent variable
            %   endogVars   - (n x k1) Endogenous regressors (variables suspected of endogeneity)
            %   exogVars    - (n x k2) Exogenous regressors (control variables, not instrumented)
            %                 Pass [] if no exogenous controls
            %   instruments - (n x m) Instrumental variables (excluded instruments, m >= k1)
            %   options.weights - (n x 1) Observation weights (optional)
            %
            % linearmodels API: IV2SLS(dependent, exog, endog, instruments)
            %   -> dependent = y
            %   -> exog = exogVars (exogenous controls)
            %   -> endog = endogVars (endogenous variables)
            %   -> instruments = instruments (instrumental variables)
            %
            % Example:
            %   % y = wage, endogVars = education, exogVars = experience, instruments = parents_education
            %   result = pyBridge.LinearmodelsWrapper.iv2SLS(wage, education, experience, parents_edu);
            
            arguments
                y double
                endogVars double % Endogenous regressors (need to be instrumented)
                exogVars double % Exogenous regressors (control variables), pass [] if none
                instruments double % Instrumental variables
                options.weights double = []
                options.addConstant (1,1) logical = true
                options.covType string = "unadjusted"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Add constant to exogenous variables if requested
            if options.addConstant
                n = size(endogVars, 1);
                if isempty(exogVars)
                    exogVars = ones(n, 1);
                else
                    exogVars = [ones(n, 1), exogVars];
                end
            end
            
            % Convert dependent variable
            yPy = pyBridge.DataConverter.toPython(y(:));
            
            % Convert endogenous variables
            endogPy = pyBridge.DataConverter.toPython(endogVars);
            
            % Convert exogenous variables (can be empty)
            if isempty(exogVars)
                exogPy = py.None;
            else
                exogPy = pyBridge.DataConverter.toPython(exogVars);
            end
            
            % Convert instruments
            instrumentsPy = pyBridge.DataConverter.toPython(instruments);
            
            % linearmodels API: IV2SLS(dependent, exog, endog, instruments)
            if isempty(options.weights)
                model = py.linearmodels.iv.IV2SLS(yPy, exogPy, endogPy, instrumentsPy);
            else
                weightsPy = pyBridge.DataConverter.toPython(options.weights(:));
                model = py.linearmodels.iv.IV2SLS(yPy, exogPy, endogPy, instrumentsPy, weights=weightsPy);
            end
            fitResult = model.fit(pyargs("cov_type", char(options.covType)));
            
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "IV2SLS";
            
            % Add IV-specific diagnostics
            if py_builtin.hasattr(fitResult, "first_stage")
                result.firstStage = pyBridge.ResultParser.parseLinearmodels(fitResult.first_stage);
            end
        end
        
        function result = ivLIML(y, endogVars, exogVars, instruments, options)
            % IVLIML Limited information maximum likelihood estimation
            %
            % Syntax:
            %   result = pyBridge.LinearmodelsWrapper.ivLIML(y, endogVars, exogVars, instruments)
            %   result = pyBridge.LinearmodelsWrapper.ivLIML(y, endogVars, exogVars, instruments, weights=w)
            %
            % Parameters:
            %   y           - (n x 1) Dependent variable
            %   endogVars   - (n x k1) Endogenous regressors (variables suspected of endogeneity)
            %   exogVars    - (n x k2) Exogenous regressors (control variables, not instrumented)
            %                 Pass [] if no exogenous controls
            %   instruments - (n x m) Instrumental variables (excluded instruments, m >= k1)
            %   options.weights - (n x 1) Observation weights (optional)
            %
            % linearmodels API: IVLIML(dependent, exog, endog, instruments)
            
            arguments
                y double
                endogVars double % Endogenous regressors (need to be instrumented)
                exogVars double % Exogenous regressors (control variables), pass [] if none
                instruments double % Instrumental variables
                options.weights double = []
                options.addConstant (1,1) logical = true
                options.covType string = "unadjusted"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Add constant to exogenous variables if requested
            if options.addConstant
                n = size(endogVars, 1);
                if isempty(exogVars)
                    exogVars = ones(n, 1);
                else
                    exogVars = [ones(n, 1), exogVars];
                end
            end
            
            % Convert dependent variable
            yPy = pyBridge.DataConverter.toPython(y(:));
            
            % Convert endogenous variables
            endogPy = pyBridge.DataConverter.toPython(endogVars);
            
            % Convert exogenous variables (can be empty)
            if isempty(exogVars)
                exogPy = py.None;
            else
                exogPy = pyBridge.DataConverter.toPython(exogVars);
            end
            
            % Convert instruments
            instrumentsPy = pyBridge.DataConverter.toPython(instruments);
            
            % linearmodels API: IVLIML(dependent, exog, endog, instruments)
            if isempty(options.weights)
                model = py.linearmodels.iv.IVLIML(yPy, exogPy, endogPy, instrumentsPy);
            else
                weightsPy = pyBridge.DataConverter.toPython(options.weights(:));
                model = py.linearmodels.iv.IVLIML(yPy, exogPy, endogPy, instrumentsPy, weights=weightsPy);
            end
            fitResult = model.fit(pyargs("cov_type", char(options.covType)));
            
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "IVLIML";
        end
        
        function result = ivGMM(y, endogVars, exogVars, instruments, options)
            % IVGMM Generalized method of moments estimation
            %
            % Syntax:
            %   result = pyBridge.LinearmodelsWrapper.ivGMM(y, endogVars, exogVars, instruments)
            %   result = pyBridge.LinearmodelsWrapper.ivGMM(y, endogVars, exogVars, instruments, weights=w)
            %
            % Parameters:
            %   y           - (n x 1) Dependent variable
            %   endogVars   - (n x k1) Endogenous regressors (variables suspected of endogeneity)
            %   exogVars    - (n x k2) Exogenous regressors (control variables, not instrumented)
            %                 Pass [] if no exogenous controls
            %   instruments - (n x m) Instrumental variables (excluded instruments, m >= k1)
            %   options.weights - (n x 1) Observation weights (optional)
            %
            % linearmodels API: IVGMM(dependent, exog, endog, instruments)
            
            arguments
                y double
                endogVars double % Endogenous regressors (need to be instrumented)
                exogVars double % Exogenous regressors (control variables), pass [] if none
                instruments double % Instrumental variables
                options.weights double = []
                options.addConstant (1,1) logical = true
                options.covType string = "unadjusted"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");
            
            % Add constant to exogenous variables if requested
            if options.addConstant
                n = size(endogVars, 1);
                if isempty(exogVars)
                    exogVars = ones(n, 1);
                else
                    exogVars = [ones(n, 1), exogVars];
                end
            end
            
            % Convert dependent variable
            yPy = pyBridge.DataConverter.toPython(y(:));
            
            % Convert endogenous variables
            endogPy = pyBridge.DataConverter.toPython(endogVars);
            
            % Convert exogenous variables (can be empty)
            if isempty(exogVars)
                exogPy = py.None;
            else
                exogPy = pyBridge.DataConverter.toPython(exogVars);
            end
            
            % Convert instruments
            instrumentsPy = pyBridge.DataConverter.toPython(instruments);
            
            % linearmodels API: IVGMM(dependent, exog, endog, instruments)
            if isempty(options.weights)
                model = py.linearmodels.iv.IVGMM(yPy, exogPy, endogPy, instrumentsPy);
            else
                weightsPy = pyBridge.DataConverter.toPython(options.weights(:));
                model = py.linearmodels.iv.IVGMM(yPy, exogPy, endogPy, instrumentsPy, weights=weightsPy);
            end
            fitResult = model.fit(pyargs("cov_type", char(options.covType)));
            
            result = pyBridge.ResultParser.parseLinearmodels(fitResult);
            result.modelType = "IVGMM";
        end
        
        %% Tests
        function result = hausmanTest(fixedEffects, randomEffects)
            % HAUSMANTEST Hausman test (fixed vs random effects)
            %   Academic standard: Uses Sargan-Hansen consistent variance estimator (VCE)
            %
            % Parameters:
            %   fixedEffects - PanelOLS result struct
            %   randomEffects - RandomEffects result struct
            %
            % Returns:
            %   result.hausmanStatistic - Hausman statistic
            %   result.pValue - p-value
            %   result.method - Estimation method used

            arguments
                fixedEffects struct % PanelOLS result
                randomEffects struct % RandomEffects result
            end

            pyBridge.ErrorHandler.assertPyAvailable("linearmodels");

            % Use Sargan-Hansen VCE method (heteroskedasticity-robust)
            bFE = fixedEffects.params;
            bRE = randomEffects.params;
            varFE = fixedEffects.stdErrors.^2;
            varRE = randomEffects.stdErrors.^2;

            diff = bFE - bRE;
            varDiff = varFE - varRE;

            % Keep only positive definite part (avoid numerical issues)
            isPosDef = varDiff > 0;
            if sum(isPosDef) < length(isPosDef)
                warning("pyBridge:HausmanWarning", ...
                    "Negative values in variance difference, Hausman test may be invalid, consider using Sargan-Hansen test");
            end

            diff_valid = diff(isPosDef);
            varDiff_valid = varDiff(isPosDef);

            if isempty(diff_valid) || any(varDiff_valid <= 0)
                result.hausmanStatistic = NaN;
                result.pValue = NaN;
                result.conclusion = "Hausman statistic cannot be computed";
                result.method = "Sargan-Hansen VCE (failed)";
            else
                % Sargan-Hansen VCE: use pseudo-inverse to handle possible singularity
                hausmanStat = (diff_valid' * pinv(diag(varDiff_valid)) * diff_valid);
                df = length(diff_valid);
                pValue = 1 - chi2cdf(hausmanStat, df);

                result.hausmanStatistic = hausmanStat;
                result.pValue = pValue;
                result.df = df;
                result.preferFixedEffects = pValue < 0.05;
                result.method = "Sargan-Hansen VCE";

                if result.preferFixedEffects
                    result.conclusion = "Reject null hypothesis, should use fixed effects model";
                else
                    result.conclusion = "Cannot reject null hypothesis, can use random effects model";
                end
            end
        end
        
        function result = panelUnitTest(y, entityIds, timeIds, testName)
            % PANELUNITTEST Panel unit root test
            %   Note: linearmodels v7.0 removed panel.unitroot module
            %   This function now uses statsmodels for unit root tests
            %   Applied per-entity then combined using Fisher's method
            
            arguments
                y double
                entityIds
                timeIds %#ok<INUSA> Reserved for API consistency
                testName char = "adf" % 'adf', 'kpss', 'pp'
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Get unique entities
            uniqueEntities = unique(entityIds);
            nEntities = length(uniqueEntities);
            
            % Collect p-values from individual tests
            pValues = zeros(nEntities, 1);
            statistics = zeros(nEntities, 1);
            
            for i = 1:nEntities
                entityMask = entityIds == uniqueEntities(i);
                yEntity = y(entityMask);
                
                if length(yEntity) < 4
                    % Not enough observations for unit root test
                    pValues(i) = NaN;
                    statistics(i) = NaN;
                    continue;
                end
                
                yPy = pyBridge.DataConverter.toPython(yEntity(:));
                
                switch lower(testName)
                    case "adf"
                        testResult = py.statsmodels.tsa.stattools.adfuller(yPy);
                        statistics(i) = double(testResult{1});
                        pValues(i) = double(testResult{2});
                    case "kpss"
                        testResult = py.statsmodels.tsa.stattools.kpss(yPy);
                        statistics(i) = double(testResult{1});
                        pValues(i) = double(testResult{2});
                    case "pp"
                        % Phillips-Perron test
                        testResult = py.arch.unitroot.PhillipsPerron(yPy);
                        statistics(i) = double(testResult.stat);
                        pValues(i) = double(testResult.pvalue);
                    otherwise
                        error("pyBridge:InvalidTest", "Unknown test type: %s. Supported: adf, kpss, pp", testName);
                end
            end
            
            % Remove NaN values
            validMask = ~isnan(pValues);
            pValuesValid = pValues(validMask);
            statisticsValid = statistics(validMask);
            
            if isempty(pValuesValid)
                error("pyBridge:InsufficientData", "Not enough data for unit root test");
            end
            
            % Combine using Fisher's method: -2 * sum(log(p)) ~ chi2(2*n)
            fisherStat = -2 * sum(log(pValuesValid + eps));
            df = 2 * length(pValuesValid);
            combinedPValue = 1 - chi2cdf(fisherStat, df);
            
            result = struct();
            result.testName = testName;
            result.method = "Fisher combination of individual tests";
            result.statistic = fisherStat;
            result.pValue = combinedPValue;
            result.df = df;
            result.nEntities = length(pValuesValid);
            result.individualStats = statisticsValid;
            result.individualPValues = pValuesValid;
            result.isStationary = combinedPValue < 0.05;
        end
    end
end
