classdef CovarianceTypes
    % COVARIANCETYPES Covariance matrix types and standard error calculation
    %   Provides various robust standard error types: HAC, clustered, multiway clustered, etc.
    %
    % Static Methods:
    %   hac - HAC (Newey-West) standard errors
    %   clustered - Clustered standard errors
    %   multiwayClustered - Multiway clustered standard errors
    %   heteroskedastic - Heteroskedasticity-robust standard errors (HC0-HC3)
    %   mapKernelName - Map kernel names to statsmodels-accepted names
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    methods(Static)
        function kernelOut = mapKernelName(kernelIn)
            % MAPKERNELNAME Map kernel names to statsmodels-accepted names
            %   Maps various kernel name aliases to the names accepted by statsmodels:
            %   - 'bartlett', 'newey-west', 'neweywest' -> 'bartlett'
            %   - 'parzen' -> 'bartlett' (with warning, parzen not supported in discrete models)
            %   - 'quadratic spectral', 'qs' -> 'bartlett' (with warning)
            %   - 'uniform' -> 'uniform'
            %
            % Note: statsmodels discrete models (MNLogit, Logit, etc.) only support
            %       'bartlett' and 'uniform' kernels for HAC standard errors.
            %       Other kernels will be mapped to 'bartlett' with a warning.
            %
            % Parameters:
            %   kernelIn - Input kernel name
            %
            % Returns:
            %   kernelOut - statsmodels-accepted kernel name
            
            kernelLower = lower(char(kernelIn));
            
            % Map to statsmodels-accepted kernel names
            switch kernelLower
                case {'bartlett', 'newey-west', 'neweywest', 'newey_west', 'nw'}
                    kernelOut = 'bartlett';
                case 'uniform'
                    kernelOut = 'uniform';
                case 'parzen'
                    % Parzen not supported in discrete models, use bartlett
                    warning('pyBridge:UnsupportedKernel', ...
                        'Parzen kernel not supported for discrete models, using Bartlett instead.');
                    kernelOut = 'bartlett';
                case {'quadratic spectral', 'qs', 'quadraticspectral'}
                    % QS not supported in discrete models, use bartlett
                    warning('pyBridge:UnsupportedKernel', ...
                        'Quadratic spectral kernel not supported for discrete models, using Bartlett instead.');
                    kernelOut = 'bartlett';
                otherwise
                    % Use bartlett as default for unrecognized kernels
                    warning('pyBridge:UnrecognizedKernel', ...
                        'Unrecognized kernel "%s", using Bartlett instead.', kernelIn);
                    kernelOut = 'bartlett';
            end
        end
        
        function covMatrix = hac(residuals, X, options)
            % HAC HAC (Newey-West) standard errors
            %   Handles heteroskedasticity and autocorrelation
            %
            % Parameters:
            %   residuals - Regression residuals
            %   X - Design matrix
            %   options.lag - Lag order for HAC (default: floor(4*(n/100)^(2/9)))
            %   options.kernel - Kernel function: 'bartlett', 'newey-west', 'parzen', 'qs'
            %
            % Returns:
            %   covMatrix - HAC covariance matrix
            %
            % Example:
            %   cov = pyBridge.internal.CovarianceTypes.hac(resid, X);
            
            arguments
                residuals double
                X double
                options.lag double = []
                options.kernel char = "bartlett"  % 'bartlett' (Newey-West), 'parzen', 'qs'
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Map kernel name to statsmodels-accepted name (validates input)
            pyBridge.internal.CovarianceTypes.mapKernelName(options.kernel);
            
            % Compute HAC covariance matrix directly using Newey-West formula
            % V = (X'X)^{-1} * S * (X'X)^{-1} where S is the HAC estimator
            n = size(X, 1);
            k = size(X, 2);
            
            % Auto-select lags using Newey-West rule if not specified
            if isempty(options.lag)
                nLags = floor(4 * (n/100)^(2/9));
            else
                nLags = options.lag;
            end
            
            % Compute (X'X)^{-1}
            XtX_inv = inv(X' * X);
            
            % Compute score matrix: u_i * x_i
            scores = X .* residuals(:);
            
            % Compute HAC middle matrix S using Bartlett kernel
            S = zeros(k, k);
            for lag = 0:nLags
                % Bartlett kernel weight
                weight = 1 - lag / (nLags + 1);
                
                if lag == 0
                    Gamma = scores' * scores;
                else
                    Gamma = scores(1:end-lag, :)' * scores(lag+1:end, :);
                    Gamma = Gamma + Gamma';  % Symmetrize
                end
                
                S = S + weight * Gamma;
            end
            
            % HAC covariance matrix: sandwich form
            covMatrix = XtX_inv * S * XtX_inv; %#ok<MINV> Sandwich form for formula clarity
        end
        
        function covMatrix = clustered(residuals, X, clusterIds, options)
            % CLUSTERED Clustered robust standard errors
            %
            % Parameters:
            %   residuals - Regression residuals
            %   X - Design matrix
            %   clusterIds - Cluster identifier vector
            %   options.useCorrection - Whether to use small sample correction
            %
            % Example:
            %   cov = pyBridge.internal.CovarianceTypes.clustered(resid, X, firmIds);
            
            arguments
                residuals double
                X double
                clusterIds
                options.useCorrection logical = true
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            residPy = pyBridge.DataConverter.toPython(residuals(:));
            XPy = pyBridge.DataConverter.toPython(X);
            clusterPy = pyBridge.DataConverter.toPython(clusterIds(:));
            
            % Create a simple OLS model to get results object for robust cov
            % Reconstruct y from X * beta + residuals (use zero beta for simplicity)
            yPy = residPy;  % Use residuals as y (since beta=0)
            model = py.statsmodels.api.OLS(yPy, XPy);
            results = model.fit();
            
            % Cluster covariance - use get_robustcov_results
            groups = py.pandas.Series(clusterPy);
            getRobustCovFunc = py.getattr(results, 'get_robustcov_results');
            robustResults = getRobustCovFunc(pyargs(...
                'cov_type', 'cluster', 'groups', groups, ...
                'use_correction', options.useCorrection));
            
            covParamsFunc = py.getattr(robustResults, 'cov_params');
            covPy = covParamsFunc();
            
            covMatrix = double(covPy);
        end
        
        function covMatrix = multiwayClustered(fitResult, clusterGroups, options)
            % MULTIWAYCLUSTERED Multiway clustered standard errors
            %   Cluster by multiple dimensions simultaneously (e.g. firm x year)
            %
            % Parameters:
            %   fitResult - statsmodels OLS results object (from model.fit())
            %   clusterGroups - Cell array containing multiple cluster identifiers
            %       e.g.: {firmIds, yearIds}
            %   options.useCorrection - Small sample correction method
            %
            % Example:
            %   cov = pyBridge.internal.CovarianceTypes.multiwayClustered(...
            %       fitResult, {firmIds, yearIds});
            %   % Two-way clustering: firm dimension + time dimension
            
            arguments
                fitResult  % statsmodels OLS results object
                clusterGroups cell % {cluster1, cluster2, ...}
                options.useCorrection logical = true
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            % Convert cluster groups to pandas Series
            nClusters = length(clusterGroups);
            groupsPy = py.list();
            
            for i = 1:nClusters
                clusterPy = pyBridge.DataConverter.toPython(clusterGroups{i}(:));
                groupsPy.append(py.numpy.array(clusterPy));
            end
            
            % Multiway cluster covariance
            % Use Cameron-Gelbach-Miller (2011) formula: V = V1 + V2 - V_interaction
            if nClusters == 1
                % Single-way clustering - use get_robustcov_results
                group0 = py.operator.getitem(groupsPy, int64(0));
                getRobustCovFunc = py.getattr(fitResult, 'get_robustcov_results');
                fitRobust = getRobustCovFunc(pyargs('cov_type', 'cluster', 'groups', group0, ...
                    'use_correction', options.useCorrection));
                covParamsFunc = py.getattr(fitRobust, 'cov_params');
                covMatrix = double(covParamsFunc());
            elseif nClusters == 2
                % Two-way clustering using CGM formula: V = V1 + V2 - V12
                group0 = py.operator.getitem(groupsPy, int64(0));
                group1 = py.operator.getitem(groupsPy, int64(1));
                
                getRobustCovFunc = py.getattr(fitResult, 'get_robustcov_results');
                
                % Compute V1 (cluster by first dimension)
                fitRobust1 = getRobustCovFunc(pyargs('cov_type', 'cluster', 'groups', group0, ...
                    'use_correction', false));
                covFunc1 = py.getattr(fitRobust1, 'cov_params');
                V1 = double(covFunc1());
                
                % Compute V2 (cluster by second dimension)
                fitRobust2 = getRobustCovFunc(pyargs('cov_type', 'cluster', 'groups', group1, ...
                    'use_correction', false));
                covFunc2 = py.getattr(fitRobust2, 'cov_params');
                V2 = double(covFunc2());
                
                % Create interaction groups (firm x year pairs)
                g0_arr = double(group0);
                g1_arr = double(group1);
                interactionGroups = g0_arr * (max(g1_arr) + 1) + g1_arr;
                interactionPy = py.numpy.array(interactionGroups);
                
                % Compute V12 (cluster by interaction)
                fitRobust12 = getRobustCovFunc(pyargs('cov_type', 'cluster', 'groups', interactionPy, ...
                    'use_correction', false));
                covFunc12 = py.getattr(fitRobust12, 'cov_params');
                V12 = double(covFunc12());
                
                % CGM formula: V = V1 + V2 - V12
                covMatrix = V1 + V2 - V12;
                
                % Small sample correction if requested
                if options.useCorrection
                    residuals = double(py.getattr(fitResult, 'resid'));
                    params = double(py.getattr(fitResult, 'params'));
                    n = length(residuals);
                    k = length(params);
                    nClusters0 = length(unique(g0_arr));
                    nClusters1 = length(unique(g1_arr));
                    minClusters = min(nClusters0, nClusters1);
                    correction = (minClusters / (minClusters - 1)) * ((n - 1) / (n - k));
                    covMatrix = covMatrix * correction;
                end
            else
                % Multiway clustering (3 or more) - not commonly used
                error('pyBridge:UnsupportedClustering', ...
                    'Clustering with more than 2 dimensions is not currently supported');
            end
        end
        
        function covMatrix = heteroskedastic(residuals, X, type)
            % HETEROSKEDASTIC Heteroskedasticity-robust standard errors (HC series)
            %
            % Parameters:
            %   residuals - Regression residuals
            %   X - Design matrix
            %   type - HC type: 'HC0', 'HC1', 'HC2', 'HC3'
            %       HC0 - White standard errors
            %       HC1 - Stata small sample correction
            %       HC2 - Robust, suitable for small samples
            %       HC3 - Most robust, extreme heteroskedasticity
            %
            % Example:
            %   cov = pyBridge.internal.CovarianceTypes.heteroskedastic(resid, X, 'HC1');
            
            arguments
                residuals double
                X double
                type char = "HC1"
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            residPy = pyBridge.DataConverter.toPython(residuals(:));
            XPy = pyBridge.DataConverter.toPython(X);
            
            % Create a simple OLS model to get results object
            yPy = residPy;  % Use residuals as y
            model = py.statsmodels.api.OLS(yPy, XPy);
            results = model.fit();
            
            % Use get_robustcov_results for heteroskedasticity-robust covariance
            getRobustCovFunc = py.getattr(results, 'get_robustcov_results');
            robustResults = getRobustCovFunc(pyargs('cov_type', upper(type)));
            covParamsFunc = py.getattr(robustResults, 'cov_params');
            covPy = covParamsFunc();
            
            covMatrix = double(covPy);
        end
        
        function result = testHAC(residuals, X, options)
            % TESTHAC HAC standard error testing and comparison
            %   Compare results from different kernel functions and lag orders
            
            arguments
                residuals double
                X double
                options.lagRange double = 0:10
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("statsmodels");
            
            kernels = {"bartlett", "parzen", "qs"};  % Valid statsmodels kernel names
            nLags = length(options.lagRange);
            
            result = struct();
            result.lags = options.lagRange;
            
            for k = 1:length(kernels)
                kernel = kernels{k};
                seMatrix = zeros(size(X, 2), nLags);
                
                for l = 1:nLags
                    lag = options.lagRange(l);
                    if lag == 0
                        % lag=0 is equivalent to White standard errors
                        covMatrix = pyBridge.internal.CovarianceTypes.heteroskedastic(...
                            residuals, X, "HC0");
                    else
                        covMatrix = pyBridge.internal.CovarianceTypes.hac(...
                            residuals, X, lag=lag, kernel=kernel);
                    end
                    seMatrix(:, l) = sqrt(diag(covMatrix));
                end
                
                result.("se_" + kernel) = seMatrix;
            end
            
            result.note = "Comparison of standard errors across different lag orders";
        end
        
        %% Robust standard errors for MLE models (for Logit/Probit/MLogit etc.)
        function covMatrix = mleRobust(hessian, scoreMatrix, options)
            % MLE ROBUST Robust standard errors for MLE models (sandwich estimator)
            %
            % Parameters:
            %   hessian - Hessian matrix (second derivative of negative log-likelihood)
            %   scoreMatrix - Score matrix (N x p, score vector for each observation)
            %   options.covType - Standard error type:
            %       'sandwich' - Robust standard errors (Huber-White)
            %       'cluster' - Clustered standard errors (requires clusterIds)
            %       'hac' - HAC standard errors (requires maxLags)
            %   options.clusterIds - Cluster identifiers (for cluster type)
            %   options.maxLags - HAC maximum lag order (for hac type)
            %
            % Formula:
            %   V_sandwich = H^{-1} * (S'S) * H^{-1}
            %   where H = -d^2 lnL/d beta^2, S = d lnL/d beta
            %
            % Example:
            %   cov = pyBridge.internal.CovarianceTypes.mleRobust(H, scores);
            %   cov = pyBridge.internal.CovarianceTypes.mleRobust(H, scores, covType="cluster", clusterIds=firmIds);
            
            arguments
                hessian double
                scoreMatrix double
                options.covType char = "sandwich"
                options.clusterIds double = []
                options.maxLags double = 4
            end
            
            % Hessian matrix inverse
            Hinv = inv(hessian);
            
            switch lower(options.covType)
                case "sandwich"
                    % Standard sandwich estimator
                    % V = H^{-1} * B * H^{-1}, where B = S'S
                    B = scoreMatrix' * scoreMatrix;
                    covMatrix = Hinv * B * Hinv; %#ok<MINV> Sandwich form for formula clarity
                    
                case "cluster"
                    % Clustered robust standard errors
                    % B = sum_g (S_g' * S_g)
                    if isempty(options.clusterIds)
                        error("pyBridge:MissingClusterIds", "Clustered standard errors require clusterIds parameter");
                    end
                    
                    uniqueClusters = unique(options.clusterIds);
                    nClusters = length(uniqueClusters);
                    p = size(scoreMatrix, 2);
                    B = zeros(p, p);
                    
                    for g = 1:nClusters
                        idx = options.clusterIds == uniqueClusters(g);
                        Sg = scoreMatrix(idx, :);
                        B = B + Sg' * Sg;
                    end
                    
                    % Small sample correction
                    correction = (nClusters * (nClusters - 1)) / ((nClusters - 1)^2);
                    covMatrix = correction * Hinv * B * Hinv; %#ok<MINV> Sandwich form for formula clarity
                    
                case "hac"
                    % HAC standard errors (Newey-West for MLE)
                    % Need to compute autocovariance of scores
                    p = size(scoreMatrix, 2);
                    S = zeros(p, p);
                    
                    % Score mean
                    scoreMean = mean(scoreMatrix, 1);
                    scoreCentered = scoreMatrix - scoreMean;
                    
                    % Weighted sum of autocovariances
                    for lag = 0:options.maxLags
                        weight = 1 - lag / (options.maxLags + 1);
                        
                        if lag == 0
                            Gamma = scoreCentered' * scoreCentered;
                        else
                            Gamma = scoreCentered(1:end-lag, :)' * scoreCentered(lag+1:end, :);
                            Gamma = Gamma + Gamma';  % Symmetrize
                        end
                        
                        S = S + weight * Gamma;
                    end
                    
                    covMatrix = Hinv * S * Hinv; %#ok<MINV> Sandwich form for formula clarity
                    
                otherwise
                    error("pyBridge:InvalidCovType", "Unsupported covariance type: %s", options.covType);
            end
        end
        
        function covMatrix = mleClustered(hessian, scoreMatrix, clusterIds, options)
            % MLECLUSTERED Clustered standard errors for MLE models (simplified interface)
            %
            % Parameters:
            %   hessian - Hessian matrix
            %   scoreMatrix - Score matrix (N x p)
            %   clusterIds - Cluster identifiers
            %   options.dfCorrection - Degrees of freedom correction (default true)
            %
            % Example:
            %   cov = pyBridge.internal.CovarianceTypes.mleClustered(H, scores, firmIds);
            
            arguments
                hessian double
                scoreMatrix double
                clusterIds double
                options.dfCorrection logical = true
            end
            
            Hinv = inv(hessian);
            
            uniqueClusters = unique(clusterIds);
            nClusters = length(uniqueClusters);
            p = size(scoreMatrix, 2);
            B = zeros(p, p);
            
            for g = 1:nClusters
                idx = clusterIds == uniqueClusters(g);
                Sg = scoreMatrix(idx, :);
                B = B + Sg' * Sg;
            end
            
            if options.dfCorrection
                % Small sample correction factor
                N = size(scoreMatrix, 1);
                correction = (nClusters / (nClusters - 1)) * ((N - 1) / (N - p));
                covMatrix = correction * Hinv * B * Hinv; %#ok<MINV> Sandwich form for formula clarity
            else
                covMatrix = Hinv * B * Hinv; %#ok<MINV> Sandwich form for formula clarity
            end
        end
    end
end
