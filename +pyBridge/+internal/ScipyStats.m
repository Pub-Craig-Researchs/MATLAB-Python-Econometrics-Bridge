classdef ScipyStats
    % SCIPYSTATS scipy.stats module wrapper
    %   Provides common statistical distributions and hypothesis tests
    %
    % Methods:
    %   normPDF, normCDF, normPPF - Normal distribution
    %   tTest, chi2Test, fTest - Hypothesis tests
    %   describe - Descriptive statistics
    %   correlation - Correlation analysis
    %
    % Example:
    %   stats = pyBridge.internal.ScipyStats();
    %   pdf = stats.normPDF(linspace(-3,3,100), 0, 1);
    %   tStat, pValue = stats.tTest(data1, data2);
    %
    
    methods
        function obj = ScipyStats()
            % Constructor - verify scipy is available
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
        end
        
        %% Normal Distribution
        function y = normPDF(~, x, mu, sigma)
            % NORMPDF Normal distribution probability density function
            
            arguments
                ~
                x double
                mu double = 0
                sigma double = 1
            end
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            % Use frozen distribution with mu and sigma
            dist = py.scipy.stats.norm(mu, sigma);
            y = double(dist.pdf(xPy));
            
            if ~isscalar(x)
                y = reshape(y, size(x));
            end
        end
        
        function y = normCDF(~, x, mu, sigma)
            % NORMCDF Normal distribution cumulative distribution function
            
            arguments
                ~
                x double
                mu double = 0
                sigma double = 1
            end
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            % Use frozen distribution with mu and sigma
            dist = py.scipy.stats.norm(mu, sigma);
            y = double(dist.cdf(xPy));
            
            if ~isscalar(x)
                y = reshape(y, size(x));
            end
        end
        
        function x = normPPF(~, p, mu, sigma)
            % NORMPPF Normal distribution quantile function
            
            arguments
                ~
                p double
                mu double = 0
                sigma double = 1
            end
            
            pPy = pyBridge.DataConverter.toPython(p(:));
            % Use frozen distribution with mu and sigma
            dist = py.scipy.stats.norm(mu, sigma);
            x = double(dist.ppf(pPy));
            
            if ~isscalar(p)
                x = reshape(x, size(p));
            end
        end
        
        %% t-distribution
        function y = tPDF(~, x, df)
            % TPDF t-distribution probability density function
            
            arguments
                ~
                x double
                df double
            end
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            y = double(py.scipy.stats.t(df).pdf(xPy));
            
            if ~isscalar(x)
                y = reshape(y, size(x));
            end
        end
        
        function y = tCDF(~, x, df)
            % TCDF t-distribution cumulative distribution function
            
            arguments
                ~
                x double
                df double
            end
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            y = double(py.scipy.stats.t(df).cdf(xPy));
            
            if ~isscalar(x)
                y = reshape(y, size(x));
            end
        end
        
        %% Hypothesis Tests
        function result = tTest(~, data1, data2, options)
            % TTEST t-test
            
            arguments
                ~
                data1 double
                data2 double = []
                options.equalVar logical = true
                options.alternative char = "two-sided" % 'two-sided', 'less', 'greater'
            end
            
            data1Py = pyBridge.DataConverter.toPython(data1);
            
            if ~isempty(data2)
                data2Py = pyBridge.DataConverter.toPython(data2);
                pyResult = py.scipy.stats.ttest_ind(data1Py, data2Py, ...
                    equal_var=options.equalVar, alternative=options.alternative);
            else
                pyResult = py.scipy.stats.ttest_1samp(data1Py, pyargs('popmean', 0, ...
                    'alternative', options.alternative));
            end
            
            result = struct();
            result.tStatistic = double(py.getattr(pyResult, 'statistic'));
            result.pValue = double(py.getattr(pyResult, 'pvalue'));
            result.significant = result.pValue < 0.05;
        end
        
        function result = tTestPaired(~, data1, data2)
            % TTESTPAIRED Paired t-test
            
            arguments
                ~
                data1 double
                data2 double
            end
            
            data1Py = pyBridge.DataConverter.toPython(data1);
            data2Py = pyBridge.DataConverter.toPython(data2);
            
            pyResult = py.scipy.stats.ttest_rel(data1Py, data2Py);
            
            result = struct();
            result.tStatistic = double(py.getattr(pyResult, 'statistic'));
            result.pValue = double(py.getattr(pyResult, 'pvalue'));
            result.significant = result.pValue < 0.05;
        end
        
        function result = chi2Test(~, observed, expected)
            % CHI2TEST Chi-square test
            
            arguments
                ~
                observed double
                expected double = []
            end
            
            % Ensure 1D array for chisquare
            observedPy = py.numpy.ravel(pyBridge.DataConverter.toPython(observed));
            
            if isempty(expected)
                pyResult = py.scipy.stats.chisquare(observedPy);
            else
                expectedPy = py.numpy.ravel(pyBridge.DataConverter.toPython(expected));
                pyResult = py.scipy.stats.chisquare(observedPy, f_exp=expectedPy);
            end
            
            result = struct();
            result.chi2Statistic = double(py.getattr(pyResult, 'statistic'));
            result.pValue = double(py.getattr(pyResult, 'pvalue'));
            result.significant = result.pValue < 0.05;
        end
        
        function result = fTest(~, data1, data2)
            % FTEST F-test (variance homogeneity test)
            
            arguments
                ~
                data1 double
                data2 double
            end
            
            data1Py = pyBridge.DataConverter.toPython(data1);
            data2Py = pyBridge.DataConverter.toPython(data2);
            
            pyResult = py.scipy.stats.f_oneway(data1Py, data2Py);
            
            result = struct();
            result.fStatistic = double(py.getattr(pyResult, 'statistic'));
            result.pValue = double(py.getattr(pyResult, 'pvalue'));
            result.significant = result.pValue < 0.05;
        end
        
        function result = kruskalWallis(~, varargin)
            % KRUSKALWALLIS Kruskal-Wallis H-test
            
            % Convert all inputs
            dataPy = cell(length(varargin), 1);
            for i = 1:length(varargin)
                dataPy{i} = pyBridge.DataConverter.toPython(varargin{i});
            end
            
            pyResult = py.scipy.stats.kruskal(dataPy{:});
            
            result = struct();
            result.hStatistic = double(py.getattr(pyResult, 'statistic'));
            result.pValue = double(py.getattr(pyResult, 'pvalue'));
            result.significant = result.pValue < 0.05;
        end
        
        function result = mannWhitneyU(~, data1, data2, options)
            % MANNWHITNEYU Mann-Whitney U-test
            
            arguments
                ~
                data1 double
                data2 double
                options.alternative char = "two-sided"
            end
            
            data1Py = pyBridge.DataConverter.toPython(data1);
            data2Py = pyBridge.DataConverter.toPython(data2);
            
            pyResult = py.scipy.stats.mannwhitneyu(data1Py, data2Py, ...
                alternative=options.alternative);
            
            result = struct();
            result.uStatistic = double(py.getattr(pyResult, 'statistic'));
            result.pValue = double(py.getattr(pyResult, 'pvalue'));
            result.significant = result.pValue < 0.05;
        end
        
        function result = normalityTest(~, data, testName)
            % NORMALITYTEST Normality test
            
            arguments
                ~
                data double
                testName char = "shapiro" % 'shapiro', 'normaltest', 'kstest'
            end
            
            dataPy = pyBridge.DataConverter.toPython(data);
            
            switch lower(testName)
                case "shapiro"
                    pyResult = py.scipy.stats.shapiro(dataPy);
                case "normaltest"
                    pyResult = py.scipy.stats.normaltest(dataPy);
                case "kstest"
                    pyResult = py.scipy.stats.kstest(dataPy, "norm");
                otherwise
                    error("pyBridge:InvalidTest", "Unknown test type: %s", testName);
            end
            
            result = struct();
            result.testName = testName;
            result.statistic = double(py.getattr(pyResult, 'statistic'));
            result.pValue = double(py.getattr(pyResult, 'pvalue'));
            result.isNormal = result.pValue >= 0.05;
        end
        
        %% Descriptive Statistics
        function result = describe(~, data, options)
            % DESCRIBE Descriptive statistics
            
            arguments
                ~
                data
                options.axis double = 0
                options.nanPolicy char = "omit" % 'propagate', 'omit', 'raise'
            end
            
            if istable(data)
                dataPy = pyBridge.DataConverter.table2Df(data);
            else
                dataPy = pyBridge.DataConverter.toPython(data);
            end
            
            descResult = py.scipy.stats.describe(dataPy, ...
                axis=int32(options.axis), nan_policy=options.nanPolicy);
            
            result = struct();
            result.nObs = double(py.getattr(descResult, 'nobs'));
            minmax = py.getattr(descResult, 'minmax');
            minmaxCell = cell(minmax);
            result.min = double(minmaxCell{1});
            result.max = double(minmaxCell{2});
            result.mean = double(py.getattr(descResult, 'mean'));
            result.variance = double(py.getattr(descResult, 'variance'));
            result.skewness = double(py.getattr(descResult, 'skewness'));
            result.kurtosis = double(py.getattr(descResult, 'kurtosis'));
        end
        
        function result = correlation(~, data1, data2, method)
            % CORRELATION Correlation analysis
            
            arguments
                ~
                data1 double
                data2 double = []
                method char = "pearson" % 'pearson', 'spearman', 'kendall'
            end
            
            data1Py = pyBridge.DataConverter.toPython(data1);
            
            if ~isempty(data2)
                data2Py = pyBridge.DataConverter.toPython(data2);
                
                switch lower(method)
                    case "pearson"
                        pyResult = py.scipy.stats.pearsonr(data1Py, data2Py);
                    case "spearman"
                        pyResult = py.scipy.stats.spearmanr(data1Py, data2Py);
                    case "kendall"
                        pyResult = py.scipy.stats.kendalltau(data1Py, data2Py);
                    otherwise
                        error("pyBridge:InvalidMethod", "Unknown correlation method: %s", method);
                end
                
                result = struct();
                result.correlation = double(py.getattr(pyResult, 'statistic'));
                result.pValue = double(py.getattr(pyResult, 'pvalue'));
                result.method = method;
            else
                % Calculate correlation matrix
                if size(data1, 2) < 2
                    error("pyBridge:InvalidData", "Need multi-column data to calculate correlation matrix");
                end
                
                switch lower(method)
                    case "pearson"
                        % pearsonr does not support matrix input, use numpy
                        r = py.numpy.corrcoef(data1Py, rowvar=false);
                    case "spearman"
                        pyResult = py.scipy.stats.spearmanr(data1Py);
                        r = py.getattr(pyResult, 'statistic');
                        pValueMat = py.getattr(pyResult, 'pvalue');
                    otherwise
                        error("pyBridge:InvalidMethod", "Matrix correlation only supports pearson and spearman");
                end
                
                result = struct();
                result.correlationMatrix = double(r);
                if exist('pValueMat', 'var')
                    result.pValueMatrix = double(pValueMat);
                end
                result.method = method;
            end
        end
        
        %% Distribution Fitting
        function result = fitDistribution(~, data, distName)
            % FITDISTRIBUTION Fit distribution parameters
            
            arguments
                ~
                data double
                distName char = "norm" % 'norm', 't', 'gamma', 'beta', etc.
            end
            
            dataPy = pyBridge.DataConverter.toPython(data);
            
            % Get distribution object
            dist = py.scipy.stats.(distName);
            
            % Fit parameters
            params = dist.fit(dataPy);
            
            result = struct();
            result.distribution = distName;
            result.parameters = double(params);
            
            % Add common parameter names
            switch lower(distName)
                case "norm"
                    result.mu = result.parameters(1);
                    result.sigma = result.parameters(2);
                case "t"
                    result.df = result.parameters(1);
                    result.loc = result.parameters(2);
                    result.scale = result.parameters(3);
            end
        end
        
        %% Quantiles and Percentiles
        function q = percentile(~, data, percentiles)
            % PERCENTILE Calculate percentiles
            
            arguments
                ~
                data double
                percentiles double
            end
            
            dataPy = pyBridge.DataConverter.toPython(data);
            percentilesPy = pyBridge.DataConverter.toPython(percentiles);
            
            q = double(py.numpy.percentile(dataPy, percentilesPy));
        end
        
        function q = quantile(~, data, quantiles)
            % QUANTILE Calculate quantiles
            
            arguments
                ~
                data double
                quantiles double
            end
            
            dataPy = pyBridge.DataConverter.toPython(data);
            quantilesPy = pyBridge.DataConverter.toPython(quantiles);
            
            q = double(py.numpy.quantile(dataPy, quantilesPy));
        end
    end
end
