classdef ScipyWrapper
    % SCIPYWRAPPER scipy library wrapper
    %   Provides scipy.stats, scipy.optimize, scipy.signal and other common functions
    %
    % Static Methods:
    %   stats - Statistics functions
    %   optimize - Optimization functions
    %   signal - Signal processing functions
    %   linalg - Linear algebra functions
    %   integrate - Integration functions
    %
    % Example:
    %   % Statistical distributions
    %   pdf = pyBridge.ScipyWrapper.stats.normPDF(x, 0, 1);
    %   cdf = pyBridge.ScipyWrapper.stats.normCDF(x, 0, 1);
    %
    %   % Optimization
    %   result = pyBridge.ScipyWrapper.optimize.minimize(objective, x0);
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    properties(Constant)
        Stats = pyBridge.internal.ScipyStats()
        Optimize = pyBridge.internal.ScipyOptimize()
        Signal = pyBridge.internal.ScipySignal()
    end
    
    methods(Static)
        function result = stats(functionName, varargin)
            % STATS Call scipy.stats functions
            %   functionName - Function name
            %   varargin - Function parameters
            
            arguments
                functionName char
            end
            arguments (Repeating)
                varargin
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
            
            % Get stats module
            statsModule = py.scipy.stats;
            
            % Call function
            result = pyBridge.ErrorHandler.wrapCall(...
                @() statsModule.(functionName)(varargin{:}));
            
            % Convert result
            if ~isempty(result)
                result = pyBridge.DataConverter.toMatlab(result);
            end
        end
        
        function result = optimize(functionName, varargin)
            % OPTIMIZE Call scipy.optimize functions
            
            arguments
                functionName char
            end
            arguments (Repeating)
                varargin
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
            
            optimizeModule = py.scipy.optimize;
            
            result = pyBridge.ErrorHandler.wrapCall(...
                @() optimizeModule.(functionName)(varargin{:}));
            
            if ~isempty(result)
                result = pyBridge.ResultParser.parse(result, "scipy_optimize");
            end
        end
        
        function result = signal(functionName, varargin)
            % SIGNAL Call scipy.signal functions
            
            arguments
                functionName char
            end
            arguments (Repeating)
                varargin
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
            
            signalModule = py.scipy.signal;
            
            result = pyBridge.ErrorHandler.wrapCall(...
                @() signalModule.(functionName)(varargin{:}));
            
            if ~isempty(result)
                result = pyBridge.DataConverter.toMatlab(result);
            end
        end
        
        function result = linalg(functionName, varargin)
            % LINALG Call scipy.linalg functions
            
            arguments
                functionName char
            end
            arguments (Repeating)
                varargin
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
            
            linalgModule = py.scipy.linalg;
            
            result = pyBridge.ErrorHandler.wrapCall(...
                @() linalgModule.(functionName)(varargin{:}));
            
            if ~isempty(result)
                result = pyBridge.DataConverter.toMatlab(result);
            end
        end
        
        function result = integrate(functionName, varargin)
            % INTEGRATE Call scipy.integrate functions
            
            arguments
                functionName char
            end
            arguments (Repeating)
                varargin
            end
            
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
            
            integrateModule = py.scipy.integrate;
            
            result = pyBridge.ErrorHandler.wrapCall(...
                @() integrateModule.(functionName)(varargin{:}));
            
            if ~isempty(result)
                result = pyBridge.DataConverter.toMatlab(result);
            end
        end
    end
end
