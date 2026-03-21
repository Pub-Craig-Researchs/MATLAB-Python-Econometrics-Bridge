classdef FitFunctionWrapper < handle
    % FITFUNCTIONWRAPPER Curve fitting function wrapper
    %   Specifically designed for scipy.optimize.curve_fit function wrapping
    %   Signature: f(x, p1, p2, ..., pn) -> y
    %
    % Usage:
    %   fun = @(x, a, b) a * x + b;
    %   pyFun = py.pyfunc.FitFunctionWrapper(fun);
    %   py.scipy.optimize.curve_fit(pyFun, xdata, ydata);
    %
    
    properties
        matlabFunction function_handle
    end
    
    methods
        function obj = FitFunctionWrapper(fun)
            arguments
                fun function_handle
            end
            
            obj.matlabFunction = fun;
        end
        
        function y = call(obj, x, varargin)
            % CALL Python call entry point
            %   First parameter is independent variable x, subsequent parameters are fit parameters
            
            try
                % Convert x
                xMatlab = pyBridge.DataConverter.toMatlab(x);
                
                % Convert parameters
                params = cell(size(varargin));
                for i = 1:length(varargin)
                    params{i} = pyBridge.DataConverter.toMatlab(varargin{i});
                    if isscalar(params{i})
                        % Parameters are usually scalars
                        params{i} = double(params{i});
                    end
                end
                
                % Call MATLAB function
                y = obj.matlabFunction(xMatlab, params{:});
                
                % Convert output
                y = pyBridge.DataConverter.toPython(y(:));
                
            catch ME
                error("pyBridge:FitCallbackError", ...
                    "Fit function execution error:\n%s", ME.message);
            end
        end
    end
end
