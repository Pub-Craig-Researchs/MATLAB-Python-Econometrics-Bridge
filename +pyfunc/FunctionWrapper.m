classdef FunctionWrapper < handle
    % FUNCTIONWRAPPER MATLAB function wrapper
    %   Wraps MATLAB function handle as Python callable object
    %   Used for optimization, root finding and other callback scenarios
    %
    % Usage:
    %   fun = @(x) x(1)^2 + x(2)^2;
    %   pyFun = py.pyfunc.FunctionWrapper(fun, 'scalar');
    %   py.scipy.optimize.minimize(pyFun, [1; 1]);
    %
    
    properties
        matlabFunction function_handle
        outputType char = "scalar" % 'scalar' or 'vector'
    end
    
    methods
        function obj = FunctionWrapper(fun, outputType)
            arguments
                fun function_handle
                outputType char = "scalar"
            end
            
            obj.matlabFunction = fun;
            obj.outputType = outputType;
        end
        
        function result = call(obj, varargin)
            % CALL Python call entry point
            %   Receives parameters from Python, converts to MATLAB format, calls function, returns Python format
            
            try
                % Convert input parameters
                matlabArgs = cell(size(varargin));
                for i = 1:length(varargin)
                    matlabArgs{i} = pyBridge.DataConverter.toMatlab(varargin{i});
                end
                
                % Call MATLAB function
                if strcmpi(obj.outputType, 'scalar')
                    % Scalar output
                    result = obj.matlabFunction(matlabArgs{:});
                    result = double(result);
                else
                    % Vector output
                    result = obj.matlabFunction(matlabArgs{:});
                    result = pyBridge.DataConverter.toPython(result(:));
                end
                
            catch ME
                % Error handling
                error("pyBridge:CallbackError", ...
                    "MATLAB callback function execution error:\n%s", ME.message);
            end
        end
    end
end
