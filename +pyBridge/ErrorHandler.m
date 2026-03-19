classdef ErrorHandler
    % ERRORHANDLER Unified error handling module
    %   Provides Python exception catching and MATLAB-style error message conversion
    %
    % Static Methods:
    %   wrapCall      - Wrap function calls, automatically catch exceptions
    %   handlePyError - Handle Python exceptions
    %   getPyTraceback - Get Python error traceback
    %   formatError   - Format error messages
    %
    % Example:
    %   result = pyBridge.ErrorHandler.wrapCall(@() py.scipy.stats.norm.pdf(x));
    %   % Or handle manually
    %   try
    %       result = pyFunction();
    %   catch ME
    %       pyBridge.ErrorHandler.handlePyError(ME);
    %   end
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    properties(Constant)
        ErrorTypes = dictionary( ...
            "ImportError", "Import Error", ...
            "ModuleNotFoundError", "Module Not Found", ...
            "ValueError", "Value Error", ...
            "TypeError", "Type Error", ...
            "AttributeError", "Attribute Error", ...
            "KeyError", "Key Error", ...
            "IndexError", "Index Error", ...
            "RuntimeError", "Runtime Error", ...
            "LinAlgError", "Linear Algebra Error" ...
        )
    end
    
    methods(Static)
        function result = wrapCall(func, varargin)
            % WRAPCALL Wrap function calls, automatically catch and handle exceptions
            %   func - Function handle
            %   varargin - Arguments to pass to func
            %
            % Returns: Function execution result
            %
            % Example:
            %   result = pyBridge.ErrorHandler.wrapCall(@() py.scipy.stats.norm.pdf(x));
            
            arguments
                func function_handle
            end
            arguments (Repeating)
                varargin
            end
            
            try
                if ~isempty(varargin)
                    result = func(varargin{:});
                else
                    result = func();
                end
                
            catch ME
                % Handle error
                pyBridge.ErrorHandler.handlePyError(ME);
            end
        end
        
        function handlePyError(ME, options)
            % HANDLEPYERROR Handle Python exceptions
            %   ME - Caught MATLAB exception
            %   options - Optional parameters
            %       .throw - Whether to throw error (true) or just print (false)
            %       .verbose - Whether to show detailed stack info
            %       .prefix - Error message prefix
            
            arguments
                ME
                options.throw logical = true
                options.verbose logical = false
                options.prefix char = ""
            end
            
            % Get error info
            errorInfo = pyBridge.ErrorHandler.parseError(ME);
            
            % Format error message
            errorMsg = pyBridge.ErrorHandler.formatError(errorInfo, options);
            
            % Output or throw
            if options.throw
                error("pyBridge:PythonError", errorMsg);
            else
                fprintf(2, "%s\n", errorMsg);
                if options.verbose
                    fprintf(2, "\n=== Detailed Error Info ===\n");
                    fprintf(2, "MATLAB Exception ID: %s\n", ME.identifier);
                    fprintf(2, "Exception Type: %s\n", errorInfo.exceptionType);
                    if ~isempty(errorInfo.traceback)
                        fprintf(2, "\nPython Traceback:\n%s\n", errorInfo.traceback);
                    end
                end
            end
        end
        
        function errorInfo = parseError(ME)
            % PARSEERROR Parse error information
            %   Extract Python error details from MATLAB exception
            
            errorInfo = struct();
            
            % MATLAB error info
            errorInfo.matlabMessage = ME.message;
            errorInfo.matlabIdentifier = ME.identifier;
            
            % Try to extract Python error type
            if contains(ME.message, "Python Error:")
                % Parse Python error format
                parts = strsplit(ME.message, "Python Error:", ...
                    'CollapseDelimiters', true);
                
                if length(parts) >= 2
                    pyErrorMsg = strtrim(parts{2});
                    
                    % Extract exception type and message
                    colonIdx = strfind(pyErrorMsg, ":");
                    if ~isempty(colonIdx)
                        errorInfo.exceptionType = strtrim(pyErrorMsg(1:colonIdx(1)-1));
                        errorInfo.exceptionMessage = strtrim(pyErrorMsg(colonIdx(1)+1:end));
                    else
                        errorInfo.exceptionType = "Unknown";
                        errorInfo.exceptionMessage = pyErrorMsg;
                    end
                end
            else
                % May be MATLAB error
                errorInfo.exceptionType = "MatlabError";
                errorInfo.exceptionMessage = ME.message;
            end
            
            % Get translated error type
            if isfield(errorInfo, 'exceptionType')
                pyType = string(errorInfo.exceptionType);
                if isKey(pyBridge.ErrorHandler.ErrorTypes, pyType)
                    errorInfo.englishType = pyBridge.ErrorHandler.ErrorTypes(pyType);
                else
                    errorInfo.englishType = errorInfo.exceptionType;
                end
            else
                errorInfo.exceptionType = "Unknown";
                errorInfo.exceptionMessage = ME.message;
                errorInfo.englishType = "Unknown Error";
            end
            
            % Try to get Python traceback
            errorInfo.traceback = pyBridge.ErrorHandler.getPyTraceback();
        end
        
        function traceback = getPyTraceback()
            % GETPYTRACEBACK Get Python error traceback
            %   Returns: Python traceback string
            
            traceback = "";
            
            try
                % Try to get the latest exception from sys module
                excInfo = py.sys.exc_info();
                
                if ~py_builtin.isinstance(excInfo{1}, py.types.NoneType)
                    % Extract traceback
                    tbList = py.traceback.format_exception(excInfo{1}, excInfo{2}, excInfo{3});
                    traceback = strjoin(cell(tbList), "");
                end
                
            catch
                % Unable to get traceback
                traceback = "";
            end
        end
        
        function errorMsg = formatError(errorInfo, options)
            % FORMATERROR Format error message
            %   errorInfo - Error info struct
            %   options - Formatting options
            
            arguments
                errorInfo struct
                options.prefix char = ""
            end
            
            % Build error message
            if ~isempty(options.prefix)
                msgParts = {options.prefix, ""};
            else
                msgParts = {};
            end
            
            % Add error type and message
            if isfield(errorInfo, 'englishType')
                msgParts{end+1} = sprintf("[%s] %s", ...
                    errorInfo.englishType, errorInfo.exceptionType);
            end
            
            if isfield(errorInfo, 'exceptionMessage')
                msgParts{end+1} = sprintf("Details: %s", ...
                    errorInfo.exceptionMessage);
            end
            
            errorMsg = strjoin(msgParts, "\n");
        end
        
        function result = safeCall(func, defaultResult, varargin)
            % SAFECALL Safe call, returns default value on failure
            %   func - Function handle
            %   defaultResult - Default return value on failure
            %   varargin - Arguments to pass to func
            
            arguments
                func function_handle
                defaultResult = []
            end
            arguments (Repeating)
                varargin
            end
            
            try
                if ~isempty(varargin)
                    result = func(varargin{:});
                else
                    result = func();
                end
            catch ME
                % Log error but don't throw
                fprintf(2, "[PyBridge Warning] Call failed: %s\n", ME.message);
                result = defaultResult;
            end
        end
        
        function validateInputs(varargin)
            % VALIDATEINPUTS Validate input parameters
            %   Uses MATLAB parameter validation style
            %   Each parameter is in [value, name, type, required] format
            
            for i = 1:length(varargin)
                arg = varargin{i};
                
                if length(arg) >= 3
                    value = arg{1};
                    name = arg{2};
                    argType = arg{3};
                    required = false;
                    if length(arg) >= 4
                        required = arg{4};
                    end
                    
                    % Check if empty
                    if required && isempty(value)
                        error("pyBridge:InvalidInput", ...
                            "Parameter '%s' cannot be empty", name);
                    end
                    
                    % Check type
                    if ~isempty(value) && ~isempty(argType)
                        if ~isa(value, argType)
                            error("pyBridge:InvalidInput", ...
                                "Parameter '%s' type error, expected %s, got %s", ...
                                name, argType, class(value));
                        end
                    end
                end
            end
        end
        
        function printPyError(ME)
            % PRINTPYERROR Print Python error details (for debugging)
            
            fprintf("\n=== Python Error Details ===\n");
            
            % MATLAB exception info
            fprintf("MATLAB Exception ID: %s\n", ME.identifier);
            fprintf("MATLAB Message: %s\n", ME.message);
            
            % Parse Python error
            errorInfo = pyBridge.ErrorHandler.parseError(ME);
            
            fprintf("\nPython Exception Type: %s\n", errorInfo.exceptionType);
            fprintf("Error Type: %s\n", errorInfo.englishType);
            fprintf("Error Message: %s\n", errorInfo.exceptionMessage);
            
            % Stack info
            if ~isempty(errorInfo.traceback)
                fprintf("\n=== Python Traceback ===\n");
                fprintf("%s\n", errorInfo.traceback);
            end
            
            % MATLAB stack
            if ~isempty(ME.stack)
                fprintf("\n=== MATLAB Call Stack ===\n");
                for i = 1:length(ME.stack)
                    fprintf("  %s (line %d)\n", ME.stack(i).name, ME.stack(i).line);
                end
            end
        end
        
        function success = tryImport(moduleName)
            % TRYIMPORT Try to import Python module
            %   moduleName - Module name
            %   Returns: Whether import was successful
            
            arguments
                moduleName char
            end
            
            try
                pyBridge.ErrorHandler.wrapCall(@() py.importlib.import_module(moduleName));
                success = true;
            catch
                success = false;
            end
        end
        
        function assertPyAvailable(libName)
            % ASSERTPYAVAILABLE Ensure Python library is available
            %   Throws friendly error if library is not available
            
            arguments
                libName char
            end
            
            config = pyBridge.PyBridgeConfig.getInstance();
            
            if ~config.checkLibrary(libName)
                error("pyBridge:LibraryNotFound", ...
                    "Python library '%s' is not installed.\n\nInstallation:\n  pip install %s\n\n" + ...
                    "Or using conda:\n  conda install -c conda-forge %s", ...
                    libName, libName, libName);
            end
        end
    end
end
