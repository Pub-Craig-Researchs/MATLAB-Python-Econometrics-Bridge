classdef PyBridgeConfig < handle
    % PYBRIDGECONFIG Python environment configuration and library validation class
    %   Manages Python environment configuration, detects and validates required Python libraries
    %
    % Properties:
    %   pythonPath     - Python executable path
    %   pythonVersion  - Python version information
    %   isInitialized - Whether Python environment is initialized
    %   installedLibs  - List of installed libraries
    %
    % Methods:
    %   PyBridgeConfig - Constructor
    %   initialize     - Initialize Python environment
    %   checkLibrary   - Check if specified library is installed
    %   getLibVersion  - Get library version information
    %   verifyAll      - Verify all required libraries
    %
    % Example:
    %   config = pyBridge.PyBridgeConfig();
    %   config.initialize();
    %   config.verifyAll();
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    properties
        pythonPath char = ""
        pythonVersion char = ""
        isInitialized logical = false
        installedLibs dictionary = dictionary(string,string)
    end
    
    properties(Constant)
        RequiredLibs = ["scipy", "statsmodels", "linearmodels", "econml", ...
                        "numpy", "pandas", "sklearn"]
    end
    
    methods
        function obj = PyBridgeConfig(pythonPath)
            % PYBRIDGECONFIG Constructor
            %   Optional parameter: pythonPath - Python executable path
            
            arguments
                pythonPath char = ""
            end
            
            obj.pythonPath = pythonPath;
            
            % Try to initialize
            obj.initialize();
        end
        
        function initialize(obj)
            % INITIALIZE Initialize Python environment
            %   Configure MATLAB's Python interface and verify environment
            
            % Auto-detect Python path if not specified
            if isempty(obj.pythonPath) || obj.pythonPath == ""
                pe = pyenv;
                if pe.Version ~= ""
                    obj.pythonPath = char(pe.Executable);
                end
            end
            
            try
                % Check if already configured
                if ~isempty(pyenv().Executable)
                    currentPython = char(pyenv().Executable);
                    if contains(currentPython, obj.pythonPath)
                        obj.isInitialized = true;
                        obj.pythonVersion = char(py.sys.version);
                        return;
                    end
                end
                
                % Configure Python environment
                % Use 'Version' parameter for better MATLAB version compatibility
                % ('Executable' is not supported in older MATLAB versions)
                pyenv("Version", obj.pythonPath);
                
                % Get version information
                obj.pythonVersion = char(py.sys.version);
                obj.isInitialized = true;
                
                fprintf("Python environment initialized successfully\n");
                fprintf("  Python version: %s\n", obj.pythonVersion(1:min(40,length(obj.pythonVersion))));
                
            catch ME
                obj.isInitialized = false;
                error("pyBridge:InitFailed", ...
                    "Python environment initialization failed:\n%s", ME.message);
            end
        end
        
        function isAvailable = checkLibrary(obj, libName)
            % CHECKLIBRARY Check if specified library is installed
            %   libName - Library name (e.g. 'scipy', 'statsmodels')
            %   Returns: boolean
            
            arguments
                obj
                libName char
            end
            
            if ~obj.isInitialized
                error("pyBridge:NotInitialized", ...
                    "Python environment not initialized, please call initialize() first");
            end
            
            try
                % Try to import library using importlib (compatible with Python 3.13+)
                mod = py.importlib.import_module(libName);
                
                % Try to get version information using py.getattr
                try
                    versionStr = string(py.getattr(mod, '__version__'));
                catch
                    versionStr = "installed";
                end
                
                % Store version information
                obj.installedLibs(libName) = versionStr;
                isAvailable = true;
                
            catch
                isAvailable = false;
            end
        end
        
        function version = getLibVersion(obj, libName)
            % GETLIBVERSION Get library version information
            %   libName - Library name
            %   Returns: Version string
            
            arguments
                obj
                libName char
            end
            
            % If cached, return directly
            if isKey(obj.installedLibs, libName)
                version = obj.installedLibs(libName);
                return;
            end
            
            % Otherwise check and get
            if obj.checkLibrary(libName)
                version = obj.installedLibs(libName);
            else
                version = "Not installed";
            end
        end
        
        function results = verifyAll(obj)
            % VERIFYALL Verify all required libraries
            %   Returns: Table containing verification results
            
            libNames = obj.RequiredLibs;
            numLibs = length(libNames);
            
            libNameCell = cell(numLibs, 1);
            statusCell = cell(numLibs, 1);
            versionCell = cell(numLibs, 1);
            
            fprintf("\n=== Python Library Verification ===\n");
            
            for i = 1:numLibs
                libName = libNames(i);
                libNameCell{i} = libName;
                
                isAvailable = obj.checkLibrary(libName);
                
                if isAvailable
                    statusCell{i} = "Installed";
                    versionCell{i} = obj.getLibVersion(libName);
                    fprintf("  %-15s [OK] (v%s)\n", libName, versionCell{i});
                else
                    statusCell{i} = "Not installed";
                    versionCell{i} = "-";
                    fprintf("  %-15s Not installed\n", libName);
                end
            end
            
            results = table(libNameCell, statusCell, versionCell, ...
                VariableNames=["Library", "Status", "Version"]);
        end
        
        function printInfo(obj)
            % PRINTINFO Print environment information
            
            fprintf("\n=== PyBridge Environment Info ===\n");
            fprintf("Python path: %s\n", obj.pythonPath);
            fprintf("Python version: %s\n", obj.pythonVersion);
            fprintf("Init status: %s\n", ...
                ternary(obj.isInitialized, "Initialized", "Not initialized"));
            
            if obj.isInitialized
                obj.verifyAll();
            end
        end
    end
    
    methods(Static)
        function config = getInstance()
            % GETINSTANCE Get singleton instance (factory method)
            persistent instance;
            
            if isempty(instance) || ~isvalid(instance)
                instance = pyBridge.PyBridgeConfig();
            end
            
            config = instance;
        end
        
        function success = quickCheck()
            % QUICKCHECK Quick check if environment is ready
            %   Returns: boolean
            
            try
                config = pyBridge.PyBridgeConfig.getInstance();
                success = config.isInitialized && ...
                    all(arrayfun(@(x) config.checkLibrary(x), config.RequiredLibs));
            catch
                success = false;
            end
        end
    end
end

function result = ternary(condition, trueVal, falseVal)
    % TERNARY Ternary operator helper function
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end
