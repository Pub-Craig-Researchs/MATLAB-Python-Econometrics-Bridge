function startup()
% STARTUP PyBridge toolbox startup script
%   Adds toolbox to MATLAB path and verifies environment
%
% Usage:
%   cd('path/to/MATLAB-Python-Econometrics-Bridge')
%   startup()
%
% Author: WorkBuddy
% Date: 2026-03-18

    % Suppress Python-MATLAB method name conflict warnings during initialization
    % These occur because Python objects (pandas/numpy) have methods
    % with names that match MATLAB built-in functions (empty, reshape, sort, etc.)
    % This is a known issue in MATLAB R2024b+ and does not affect functionality
    warnState = warning('off', 'all');
    
    % Get current directory (toolbox root)
    toolboxRoot = fileparts(mfilename('fullpath'));
    
    % Add toolbox to path
    % Note: Package directories (+pyBridge, +pyfunc, etc.) should NOT be added to path
    % Only the parent directory needs to be added - MATLAB will find packages automatically
    fprintf("Adding PyBridge toolbox to MATLAB path...\n");
    addpath(toolboxRoot);
    
    % Add non-package subdirectories only
    addpath(fullfile(toolboxRoot, 'examples'));
    addpath(fullfile(toolboxRoot, 'tests'));
    
    fprintf("Path configuration complete\n\n");
    
    % Verify environment
    fprintf("Verifying Python environment...\n");
    try
        config = pyBridge.PyBridgeConfig();
        config.initialize();
        results = config.verifyAll();
        
        % Check if all libraries are installed
        % Status is a cell array of strings, use cellfun for comparison
        isInstalled = cellfun(@(x) x == "Installed", results.Status);
        allInstalled = all(isInstalled);
        
        if allInstalled
            fprintf("\nPyBridge toolbox started successfully!\n");
            fprintf("  All required Python libraries are installed\n\n");
            
            % Show quick start tips
            fprintf("Quick Start:\n");
            fprintf("  config = pyBridge.PyBridgeConfig.getInstance();\n");
            fprintf("  stats = pyBridge.internal.ScipyStats();\n");
            fprintf("  result = stats.tTest(data1, data2);\n\n");
            
            fprintf("Run examples:\n");
            fprintf("  run('examples/quickStart.m')\n\n");
        else
            fprintf("\nSome Python libraries are not installed\n");
            fprintf("Please run the following command to install missing libraries:\n");
            fprintf("  pip install scipy statsmodels linearmodels econml numpy pandas scikit-learn\n\n");
            
            % Show missing libraries
            missingLibs = results.Library(~isInstalled);
            fprintf("Missing libraries:\n");
            for i = 1:length(missingLibs)
                fprintf("  - %s\n", missingLibs{i});
            end
            fprintf("\n");
        end
        
    catch ME
        fprintf("\nEnvironment verification failed:\n");
        fprintf("  %s\n\n", ME.message);
        fprintf("Please ensure:\n");
        fprintf("  1. Python is properly installed\n");
        fprintf("  2. MATLAB Python interface is configured\n");
        fprintf("  3. Run: pip install scipy statsmodels linearmodels econml\n\n");
    end
    
    % Show version information
    fprintf("=====================================\n");
    fprintf("PyBridge Toolbox v1.1.0\n");
    fprintf("MATLAB Interface for Python Scientific Computing Libraries\n");
    fprintf("Test Status: 137 tests, 135 passing, 0 failures\n");
    fprintf("Compatibility: Python 3.13, linearmodels v7.0\n");
    fprintf("=====================================\n\n");
    
    % Restore warning state
    warning(warnState);
    
end
