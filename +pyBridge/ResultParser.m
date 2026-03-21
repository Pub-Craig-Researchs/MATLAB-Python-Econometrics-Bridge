classdef ResultParser
    % RESULTPARSER Python return result parser
    %   Converts complex objects returned by Python libraries to MATLAB-friendly formats
    %
    % Static Methods:
    %   parse          - Auto-parse Python objects
    %   parseStatsmodels - Parse statsmodels return objects
    %   parseEconML    - Parse econml return objects
    %   parseSummary   - Parse statistical summary results
    %   parseArray     - Parse array-type results
    %   extractAttributes - Extract Python object attributes
    %
    % Example:
    %   % Parse statsmodels regression result
    %   result = pyBridge.ResultParser.parseStatsmodels(modelResult);
    %   % Parse econml causal inference result
    %   ate = pyBridge.ResultParser.parseEconML(ateResult);
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    properties(Constant)
        % Common Python object type parsing strategies
        ParseStrategies = dictionary( ...
            "statsmodels.iolib.summary.Summary", "summary", ...
            "statsmodels.regression.linear_model.RegressionResults", "regression", ...
            "econml.dml.DML", "econml_model", ...
            "econml.dr.DRLearner", "econml_model", ...
            "numpy.ndarray", "array", ...
            "pandas.DataFrame", "dataframe" ...
        )
    end
    
    methods(Static)
        function result = parse(pyObj, parseType)
            % PARSE Auto-parse Python objects
            %   pyObj - Python return object
            %   parseType - Optional parsing type specification
            %
            % Returns: MATLAB structure or array
            
            arguments
                pyObj
                parseType char = "auto"
            end
            
            if isempty(pyObj)
                result = [];
                return;
            end
            
            % Auto-detect type
            if strcmpi(parseType, "auto")
                pyType = char(py.getattr(py.type(pyObj), '__name__'));
                pyModule = char(py.getattr(py.type(pyObj), '__module__'));
                fullType = pyModule + "." + pyType;
                
                % Check known types
                if isKey(pyBridge.ResultParser.ParseStrategies, fullType)
                    parseType = pyBridge.ResultParser.ParseStrategies(fullType);
                else
                    % Infer from module
                    if startsWith(pyModule, "statsmodels")
                        parseType = "statsmodels";
                    elseif startsWith(pyModule, "econml")
                        parseType = "econml";
                    elseif startsWith(pyModule, "linearmodels")
                        parseType = "linearmodels";
                    else
                        parseType = "generic";
                    end
                end
            end
            
            % Parse by type
            switch lower(parseType)
                case "summary"
                    result = pyBridge.ResultParser.parseSummary(pyObj);
                case "regression"
                    result = pyBridge.ResultParser.parseStatsmodels(pyObj);
                case "statsmodels"
                    result = pyBridge.ResultParser.parseStatsmodels(pyObj);
                case "econml"
                    result = pyBridge.ResultParser.parseEconML(pyObj);
                case "econml_model"
                    result = pyBridge.ResultParser.parseEconMLModel(pyObj);
                case "linearmodels"
                    result = pyBridge.ResultParser.parseLinearmodels(pyObj);
                case "array"
                    result = pyBridge.ResultParser.parseArray(pyObj);
                case "dataframe"
                    result = pyBridge.DataConverter.df2Table(pyObj);
                otherwise
                    result = pyBridge.ResultParser.parseGeneric(pyObj);
            end
        end
        
        function result = parseStatsmodels(pyObj)
            % PARSESTATSMODELS Parse statsmodels return object
            %   Automatically extracts regression results, statistics and diagnostic info
            %   Uses py.getattr for safe attribute access to handle Python property issues
            
            result = struct();
            
            try
                % Save original Python object (for marginal effects calculation, etc.)
                result.pyModel = pyObj;
                
                % Basic information - use py.getattr for safe access
                nobsAttr = py.getattr(pyObj, 'nobs', py.None);
                if ~isequal(nobsAttr, py.None)
                    result.nObs = double(nobsAttr);
                end
                
                dfModelAttr = py.getattr(pyObj, 'df_model', py.None);
                if ~isequal(dfModelAttr, py.None)
                    result.dfModel = double(dfModelAttr);
                end
                
                dfResidAttr = py.getattr(pyObj, 'df_resid', py.None);
                if ~isequal(dfResidAttr, py.None)
                    result.dfResiduals = double(dfResidAttr);
                end
                
                % Fit statistics
                rsquaredAttr = py.getattr(pyObj, 'rsquared', py.None);
                if ~isequal(rsquaredAttr, py.None)
                    result.rSquared = double(rsquaredAttr);
                    rsquaredAdjAttr = py.getattr(pyObj, 'rsquared_adj', py.None);
                    if ~isequal(rsquaredAdjAttr, py.None)
                        result.adjRSquared = double(rsquaredAdjAttr);
                    end
                end
                
                aicAttr = py.getattr(pyObj, 'aic', py.None);
                if ~isequal(aicAttr, py.None)
                    result.aic = double(aicAttr);
                    bicAttr = py.getattr(pyObj, 'bic', py.None);
                    if ~isequal(bicAttr, py.None)
                        result.bic = double(bicAttr);
                    end
                end
                
                llfAttr = py.getattr(pyObj, 'llf', py.None);
                if ~isequal(llfAttr, py.None)
                    result.logLikelihood = double(llfAttr);

                    % McFadden Pseudo-R-squared (common academic metric for discrete models)
                    llnullAttr = py.getattr(pyObj, 'llnull', py.None);
                    if ~isequal(llnullAttr, py.None)
                        llNull = double(llnullAttr);
                        if llNull ~= 0
                            result.pseudoRSquared = 1 - result.logLikelihood / llNull;
                            result.pseudoRSquaredMethod = "McFadden";
                            % Adjusted McFadden Pseudo R-squared
                            dfModelAttr = py.getattr(pyObj, 'df_model', py.None);
                            if ~isequal(dfModelAttr, py.None)
                                k = double(dfModelAttr);
                                result.adjPseudoRSquared = 1 - (result.logLikelihood - k) / llNull;
                            end
                        end
                    end
                end
                
                % Coefficients and statistics - use py.getattr for safe access
                paramsAttr = py.getattr(pyObj, 'params', py.None);
                if ~isequal(paramsAttr, py.None)
                    paramsData = pyBridge.DataConverter.numpy2Array(paramsAttr);
                    % Ensure params is always a column vector
                    result.params = paramsData(:);
                end
                
                bseAttr = py.getattr(pyObj, 'bse', py.None);
                if ~isequal(bseAttr, py.None)
                    stdData = pyBridge.DataConverter.numpy2Array(bseAttr);
                    % Ensure stdErrors is always a column vector
                    result.stdErrors = stdData(:);
                end
                
                tvaluesAttr = py.getattr(pyObj, 'tvalues', py.None);
                if ~isequal(tvaluesAttr, py.None)
                    tData = pyBridge.DataConverter.numpy2Array(tvaluesAttr);
                    % Ensure tStatistics is always a column vector
                    result.tStatistics = tData(:);
                end
                
                pvaluesAttr = py.getattr(pyObj, 'pvalues', py.None);
                if ~isequal(pvaluesAttr, py.None)
                    pData = pyBridge.DataConverter.numpy2Array(pvaluesAttr);
                    % Ensure pValues is always a column vector
                    result.pValues = pData(:);
                end
                
                % Confidence intervals
                confIntMethod = py.getattr(pyObj, 'conf_int', py.None);
                if ~isequal(confIntMethod, py.None) && py.builtins.callable(confIntMethod)
                    try
                        ci = confIntMethod();
                        % Convert DataFrame to numpy array first, then index in MATLAB
                        ciArray = pyBridge.DataConverter.numpy2Array(ci.to_numpy());
                        result.confInt = struct();
                        result.confInt.lower = ciArray(:, 1);
                        result.confInt.upper = ciArray(:, 2);
                    catch
                        % Confidence interval calculation may fail
                    end
                end
                
                % Parameter names
                if isfield(result, 'params')
                    try
                        paramsObj = py.getattr(pyObj, 'params');
                        indexObj = py.getattr(paramsObj, 'index', py.None);
                        if ~isequal(indexObj, py.None)
                            result.paramNames = cell(indexObj.tolist());
                        end
                    catch
                        modelAttr = py.getattr(pyObj, 'model', py.None);
                        if ~isequal(modelAttr, py.None)
                            exogNamesAttr = py.getattr(modelAttr, 'exog_names', py.None);
                            if ~isequal(exogNamesAttr, py.None)
                                result.paramNames = cell(exogNamesAttr.tolist());
                            end
                        end
                    end
                end
                
                % Residuals (use py.getattr for safe attribute access)
                residObj = py.getattr(pyObj, 'resid', py.None);
                if ~isequal(residObj, py.None)
                    result.residuals = pyBridge.DataConverter.numpy2Array(residObj);
                end
                
                % Fitted values
                fittedAttr = py.getattr(pyObj, 'fittedvalues', py.None);
                if ~isequal(fittedAttr, py.None)
                    result.fittedValues = pyBridge.DataConverter.numpy2Array(fittedAttr);
                end
                
                % F-statistic
                fvalueAttr = py.getattr(pyObj, 'fvalue', py.None);
                if ~isequal(fvalueAttr, py.None)
                    result.fStatistic = double(fvalueAttr);
                    fpvalueAttr = py.getattr(pyObj, 'f_pvalue', py.None);
                    if ~isequal(fpvalueAttr, py.None)
                        result.fPValue = double(fpvalueAttr);
                    end
                end
                
                % Marginal effects (for discrete choice models)
                getMargeffMethod = py.getattr(pyObj, 'get_margeff', py.None);
                if ~isequal(getMargeffMethod, py.None) && py.builtins.callable(getMargeffMethod)
                    try
                        margEffObj = getMargeffMethod();
                        margeffAttr = py.getattr(margEffObj, 'margeff', py.None);
                        if ~isequal(margeffAttr, py.None)
                            result.marginalEffects = pyBridge.DataConverter.numpy2Array(margeffAttr);
                        end
                        margeffSeAttr = py.getattr(margEffObj, 'margeff_se', py.None);
                        if ~isequal(margeffSeAttr, py.None)
                            result.marginalEffectsSE = pyBridge.DataConverter.numpy2Array(margeffSeAttr);
                        end
                        margeffPvAttr = py.getattr(margEffObj, 'pvalues', py.None);
                        if ~isequal(margeffPvAttr, py.None)
                            result.marginalEffectsP = pyBridge.DataConverter.numpy2Array(margeffPvAttr);
                        end
                        margeffTAttr = py.getattr(margEffObj, 'tvalues', py.None);
                        if ~isequal(margeffTAttr, py.None)
                            result.marginalEffectsT = pyBridge.DataConverter.numpy2Array(margeffTAttr);
                        end
                    catch
                        % Marginal effects not available for this model
                        result.marginalEffects = [];
                    end
                else
                    result.marginalEffects = [];
                end
                
            catch ME
                % Try generic parsing
                result = pyBridge.ResultParser.parseGeneric(pyObj);
                result.parseError = ME.message;
            end
        end
        
        function result = parseEconML(pyResult)
            % PARSEECONML Parse econml return result
            %   Processes causal inference results like Average Treatment Effect (ATE)
            
            result = struct();
            
            try
                % Check if dictionary type result
                if py_builtin.hasattr(pyResult, 'items')
                    % Dictionary format
                    keys = cell(pyResult.keys());
                    for i = 1:length(keys)
                        key = keys{i};
                        value = py.operator.getitem(pyResult, key);
                        
                        if isa(value, 'py.numpy.ndarray')
                            result.(key) = pyBridge.DataConverter.numpy2Array(value);
                        elseif isa(value, 'py.pandas.DataFrame')
                            result.(key) = pyBridge.DataConverter.df2Table(value);
                        else
                            result.(key) = pyBridge.DataConverter.toMatlab(value);
                        end
                    end
                else
                    % Array or other types
                    result = pyBridge.DataConverter.toMatlab(pyResult);
                end
                
            catch ME
                result = pyBridge.ResultParser.parseGeneric(pyResult);
                result.parseError = ME.message;
            end
        end
        
        function result = parseEconMLModel(pyModel)
            % PARSEECONMLMODEL Parse econml model object
            %   Extracts model information and estimation results
            
            result = struct();
            
            try
                % Model basic information
                result.modelType = char(py.getattr(py.type(pyModel), '__name__'));
                
                % Extract common attributes (use py.getattr for underscore-prefixed attrs)
                if py_builtin.hasattr(pyModel, '_cate_treatment_names')
                    treatNames = py.getattr(pyModel, '_cate_treatment_names');
                    result.treatmentNames = cell(treatNames.tolist());
                end
                
                if py_builtin.hasattr(pyModel, '_cate_outcome_names')
                    outcomeNames = py.getattr(pyModel, '_cate_outcome_names');
                    result.outcomeNames = cell(outcomeNames.tolist());
                end
                
                % Try to extract feature importance
                if py_builtin.hasattr(pyModel, 'feature_importances_')
                    result.featureImportance = ...
                        pyBridge.DataConverter.numpy2Array(pyModel.feature_importances_);
                end
                
            catch ME
                result.parseError = ME.message;
            end
        end
        
        function result = parseLinearmodels(pyObj)
            % PARSELINEARMODELS Parse linearmodels return object
            %   Panel data regression, IV regression and other results
            %   Uses py.getattr for safe attribute access with isa check
            
            result = struct();
            
            % Basic information
            try
                nobsAttr = py.getattr(pyObj, 'nobs', py.None);
                if ~isa(nobsAttr, 'py.NoneType')
                    result.nObs = double(nobsAttr);
                end
            catch
            end
            
            try
                entityInfoAttr = py.getattr(pyObj, 'entity_info', py.None);
                if ~isa(entityInfoAttr, 'py.NoneType')
                    totalAttr = py.getattr(entityInfoAttr, 'total', py.None);
                    if ~isa(totalAttr, 'py.NoneType')
                        result.nEntities = double(totalAttr);
                    end
                end
            catch
            end
            
            try
                timeInfoAttr = py.getattr(pyObj, 'time_info', py.None);
                if ~isa(timeInfoAttr, 'py.NoneType')
                    totalAttr = py.getattr(timeInfoAttr, 'total', py.None);
                    if ~isa(totalAttr, 'py.NoneType')
                        result.nTimes = double(totalAttr);
                    end
                end
            catch
            end
            
            % Fit statistics
            try
                rsquaredAttr = py.getattr(pyObj, 'rsquared', py.None);
                if ~isa(rsquaredAttr, 'py.NoneType')
                    result.rSquared = double(rsquaredAttr);
                end
            catch
            end
            
            try
                rsquaredWithinAttr = py.getattr(pyObj, 'rsquared_within', py.None);
                if ~isa(rsquaredWithinAttr, 'py.NoneType')
                    result.rSquaredWithin = double(rsquaredWithinAttr);
                end
            catch
            end
            
            try
                rsquaredBetweenAttr = py.getattr(pyObj, 'rsquared_between', py.None);
                if ~isa(rsquaredBetweenAttr, 'py.NoneType')
                    result.rSquaredBetween = double(rsquaredBetweenAttr);
                end
            catch
            end
            
            try
                rsquaredOverallAttr = py.getattr(pyObj, 'rsquared_overall', py.None);
                if ~isa(rsquaredOverallAttr, 'py.NoneType')
                    result.rSquaredOverall = double(rsquaredOverallAttr);
                end
            catch
            end
            
            % Coefficients
            try
                paramsAttr = py.getattr(pyObj, 'params', py.None);
                if ~isa(paramsAttr, 'py.NoneType')
                    result.params = pyBridge.DataConverter.numpy2Array(paramsAttr);
                    % Parameter names
                    try
                        indexAttr = py.getattr(paramsAttr, 'index', py.None);
                        if ~isa(indexAttr, 'py.NoneType')
                            result.paramNames = cell(indexAttr.tolist());
                        end
                    catch
                    end
                end
            catch
            end
            
            try
                stdErrorsAttr = py.getattr(pyObj, 'std_errors', py.None);
                if ~isa(stdErrorsAttr, 'py.NoneType')
                    result.stdErrors = pyBridge.DataConverter.numpy2Array(stdErrorsAttr);
                end
            catch
            end
            
            try
                tstatsAttr = py.getattr(pyObj, 'tstats', py.None);
                if ~isa(tstatsAttr, 'py.NoneType')
                    result.tStatistics = pyBridge.DataConverter.numpy2Array(tstatsAttr);
                end
            catch
            end
            
            try
                pvaluesAttr = py.getattr(pyObj, 'pvalues', py.None);
                if ~isa(pvaluesAttr, 'py.NoneType')
                    result.pValues = pyBridge.DataConverter.numpy2Array(pvaluesAttr);
                end
            catch
            end
            
            % F-statistic
            try
                fstatAttr = py.getattr(pyObj, 'f_statistic', py.None);
                if ~isa(fstatAttr, 'py.NoneType')
                    statAttr = py.getattr(fstatAttr, 'stat', py.None);
                    if ~isa(statAttr, 'py.NoneType')
                        result.fStatistic = double(statAttr);
                    end
                    pvalAttr = py.getattr(fstatAttr, 'pval', py.None);
                    if ~isa(pvalAttr, 'py.NoneType')
                        result.fPValue = double(pvalAttr);
                    end
                end
            catch
            end
        end
        
        function result = parseSummary(pyObj)
            % PARSESUMMARY Parse statsmodels summary object
            %   Extracts text summary and table data
            
            result = struct();
            
            try
                % Text summary
                result.text = char(pyObj.as_text());
                
                % Table data
                if py_builtin.hasattr(pyObj, 'tables')
                    numTables = length(pyObj.tables);
                    result.tables = cell(numTables, 1);
                    
                    for i = 1:numTables
                        tableObj = py.operator.getitem(pyObj.tables, int32(i-1));
                        if isa(tableObj, 'py.pandas.DataFrame')
                            result.tables{i} = pyBridge.DataConverter.df2Table(tableObj);
                        else
                            result.tables{i} = char(tableObj);
                        end
                    end
                end
                
            catch ME
                result.parseError = ME.message;
                result.text = char(pyObj);
            end
        end
        
        function result = parseArray(pyObj)
            % PARSEARRAY Parse NumPy array
            %   Converts to MATLAB array and extracts statistics
            
            result = struct();
            
            % Numeric array
            result.data = pyBridge.DataConverter.numpy2Array(pyObj);
            
            % Dimension information
            result.shape = cellfun(@double, cell(pyObj.shape));
            result.ndim = double(pyObj.ndim);
            result.dtype = char(pyObj.dtype);
            
            % Basic statistics
            if pyObj.ndim == 2 && int64(py.operator.getitem(pyObj.shape, int32(1))) == 1
                % Column vector
                result.mean = double(pyObj.mean());
                result.std = double(pyObj.std());
                result.min = double(pyObj.min());
                result.max = double(pyObj.max());
            end
        end
        
        function result = parseGeneric(pyObj)
            % PARSEGENERIC Generic parsing method
            %   Attempts to extract all accessible attributes
            
            result = struct();
            
            try
                % Get object type
                result.pythonType = char(py.getattr(py.type(pyObj), '__name__'));
                result.pythonModule = char(py.getattr(py.type(pyObj), '__module__'));
                
                % Try to convert to MATLAB type
                try
                    mlData = pyBridge.DataConverter.toMatlab(pyObj);
                    result.data = mlData;
                catch
                    % Direct conversion failed, extract attributes
                end
                
                % Extract public attributes
                attrNames = py_builtin.dir(pyObj);
                numAttrs = length(attrNames);
                
                for i = 1:numAttrs
                    attrName = char(py.operator.getitem(attrNames, int32(i-1)));
                    
                    % Skip private attributes and special methods
                    if startsWith(attrName, "_")
                        continue;
                    end
                    
                    try
                        attrValue = pyObj.(attrName);
                        
                        % Skip methods
                        if py_builtin.callable(attrValue)
                            continue;
                        end
                        
                        % Convert attribute value
                        result.(attrName) = pyBridge.DataConverter.toMatlab(attrValue);
                        
                    catch
                        % Ignore inaccessible attributes
                    end
                end
                
            catch ME
                result.parseError = ME.message;
                result.originalObject = pyObj;
            end
        end
        
        function attrValue = extractAttributes(pyObj, attrNames)
            % EXTRACTATTRIBUTES Extract specified attributes
            %   pyObj - Python object
            %   attrNames - Attribute name list (cell array or string array)
            %
            % Returns: Structure containing specified attributes
            
            arguments
                pyObj
                attrNames
            end
            
            if ischar(attrNames)
                attrNames = {attrNames};
            elseif isstring(attrNames)
                attrNames = cellstr(attrNames);
            end
            
            attrValue = struct();
            
            for i = 1:length(attrNames)
                attrName = attrNames{i};
                
                try
                    value = pyObj.(attrName);
                    attrValue.(attrName) = pyBridge.DataConverter.toMatlab(value);
                catch
                    attrValue.(attrName) = NaN; % Attribute does not exist
                end
            end
        end
        
        function printResult(result, maxDepth)
            % PRINTRESULT Print parsed result
            %   maxDepth - Maximum recursion depth (default 2)
            
            arguments
                result
                maxDepth double = 2
            end
            
            pyBridge.ResultParser.printStruct(result, 0, maxDepth);
        end
        
        function printStruct(s, depth, maxDepth)
            % PRINTSTRUCT Recursively print structure
            
            if depth > maxDepth
                fprintf("%s...\n", repmat("  ", 1, depth));
                return;
            end
            
            fn = fieldnames(s);
            for i = 1:length(fn)
                value = s.(fn{i});
                
                if isstruct(value)
                    fprintf("%s%s: struct\n", repmat("  ", 1, depth), fn{i});
                    pyBridge.ResultParser.printStruct(value, depth+1, maxDepth);
                elseif istable(value)
                    fprintf("%s%s: table [%dx%d]\n", repmat("  ", 1, depth), ...
                        fn{i}, height(value), width(value));
                elseif isnumeric(value)
                    if isscalar(value)
                        fprintf("%s%s: %.6f\n", repmat("  ", 1, depth), fn{i}, value);
                    else
                        fprintf("%s%s: array [%s]\n", repmat("  ", 1, depth), ...
                            fn{i}, strjoin(cellfun(@num2str, num2cell(size(value)), ...
                            'UniformOutput', false), "x"));
                    end
                elseif ischar(value) || isstring(value)
                    valueStr = char(value);
                    truncLen = min(50, length(valueStr));
                    fprintf('%s%s: "%s"\n', repmat('  ', 1, depth), fn{i}, valueStr(1:truncLen));
                else
                    fprintf('%s%s: %s\n', repmat('  ', 1, depth), fn{i}, class(value));
                end
            end
        end
    end
end
