classdef RegressionTable
    % REGRESSIONTABLE Outreg2-style regression results table output class
    %   Consolidates multiple regression model results into publication-ready tables, similar to Stata outreg2
    %
    % Features:
    %   - Coefficients centered, standard errors in parentheses
    %   - Significance stars: * p<0.1, ** p<0.05, *** p<0.01
    %   - Supports MLogit: one column per category (excluding reference level)
    %   - Supports ZINB: displays count and zero-inflation parts separately
    %   - Supports multiple output formats: console, CSV, LaTeX, Excel
    %   - Excel output defaults to Times New Roman font
    %
    % Example:
    %   tbl = pyBridge.RegressionTable();
    %   tbl.addModel("OLS", result1, varNames={"X1", "X2"});
    %   tbl.addMLogit("MLogit", mlogitResult, categoryNames={"Category1", "Category2"});
    %   tbl.addZINB("ZINB", zinbResult, varNames={"X1", "X2"});
    %   tbl.display();
    %   tbl.toExcel("results.xlsx");
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    properties
        models cell = {}            % Store all model results
        modelNames string = []      % Model names
        allVarNames string = []     % All variable names (union)
        starLevels = [0.1, 0.05, 0.01]  % Significance levels
        starSymbols = {"*", "**", "***"}
        decimalCoef = 3             % Coefficient decimal places
        decimalSE = 3               % Standard error decimal places
        decimalStat = 2             % Statistics decimal places
        showStars logical = true    % Whether to show stars
        fontName char = "Times New Roman"  % Excel default font
    end
    
    properties(Access = private)
        varOrder string = []        % Variable display order
        columnHeaders cell = {}     % Column headers (supports multiple rows, e.g. MLogit category names)
        columnSubHeaders cell = {}  % Column sub-headers
        isMultiColumnModel logical = false  % Whether there are multi-column models (e.g. MLogit)
    end
    
    methods
        function obj = RegressionTable()
            % REGRESSIONTABLE Constructor
        end
        
        function obj = addModel(obj, modelName, result, options)
            % ADDMODEL Add regression model result (standard single-column model)
            %
            % Parameters:
            %   modelName - Model name (e.g. "OLS", "FE", "IV")
            %   result - Regression result structure (from StatsmodelsWrapper)
            %   options.varNames - Variable names (cell or string array)
            %   options.note - Model note (e.g. "Firm fixed effects")
            %   options.covType - Standard error type description
            
            arguments
                obj
                modelName char
                result struct
                options.varNames cell = {}
                options.note char = ""
                options.covType char = ""
            end
            
            % Extract coefficients and standard errors
            if isfield(result, 'params')
                params = result.params;
            else
                error("pyBridge:MissingParams", "Missing params field in result");
            end
            
            % Standard errors: prefer stdErrors, then bse
            if isfield(result, 'stdErrors')
                stdErrors = result.stdErrors;
            elseif isfield(result, 'bse')
                stdErrors = result.bse;
            else
                stdErrors = zeros(size(params));
            end
            
            % Ensure vectors
            params = params(:);
            stdErrors = stdErrors(:);
            
            % p-values
            if isfield(result, 'pValues')
                pValues = result.pValues(:);
            elseif isfield(result, 'tValues') && isfield(result, 'dfResiduals')
                tStats = result.tValues(:);
                pValues = 2 * (1 - tcdf(abs(tStats), result.dfResiduals));
            else
                pValues = ones(size(params));
            end
            
            % Variable names
            if ~isempty(options.varNames)
                varNames = string(options.varNames);
            elseif isfield(result, 'paramNames')
                varNames = string(result.paramNames);
            else
                varNames = "X" + string(1:length(params));
            end
            
            % Statistics
            stats = obj.extractStats(result);
            
            % Model type
            if isfield(result, 'modelType')
                modelType = result.modelType;
            else
                modelType = "Unknown";
            end
            
            % Store model
            modelData = struct();
            modelData.name = string(modelName);
            modelData.params = params;
            modelData.stdErrors = stdErrors;
            modelData.pValues = pValues;
            modelData.varNames = varNames;
            modelData.stats = stats;
            modelData.note = options.note;
            modelData.covType = options.covType;
            modelData.modelType = modelType;
            modelData.nColumns = 1;  % Single column model
            modelData.columnNames = {string(modelName)};
            
            obj.models{end+1} = modelData;
            obj.modelNames = [obj.modelNames, string(modelName)];
            
            % Update variable list
            obj = obj.updateVarList(varNames);
        end
        
        function obj = addMLogit(obj, modelName, result, options)
            % ADDMLOGIT Add multinomial logit model result
            %   One column per category (excluding reference level), similar to outreg2 style
            %
            % Parameters:
            %   modelName - Model name prefix
            %   result - MLogit regression result
            %   options.varNames - Variable names (excluding constant)
            %   options.categoryNames - Category names (cell array, excluding reference)
            %   options.referenceName - Reference category name (for display)
            %   options.covType - Standard error type description
            %
            % Example:
            %   tbl.addMLogit("MLogit", result, varNames={"X1", "X2"}, ...
            %       categoryNames={"Medium", "High"});
            
            arguments
                obj
                modelName char
                result struct
                options.varNames cell = {}
                options.categoryNames cell = {}
                options.referenceName char = "Baseline"
                options.note char = ""
                options.covType char = ""
            end
            
            % Check model type
            modelType = "MultinomialLogit";
            if isfield(result, 'modelType')
                modelType = result.modelType;
            end
            
            % Extract parameter matrix
            if isfield(result, 'params')
                paramsMatrix = result.params;
            else
                error("pyBridge:MissingParams", "Missing params field in result");
            end
            
            % If params is a vector, convert to matrix
            if isvector(paramsMatrix)
                % Try to infer number of categories
                if isfield(result, 'nCategories')
                    nCategories = result.nCategories;
                    nCols = nCategories - 1;
                else
                    nCols = 1;
                end
                paramsMatrix = reshape(paramsMatrix, [], nCols);
            end
            
            [nParamsPerCat, nCategories] = size(paramsMatrix);
            
            % Standard errors
            if isfield(result, 'stdErrors')
                seMatrix = result.stdErrors;
                if isvector(seMatrix)
                    seMatrix = reshape(seMatrix, [], nCategories);
                end
            elseif isfield(result, 'bse')
                seMatrix = result.bse;
                if isvector(seMatrix)
                    seMatrix = reshape(seMatrix, [], nCategories);
                end
            else
                seMatrix = zeros(nParamsPerCat, nCategories);
            end
            
            % p-values
            if isfield(result, 'pValues')
                pMatrix = result.pValues;
                if isvector(pMatrix)
                    pMatrix = reshape(pMatrix, [], nCategories);
                end
            else
                pMatrix = ones(nParamsPerCat, nCategories);
            end
            
            % Variable names
            if ~isempty(options.varNames)
                varNames = string(options.varNames);
                % Add constant term
                if length(varNames) < nParamsPerCat
                    varNames = ["const", varNames];
                end
            elseif isfield(result, 'paramNames')
                varNames = string(result.paramNames);
            else
                varNames = "X" + string(1:nParamsPerCat);
            end
            
            % Category names
            if ~isempty(options.categoryNames)
                categoryNames = string(options.categoryNames);
            else
                categoryNames = "Cat" + string(1:nCategories);
            end
            
            % Statistics
            stats = obj.extractStats(result);
            
            % Store as multi-column model
            modelData = struct();
            modelData.name = string(modelName);
            modelData.modelType = modelType;
            modelData.nColumns = nCategories;
            modelData.columnNames = cellstr(categoryNames);
            
            % Parameters for each category
            modelData.paramsByColumn = cell(nCategories, 1);
            modelData.stdErrorsByColumn = cell(nCategories, 1);
            modelData.pValuesByColumn = cell(nCategories, 1);
            
            for k = 1:nCategories
                modelData.paramsByColumn{k} = paramsMatrix(:, k);
                modelData.stdErrorsByColumn{k} = seMatrix(:, k);
                modelData.pValuesByColumn{k} = pMatrix(:, k);
            end
            
            modelData.varNames = varNames;
            modelData.stats = stats;
            modelData.note = options.note;
            modelData.covType = options.covType;
            modelData.referenceName = options.referenceName;
            
            obj.models{end+1} = modelData;
            obj.isMultiColumnModel = true;
            
            % Model names: add column name for each category
            for k = 1:nCategories
                obj.modelNames = [obj.modelNames, modelName + "_" + categoryNames(k)];
            end
            
            obj = obj.updateVarList(varNames);
        end
        
        function obj = addZINB(obj, modelName, result, options)
            % ADDZINB Add zero-inflated negative binomial model result
            %   Displays in two parts: count part and zero-inflation part
            %
            % Parameters:
            %   modelName - Model name
            %   result - ZINB regression result
            %   options.varNames - Variable names (count part)
            %   options.inflateVarNames - Zero-inflation part variable names (optional)
            %   options.note - Model note
            %   options.covType - Standard error type description
            
            arguments
                obj
                modelName char
                result struct
                options.varNames cell = {}
                options.inflateVarNames cell = {}
                options.note char = ""
                options.covType char = ""
            end
            
            % Extract count part parameters
            if isfield(result, 'paramsCount')
                paramsCount = result.paramsCount(:);
            elseif isfield(result, 'params')
                % Assume first nCount are count part
                paramsCount = result.params(:);
            else
                error("pyBridge:MissingParams", "Missing paramsCount field in result");
            end
            
            nCount = length(paramsCount);
            
            % Zero-inflation part parameters
            if isfield(result, 'paramsInflate')
                paramsInflate = result.paramsInflate(:);
            else
                paramsInflate = [];
            end
            nInflate = length(paramsInflate);
            
            % Standard errors
            if isfield(result, 'stdErrors')
                allSE = result.stdErrors(:);
                seCount = allSE(1:nCount);
                if nInflate > 0 && length(allSE) >= nCount + nInflate
                    seInflate = allSE(nCount+1:nCount+nInflate);
                else
                    seInflate = zeros(nInflate, 1);
                end
            else
                seCount = zeros(nCount, 1);
                seInflate = zeros(nInflate, 1);
            end
            
            % p-values
            if isfield(result, 'pValues')
                allP = result.pValues(:);
                pCount = allP(1:nCount);
                if nInflate > 0 && length(allP) >= nCount + nInflate
                    pInflate = allP(nCount+1:nCount+nInflate);
                else
                    pInflate = ones(nInflate, 1);
                end
            else
                pCount = ones(nCount, 1);
                pInflate = ones(nInflate, 1);
            end
            
            % Variable names
            if ~isempty(options.varNames)
                varNamesCount = string(options.varNames);
                if length(varNamesCount) < nCount
                    varNamesCount = ["const", varNamesCount];
                end
            else
                varNamesCount = "X" + string(1:nCount);
            end
            
            if ~isempty(options.inflateVarNames)
                varNamesInflate = string(options.inflateVarNames);
            elseif isfield(result, 'inflateParamNames')
                varNamesInflate = string(result.inflateParamNames);
            else
                varNamesInflate = "Z_" + string(1:nInflate);
            end
            
            % Statistics
            stats = obj.extractStats(result);
            if isfield(result, 'alpha')
                stats.alpha = result.alpha;
            end
            if isfield(result, 'vuongTest')
                stats.vuongStat = result.vuongTest.statistic;
            end
            
            % Store model
            modelData = struct();
            modelData.name = string(modelName);
            modelData.modelType = "ZeroInflatedNB";
            modelData.nColumns = 1;
            modelData.columnNames = {string(modelName)};
            
            % Count part
            modelData.params = paramsCount;
            modelData.stdErrors = seCount;
            modelData.pValues = pCount;
            modelData.varNames = varNamesCount;
            
            % Zero-inflation part
            modelData.hasInflatePart = nInflate > 0;
            modelData.paramsInflate = paramsInflate;
            modelData.stdErrorsInflate = seInflate;
            modelData.pValuesInflate = pInflate;
            modelData.varNamesInflate = varNamesInflate;
            
            modelData.stats = stats;
            modelData.note = options.note;
            modelData.covType = options.covType;
            
            obj.models{end+1} = modelData;
            obj.modelNames = [obj.modelNames, string(modelName)];
            
            obj = obj.updateVarList(varNamesCount);
            if nInflate > 0
                obj = obj.updateVarList(varNamesInflate);
            end
        end
        
        function obj = setVarOrder(obj, varNames)
            % SETVARORDER Set variable display order
            
            arguments
                obj
                varNames
            end
            
            if iscell(varNames) || isnumeric(varNames)
                obj.varOrder = string(varNames);
            else
                obj.varOrder = varNames(:)';  % Already string, ensure row vector
            end
        end
        
        function display(obj) %#ok<DISPLAY>
            % DISPLAY Display table in console (overloaded intentionally)
            
            if isempty(obj.models)
                disp("No model data");
                return;
            end
            
            tableStr = obj.buildConsoleTable();
            disp(tableStr);
        end
        
        function toCSV(obj, filePath)
            % TOCSV Export to CSV file
            
            arguments
                obj
                filePath char
            end
            
            if isempty(obj.models)
                error("pyBridge:NoData", "No model data to export");
            end
            
            [data, headers] = obj.buildTableData();
            
            fid = fopen(filePath, "w", "n", "UTF-8");
            if fid == -1
                error("pyBridge:FileError", "Cannot create file: %s", filePath);
            end
            
            fprintf(fid, "%s\n", strjoin(headers, ","));
            for i = 1:size(data, 1)
                fprintf(fid, "%s\n", strjoin(data(i, :), ","));
            end
            fclose(fid);
            fprintf("Exported to: %s\n", filePath);
        end
        
        function toLaTeX(obj, filePath, options)
            % TOLATEX Export to LaTeX table
            
            arguments
                obj
                filePath char
                options.caption char = "Regression Results"
                options.label char = "tab:regression"
                options.booktabs logical = true
            end
            
            if isempty(obj.models)
                error("pyBridge:NoData", "No model data to export");
            end
            
            latexStr = obj.buildLaTeXTable(options);
            
            fid = fopen(filePath, "w", "n", "UTF-8");
            fprintf(fid, "%s", latexStr);
            fclose(fid);
            fprintf("Exported LaTeX table to: %s\n", filePath);
        end
        
        function toExcel(obj, filePath)
            % TOEXCEL Export to Excel file (with formatting)
            
            arguments
                obj
                filePath char
            end
            
            if isempty(obj.models)
                error("pyBridge:NoData", "No model data to export");
            end
            
            obj.writeExcelWithFormat(filePath);
            fprintf("Exported Excel to: %s\n", filePath);
        end
    end
    
    methods(Access = private)
        function stats = extractStats(~, result)
            % Extract statistics from result
            stats = struct();
            if isfield(result, 'nObs')
                stats.nObs = result.nObs;
            end
            if isfield(result, 'rSquared')
                stats.rSquared = result.rSquared;
            end
            if isfield(result, 'adjRSquared')
                stats.adjRSquared = result.adjRSquared;
            end
            if isfield(result, 'aic')
                stats.aic = result.aic;
            end
            if isfield(result, 'bic')
                stats.bic = result.bic;
            end
            if isfield(result, 'fStatistic')
                stats.fStatistic = result.fStatistic;
            end
            if isfield(result, 'nClusters')
                stats.nClusters = result.nClusters;
            end
            if isfield(result, 'nCategories')
                stats.nCategories = result.nCategories;
            end
        end
        
        function obj = updateVarList(obj, newVarNames)
            % Update variable list
            for i = 1:length(newVarNames)
                vn = newVarNames(i);
                if ~ismember(vn, obj.allVarNames)
                    obj.allVarNames = [obj.allVarNames, vn];
                end
            end
            if isempty(obj.varOrder)
                obj.varOrder = obj.allVarNames;
            end
        end
        
        function [nTotalCols, colModelIdx, colSubIdx] = getTotalColumns(obj)
            % Calculate total columns
            nTotalCols = 0;
            modelIdx = {};
            subIdx = {};
            
            for j = 1:length(obj.models)
                model = obj.models{j};
                nCols = model.nColumns;
                for k = 1:nCols
                    nTotalCols = nTotalCols + 1;
                    modelIdx{end+1} = j; %#ok<AGROW>
                    subIdx{end+1} = k; %#ok<AGROW>
                end
            end
            
            colModelIdx = cell2mat(modelIdx);
            colSubIdx = cell2mat(subIdx);
        end
        
        function tableStr = buildConsoleTable(obj)
            % Build console table
            
            [nTotalCols, colModelIdx, colSubIdx] = obj.getTotalColumns();
            nVars = length(obj.varOrder);
            
            % Calculate column width
            varColWidth = max([12, max(strlength(obj.varOrder)) + 2]);
            modelColWidth = 14;
            
            % Collect column titles
            colTitles = cell(nTotalCols, 1);
            colSubTitles = cell(nTotalCols, 1);
            
            colIdx = 1;
            for j = 1:length(obj.models)
                model = obj.models{j};
                if model.nColumns == 1
                    colTitles{colIdx} = string(model.name);
                    colSubTitles{colIdx} = "";
                    colIdx = colIdx + 1;
                else
                    % MLogit: main title is model name, subtitle is category name
                    for k = 1:model.nColumns
                        colTitles{colIdx} = string(model.name);
                        colSubTitles{colIdx} = model.columnNames{k};
                        colIdx = colIdx + 1;
                    end
                end
            end
            
            headerLine = repmat("-", 1, varColWidth + (modelColWidth + 2) * nTotalCols);
            
            lines = {};
            lines{end+1} = "";
            lines{end+1} = headerLine;
            
            % Main title row
            headerStr = sprintf("%-*s", varColWidth, "");
            for c = 1:nTotalCols
                headerStr = headerStr + sprintf("  %*s", modelColWidth, colTitles{c});
            end
            lines{end+1} = headerStr;
            
            % Sub-title row (if MLogit exists)
            if obj.isMultiColumnModel
                subHeaderStr = sprintf("%-*s", varColWidth, "");
                for c = 1:nTotalCols
                    subHeaderStr = subHeaderStr + sprintf("  %*s", modelColWidth, colSubTitles{c});
                end
                lines{end+1} = subHeaderStr;
            end
            
            lines{end+1} = headerLine;
            
            % Variable coefficients and standard errors
            for i = 1:nVars
                varName = obj.varOrder(i);
                
                % Coefficient row
                coefStr = sprintf("%-*s", varColWidth, varName);
                seStr = sprintf("%-*s", varColWidth, "");
                
                for c = 1:nTotalCols
                    modelIdx = colModelIdx(c);
                    subIdx = colSubIdx(c);
                    model = obj.models{modelIdx};
                    
                    [coef, se, p] = obj.getCoefSE(model, varName, subIdx);
                    
                    if ~isempty(coef)
                        stars = obj.getStars(p);
                        if obj.showStars
                            coefFormatted = sprintf("%.*f%s", obj.decimalCoef, coef, stars);
                        else
                            coefFormatted = sprintf("%.*f", obj.decimalCoef, coef);
                        end
                        seFormatted = sprintf("(%.*f)", obj.decimalSE, se);
                        
                        coefStr = coefStr + sprintf("  %*s", modelColWidth, coefFormatted);
                        seStr = seStr + sprintf("  %*s", modelColWidth, seFormatted);
                    else
                        coefStr = coefStr + sprintf("  %*s", modelColWidth, "");
                        seStr = seStr + sprintf("  %*s", modelColWidth, "");
                    end
                end
                
                lines{end+1} = coefStr; %#ok<AGROW>
                lines{end+1} = seStr; %#ok<AGROW>
            end
            
            % Separator line
            lines{end+1} = headerLine;
            
            % Statistics
            lines = obj.addStatsToConsole(lines, varColWidth, modelColWidth, nTotalCols, colModelIdx);
            
            lines{end+1} = headerLine;
            
            if obj.showStars
                lines{end+1} = sprintf("*** p<0.01, ** p<0.05, * p<0.1");
            end
            
            tableStr = strjoin(lines, "\n");
        end
        
        function [coef, se, p] = getCoefSE(~, model, varName, subIdx)
            % Get coefficient and standard error for specified variable from model
            coef = [];
            se = [];
            p = [];
            
            if isfield(model, 'paramsByColumn') && subIdx > 0
                % Multi-column model (MLogit)
                idx = find(model.varNames == varName);
                if ~isempty(idx)
                    coef = model.paramsByColumn{subIdx}(idx(1));
                    se = model.stdErrorsByColumn{subIdx}(idx(1));
                    p = model.pValuesByColumn{subIdx}(idx(1));
                end
            else
                % Single column model
                idx = find(model.varNames == varName);
                if ~isempty(idx)
                    coef = model.params(idx(1));
                    se = model.stdErrors(idx(1));
                    p = model.pValues(idx(1));
                end
            end
        end
        
        function lines = addStatsToConsole(obj, lines, varColWidth, modelColWidth, nTotalCols, colModelIdx)
            % Add statistics to console output
            
            statLabels = struct();
            statLabels.nObs = "Observations";
            statLabels.rSquared = "R-squared";
            statLabels.adjRSquared = "Adj R-squared";
            statLabels.aic = "AIC";
            statLabels.bic = "BIC";
            statLabels.alpha = "Alpha";
            statLabels.nClusters = "N Clusters";
            
            statFields = fieldnames(statLabels);
            
            for k = 1:length(statFields)
                field = statFields{k};
                label = statLabels.(field);
                
                % Check if any model has this statistic
                hasStat = any(cellfun(@(m) isfield(m.stats, field), obj.models));
                
                if hasStat
                    statStr = sprintf("%-*s", varColWidth, label);
                    
                    for c = 1:nTotalCols
                        modelIdx = colModelIdx(c);
                        model = obj.models{modelIdx};
                        
                        if isfield(model.stats, field)
                            val = model.stats.(field);
                            if strcmp(field, "nObs") || strcmp(field, "nClusters")
                                statStr = statStr + sprintf("  %*d", modelColWidth, round(val));
                            else
                                statStr = statStr + sprintf("  %*.3f", modelColWidth, val);
                            end
                        else
                            statStr = statStr + sprintf("  %*s", modelColWidth, "");
                        end
                    end
                    lines{end+1} = statStr; %#ok<AGROW>
                end
            end
        end
        
        function [data, headers] = buildTableData(obj)
            % Build table data (CSV format)
            
            [nTotalCols, colModelIdx, colSubIdx] = obj.getTotalColumns();
            nVars = length(obj.varOrder);
            
            % Header
            headers = "Variable";
            for c = 1:nTotalCols
                model = obj.models{colModelIdx(c)};
                if model.nColumns == 1
                    headers = [headers, string(model.name)]; %#ok<AGROW>
                else
                    headers = [headers, string(model.name) + "_" + model.columnNames{colSubIdx(c)}]; %#ok<AGROW>
                end
            end
            
            data = {};
            
            % Variable coefficients and standard errors
            for i = 1:nVars
                varName = obj.varOrder(i);
                
                row = {varName};
                rowSE = {""};
                
                for c = 1:nTotalCols
                    [coef, se, p] = obj.getCoefSE(obj.models{colModelIdx(c)}, varName, colSubIdx(c));
                    
                    if ~isempty(coef)
                        stars = obj.getStars(p);
                        if obj.showStars
                            row{end+1} = sprintf("%.*f%s", obj.decimalCoef, coef, stars); %#ok<AGROW>
                        else
                            row{end+1} = sprintf("%.*f", obj.decimalCoef, coef); %#ok<AGROW>
                        end
                        rowSE{end+1} = sprintf("(%.*f)", obj.decimalSE, se); %#ok<AGROW>
                    else
                        row{end+1} = ""; %#ok<AGROW>
                        rowSE{end+1} = ""; %#ok<AGROW>
                    end
                end
                data{end+1, :} = row; %#ok<AGROW>
                data{end+1, :} = rowSE; %#ok<AGROW>
            end
            
            % Statistics
            statLabels = struct();
            statLabels.nObs = "Observations";
            statLabels.aic = "AIC";
            statLabels.bic = "BIC";
            
            statFields = fieldnames(statLabels);
            for k = 1:length(statFields)
                field = statFields{k};
                label = statLabels.(field);
                
                row = {label};
                for c = 1:nTotalCols
                    model = obj.models{colModelIdx(c)};
                    if isfield(model.stats, field)
                        val = model.stats.(field);
                        row{end+1} = sprintf("%.2f", val); %#ok<AGROW>
                    else
                        row{end+1} = ""; %#ok<AGROW>
                    end
                end
                data{end+1, :} = row; %#ok<AGROW>
            end
            
            if obj.showStars
                data{end+1, :} = {"*** p<0.01, ** p<0.05, * p<0.1", repmat({""}, 1, nTotalCols)};
            end
        end
        
        function latexStr = buildLaTeXTable(obj, options)
            % Build LaTeX table
            
            [nTotalCols, colModelIdx, colSubIdx] = obj.getTotalColumns();
            nVars = length(obj.varOrder);
            
            if options.booktabs
                topRule = "\toprule";
                midRule = "\midrule";
                bottomRule = "\bottomrule";
            else
                topRule = "\hline";
                midRule = "\hline";
                bottomRule = "\hline";
            end
            
            lines = {};
            lines{end+1} = sprintf("\\begin{table}[htbp]");
            lines{end+1} = sprintf("\\centering");
            lines{end+1} = sprintf("\\caption{%s}", options.caption);
            lines{end+1} = sprintf("\\label{%s}", options.label);
            lines{end+1} = sprintf("\\begin{tabular}{l%s}", repmat("c", 1, nTotalCols));
            lines{end+1} = topRule;
            
            % Header
            headerStr = "";
            for c = 1:nTotalCols
                model = obj.models{colModelIdx(c)};
                if model.nColumns == 1
                    headerStr = headerStr + " & " + string(model.name);
                else
                    headerStr = headerStr + " & " + model.columnNames{colSubIdx(c)};
                end
            end
            headerStr = headerStr + " \\\\";
            lines{end+1} = headerStr;
            lines{end+1} = midRule;
            
            % Variable coefficients
            for i = 1:nVars
                varName = obj.varOrder(i);
                varNameEsc = obj.escapeLaTeX(varName);
                
                coefStr = sprintf("%s", varNameEsc);
                for c = 1:nTotalCols
                    [coef, ~, p] = obj.getCoefSE(obj.models{colModelIdx(c)}, varName, colSubIdx(c));
                    if ~isempty(coef)
                        stars = obj.getStarsLaTeX(p);
                        coefStr = coefStr + sprintf(" & %.*f%s", obj.decimalCoef, coef, stars);
                    else
                        coefStr = coefStr + " & ";
                    end
                end
                coefStr = coefStr + " \\\\";
                lines{end+1} = coefStr; %#ok<AGROW>
                
                seStr = "";
                for c = 1:nTotalCols
                    [~, se] = obj.getCoefSE(obj.models{colModelIdx(c)}, varName, colSubIdx(c));
                    if ~isempty(se)
                        seStr = seStr + sprintf(" & (%.*f)", obj.decimalSE, se);
                    else
                        seStr = seStr + " & ";
                    end
                end
                seStr = seStr + " \\\\";
                lines{end+1} = seStr; %#ok<AGROW>
            end
            
            lines{end+1} = midRule;
            
            % Statistics
            if any(cellfun(@(m) isfield(m.stats, 'nObs'), obj.models))
                statStr = "Observations";
                for c = 1:nTotalCols
                    model = obj.models{colModelIdx(c)};
                    if isfield(model.stats, 'nObs')
                        statStr = statStr + sprintf(" & %d", round(model.stats.nObs));
                    else
                        statStr = statStr + " & ";
                    end
                end
                statStr = statStr + " \\\\";
                lines{end+1} = statStr;
            end
            
            lines{end+1} = bottomRule;
            
            if obj.showStars
                lines{end+1} = sprintf("\\multicolumn{%d}{l}{\\footnotesize *** p<0.01, ** p<0.05, * p<0.1} \\\\", nTotalCols + 1);
            end
            
            lines{end+1} = sprintf("\\end{tabular}");
            lines{end+1} = sprintf("\\end{table}");
            
            latexStr = strjoin(lines, "\n");
        end
        
        function writeExcelWithFormat(obj, filePath)
            % Write to Excel and apply formatting
            
            [nTotalCols, colModelIdx, colSubIdx] = obj.getTotalColumns();
            nVars = length(obj.varOrder);
            
            data = cell(nVars * 2 + 15, nTotalCols + 1);
            rowIdx = 1;
            
            % Header
            data{rowIdx, 1} = "";
            for c = 1:nTotalCols
                model = obj.models{colModelIdx(c)};
                if model.nColumns == 1
                    data{rowIdx, c+1} = string(model.name);
                else
                    data{rowIdx, c+1} = model.columnNames{colSubIdx(c)};
                end
            end
            rowIdx = rowIdx + 1;
            
            % MLogit sub-title row
            if obj.isMultiColumnModel
                data{rowIdx, 1} = "";
                for c = 1:nTotalCols
                    model = obj.models{colModelIdx(c)};
                    if model.nColumns > 1
                        data{rowIdx, c+1} = model.columnNames{colSubIdx(c)};
                    end
                end
                rowIdx = rowIdx + 1;
            end
            
            % Variable coefficients and standard errors
            for i = 1:nVars
                varName = obj.varOrder(i);
                
                data{rowIdx, 1} = varName;
                for c = 1:nTotalCols
                    [coef, ~, p] = obj.getCoefSE(obj.models{colModelIdx(c)}, varName, colSubIdx(c));
                    if ~isempty(coef)
                        stars = obj.getStars(p);
                        data{rowIdx, c+1} = sprintf("%.*f%s", obj.decimalCoef, coef, stars);
                    end
                end
                rowIdx = rowIdx + 1;
                
                data{rowIdx, 1} = "";
                for c = 1:nTotalCols
                    [~, se] = obj.getCoefSE(obj.models{colModelIdx(c)}, varName, colSubIdx(c));
                    if ~isempty(se)
                        data{rowIdx, c+1} = sprintf("(%.*f)", obj.decimalSE, se);
                    end
                end
                rowIdx = rowIdx + 1;
            end
            
            rowIdx = rowIdx + 1;
            
            % Statistics
            statLabels = struct();
            statLabels.nObs = "Observations";
            statLabels.rSquared = "R-squared";
            statLabels.aic = "AIC";
            statLabels.bic = "BIC";
            statLabels.alpha = "Alpha";
            
            statFields = fieldnames(statLabels);
            for k = 1:length(statFields)
                field = statFields{k};
                label = statLabels.(field);
                
                if any(cellfun(@(m) isfield(m.stats, field), obj.models))
                    data{rowIdx, 1} = label;
                    for c = 1:nTotalCols
                        model = obj.models{colModelIdx(c)};
                        if isfield(model.stats, field)
                            data{rowIdx, c+1} = model.stats.(field);
                        end
                    end
                    rowIdx = rowIdx + 1;
                end
            end
            
            if obj.showStars
                rowIdx = rowIdx + 1;
                data{rowIdx, 1} = "*** p<0.01, ** p<0.05, * p<0.1";
            end
            
            data = data(1:rowIdx, :);
            writecell(data, filePath, Sheet="Regression Results");
            
            try
                obj.applyExcelFormat(filePath);
            catch
            end
        end
        
        function applyExcelFormat(obj, filePath)
            % Apply Excel formatting
            
            excel = actxserver("Excel.Application");
            excel.Visible = false;
            
            try
                workbook = excel.Workbooks.Open(fullfile(pwd, filePath));
                sheet = workbook.Sheets.Item("Regression Results");
                usedRange = sheet.UsedRange;
                nRows = usedRange.Rows.Count;
                nCols = usedRange.Columns.Count;
                
                usedRange.Font.Name = obj.fontName;
                usedRange.Font.Size = 11;
                
                headerRange = sheet.Range(sheet.Cells(1, 1), sheet.Cells(1, nCols));
                headerRange.Font.Bold = true;
                
                varColRange = sheet.Range(sheet.Cells(2, 1), sheet.Cells(nRows, 1));
                varColRange.HorizontalAlignment = -4131;
                
                if nCols > 1
                    dataRange = sheet.Range(sheet.Cells(2, 2), sheet.Cells(nRows, nCols));
                    dataRange.HorizontalAlignment = -4108;
                end
                
                usedRange.Borders.LineStyle = 1;
                usedRange.Borders.Weight = 2;
                usedRange.Columns.AutoFit;
                
                workbook.Save;
                workbook.Close;
            catch ME
                warning("Excel formatting failed: %s", ME.message);
            end
            
            excel.Quit;
            delete(excel);
        end
        
        function stars = getStars(obj, p)
            stars = "";
            if ~obj.showStars || isempty(p)
                return;
            end
            if p < obj.starLevels(3)
                stars = obj.starSymbols{3};
            elseif p < obj.starLevels(2)
                stars = obj.starSymbols{2};
            elseif p < obj.starLevels(1)
                stars = obj.starSymbols{1};
            end
        end
        
        function stars = getStarsLaTeX(obj, p)
            stars = "";
            if ~obj.showStars || isempty(p)
                return;
            end
            if p < obj.starLevels(3)
                stars = "$^{***}$";
            elseif p < obj.starLevels(2)
                stars = "$^{**}$";
            elseif p < obj.starLevels(1)
                stars = "$^{*}$";
            end
        end
        
        function escaped = escapeLaTeX(~, str)
            escaped = str;
            escaped = strrep(escaped, "\", "\textbackslash");
            escaped = strrep(escaped, "%", "\%");
            escaped = strrep(escaped, "$", "\$");
            escaped = strrep(escaped, "#", "\#");
            escaped = strrep(escaped, "_", "\_");
            escaped = strrep(escaped, "{", "\{");
            escaped = strrep(escaped, "}", "\}");
            escaped = strrep(escaped, "&", "\&");
        end
    end
end
