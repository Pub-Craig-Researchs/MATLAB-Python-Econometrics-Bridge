classdef DataConverter
    % DATACONVERTER MATLAB-Python bidirectional data conversion tool
    %   Provides automatic conversion between MATLAB data types and Python objects
    %
    % Static Methods:
    %   toPython    - Convert MATLAB data to Python objects
    %   toMatlab    - Convert Python objects to MATLAB data
    %   array2Numpy - MATLAB array to NumPy array
    %   numpy2Array - NumPy array to MATLAB array
    %   table2Df    - MATLAB table to Pandas DataFrame
    %   df2Table    - Pandas DataFrame to MATLAB table
    %   struct2Dict - MATLAB struct to Python dictionary
    %   dict2Struct - Python dictionary to MATLAB struct
    %
    % Example:
    %   % MATLAB -> Python
    %   pyArray = pyBridge.DataConverter.toPython(magic(3));
    %   pyDf = pyBridge.DataConverter.table2Df(tableData);
    %
    %   % Python -> MATLAB
    %   mlArray = pyBridge.DataConverter.toMatlab(pyArray);
    %   mlTable = pyBridge.DataConverter.df2Table(pyDf);
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    methods(Static)
        function pyObj = toPython(mlData, options)
            % TOPYTHON Convert MATLAB data to Python objects
            %   mlData - MATLAB data (array, table, struct, cell, etc.)
            %   options - Optional parameter struct
            %       .dtype - Specify NumPy data type
            %       .copy - Whether to create a copy (default false)
            %
            % Returns: Python object (NumPy array, Pandas DataFrame or dict)
            
            arguments
                mlData
                options.dtype char = ""
                options.copy logical = false
            end
            
            % Check input type
            if isa(mlData, 'numeric')
                % Numeric array
                optArgs = namedargs2cell(options);
                pyObj = pyBridge.DataConverter.array2Numpy(mlData, optArgs{:});
                
            elseif istable(mlData)
                % Table
                pyObj = pyBridge.DataConverter.table2Df(mlData);
                
            elseif isstruct(mlData)
                % Struct
                pyObj = pyBridge.DataConverter.struct2Dict(mlData);
                
            elseif iscell(mlData)
                % Cell array
                pyObj = pyBridge.DataConverter.cell2List(mlData);
                
            elseif ischar(mlData) || isstring(mlData)
                % String
                pyObj = py.str(mlData);
                
            elseif islogical(mlData)
                % Logical array
                optArgs = namedargs2cell(options);
                pyObj = pyBridge.DataConverter.array2Numpy(mlData, optArgs{:});
                
            else
                % Other types, try to pass directly
                pyObj = mlData;
            end
        end
        
        function mlData = toMatlab(pyObj, options)
            % TOMATLAB Convert Python objects to MATLAB data
            %   pyObj - Python object (NumPy array, Pandas DataFrame, dict, etc.)
            %   options - Optional parameter struct
            %       .type - Specify output type ('array', 'table', 'struct')
            %       .rowNames - Whether to preserve row names when converting DataFrame
            %
            % Returns: MATLAB data
            
            arguments
                pyObj
                options.type char = "auto"
                options.rowNames logical = false
            end
            
            if isempty(pyObj)
                mlData = [];
                return;
            end
            
            % Get Python type
            pyType = char(py.getattr(py.type(pyObj), '__name__'));
            
            switch pyType
                case 'ndarray'
                    mlData = pyBridge.DataConverter.numpy2Array(pyObj);
                    
                case 'DataFrame'
                    if strcmpi(options.type, 'array')
                        mlData = pyBridge.DataConverter.numpy2Array(pyObj.values);
                    else
                        optArgs = namedargs2cell(options);
                        mlData = pyBridge.DataConverter.df2Table(pyObj, optArgs{:});
                    end
                    
                case 'Series'
                    mlData = pyBridge.DataConverter.numpy2Array(pyObj.values);
                    
                case 'dict'
                    if strcmpi(options.type, 'struct')
                        mlData = pyBridge.DataConverter.dict2Struct(pyObj);
                    else
                        mlData = pyBridge.DataConverter.dict2Cell(pyObj);
                    end
                    
                case {'list', 'tuple'}
                    mlData = pyBridge.DataConverter.list2Cell(pyObj);
                    
                case 'str'
                    mlData = char(pyObj);
                    
                otherwise
                    % Try to convert
                    try
                        mlData = double(pyObj);
                    catch
                        mlData = pyObj;
                    end
            end
        end
        
        function pyArray = array2Numpy(mlArray, options)
            % ARRAY2NUMPY MATLAB array to NumPy array
            %   Supports automatic type inference and dimension handling
            %   Preserves array dimensions correctly (fixes AxisError issues)
            %
            %   Dimension handling rules:
            %   - Column vectors (n, 1) -> 1D array (n,) - for statsmodels y vectors
            %   - Row vectors (1, n) -> 1D array (n,) - treated as 1D
            %   - 2D matrices (n, k) where k > 1 -> 2D array (n, k)
            %   - Higher dimensional arrays -> preserve original shape
            
            arguments
                mlArray
                options.dtype char = ""
                options.copy logical = false
            end
            
            % Handle empty array
            if isempty(mlArray)
                pyArray = py.numpy.array([]);
                return;
            end
            
            % Get original shape before flattening
            origShape = size(mlArray);
            
            % Flatten to 1D first to avoid MATLAB-Python dimension issues
            flatArray = mlArray(:);
            
            % Convert to NumPy array
            if ~isempty(options.dtype)
                pyArray = py.numpy.array(flatArray, py.dtype(options.dtype));
            else
                pyArray = py.numpy.array(flatArray);
            end
            
            % Reshape based on original dimensions
            % Use Fortran order to match MATLAB's column-major storage
            if numel(origShape) == 2
                if origShape(2) > 1
                    % 2D matrix (n x k, k > 1) - reshape to 2D
                    pyShape = py.tuple(num2cell(int64(origShape)));
                    pyArray = pyArray.reshape(pyShape, pyargs('order', 'F'));
                else
                    % Column vector (n, 1) - ensure it's truly 1D (n,)
                    % MATLAB's py.numpy.array may keep it as (n, 1), so flatten
                    pyArray = pyArray.flatten();
                end
            elseif numel(origShape) > 2
                % Higher dimensional array - reshape to original
                pyShape = py.tuple(num2cell(int64(origShape)));
                pyArray = pyArray.reshape(pyShape, pyargs('order', 'F'));
            end
            % Row vectors (1, n) and scalars remain as 1D after flatten
            
            % Whether to create a copy
            if options.copy
                pyArray = pyArray.copy();
            end
        end
        
        function mlArray = numpy2Array(pyArray)
            % NUMPY2ARRAY NumPy array to MATLAB array
            %   Automatically handles data types and dimensions
            %   Uses double() for direct conversion which is more reliable
            
            if isempty(pyArray)
                mlArray = [];
                return;
            end
            
            % Handle py.None (use isa for safe check)
            if isa(pyArray, 'py.NoneType')
                mlArray = [];
                return;
            end
            
            % Use direct double() conversion which handles py.numpy.ndarray well
            try
                mlArray = double(pyArray);
            catch
                % Fallback: convert via values attribute for pandas Series
                try
                    values = py.getattr(pyArray, 'values', py.None);
                    if ~isa(values, 'py.NoneType')
                        mlArray = double(values);
                    else
                        % Last resort: use tolist
                        tolistMethod = py.getattr(pyArray, 'tolist', py.None);
                        if ~isa(tolistMethod, 'py.NoneType') && py.builtins.callable(tolistMethod)
                            mlArray = cell2mat(cell(tolistMethod()));
                        else
                            mlArray = [];
                        end
                    end
                catch
                    mlArray = [];
                end
            end
        end
        
        function pyDf = table2Df(mlTable)
            % TABLE2DF MATLAB table to Pandas DataFrame
            %   Preserves column names and data types
            
            arguments
                mlTable table
            end
            
            % Get column names
            colNames = mlTable.Properties.VariableNames;
            
            % Create dictionary
            dataDict = py.dict();
            for i = 1:width(mlTable)
                colData = mlTable.(colNames{i});
                
                % Handle different data types
                if iscell(colData)
                    % Cell column to list
                    pyData = pyBridge.DataConverter.cell2List(colData);
                else
                    % Numeric column to NumPy (use array2Numpy for consistency)
                    pyData = pyBridge.DataConverter.array2Numpy(colData);
                end
                
                py.operator.setitem(dataDict, colNames{i}, pyData);
            end
            
            % Create DataFrame
            pyDf = py.pandas.DataFrame(dataDict);
            
            % Add row names (if any)
            if ~isempty(mlTable.Properties.RowNames)
                pyDf.index = py.list(mlTable.Properties.RowNames);
            end
        end
        
        function mlTable = df2Table(pyDf, options)
            % DF2TABLE Pandas DataFrame to MATLAB table
            %   options.rowNames - Whether to preserve row names
            
            arguments
                pyDf
                options.rowNames logical = false
            end
            
            % Get column names
            colNames = cell(pyDf.columns.tolist());
            numCols = length(colNames);
            
            % Get data
            dataCell = cell(1, numCols);
            for i = 1:numCols
                colData = py.operator.getitem(pyDf, colNames{i});
                
                if isa(colData, 'py.pandas.Series')
                    % Check data type
                    dtype = char(colData.dtype);
                    
                    if contains(dtype, 'object') || contains(dtype, 'str')
                        % String or mixed type
                        dataCell{i} = cell(colData.tolist());
                    else
                        % Numeric type
                        dataCell{i} = double(colData.values);
                    end
                else
                    dataCell{i} = double(colData);
                end
            end
            
            % Create table
            mlTable = table(dataCell{:}, VariableNames=colNames);
            
            % Add row names
            if options.rowNames
                rowNames = cell(pyDf.index.tolist());
                mlTable.Properties.RowNames = rowNames;
            end
        end
        
        function pyDict = struct2Dict(mlStruct)
            % STRUCT2DICT MATLAB struct to Python dictionary
            %   Supports nested structs
            
            arguments
                mlStruct struct
            end
            
            if isscalar(mlStruct)
                % Scalar struct
                pyDict = py.dict();
                fn = fieldnames(mlStruct);
                
                for i = 1:length(fn)
                    fieldValue = mlStruct.(fn{i});
                    
                    % Recursively handle nested structs
                    if isstruct(fieldValue)
                        fieldValue = pyBridge.DataConverter.struct2Dict(fieldValue);
                    elseif istable(fieldValue)
                        fieldValue = pyBridge.DataConverter.table2Df(fieldValue);
                    elseif iscell(fieldValue)
                        fieldValue = pyBridge.DataConverter.cell2List(fieldValue);
                    end
                    
                    py.operator.setitem(pyDict, fn{i}, fieldValue);
                end
            else
                % Struct array -> list of dicts
                numStructs = length(mlStruct);
                dictList = py.list();
                
                for j = 1:numStructs
                    dictList.append(pyBridge.DataConverter.struct2Dict(mlStruct(j)));
                end
                
                pyDict = dictList;
            end
        end
        
        function mlStruct = dict2Struct(pyDict)
            % DICT2STRUCT Python dictionary to MATLAB struct
            %   Supports nested dictionaries
            
            arguments
                pyDict
            end
            
            % Get keys
            keys = cell(pyDict.keys());
            
            % Create struct
            for i = 1:length(keys)
                key = keys{i};
                value = py.operator.getitem(pyDict, key);
                
                % Recursively handle nested dictionaries
                pyType = char(py.getattr(py.type(value), '__name__'));
                
                switch pyType
                    case 'dict'
                        mlStruct.(key) = pyBridge.DataConverter.dict2Struct(value);
                    case 'DataFrame'
                        mlStruct.(key) = pyBridge.DataConverter.df2Table(value);
                    case 'ndarray'
                        mlStruct.(key) = pyBridge.DataConverter.numpy2Array(value);
                    case {'list', 'tuple'}
                        mlStruct.(key) = pyBridge.DataConverter.list2Cell(value);
                    otherwise
                        mlStruct.(key) = pyBridge.DataConverter.toMatlab(value);
                end
            end
        end
        
        function pyList = cell2List(mlCell)
            % CELL2LIST MATLAB cell array to Python list
            %   Supports mixed types
            
            arguments
                mlCell cell
            end
            
            pyList = py.list();
            
            for i = 1:length(mlCell)
                item = mlCell{i};
                
                % Recursive conversion
                if iscell(item)
                    item = pyBridge.DataConverter.cell2List(item);
                elseif isstruct(item)
                    item = pyBridge.DataConverter.struct2Dict(item);
                elseif istable(item)
                    item = pyBridge.DataConverter.table2Df(item);
                else
                    item = pyBridge.DataConverter.toPython(item);
                end
                
                pyList.append(item);
            end
        end
        
        function mlCell = list2Cell(pyList)
            % LIST2CELL Python list to MATLAB cell array
            %   Uses py_builtin.len() to ensure compatibility with Python 0-based indexing

            arguments
                pyList
            end

            % Use Python iterator for safety
            numItems = int64(double(py_builtin.len(pyList)));
            mlCell = cell(1, numItems);

            for i = int64(0):(numItems-1)
                item = pyList{i};  % Use Python 0-based indexing directly
                mlCell{i+1} = pyBridge.DataConverter.toMatlab(item);
            end
        end
        
        function mlCell = dict2Cell(pyDict)
            % DICT2CELL Python dictionary to key-value pair cell array
            
            arguments
                pyDict
            end
            
            keys = cell(pyDict.keys());
            values = cell(pyDict.values());
            
            mlCell = cell(2, length(keys));
            mlCell(1, :) = keys;
            mlCell(2, :) = values;
        end
        
        function mlArray = ensureColumnVector(mlArray)
            % ENSURECOLUMNVECTOR Ensure array is a column vector
            
            if isrow(mlArray)
                mlArray = mlArray';
            end
        end
        
        function printConversionInfo(pyObj)
            % PRINTCONVERSIONINFO Print Python object conversion info
            
            fprintf('Python object type: %s\n', char(py.getattr(py.type(pyObj), '__name__')));
            
            if isa(pyObj, 'py.numpy.ndarray')
                fprintf('  Shape: [%s]\n', strjoin(cellfun(@num2str, ...
                    cell(py.numpy.shape(pyObj)), 'UniformOutput', false), ', '));
                fprintf('  Data type: %s\n', char(pyObj.dtype));
            elseif isa(pyObj, 'py.pandas.DataFrame')
                fprintf('  Rows: %d, Columns: %d\n', pyObj.shape{1}, pyObj.shape{2});
                fprintf('  Column names: %s\n', strjoin(cell(pyObj.columns.tolist()), ', '));
            end
        end
    end
end
