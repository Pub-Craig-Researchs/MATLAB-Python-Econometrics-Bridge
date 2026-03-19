function result = len(pyObj)
    % LEN Get length of Python object
    %   Wrapper for py.builtins.len to provide py_builtin compatibility
    %   Returns MATLAB int64 for safe indexing
    %
    % Example:
    %   n = py_builtin.len(pyList);
    
    try
        % Get Python int and convert to MATLAB int64
        pyLen = py.builtins.len(pyObj);
        result = int64(double(pyLen));
    catch
        % Fallback: try __len__ method
        try
            lenMethod = py.getattr(pyObj, '__len__');
            result = int64(double(lenMethod()));
        catch
            error('py_builtin:NoLength', 'Object has no length');
        end
    end
end
