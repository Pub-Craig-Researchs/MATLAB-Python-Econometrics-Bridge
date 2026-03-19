function result = dir(pyObj)
    % DIR Get directory listing of Python object attributes
    %   Wrapper for py.builtins.dir to provide py_builtin compatibility
    %   Returns Python list (for compatibility with existing code)
    %
    % Example:
    %   attrs = py_builtin.dir(pyObj);
    
    try
        result = py.builtins.dir(pyObj);
    catch
        % Fallback: empty list
        result = py.list();
    end
end
