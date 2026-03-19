function result = callable(pyObj)
    % CALLABLE Check if Python object is callable
    %   Wrapper for py.builtins.callable to provide py_builtin compatibility
    %
    % Example:
    %   if py_builtin.callable(pyObj)
    %       result = pyObj();
    %   end
    
    try
        result = py.builtins.callable(pyObj);
    catch
        % Fallback: try to check for __call__ method
        try
            py.getattr(pyObj, '__call__');
            result = true;
        catch
            result = false;
        end
    end
end
