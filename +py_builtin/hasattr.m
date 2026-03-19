function result = hasattr(pyObj, attrName)
    % HASATTR Check if Python object has an attribute
    %   Wrapper for py.builtins.hasattr to provide py_builtin compatibility
    %
    % Example:
    %   if py_builtin.hasattr(pyObj, 'predict')
    %       result = pyObj.predict();
    %   end
    
    try
        result = py.builtins.hasattr(pyObj, attrName);
    catch
        % Fallback: try to get the attribute
        try
            py.getattr(pyObj, attrName);
            result = true;
        catch
            result = false;
        end
    end
end
