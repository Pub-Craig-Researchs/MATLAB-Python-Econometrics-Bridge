classdef ScipyOptimize
    % SCIPYOPTIMIZE scipy.optimize module wrapper
    %   Provides optimization, root finding, least squares and other functions
    %
    % Methods:
    %   minimize - Multivariate function minimization
    %   minimizeScalar - Univariate function minimization
    %   root - Equation root finding
    %   curveFit - Curve fitting
    %   leastSquares - Least squares
    %
    % Example:
    %   opt = pyBridge.internal.ScipyOptimize();
    %   result = opt.minimize(@(x) x(1)^2 + x(2)^2, [1, 1]);
    %
    % Author: WorkBuddy
    % Date: 2026-03-18
    
    methods
        function obj = ScipyOptimize()
            % Constructor
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
        end
        
        %% Minimization
        function result = minimize(~, fun, x0, options)
            % MINIMIZE Multivariate function minimization
            
            arguments
                ~
                fun function_handle
                x0 double
                options.method char = "BFGS"
                options.maxIter double = 1000
                options.tol double = 1e-6
                options.display logical = false
            end
            
            % Convert initial values
            x0Py = pyBridge.DataConverter.toPython(x0(:));
            
            % Create Python callback function
            pyFun = py.pyfunc.FunctionWrapper(fun, 'scalar');
            
            % Set optimization options
            optOptions = py.dict();
            optOptions{"maxiter"} = options.maxIter;
            optOptions{"disp"} = options.display;
            
            % Execute optimization
            optResult = py.scipy.optimize.minimize(pyFun, x0Py, ...
                method=options.method, tol=options.tol, options=optOptions);
            
            % Parse results
            result = struct();
            result.x = double(optResult.x);
            result.fun = double(optResult.fun);
            result.success = logical(optResult.success);
            result.message = char(optResult.message);
            result.nIter = double(optResult.nit);
            
            if py_builtin.hasattr(optResult, 'hess_inv')
                result.hessInv = double(optResult.hess_inv);
            end
            
            if py_builtin.hasattr(optResult, 'jac')
                result.jacobian = double(optResult.jac);
            end
        end
        
        function result = minimizeScalar(~, fun, bounds, options)
            % MINIMIZESCALAR Univariate function minimization
            
            arguments
                ~
                fun function_handle
                bounds double = []
                options.method char = "brent"
                options.tol double = 1e-6
            end
            
            % Create Python callback
            pyFun = py.pyfunc.FunctionWrapper(fun, 'scalar');
            
            if isempty(bounds)
                optResult = py.scipy.optimize.minimize_scalar(pyFun, ...
                    method=options.method, options=py.dict("xatol", options.tol));
            else
                boundsPy = py.tuple({bounds(1), bounds(2)});
                optResult = py.scipy.optimize.minimize_scalar(pyFun, ...
                    bounds=boundsPy, method="bounded");
            end
            
            result = struct();
            result.x = double(optResult.x);
            result.fun = double(optResult.fun);
            result.success = logical(optResult.success);
            result.nIter = double(optResult.nit);
        end
        
        function result = linprog(~, c, A, b, Aeq, beq, bounds, options)
            % LINPROG Linear programming
            %   min c'*x
            %   s.t. A*x <= b, Aeq*x = beq, lb <= x <= ub
            
            arguments
                ~
                c double
                A double = []
                b double = []
                Aeq double = []
                beq double = []
                bounds double = [] % [lb, ub] or [lb, ub] for each row
                options.method char = "highs"
            end
            
            cPy = pyBridge.DataConverter.toPython(c(:));
            
            % Inequality constraints
            if isempty(A)
                Aub = py.None;
                bub = py.None;
            else
                Aub = pyBridge.DataConverter.toPython(A);
                bub = pyBridge.DataConverter.toPython(b(:));
            end
            
            % Equality constraints
            if isempty(Aeq)
                AeqPy = py.None;
                beqPy = py.None;
            else
                AeqPy = pyBridge.DataConverter.toPython(Aeq);
                beqPy = pyBridge.DataConverter.toPython(beq(:));
            end
            
            % Bounds
            if isempty(bounds)
                boundsPy = py.list();
            else
                if size(bounds, 1) == 1
                    % Uniform bounds
                    boundsPy = py.tuple({bounds(1), bounds(2)});
                else
                    % Different bounds for each variable
                    boundsPy = py.list();
                    for i = 1:size(bounds, 1)
                        boundsPy.append(py.tuple({bounds(i,1), bounds(i,2)}));
                    end
                end
            end
            
            % Solve
            optResult = py.scipy.optimize.linprog(cPy, A_ub=Aub, b_ub=bub, ...
                A_eq=AeqPy, b_eq=beqPy, bounds=boundsPy, method=options.method);
            
            result = struct();
            result.x = double(optResult.x);
            result.fun = double(optResult.fun);
            result.success = logical(optResult.success);
            result.message = char(optResult.message);
        end
        
        %% Root Finding
        function result = root(~, fun, x0, options)
            % ROOT Equation root finding
            
            arguments
                ~
                fun function_handle
                x0 double
                options.method char = "hybr"
                options.tol double = 1e-6
            end
            
            x0Py = pyBridge.DataConverter.toPython(x0(:));
            
            pyFun = py.pyfunc.FunctionWrapper(fun, 'vector');
            
            rootResult = py.scipy.optimize.root(pyFun, x0Py, ...
                method=options.method, tol=options.tol);
            
            result = struct();
            result.x = double(rootResult.x);
            result.fun = double(rootResult.fun);
            result.success = logical(rootResult.success);
            result.message = char(rootResult.message);
        end
        
        function x = fsolve(obj, fun, x0, options)
            % FSOLVE Solve nonlinear system of equations (MATLAB style)
            
            arguments
                obj
                fun function_handle
                x0 double
                options.tol double = 1e-6
            end
            
            result = obj.root(fun, x0, "method", "hybr", "tol", options.tol);
            x = result.x;
        end
        
        %% Curve Fitting
        function result = curveFit(~, fun, xData, yData, p0, bounds)
            % CURVEFIT Curve fitting
            
            arguments
                ~
                fun function_handle
                xData double
                yData double
                p0 double = []
                bounds double = []
            end
            
            xDataPy = pyBridge.DataConverter.toPython(xData(:));
            yDataPy = pyBridge.DataConverter.toPython(yData(:));
            
            % Create fitting function
            pyFun = py.pyfunc.FitFunctionWrapper(fun);
            
            % Initial parameters
            if isempty(p0)
                p0Py = py.None;
            else
                p0Py = pyBridge.DataConverter.toPython(p0(:));
            end
            
            % Parameter bounds
            if isempty(bounds)
                boundsPy = py.tuple({py.array.array('d'), py.array.array('d')});
            else
                lowerBounds = bounds(:,1);
                upperBounds = bounds(:,2);
                boundsPy = py.tuple({...
                    pyBridge.DataConverter.toPython(lowerBounds), ...
                    pyBridge.DataConverter.toPython(upperBounds)});
            end
            
            % Execute fitting
            [popt, pcov] = py.scipy.optimize.curve_fit(pyFun, xDataPy, yDataPy, ...
                p0=p0Py, bounds=boundsPy);
            
            result = struct();
            result.parameters = double(popt);
            result.covariance = double(pcov);
            result.stdErrors = sqrt(diag(result.covariance));
        end
        
        %% Least Squares
        function result = leastSquares(~, fun, x0, bounds, options)
            % LEASTSQUARES Nonlinear least squares
            
            arguments
                ~
                fun function_handle
                x0 double
                bounds double = []
                options.method char = "trf"
                options.maxIter double = 1000
            end
            
            x0Py = pyBridge.DataConverter.toPython(x0(:));
            
            pyFun = py.pyfunc.FunctionWrapper(fun, 'vector');
            
            % Bounds
            if isempty(bounds)
                lbPy = py.None;
                ubPy = py.None;
            else
                lbPy = pyBridge.DataConverter.toPython(bounds(:,1));
                ubPy = pyBridge.DataConverter.toPython(bounds(:,2));
            end
            
            % Solve
            lsResult = py.scipy.optimize.least_squares(pyFun, x0Py, ...
                bounds=py.tuple({lbPy, ubPy}), method=options.method, ...
                max_nfev=options.maxIter);
            
            result = struct();
            result.x = double(lsResult.x);
            result.cost = double(lsResult.cost);
            result.fun = double(lsResult.fun);
            result.optimality = double(lsResult.optimality);
            result.success = logical(lsResult.success);
            result.nFeval = double(lsResult.nfev);
        end
        
        %% Global Optimization
        function result = differentialEvolution(~, fun, bounds, options)
            % DIFFERENTIALEVOLUTION Differential evolution algorithm
            
            arguments
                ~
                fun function_handle
                bounds double % N x 2 matrix
                options.maxIter double = 1000
                options.popSize double = 15
                options.tol double = 1e-6
            end
            
            pyFun = py.pyfunc.FunctionWrapper(fun, 'scalar');
            
            % Convert bounds
            boundsPy = py.list();
            for i = 1:size(bounds, 1)
                boundsPy.append(py.tuple({bounds(i,1), bounds(i,2)}));
            end
            
            % Solve
            deResult = py.scipy.optimize.differential_evolution(pyFun, boundsPy, ...
                maxiter=options.maxIter, popsize=options.popSize, tol=options.tol);
            
            result = struct();
            result.x = double(deResult.x);
            result.fun = double(deResult.fun);
            result.success = logical(deResult.success);
            result.nIter = double(deResult.nit);
        end
        
        function result = basinhopping(~, fun, x0, options)
            % BASINHOPPING Basin hopping algorithm
            
            arguments
                ~
                fun function_handle
                x0 double
                options.nIter double = 100
                options.T double = 1.0
            end
            
            x0Py = pyBridge.DataConverter.toPython(x0(:));
            
            pyFun = py.pyfunc.FunctionWrapper(fun, 'scalar');
            
            bhResult = py.scipy.optimize.basinhopping(pyFun, x0Py, ...
                niter=options.nIter, T=options.T);
            
            result = struct();
            result.x = double(bhResult.x);
            result.fun = double(bhResult.fun);
            result.success = true;
        end
    end
end
