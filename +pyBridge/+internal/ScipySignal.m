classdef ScipySignal
    % SCIPYSIGNAL scipy.signal module wrapper
    %   Provides signal processing, filtering, spectral analysis and other functions
    %
    % Methods:
    %   butter - Butterworth filter design
    %   filter - Signal filtering
    %   fft - Fast Fourier transform
    %   welch - Power spectral density estimation
    %   findPeaks - Peak detection
    %
    % Example:
    %   sig = pyBridge.internal.ScipySignal();
    %   b, a = sig.butter(4, 0.1, 'low');
    %   filtered = sig.filtfilt(b, a, data);
    %
    
    methods
        function obj = ScipySignal()
            pyBridge.ErrorHandler.assertPyAvailable("scipy");
        end
        
        %% Filter Design
        function [b, a] = butter(obj, order, wn, filterType)
            % BUTTER Butterworth filter design
            
            arguments
                obj
                order double
                wn double
                filterType char = "low" % 'low', 'high', 'band', 'stop'
            end
            unused = obj; %#ok<NASGU>
            
            if isscalar(wn)
                [b, a] = py.scipy.signal.butter(order, wn, btype=filterType);
            else
                [b, a] = py.scipy.signal.butter(order, pyBridge.DataConverter.toPython(wn), ...
                    btype=filterType);
            end
            
            b = double(b);
            a = double(a);
        end
        
        function [b, a] = cheby1(obj, order, rp, wn, filterType)
            % CHEBY1 Chebyshev Type I filter design
            
            arguments
                obj
                order double
                rp double % Passband ripple (dB)
                wn double
                filterType char = "low"
            end
            unused = obj; %#ok<NASGU>
            
            if isscalar(wn)
                [b, a] = py.scipy.signal.cheby1(order, rp, wn, btype=filterType);
            else
                [b, a] = py.scipy.signal.cheby1(order, rp, ...
                    pyBridge.DataConverter.toPython(wn), btype=filterType);
            end
            
            b = double(b);
            a = double(a);
        end
        
        function [b, a] = ellip(obj, order, rp, rs, wn, filterType)
            % ELLIP Elliptic filter design
            
            arguments
                obj
                order double
                rp double % Passband ripple (dB)
                rs double % Stopband attenuation (dB)
                wn double
                filterType char = "low"
            end
            unused = obj; %#ok<NASGU>
            
            if isscalar(wn)
                [b, a] = py.scipy.signal.ellip(order, rp, rs, wn, btype=filterType);
            else
                [b, a] = py.scipy.signal.ellip(order, rp, rs, ...
                    pyBridge.DataConverter.toPython(wn), btype=filterType);
            end
            
            b = double(b);
            a = double(a);
        end
        
        %% Filtering
        function y = filter(obj, b, a, x)
            % FILTER IIR/FIR filtering
            
            arguments
                obj
                b double
                a double
                x double
            end
            unused = obj; %#ok<NASGU>
            
            bPy = pyBridge.DataConverter.toPython(b(:));
            aPy = pyBridge.DataConverter.toPython(a(:));
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            yPy = py.scipy.signal.lfilter(bPy, aPy, xPy);
            y = double(yPy);
            
            if ~isscalar(x)
                y = reshape(y, size(x));
            end
        end
        
        function y = filtfilt(obj, b, a, x)
            % FILTFILT Zero-phase filtering (forward-backward filtering)
            
            arguments
                obj
                b double
                a double
                x double
            end
            unused = obj; %#ok<NASGU>
            
            bPy = pyBridge.DataConverter.toPython(b(:));
            aPy = pyBridge.DataConverter.toPython(a(:));
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            yPy = py.scipy.signal.filtfilt(bPy, aPy, xPy);
            y = double(yPy);
            
            if ~isscalar(x)
                y = reshape(y, size(x));
            end
        end
        
        function y = resample(obj, x, num, options)
            % RESAMPLE Resample signal
            
            arguments
                obj
                x double
                num double % Number of new sample points
                options.axis double = 0
                options.window = [] % Window function
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            if isempty(options.window)
                yPy = py.scipy.signal.resample(xPy, num, axis=options.axis);
            else
                yPy = py.scipy.signal.resample(xPy, num, ...
                    window=options.window, axis=options.axis);
            end
            
            y = double(yPy);
        end
        
        %% Spectral Analysis
        function result = welch(obj, x, fs, options)
            % WELCH Welch power spectral density estimation
            
            arguments
                obj
                x double
                fs double = 1.0
                options.nperseg double = 256
                options.noverlap double = []
                options.window char = "hann"
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            if isempty(options.noverlap)
                [f, Pxx] = py.scipy.signal.welch(xPy, fs, ...
                    window=options.window, nperseg=options.nperseg);
            else
                [f, Pxx] = py.scipy.signal.welch(xPy, fs, ...
                    window=options.window, nperseg=options.nperseg, ...
                    noverlap=options.noverlap);
            end
            
            result = struct();
            result.frequencies = double(f);
            result.powerSpectralDensity = double(Pxx);
        end
        
        function result = periodogram(obj, x, fs, options)
            % PERIODOGRAM Periodogram power spectral density estimation
            
            arguments
                obj
                x double
                fs double = 1.0
                options.window char = "boxcar"
                options.nfft double = []
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            if isempty(options.nfft)
                [f, Pxx] = py.scipy.signal.periodogram(xPy, fs, window=options.window);
            else
                [f, Pxx] = py.scipy.signal.periodogram(xPy, fs, ...
                    window=options.window, nfft=options.nfft);
            end
            
            result = struct();
            result.frequencies = double(f);
            result.powerSpectralDensity = double(Pxx);
        end
        
        function result = spectrogram(obj, x, fs, options)
            % SPECTROGRAM Short-time Fourier transform
            
            arguments
                obj
                x double
                fs double = 1.0
                options.window = []
                options.nperseg double = 256
                options.noverlap double = []
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            if isempty(options.window)
                windowPy = py.scipy.signal.windows.hann(options.nperseg);
            else
                windowPy = options.window;
            end
            
            if isempty(options.noverlap)
                noverlapPy = py.None;
            else
                noverlapPy = options.noverlap;
            end
            
            [f, t, Sxx] = py.scipy.signal.spectrogram(xPy, fs, ...
                window=windowPy, nperseg=options.nperseg, noverlap=noverlapPy);
            
            result = struct();
            result.frequencies = double(f);
            result.times = double(t);
            result.spectrogram = double(Sxx);
        end
        
        %% Correlation and Convolution
        function result = correlate(obj, x, y, mode)
            % CORRELATE Cross-correlation
            
            arguments
                obj
                x double
                y double = []
                mode char = "full" % 'full', 'valid', 'same'
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            if isempty(y)
                resultPy = py.scipy.signal.correlate(xPy, mode=mode);
            else
                yPy = pyBridge.DataConverter.toPython(y(:));
                resultPy = py.scipy.signal.correlate(xPy, yPy, mode=mode);
            end
            
            result = double(resultPy);
        end
        
        function result = convolve(obj, x, y, mode)
            % CONVOLVE Convolution
            
            arguments
                obj
                x double
                y double
                mode char = "full"
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            yPy = pyBridge.DataConverter.toPython(y(:));
            
            resultPy = py.scipy.signal.convolve(xPy, yPy, mode=mode);
            result = double(resultPy);
        end
        
        function result = fftconvolve(obj, x, y, mode)
            % FFTCONVOLVE FFT convolution
            
            arguments
                obj
                x double
                y double
                mode char = "full"
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            yPy = pyBridge.DataConverter.toPython(y(:));
            
            resultPy = py.scipy.signal.fftconvolve(xPy, yPy, mode=mode);
            result = double(resultPy);
        end
        
        %% Peak Detection
        function result = findPeaks(obj, x, options)
            % FINDPEAKS Peak detection
            
            arguments
                obj
                x double
                options.height double = []
                options.threshold double = []
                options.distance double = []
                options.prominence double = []
                options.width double = []
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            % Build parameter dictionary
            kwargs = py.dict();
            if ~isempty(options.height)
                kwargs{"height"} = options.height;
            end
            if ~isempty(options.threshold)
                kwargs{"threshold"} = options.threshold;
            end
            if ~isempty(options.distance)
                kwargs{"distance"} = options.distance;
            end
            if ~isempty(options.prominence)
                kwargs{"prominence"} = options.prominence;
            end
            if ~isempty(options.width)
                kwargs{"width"} = options.width;
            end
            
            peaks = py.scipy.signal.find_peaks(xPy, kwargs);
            
            result = struct();
            result.indices = double(peaks{1});
            
            % Extract peak properties
            properties = peaks{2};
            if py_builtin.hasattr(properties, 'peak_heights')
                result.heights = double(properties{"peak_heights"});
            end
            if py_builtin.hasattr(properties, 'prominences')
                result.prominences = double(properties{"prominences"});
            end
            if py_builtin.hasattr(properties, 'widths')
                result.widths = double(properties{"widths"});
            end
        end
        
        %% Window Functions
        function w = getWindow(obj, n, windowType)
            % GETWINDOW Generate window function
            
            arguments
                obj
                n double
                windowType char = "hann"
            end
            unused = obj; %#ok<NASGU>
            
            w = double(py.scipy.signal.windows.get_window(windowType, n));
        end
        
        %% Time-Frequency Analysis
        function result = hilbert(obj, x)
            % HILBERT Hilbert transform
            
            arguments
                obj
                x double
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            analyticSignal = py.scipy.signal.hilbert(xPy);
            
            result = struct();
            result.analyticSignal = double(analyticSignal);
            result.amplitude = double(py.numpy.abs(analyticSignal));
            result.phase = double(py.numpy.angle(analyticSignal));
            result.instantaneousFrequency = double(py.numpy.diff(result.phase) / (2*pi));
        end
        
        function result = stft(obj, x, fs, options)
            % STFT Short-time Fourier transform
            
            arguments
                obj
                x double
                fs double = 1.0
                options.window char = "hann"
                options.nperseg double = 256
                options.noverlap double = []
            end
            unused = obj; %#ok<NASGU>
            
            xPy = pyBridge.DataConverter.toPython(x(:));
            
            if isempty(options.noverlap)
                [f, t, Zxx] = py.scipy.signal.stft(xPy, fs, ...
                    window=options.window, nperseg=options.nperseg);
            else
                [f, t, Zxx] = py.scipy.signal.stft(xPy, fs, ...
                    window=options.window, nperseg=options.nperseg, ...
                    noverlap=options.noverlap);
            end
            
            result = struct();
            result.frequencies = double(f);
            result.times = double(t);
            result.stft = double(Zxx);
        end
    end
end
