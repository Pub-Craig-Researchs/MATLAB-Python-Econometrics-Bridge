# PyBridge Installation Guide

## System Requirements

- **MATLAB**: R2020b or later
- **Python**: 3.7+ (Recommended: 3.9, 3.10, or **3.13**)
  - **Python 3.13 Compatibility**: Tested with MATLAB R2025b, fully functional (although not officially supported by MathWorks yet)
- **Operating System**: Windows / macOS / Linux

## Installation Steps

### 1. Install Python Environment

Miniconda or Anaconda is recommended:

```bash
# Download and install Miniconda
# https://docs.conda.io/en/latest/miniconda.html

# Create a virtual environment (optional but recommended)
conda create -n pybridge python=3.9
conda activate pybridge
```

### 2. Install Python Dependencies

```bash
pip install scipy statsmodels linearmodels econml numpy pandas scikit-learn
```

Or using conda:

```bash
conda install -c conda-forge scipy statsmodels numpy pandas scikit-learn
pip install linearmodels econml
```

### 3. Configure MATLAB Python Interface

Run in MATLAB:

```matlab
% Method 1: Direct configuration
pyenv('Version', 'path/to/python.exe')

% Method 2: If using conda
pyenv('Version', 'path/to/conda/envs/pybridge/python.exe')

% Verify
pyenv
```

### 4. Install PyBridge Toolbox

```matlab
% Method 1: Clone or download locally
% Place the toolbox in any directory, for example:
% c:/Users/YourName/Documents/GitHub/MATLAB-Python-Econometrics-Bridge

% Method 2: Run the startup script in MATLAB
cd('path/to/MATLAB-Python-Econometrics-Bridge')
startup()
```

### 5. Verify Installation

```matlab
% Run quick test
config = pyBridge.PyBridgeConfig();
config.initialize();
config.verifyAll();

% Run example script
run('examples/quickStart.m')
```

## Frequently Asked Questions

### Q1: pyenv reports "Python version incompatible"

**Solution:**
```matlab
% Check Python version
pe = pyenv;
if contains(pe.Version, '3.7') || contains(pe.Version, '3.8') || ...
   contains(pe.Version, '3.9') || contains(pe.Version, '3.10') || ...
   contains(pe.Version, '3.11') || contains(pe.Version, '3.12') || ...
   contains(pe.Version, '3.13')
    disp('Python version compatible');
else
    error('Python 3.7-3.13 required');
end
```

### Q2: Library import failed

**Solution:**
```bash
# Check if the library is installed
python -c "import scipy; print(scipy.__version__)"

# If not installed, reinstall
pip install --upgrade scipy
```

### Q3: MATLAB cannot find Python module

**Solution:**
```matlab
% Set Python path
if count(py.sys.path, 'your/module/path') == 0
    insert(py.sys.path, int32(0), 'your/module/path');
end
```

### Q4: Data conversion error

**Solution:**
- Ensure NumPy and Pandas are properly installed
- Check if the data type is supported
- Use `pyBridge.DataConverter.printConversionInfo()` for debugging

### Q5: Performance issues

**Optimization tips:**
- Avoid frequent MATLAB-Python data conversions
- Process data in batches instead of one by one
- Use chunked processing for large datasets
- Consider using `py.array.array` instead of list

## Uninstall

```matlab
% Remove from MATLAB path
rmpath('path/to/MATLAB-Python-Econometrics-Bridge');

% Delete the toolbox folder
% Simply delete it in your system file manager
```

## Update

```bash
# Update Python libraries
pip install --upgrade scipy statsmodels linearmodels econml

# Update the toolbox
# Re-download the latest version and replace the original folder
```

## Technical Support

- GitHub Issues: [Project Repository]
- Documentation: `docs/API_REFERENCE.md`
- Examples: `examples/` directory

---

**Installation Issue Feedback**: Please provide complete error messages and system environment information
