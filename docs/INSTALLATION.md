# PyBridge安装指南

## 系统要求

- **MATLAB**: R2020b或更高版本
- **Python**: 3.7+ (推荐3.9、3.10或**3.13**)
  - **Python 3.13 兼容性**: 已通过 MATLAB R2025b 测试，功能完全正常（虽然 MathWorks 官方尚未正式支持）
- **操作系统**: Windows / macOS / Linux

## 安装步骤

### 1. 安装Python环境

推荐使用Miniconda或Anaconda:

```bash
# 下载并安装Miniconda
# https://docs.conda.io/en/latest/miniconda.html

# 创建虚拟环境(可选但推荐)
conda create -n pybridge python=3.9
conda activate pybridge
```

### 2. 安装Python依赖库

```bash
pip install scipy statsmodels linearmodels econml numpy pandas scikit-learn
```

或使用conda:

```bash
conda install -c conda-forge scipy statsmodels numpy pandas scikit-learn
pip install linearmodels econml
```

### 3. 配置MATLAB Python接口

在MATLAB中运行:

```matlab
% 方法1: 直接配置
pyenv('Version', 'path/to/python.exe')

% 方法2: 如果使用conda
pyenv('Version', 'path/to/conda/envs/pybridge/python.exe')

% 验证
pyenv
```

### 4. 安装PyBridge工具箱

```matlab
% 方法1: 克隆或下载到本地
% 将工具箱放在任意目录,例如:
% c:/Users/YourName/Documents/GitHub/MATLAB-Python-Econometrics-Bridge

% 方法2: 在MATLAB中运行启动脚本
cd('path/to/MATLAB-Python-Econometrics-Bridge')
startup()
```

### 5. 验证安装

```matlab
% 运行快速测试
config = pyBridge.PyBridgeConfig();
config.initialize();
config.verifyAll();

% 运行示例脚本
run('examples/quickStart.m')
```

## 常见问题

### Q1: pyenv报错"Python版本不兼容"

**解决方法:**
```matlab
% 检查Python版本
pe = pyenv;
if contains(pe.Version, '3.7') || contains(pe.Version, '3.8') || ...
   contains(pe.Version, '3.9') || contains(pe.Version, '3.10') || ...
   contains(pe.Version, '3.11') || contains(pe.Version, '3.12') || ...
   contains(pe.Version, '3.13')
    disp('Python版本兼容');
else
    error('需要Python 3.7-3.13');
end
```

### Q2: 导入库失败

**解决方法:**
```bash
# 检查库是否已安装
python -c "import scipy; print(scipy.__version__)"

# 如果未安装,重新安装
pip install --upgrade scipy
```

### Q3: MATLAB找不到Python模块

**解决方法:**
```matlab
% 设置Python路径
if count(py.sys.path, 'your/module/path') == 0
    insert(py.sys.path, int32(0), 'your/module/path');
end
```

### Q4: 数据转换错误

**解决方法:**
- 确保NumPy和Pandas已正确安装
- 检查数据类型是否支持
- 使用`pyBridge.DataConverter.printConversionInfo()`调试

### Q5: 性能问题

**优化建议:**
- 避免频繁的MATLAB-Python数据转换
- 批量处理数据而非逐个处理
- 对大型数据集使用分块处理
- 考虑使用`py.array.array`而非list

## 卸载

```matlab
% 从MATLAB路径移除
rmpath('path/to/MATLAB-Python-Econometrics-Bridge');

% 删除工具箱文件夹
% 在系统文件管理器中删除即可
```

## 更新

```bash
# 更新Python库
pip install --upgrade scipy statsmodels linearmodels econml

# 更新工具箱
# 重新下载最新版本并替换原文件夹
```

## 技术支持

- GitHub Issues: [项目地址]
- 文档: `docs/API_REFERENCE.md`
- 示例: `examples/` 目录

---

**安装问题反馈**: 请提供完整的错误信息和系统环境
