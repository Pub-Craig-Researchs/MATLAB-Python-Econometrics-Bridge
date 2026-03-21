# PyBridge API参考文档

## 目录

1. [核心组件](#核心组件)
2. [scipy封装](#scipy封装)
3. [statsmodels封装](#statsmodels封装)
4. [linearmodels封装](#linearmodels封装)
5. [econml封装](#econml封装)
6. [高级计量方法](#高级计量方法)

---

## 核心组件

### PyBridgeConfig

环境配置和库验证类。

#### 构造函数

```matlab
config = pyBridge.PyBridgeConfig(pythonPath)
```

**参数:**
- `pythonPath` (可选): Python可执行文件路径,默认自动检测当前Python环境

#### 方法

| 方法 | 说明 |
|------|------|
| `initialize()` | 初始化Python环境 |
| `checkLibrary(libName)` | 检查指定库是否安装 |
| `getLibVersion(libName)` | 获取库版本信息 |
| `verifyAll()` | 验证所有必需库,返回table |
| `printInfo()` | 打印环境信息 |

#### 静态方法

| 方法 | 说明 |
|------|------|
| `getInstance()` | 获取单例实例 |
| `quickCheck()` | 快速检查环境是否就绪 |

---

### DataConverter

MATLAB-Python数据双向转换工具。

---

## 高级计量方法

### CovarianceTypes - 协方差矩阵计算

提供多种稳健标准误计算方法,位于 `pyBridge.internal.CovarianceTypes`。

#### HAC标准误 (Newey-West)

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.hac(residuals, X, options)
```

**参数:**
- `residuals`: 回归残差
- `X`: 设计矩阵
- `options.maxLags`: 最大滞后阶数（默认自动选择）
- `options.kernel`: 核函数类型
  - `"bartlett"` - Bartlett核（默认，等价于Newey-West）
  - `"newey-west"` - Newey-West核（内部映射为bartlett）
  - `"parzen"` - Parzen核
  - `"qs"` - 二次谱核

**注意事项:**
- `newey-west` 在内部映射为 `bartlett`，两者等价
- `parzen` 和 `qs` 核在离散选择模型（MNLogit/Logit/Probit）中不支持，会自动回退到 `bartlett` 并发出警告

**返回:**
- `covMatrix`: HAC协方差矩阵

**示例:**
```matlab
% 先拟合OLS
result = pyBridge.StatsmodelsWrapper.ols(y, X);
residuals = result.residuals;

% 计算HAC标准误（默认使用bartlett核）
covHAC = pyBridge.internal.CovarianceTypes.hac(residuals, X, ...
    maxLags=4, kernel="bartlett");
stdErrorsHAC = sqrt(diag(covHAC));
```

#### 聚类标准误

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.clustered(residuals, X, clusterIds)
```

**参数:**
- `residuals`: 回归残差
- `X`: 设计矩阵
- `clusterIds`: 聚类标识符向量
- `options.useCorrection`: 是否使用小样本校正(默认true)

**示例:**
```matlab
% 按企业聚类
firmIds = [1,1,1,2,2,2,...]; % 企业ID
covCluster = pyBridge.internal.CovarianceTypes.clustered(...
    residuals, X, firmIds);
```

#### 多维聚类标准误

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.multiwayClustered(...
    residuals, X, clusterGroups)
```

**参数:**
- `clusterGroups`: cell数组,包含多个聚类维度
  - 例如: `{firmIds, yearIds}` 表示企业×年份双向聚类

**示例:**
```matlab
% 双向聚类:企业×年份
covMultiway = pyBridge.internal.CovarianceTypes.multiwayClustered(...
    residuals, X, {firmIds, yearIds});

% 三维聚类:企业×年份×行业
covMultiway3 = pyBridge.internal.CovarianceTypes.multiwayClustered(...
    residuals, X, {firmIds, yearIds, industryIds});
```

#### 异方差稳健标准误 (HC系列)

```matlab
covMatrix = pyBridge.internal.CovarianceTypes.heteroskedastic(residuals, X, type)
```

**参数:**
- `type`: HC类型
  - `"HC0"` - White标准误
  - `"HC1"` - Stata小样本校正(推荐)
  - `"HC2"` - 适合小样本
  - `"HC3"` - 最保守,极度异方差

---

### StatsmodelsWrapper - 高级回归方法

#### Multinomial Logit

```matlab
result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, options)
```

**参数:**
- `y`: 多分类因变量（整数编码: 0, 1, 2, ..., K-1）
- `X`: 自变量矩阵
- `options.addConstant`: 是否添加常数项（默认true）
- `options.maxIter`: 最大迭代次数（默认100）
- `options.covType`: 协方差矩阵类型（可选）
  - `"nonrobust"` - 经典标准误（默认）
  - `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"` - 异方差稳健标准误
  - `"hac"` - HAC标准误（仅支持bartlett核）
- `options.covKwds`: HAC参数（当covType="hac"时使用）
  - `covKwds.maxlags`: 最大滞后阶数
  - `covKwds.kernel`: 核函数（仅支持"bartlett"）

**返回字段:**
- `params`: 系数矩阵((K-1) × p)
- `stdErrors`: 标准误
- `tStatistics`: t统计量
- `pValues`: p值
- `probabilities`: 预测概率(n × K)
- `marginalEffects`: 边际效应
- `marginalEffectsSE`: 边际效应标准误
- `marginalEffectsP`: 边际效应p值
- `marginalEffectsT`: 边际效应t统计量
- `nCategories`: 类别数量

**HAC标准误支持说明:**
- MNLogit支持HAC标准误，但仅限于 `bartlett` 核
- `parzen` 和 `qs` 核不支持，会自动回退到 `bartlett` 并发出警告

**示例:**
```matlab
% 职业选择模型（蓝领=0, 白领=1, 专业=2）
y = [0; 1; 2; 0; 1; ...];
X = [education, experience, age];
result = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X);

% 使用HAC标准误
resultHAC = pyBridge.StatsmodelsWrapper.multinomialLogit(y, X, ...
    covType="hac", covKwds=struct('maxlags', 4, 'kernel', 'bartlett'));
```

#### Ordered Logit/Probit

```matlab
result = pyBridge.StatsmodelsWrapper.orderedLogit(y, X, options)
result = pyBridge.StatsmodelsWrapper.orderedProbit(y, X, options)
```

**参数:**
- `y`: 有序因变量(整数编码: 0, 1, 2, ..., K-1)
- `X`: 自变量矩阵
- `options.addConstant` (logical, default: **false**): 是否添加常数项。注意: OrderedModel 不允许 X 中包含常数列（阈值参数作为截距）
- `options.maxIter` (double, default: **1000**): MLE优化的最大迭代次数
- `options.covType` (string, default: "nonrobust"): 协方差矩阵类型: "nonrobust", "HC0", "HC1", "HC2", "HC3"

**返回字段:**
- `coefficients`: 回归系数（不含阈值）
- `stdErrors`: 系数标准误
- `thresholds`: 实际切点（转换为Stata约定格式）
- `rawThresholds`: 原始statsmodels参数化格式（log-diff格式）
- `thresholdStdErrors`: 阈值标准误
- `params`: 系数估计（与coefficients相同）
- `tStatistics`: t统计量
- `pValues`: p值

**示例:**
```matlab
% 满意度评级(不满意=0, 一般=1, 满意=2, 非常满意=3)
y = [0; 1; 2; 3; 1; ...];
% Note: addConstant=false by default (OrderedModel uses thresholds as intercepts)
result = pyBridge.StatsmodelsWrapper.orderedLogit(y, X);
fprintf('阈值: ');
disp(result.thresholds);

% 使用稳健标准误
result = pyBridge.StatsmodelsWrapper.orderedLogit(y, X, covType="HC1");
```

#### Poisson回归

```matlab
result = pyBridge.StatsmodelsWrapper.poisson(y, X, options)
```

**参数:**
- `y`: 计数数据
- `X`: 自变量矩阵
- `options.addConstant` (logical, default: **true**): 是否添加常数项
- `options.exposure`: 暴露变量(可选)
- `options.covType` (string, default: "nonrobust"): 协方差矩阵类型: "nonrobust", "HC0", "HC1", "HC2", "HC3"

**返回字段:**
- `params`: 系数估计
- `stdErrors`: 标准误
- `tStatistics`: t统计量
- `pValues`: p值
- `overdispersionTest`: 过度离散检验统计量
- `hasOverdispersion`: 是否存在过度离散(>1.5)

**示例:**
```matlab
% 医院就诊次数
y = [0; 2; 5; 1; 3; ...];
result = pyBridge.StatsmodelsWrapper.poisson(y, X);

if result.hasOverdispersion
    fprintf('存在过度离散,建议使用负二项回归\n');
end

% 使用稳健标准误
result = pyBridge.StatsmodelsWrapper.poisson(y, X, covType="HC1");
```

#### Negative Binomial回归

```matlab
result = pyBridge.StatsmodelsWrapper.negativeBinomial(y, X)
```

**返回字段:**
- `alpha`: 过度离散参数
- 其他字段同Poisson回归

---

### StatsmodelsWrapper.ols - 高级标准误选项

OLS方法支持多种标准误类型:

```matlab
result = pyBridge.StatsmodelsWrapper.ols(y, X, options)
```

**标准误类型 (`options.covType`):**

| 类型 | 说明 | 额外参数 |
|------|------|---------|
| `"nonrobust"` | 经典标准误 | - |
| `"HC0"` | White标准误 | - |
| `"HC1"` | Stata稳健标准误 | - |
| `"HC2"` | 小样本稳健 | - |
| `"HC3"` | 最保守稳健 | - |
| `"hac"` | Newey-West HAC | `maxLags`, `kernel` |
| `"cluster"` | 单维聚类 | `clusterIds` |
| `"multiway"` | 多维聚类 | `clusterGroups` |

**完整示例:**

```matlab
%% 异方差稳健标准误
result = pyBridge.StatsmodelsWrapper.ols(y, X, covType="HC1");
fprintf('HC1标准误: %.3f\n', result.stdErrors(2));

%% HAC标准误
result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="hac", maxLags=4, kernel="newey-west");
fprintf('HAC标准误: %.3f\n', result.stdErrors(2));

%% 单维聚类标准误
result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="cluster", clusterIds=firmIds);
fprintf('聚类数: %d\n', result.nClusters);

%% 多维聚类标准误
result = pyBridge.StatsmodelsWrapper.ols(y, X, ...
    covType="multiway", clusterGroups={firmIds, yearIds});
fprintf('双向聚类标准误: %.3f\n', result.stdErrors(2));
```

---

### LinearmodelsWrapper.panelOLS - 聚类标准误选项

面板数据模型支持多种聚类标准误:

```matlab
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, entityIds, timeIds, options)
```

**标准误类型 (`options.covType`):**

| 类型 | 说明 |
|------|------|
| `"unadjusted"` | 经典标准误 |
| `"robust"` | 异方差稳健(默认) |
| `"clustered"` | 聚类标准误 |
| `"clustered_entity"` | 按个体聚类 |
| `"clustered_time"` | 按时间聚类 |
| `"clustered_both"` | 双向聚类(个体×时间) |

**示例:**

```matlab
%% 固定效应 + 企业聚类
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
    entityEffects=true, covType="clustered_entity");

%% 双向固定效应 + 双向聚类
result = pyBridge.LinearmodelsWrapper.panelOLS(y, X, firmIds, yearIds, ...
    entityEffects=true, timeEffects=true, covType="clustered_both");
```

#### 静态方法

##### toPython

```matlab
pyObj = pyBridge.DataConverter.toPython(mlData, options)
```

**参数:**
- `mlData`: MATLAB数据(数组、表、结构体、cell等)
- `options.dtype`: 指定NumPy数据类型(可选)
- `options.copy`: 是否创建副本(可选,默认false)

**返回:** Python对象(NumPy数组、Pandas DataFrame或dict)

##### toMatlab

```matlab
mlData = pyBridge.DataConverter.toMatlab(pyObj, options)
```

**参数:**
- `pyObj`: Python对象
- `options.type`: 指定输出类型('array', 'table', 'struct')
- `options.rowNames`: DataFrame转换时是否保留行名

**返回:** MATLAB数据

##### 专用转换方法

| 方法 | 说明 |
|------|------|
| `array2Numpy(mlArray)` | MATLAB数组 → NumPy数组 |
| `numpy2Array(pyArray)` | NumPy数组 → MATLAB数组 |
| `table2Df(mlTable)` | MATLAB表 → Pandas DataFrame |
| `df2Table(pyDf)` | Pandas DataFrame → MATLAB表 |
| `struct2Dict(mlStruct)` | MATLAB结构体 → Python字典 |
| `dict2Struct(pyDict)` | Python字典 → MATLAB结构体 |
| `cell2List(mlCell)` | MATLAB cell → Python list |
| `list2Cell(pyList)` | Python list → MATLAB cell |

---

### ResultParser

Python返回结果解析器。

#### 静态方法

##### parse

```matlab
result = pyBridge.ResultParser.parse(pyObj, parseType)
```

**参数:**
- `pyObj`: Python返回对象
- `parseType` (可选): 解析类型('auto', 'statsmodels', 'econml', 'linearmodels', 'array', 'dataframe')

**返回:** MATLAB结构体或数组

##### 专用解析方法

| 方法 | 说明 |
|------|------|
| `parseStatsmodels(pyObj)` | 解析statsmodels回归结果 |
| `parseEconML(pyResult)` | 解析econml因果推断结果 |
| `parseLinearmodels(pyObj)` | 解析linearmodels面板数据结果 |
| `parseSummary(pyObj)` | 解析统计摘要 |
| `parseArray(pyObj)` | 解析NumPy数组 |
| `extractAttributes(pyObj, attrNames)` | 提取指定属性 |

##### 工具方法

| 方法 | 说明 |
|------|------|
| `printResult(result, maxDepth)` | 打印解析结果 |

---

### ErrorHandler

统一的错误处理模块。

#### 静态方法

| 方法 | 说明 |
|------|------|
| `wrapCall(func, varargin)` | 包装函数调用,自动捕获异常 |
| `handlePyError(ME, options)` | 处理Python异常 |
| `getPyTraceback()` | 获取Python错误堆栈 |
| `formatError(errorInfo, options)` | 格式化错误消息 |
| `safeCall(func, defaultResult, varargin)` | 安全调用,失败返回默认值 |
| `validateInputs(varargin)` | 验证输入参数 |
| `printPyError(ME)` | 打印Python错误详情 |
| `tryImport(moduleName)` | 尝试导入Python模块 |
| `assertPyAvailable(libName)` | 确保Python库可用 |

---

## scipy封装

### ScipyStats

统计功能封装类。

#### 构造函数

```matlab
stats = pyBridge.internal.ScipyStats();
```

#### 概率分布方法

| 方法 | 说明 |
|------|------|
| `normPDF(x, mu, sigma)` | 正态分布PDF |
| `normCDF(x, mu, sigma)` | 正态分布CDF |
| `normPPF(p, mu, sigma)` | 正态分布分位数 |
| `tPDF(x, df)` | t分布PDF |
| `tCDF(x, df)` | t分布CDF |

#### 假设检验方法

| 方法 | 说明 |
|------|------|
| `tTest(data1, data2, options)` | t检验(独立/单样本) |
| `tTestPaired(data1, data2)` | 配对t检验 |
| `chi2Test(observed, expected)` | 卡方检验 |
| `fTest(data1, data2)` | F检验(ANOVA) |
| `kruskalWallis(varargin)` | Kruskal-Wallis H检验 |
| `mannWhitneyU(data1, data2, options)` | Mann-Whitney U检验 |
| `normalityTest(data, testName)` | 正态性检验 |

#### 描述性统计方法

| 方法 | 说明 |
|------|------|
| `describe(data, options)` | 描述性统计 |
| `correlation(data1, data2, method)` | 相关分析 |
| `fitDistribution(data, distName)` | 分布拟合 |
| `percentile(data, percentiles)` | 百分位数 |
| `quantile(data, quantiles)` | 分位数 |

---

### ScipyOptimize

优化功能封装类。

#### 构造函数

```matlab
opt = pyBridge.internal.ScipyOptimize();
```

#### 最小化方法

| 方法 | 说明 |
|------|------|
| `minimize(fun, x0, options)` | 多元函数最小化 |
| `minimizeScalar(fun, bounds, options)` | 一元函数最小化 |
| `linprog(c, A, b, Aeq, beq, bounds, options)` | 线性规划 |

#### 求根方法

| 方法 | 说明 |
|------|------|
| `root(fun, x0, options)` | 方程求根 |
| `fsolve(fun, x0, options)` | 求解非线性方程组(MATLAB风格) |

#### 拟合方法

| 方法 | 说明 |
|------|------|
| `curveFit(fun, xData, yData, p0, bounds)` | 曲线拟合 |
| `leastSquares(fun, x0, bounds, options)` | 非线性最小二乘 |

#### 全局优化方法

| 方法 | 说明 |
|------|------|
| `differentialEvolution(fun, bounds, options)` | 差分进化算法 |
| `basinhopping(fun, x0, options)` | 盆地跳跃算法 |

---

### ScipySignal

信号处理功能封装类。

#### 构造函数

```matlab
sig = pyBridge.internal.ScipySignal();
```

#### 滤波器设计方法

| 方法 | 说明 |
|------|------|
| `butter(order, wn, filterType)` | Butterworth滤波器 |
| `cheby1(order, rp, wn, filterType)` | Chebyshev I型滤波器 |
| `ellip(order, rp, rs, wn, filterType)` | 椭圆滤波器 |

#### 滤波方法

| 方法 | 说明 |
|------|------|
| `filter(b, a, x)` | IIR/FIR滤波 |
| `filtfilt(b, a, x)` | 零相位滤波 |
| `resample(x, num, options)` | 重采样 |

#### 频谱分析方法

| 方法 | 说明 |
|------|------|
| `welch(x, fs, options)` | Welch功率谱密度估计 |
| `periodogram(x, fs, options)` | 周期图法功率谱估计 |
| `spectrogram(x, fs, options)` | 短时傅里叶变换 |
| `stft(x, fs, options)` | STFT |

#### 其他方法

| 方法 | 说明 |
|------|------|
| `correlate(x, y, mode)` | 互相关 |
| `convolve(x, y, mode)` | 卷积 |
| `findPeaks(x, options)` | 峰值检测 |
| `getWindow(windowType, n)` | 生成窗函数 |
| `hilbert(x)` | Hilbert变换 |

---

## statsmodels封装

### StatsmodelsWrapper

回归分析、时间序列、假设检验封装类。

#### 回归分析方法

| 方法 | 说明 |
|------|------|
| `ols(y, X, options)` | 普通最小二乘回归 |
| `wls(y, X, weights, options)` | 加权最小二乘回归 |
| `glm(y, X, options)` | 广义线性模型 |
| `logistic(y, X, options)` | 逻辑回归 |
| `probit(y, X, options)` | Probit回归 |

#### 时间序列方法

| 方法 | 说明 |
|------|------|
| `arima(y, order, options)` | ARIMA时间序列模型 |
| `varModel(data, maxLags, options)` | VAR向量自回归模型 |
| `adfuller(y, options)` | ADF单位根检验 |
| `kpss(y, options)` | KPSS单位根检验 |

#### 诊断检验方法

| 方法 | 说明 |
|------|------|
| `breuschPagan(y, X, options)` | Breusch-Pagan异方差检验 |
| `whiteTest(y, X, options)` | White异方差检验 |
| `durbinWatson(residuals)` | Durbin-Watson自相关检验 |
| `vif(X, options)` | 方差膨胀因子(多重共线性) |

---

## linearmodels封装

### LinearmodelsWrapper

面板数据分析、工具变量回归封装类。

#### 面板数据方法

| 方法 | 说明 |
|------|------|
| `panelOLS(y, X, entityIds, timeIds, options)` | 固定效应面板模型 |
| `randomEffects(y, X, entityIds, timeIds, options)` | 随机效应模型 |
| `betweenOLS(y, X, entityIds, timeIds)` | 组间估计 |
| `pooledOLS(y, X, entityIds, timeIds, options)` | 混合OLS |
| `firstDifference(y, X, entityIds, timeIds)` | 一阶差分模型 |

#### 工具变量方法

| 方法 | 说明 |
|------|------|
| `iv2SLS(y, endogVars, exogVars, instruments, options)` | 两阶段最小二乘 |
| `ivLIML(y, endogVars, exogVars, instruments, options)` | 有限信息极大似然估计 |
| `ivGMM(y, endogVars, exogVars, instruments, options)` | 广义矩估计 |

**参数说明**：
- `y`：因变量
- `endogVars`：内生变量（Endogenous regressors，受内生性困扰、需要工具化的变量）
- `exogVars`：外生控制变量（Exogenous regressors，不需要工具化的控制变量），无外生控制时传 `[]`
- `instruments`：工具变量（Instrumental variables，排除限制，用于识别内生变量，数量需 >= 内生变量数量）
- `options.addConstant` (logical, default: **true**)：是否向外生变量中添加常数项
- `options.covType` (string, default: "unadjusted")：协方差矩阵类型: "unadjusted", "robust", "kernel", "clustered"
- `options.weights`：观测权重（可选）

**示例**：
```matlab
% 工资方程：教育是内生变量，用父母教育水平作为工具变量
% wage = f(education, experience), 其中 education 是内生的
result = pyBridge.LinearmodelsWrapper.iv2SLS(wage, education, experience, parents_edu);

% 无外生控制变量的情况
result = pyBridge.LinearmodelsWrapper.iv2SLS(y, endogX, [], instruments);
```

#### 检验方法

| 方法 | 说明 |
|------|------|
| `hausmanTest(fixedEffects, randomEffects)` | Hausman检验 |
| `panelUnitTest(y, entityIds, timeIds, testName)` | 面板单位根检验 |

---

## econml封装

### EconmlWrapper

因果推断和处理效应估计封装类。

#### 因果推断方法

| 方法 | 说明 |
|------|------|
| `dml(Y, T, X, W, options)` | Double Machine Learning |
| `drLearner(Y, T, X, W, options)` | Doubly Robust Learner |
| `causalForest(Y, T, X, W, options)` | 因果森林 |

#### 元学习方法

| 方法 | 说明 |
|------|------|
| `sLearner(Y, T, X, options)` | S-Learner |
| `tLearner(Y, T, X, options)` | T-Learner |
| `xLearner(Y, T, X, options)` | X-Learner |

#### 解释和预测方法

| 方法 | 说明 |
|------|------|
| `interpret(result, X, featureNames)` | 解释因果效应 |
| `predictEffect(fittedModel, Xnew)` | 预测处理效应 |
| `predictInterval(fittedModel, Xnew, alpha)` | 预测置信区间 |
| `sensitivityAnalysis(fittedModel, Y, T, X)` | 敏感性分析 |

---

## 结果结构体

### OLS回归结果

```matlab
result.nObs           % 观测数
result.dfModel        % 模型自由度
result.dfResiduals    % 残差自由度
result.rSquared       % R²
result.adjRSquared    % 调整R²
result.aic            % AIC
result.bic            % BIC
result.params         % 参数估计
result.stdErrors      % 标准误
result.tStatistics    % t统计量
result.pValues        % p值
result.confInt        % 置信区间(结构体)
  .lower              % 下界
  .upper              % 上界
result.paramNames     % 参数名称
result.residuals      % 残差
result.fittedValues   % 拟合值
result.fStatistic     % F统计量
result.fPValue        % F检验p值
```

### DML因果推断结果

```matlab
result.modelType      % 模型类型
result.ate            % 平均处理效应
result.ateConfInt     % ATE置信区间(结构体)
  .lower              % 下界
  .upper              % 上界
result.atePValue      % ATE的p值
result.cate           % 条件平均处理效应
result.fittedModel    % 拟合模型对象
```

### 面板数据结果

```matlab
result.nObs           % 观测数
result.nEntities      % 个体数
result.nTimes         % 时间点数
result.rSquared       % R²(整体)
result.rSquaredWithin % R²(组内)
result.rSquaredBetween % R²(组间)
result.params         % 参数估计
result.stdErrors      % 标准误
result.tStats         % t统计量
result.pValues        % p值
result.paramNames     % 参数名称
result.fStatistic     % F统计量
result.fPValue        % F检验p值
```

---

## 常见选项参数

### 假设检验选项

```matlab
options.equalVar      % 是否假设方差相等(默认true)
options.alternative   % 备择假设('two-sided', 'less', 'greater')
```

### 优化选项

```matlab
options.method        % 优化方法
options.maxIter       % 最大迭代次数
options.tol           % 收敛容差
options.display       % 是否显示输出
```

### 回归选项

```matlab
options.addConstant   % 是否添加常数项(默认true)
options.covType       % 协方差矩阵类型
```

### 面板数据选项

```matlab
options.entityEffects % 是否包含个体固定效应
options.timeEffects   % 是否包含时间固定效应
options.weights       % 权重
```

### 因果推断选项

```matlab
options.modelY        % Y的估计模型('linear', 'lasso', 'ridge', 'forest')
options.modelT        % T的估计模型
options.discreteTreatment % T是否为离散型
options.randomState   % 随机种子
```

---

**文档版本**: 1.0.1  
**最后更新**: 2026-03-20

---

## Python 3.13 兼容性说明

本工具箱已通过 Python 3.13 + MATLAB R2025b 环境测试，功能完全正常。

**注意事项:**
- MathWorks 官方尚未正式宣布支持 Python 3.13，但实际测试中所有功能均正常运行
- 已通过 80 个测试用例验证，包括：
  - AcademicCorrectionsTest: 10/10 ✓
  - MLogitHACSpecializedTest: 29/29 ✓
  - PyBridgeConsistencyTest: 41/41 ✓
- 如遇到兼容性问题，建议回退到 Python 3.10 或 3.11
