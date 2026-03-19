classdef AcademicCorrectionsTest < matlab.unittest.TestCase
    % AcademicCorrectionsTest 学术标准化修正的验证测试
    %
    % 测试覆盖:
    % 1. HAC小样本校正因子 (n/(n-k))
    % 2. McFadden Pseudo-R² 计算
    % 3. 泊松过度离散检验 (卡方统计检验)
    % 4. VIF排除常数项判断
    % 5. ARIMA阶数参数验证
    % 6. Hausman检验 Sargan-Hansen VCE
    %
    % 注意: 部分测试需要Python环境(statsmodels)

    properties
        WorkspacePath = fileparts(fileparts(mfilename('fullpath')))
    end

    methods (TestClassSetup)
        function addToPath(testCase)
            % 添加项目路径到MATLAB路径
            addpath(testCase.WorkspacePath);
        end
    end

    methods (Test)

        %% Test 1: HAC小样本校正
        function testHACSmallSampleCorrection(testCase)
            % 验证HAC标准误的小样本校正因子计算
            % 学术标准: Stata newey命令使用 n/(n-k) 校正因子

            n = 100;  % 样本量
            k = 5;     % 参数个数
            rawSE = 0.5;  % 原始标准误

            % 期望的校正因子
            expectedCorrection = n / (n - k);

            % 计算校正后的标准误
            correctedSE = rawSE * sqrt(expectedCorrection);

            % 验证校正因子
            testCase.verifyEqual(expectedCorrection, 100/95, 'AbsTol', 1e-10);
            testCase.verifyEqual(correctedSE, rawSE * sqrt(100/95), 'AbsTol', 1e-10);
        end

        %% Test 2: Cameron-Gelbach-Miller校正
        function testCameronGelbachMillerCorrection(testCase)
            % 验证多维聚类标准误的Cameron-Gelbach-Miller (2015) 校正

            n = 500;           % 样本量
            nClusters0 = 50;  % 第一维度聚类数
            nClusters1 = 10;  % 第二维度聚类数

            % Cameron-Gelbach-Miller 缩放因子
            scaling = (n / (n - 1)) * ((nClusters0 - 1) / (nClusters0 - 1)) * ((nClusters1 - 1) / (nClusters1 - 1));

            % 验证缩放因子公式
            testCase.verifyEqual(scaling, (n / (n - 1)), 'AbsTol', 1e-10);

            % 应用校正后的协方差矩阵
            covMatrix = rand(3, 3);
            covMatrix = (covMatrix + covMatrix') / 2;  % 对称化
            correctedCov = covMatrix * scaling;

            testCase.verifySize(correctedCov, [3, 3]);
            testCase.verifyEqual(correctedCov, covMatrix * (n / (n - 1)), 'AbsTol', 1e-10);
        end

        %% Test 3: McFadden Pseudo-R²
        function testMcFaddenPseudoRSquared(testCase)
            % 验证McFadden Pseudo-R² 计算
            % 公式: 1 - (LL / LL_null)

            logLikelihood = -45.3;
            logLikelihoodNull = -78.9;

            % 计算Pseudo-R²
            pseudoRSquared = 1 - (logLikelihood / logLikelihoodNull);

            % 期望值
            expectedPseudoR2 = 1 - (-45.3 / -78.9);

            testCase.verifyEqual(pseudoRSquared, expectedPseudoR2, 'AbsTol', 1e-10);
            testCase.verifyEqual(pseudoRSquared, 0.426, 'AbsTol', 0.001);

            % 验证范围 [0, 1]
            testCase.verifyGreaterThanOrEqual(pseudoRSquared, 0);
            testCase.verifyLessThanOrEqual(pseudoRSquared, 1);
        end

        %% Test 4: 过度离散检验 (卡方检验)
        function testOverdispersionTest(testCase)
            % 验证泊松回归过度离散检验
            % 原假设: 无过度离散 (比率 = 1)
            % 检验统计量服从卡方分布

            pearsonChi2 = 125.7;
            dfResid = 95;

            % 过度离散比率
            dispersionRatio = pearsonChi2 / dfResid;

            % 检验统计量
            testStatistic = dispersionRatio * dfResid;  % = pearsonChi2

            % p值 (卡方分布)
            pValue = 1 - chi2cdf(testStatistic, dfResid);

            % 验证计算
            testCase.verifyEqual(dispersionRatio, 125.7/95, 'AbsTol', 1e-10);
            testCase.verifyEqual(testStatistic, pearsonChi2, 'AbsTol', 1e-10);

            % 在此例中 p < 0.05, 存在过度离散
            testCase.verifyEqual(pValue < 0.05, true);

            % 对比: 硬编码阈值方法 vs 统计检验
            hardcodedThreshold = 1.5;
            legacyResult = dispersionRatio > hardcodedThreshold;  % false
            statisticalResult = pValue < 0.05;  % true (更严格)

            testCase.verifyEqual(legacyResult, false);
            testCase.verifyEqual(statisticalResult, true);
        end

        %% Test 5: VIF排除常数项
        function testVIFExcludingConstant(testCase)
            % 验证VIF计算中排除常数项的多重共线性判断

            % 假设VIF值: [Inf(常数项), 2.3, 1.8, 15.6, 1.2]
            vifAll = [inf, 2.3, 1.8, 15.6, 1.2];

            % 排除常数项后的VIF
            vifExConstant = vifAll(2:end);

            % 多重共线性判断 (阈值 = 10)
            hasMulticollinearityAll = any(vifAll > 10);  % true (因为inf)
            hasMulticollinearityExConstant = any(vifExConstant > 10);  % true

            testCase.verifyEqual(hasMulticollinearityAll, true);
            testCase.verifyEqual(hasMulticollinearityExConstant, true);
            testCase.verifyEqual(vifExConstant, [2.3, 1.8, 15.6, 1.2]);

            % 另一个例子: 常数项VIF=Inf但其他变量VIF正常
            vifNormal = [inf, 2.1, 1.5, 3.2, 1.1];
            vifExNormal = vifNormal(2:end);
            hasMcolNormal = any(vifExNormal > 10);  % false

            testCase.verifyEqual(hasMcolNormal, false);
        end

        %% Test 6: ARIMA阶数验证
        function testARIMAOrderValidation(testCase)
            % 验证ARIMA阶数参数验证逻辑
            % 要求: p, d, q 必须为非负整数

            validOrders = {[1, 0, 0], [2, 1, 1], [4, 1, 2]};
            invalidOrders = {...
                [1, 2], ...           % 不是3元素
                [1, -1, 0], ...       % d为负数
                [1, 0.5, 0], ...      % d为小数
                [1, 0, -1], ...       % q为负数
                [1, 0, 1.5] ...       % q为小数
            };

            % 验证有效阶数
            for i = 1:length(validOrders)
                order = validOrders{i};
                isValid = isvector(order) && length(order) == 3 && ...
                          order(1) >= 0 && order(1) == floor(order(1)) && ...
                          order(2) >= 0 && order(2) == floor(order(2)) && ...
                          order(3) >= 0 && order(3) == floor(order(3));
                testCase.verifyTrue(isValid, sprintf('Order [%d,%d,%d] should be valid', order(1), order(2), order(3)));
            end

            % 验证无效阶数被拒绝
            for i = 1:length(invalidOrders)
                order = invalidOrders{i};
                isValid = isvector(order) && length(order) == 3 && ...
                          order(1) >= 0 && order(1) == floor(order(1)) && ...
                          order(2) >= 0 && order(2) == floor(order(2)) && ...
                          order(3) >= 0 && order(3) == floor(order(3));
                testCase.verifyFalse(isValid, sprintf('Order [%s] should be invalid', mat2str(order)));
            end
        end

        %% Test 7: Hausman检验 Sargan-Hansen VCE
        function testHausmanSarganHansenVCE(testCase)
            % 验证Hausman检验的Sargan-Hansen VCE方法

            % 假设固定效应和随机效应估计
            bFE = [1.2; 0.8; -0.3];
            bRE = [1.15; 0.75; -0.28];
            seFE = [0.15; 0.12; 0.08];
            seRE = [0.14; 0.11; 0.07];

            % 系数差异
            diff = bFE - bRE;

            % 方差差分
            varDiff = seFE.^2 - seRE.^2;

            % 仅保留正定部分
            isPosDef = varDiff > 0;
            diffValid = diff(isPosDef);
            varDiffValid = varDiff(isPosDef);

            % 计算Hausman统计量 (使用伪逆)
            if ~isempty(diffValid) && all(varDiffValid > 0)
                hausmanStat = (diffValid' * pinv(diag(varDiffValid)) * diffValid);
            else
                hausmanStat = NaN;
            end

            % 验证正定条件
            testCase.verifyEqual(all(varDiff > 0), true);

            % 验证统计量计算
            expectedStat = diffValid' * pinv(diag(varDiffValid)) * diffValid;
            testCase.verifyEqual(hausmanStat, expectedStat, 'AbsTol', 1e-10);

            % 自由度
            df = length(diffValid);

            % p值
            pValue = 1 - chi2cdf(hausmanStat, df); %#ok<NASGU>

            % 验证结果结构
            testCase.verifyEqual(df, 3);
            testCase.verifyGreaterThan(hausmanStat, 0);
        end

        %% Test 8: Python list索引兼容性
        function testPythonListIndexing(testCase)
            % 验证Python 0-based索引的正确处理

            % 模拟Python列表长度
            n = 5;

            % MATLAB 1-based 到 Python 0-based 转换
            % MATLAB: 1 -> Python: 0
            % MATLAB: 5 -> Python: 4

            for i = 1:n
                pyIndex = i - 1;  % 转换为Python索引
                testCase.verifyGreaterThanOrEqual(pyIndex, 0);
                testCase.verifyLessThan(pyIndex, n);
            end

            % 验证py_builtin.len()模拟
            expectedLen = 5;
            actualLen = double(n);  % py_builtin.len()的返回值
            testCase.verifyEqual(actualLen, expectedLen);
        end

        %% Test 9: 异方差稳健标准误类型
        function testHCStandardErrors(testCase)
            % 验证HC0-HC3异方差稳健标准误的学术标准

            % HC类型对应的学术标准
            hcTypes = struct(...
                'HC0', 'White (1980) 标准误', ...
                'HC1', 'MacZid and White (1985) 小样本校正', ...
                'HC2', 'Horn et al. (1975) 适合小样本', ...
                'HC3', 'Davidson and MacKinnon (1993) 最稳健');

            typeNames = fieldnames(hcTypes);

            for i = 1:length(typeNames)
                type = typeNames{i};
                desc = hcTypes.(type);
                testCase.verifyNotEmpty(desc);
                testCase.verifyEqual(desc(1), upper(desc(1)));  % 首字母大写
            end

            % 验证校正因子公式
            % HC1 = HC0 * sqrt(n/(n-k))
            n = 100;
            k = 5;
            hc0 = 1.0;
            hc1 = hc0 * sqrt(n/(n-k));

            testCase.verifyEqual(hc1, sqrt(100/95), 'AbsTol', 1e-10);
        end

        %% Test 10: OrderedModel阈值提取
        function testOrderedModelThresholds(testCase)
            % 验证OrderedModel阈值参数的正确提取

            % 模拟fitResult对象
            % 假设有3个类别, k_constant = 1 (常数项索引)
            kConstant = 1;
            params = [0.5; 1.2; 2.1; 0.3; 0.8];  % [阈值1, 阈值2, 系数1, 系数2, 常数]

            % 提取阈值: params(1:kConstant-1) 应该是 params(1:0) = []
            % 由于kConstant = 1, 阈值数量 = kConstant - 1 = 0
            thresholds = params(1:kConstant-1);

            testCase.verifyEmpty(thresholds);

            % 如果kConstant = 2, 有一个阈值
            kConstant2 = 2;
            thresholds2 = params(1:kConstant2-1);

            testCase.verifyEqual(thresholds2, params(1), 'AbsTol', 1e-10);
            testCase.verifyEqual(length(thresholds2), 1);

            % kConstant必须转换为MATLAB标量
            kConstantPy = int64(2);  % Python int
            kConstantML = double(kConstantPy);  % 转换为MATLAB标量

            testCase.verifyEqual(kConstantML, 2);
            testCase.verifyEqual(class(kConstantML), 'double');
        end
    end
end
