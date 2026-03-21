# R vs MATLAB/pyBridge Cross-Validation Report

Cross-validation report using R sandwich package to verify the correctness of pyBridge (statsmodels) Logit/MLogit HAC standard error implementation.

## Test Environment

| Component | Version |
|-----------|---------|
| R | 4.5.2 |
| sandwich | 3.1.1 |
| lmtest | 0.9.40 |
| mlogit | (latest) |
| MATLAB | R2025b |
| Python | 3.13 |
| statsmodels | 0.14.6 |

---

## 1. Logit Model

### 1.1 Classical SE

| Variable | R Coef | MATLAB Coef | Match | R SE | MATLAB SE | SE Match |
|----------|--------|-------------|-------|------|-----------|----------|
| (Intercept) | -0.260447 | -0.260447 | ✅ Exact | 0.177089 | 0.177089 | ✅ Exact |
| X1 | 1.414493 | 1.414493 | ✅ Exact | 0.226665 | 0.226665 | ✅ Exact |
| X2 | -0.754087 | -0.754087 | ✅ Exact | 0.202625 | 0.202625 | ✅ Exact |
| X3 | 0.538513 | 0.538513 | ✅ Exact | 0.171293 | 0.171293 | ✅ Exact |

> **Note**: R `glm(family=binomial)` and statsmodels `Logit` coefficients and Classical SE are exactly consistent.

### 1.2 HC1 SE

| Variable | R HC1 SE | MATLAB HC1 SE | Diff |
|----------|----------|---------------|------|
| (Intercept) | 0.179024 | 0.177225 | ⚠️ ~1.0% |
| X1 | 0.238927 | 0.236526 | ⚠️ ~1.0% |
| X2 | 0.172380 | 0.170648 | ⚠️ ~1.0% |
| X3 | 0.178002 | 0.176213 | ⚠️ ~1.0% |

> **⚠️ ~1.0% Difference Explanation**: R uses `n/(n-k)` small-sample correction factor, while statsmodels uses `n/(n-1)`. This formula difference results in approximately 1.0% systematic deviation; both are valid HC1 implementations.

### 1.3 HAC SE (Bartlett kernel, lag=5)

| Variable | R HAC SE | MATLAB HAC SE | Match |
|----------|----------|---------------|-------|
| (Intercept) | 0.165843 | 0.165843 | ✅ Exact |
| X1 | 0.213944 | 0.213944 | ✅ Exact |
| X2 | 0.170837 | 0.170837 | ✅ Exact |
| X3 | 0.162235 | 0.162235 | ✅ Exact |

**R command**:
```r
library(sandwich)
library(lmtest)
m <- glm(y ~ x1 + x2 + x3, family=binomial, data=df)
coeftest(m, vcov=NeweyWest(m, lag=5, prewhite=FALSE))
```

**pyBridge command**:
```matlab
result = sm.logistic(y, X, 'covType', 'HAC', 'lag', 5, 'kernel', 'bartlett');
```

> **Conclusion**: HAC Bartlett SE are exactly consistent, verifying that statsmodels Newey-West HAC implementation fully matches the R sandwich package.

---

## 2. Multinomial Logit Model

### 2.1 Classical SE

| Equation | Variable | R Coef | MATLAB Coef | Match | R SE | MATLAB SE | SE Match |
|----------|----------|--------|-------------|-------|------|-----------|----------|
| eq1 | (Intercept) | -1.190498 | -1.190498 | ✅ Exact | 0.167517 | 0.167517 | ✅ Exact |
| eq1 | X1 | 0.563850 | 0.563850 | ✅ Exact | 0.162016 | 0.162016 | ✅ Exact |
| eq1 | X2 | -0.485545 | -0.485545 | ✅ Exact | 0.153547 | 0.153547 | ✅ Exact |
| eq2 | (Intercept) | -1.147306 | -1.147306 | ✅ Exact | 0.167092 | 0.167092 | ✅ Exact |
| eq2 | X1 | 0.533228 | 0.533228 | ✅ Exact | 0.154521 | 0.154521 | ✅ Exact |
| eq2 | X2 | 0.512542 | 0.512542 | ✅ Exact | 0.154467 | 0.154467 | ✅ Exact |

> **Note**: Must use R `mlogit` package. `nnet::multinom` does not support sandwich HAC (lacks `estfun` method).

### 2.2 HAC SE (Bartlett kernel, lag=5)

| Equation | Variable | R HAC SE | MATLAB HAC SE | Match |
|----------|----------|----------|---------------|-------|
| eq1 | (Intercept) | 0.161034 | 0.161034 | ✅ Exact |
| eq1 | X1 | 0.142057 | 0.142057 | ✅ Exact |
| eq1 | X2 | 0.157559 | 0.157559 | ✅ Exact |
| eq2 | (Intercept) | 0.154579 | 0.154579 | ✅ Exact |
| eq2 | X1 | 0.157399 | 0.157399 | ✅ Exact |
| eq2 | X2 | 0.152904 | 0.152904 | ✅ Exact |

**R command**:
```r
library(mlogit)
library(sandwich)
library(lmtest)
mdata <- dfidx(df, choice="y", idx=list(c("id","alt")))
m <- mlogit(y ~ 0 | x1 + x2, data=mdata, reflevel="1")
coeftest(m, vcov=NeweyWest(m, lag=5, prewhite=FALSE))
```

**pyBridge command**:
```matlab
result = sm.multinomialLogit(y, X, 'covType', 'HAC', 'lag', 5, 'kernel', 'bartlett');
```

> **Conclusion**: MLogit HAC SE are exactly consistent.

---

## 3. Test Statistic Type Verification

| Check | Result |
|-------|--------|
| R `coeftest` for Logit reports | "z test of coefficients" |
| R `coeftest` for MLogit reports | "z test of coefficients" |
| pyBridge field name | `zStatistics` (renamed from `tStatistics`) |

> **Conclusion**: MLE models (Logit/MLogit/Probit, etc.) use **z-values** (based on asymptotic normal distribution), not t-values. pyBridge has renamed the field from `tStatistics` to `zStatistics` for MLE models to accurately reflect the nature of the statistic.

---

## Known Differences

| Issue | Description |
|-------|-------------|
| **HC1 SE ~1.0%** | R sandwich uses `n/(n-k)` small-sample correction, while statsmodels uses `n/(n-1)`, resulting in approximately 1.0% systematic difference. Both correction methods are recognized HC1 variants in academic literature. |
| **R multinom + HAC** | `nnet::multinom` objects lack `estfun()` method, do not support sandwich HAC. Must use `mlogit` package for MLogit HAC validation instead. |
| **prewhite parameter** | R `NeweyWest` defaults to `prewhite=TRUE` (VAR(1) prewhitening), must explicitly set `prewhite=FALSE` to match statsmodels. |

---

## Summary

| Model | SE Type | Coefficients | Standard Errors |
|-------|---------|:------------:|:---------------:|
| Logit | Classical | ✅ Exact | ✅ Exact |
| Logit | HC1 | ✅ Exact | ⚠️ ~1.0% |
| Logit | HAC Bartlett | ✅ Exact | ✅ Exact |
| MLogit | Classical | ✅ Exact | ✅ Exact |
| MLogit | HAC Bartlett | ✅ Exact | ✅ Exact |

**Overall**: pyBridge (statsmodels) Logit/MLogit HAC standard error implementation is **exactly consistent** with the R sandwich package, verifying the correctness of the underlying Newey-West HAC computation. The minor HC1 differences stem from different small-sample correction conventions, which are known and acceptable implementation differences.

---

*Report generated for MATLAB-Python-Econometrics-Bridge project quality assurance.*
*Cross-validation performed against R sandwich 3.1.1 + lmtest 0.9.40 + mlogit packages.*
