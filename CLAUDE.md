# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

`scorecardpipeline` 是一个评分卡建模 Pipeline 包，封装 `toad`、`scorecardpy`、`optbinning` 等评分卡库，API 风格参考 sklearn，支持从特征筛选、分箱、WOE 编码到模型训练、评分卡生成、规则挖掘和分析、SWAP 分析、报告输出等完整流程。版本 0.1.39。

## 开发环境

- Python >= 3.8
- 安装依赖：`pip install -e .`
- 可选依赖（用于决策树可视化 PNG 输出）：`pip install scorecardpipeline[all]` 或单独安装 `pip install scorecardpipeline[graph]`
- 构建分发包：`python setup.py sdist bdist_wheel && pip install dist/scorecardpipeline-*.whl`

## 核心模块架构

### `scorecardpipeline/processing.py` — 特征处理

- `Combiner`: 特征分箱器，封装 toad + optbinning。`method` 参数支持 `"chi"`、`"dt"`、`"quantile"`、`"step"`、`"kmeans"`、`"cart"`、`"mdlp"`、`"uniform"`。`cart`/`mdlp`/`uniform` 使用 optbinning，其他使用 toad。
- `WOETransformer`: WOE 编码器，封装 toad，fit 后可 export/load JSON。
- `FeatureSelection`: 特征粗筛，支持 toad/scorecardpy 两种引擎。
- `StepwiseSelection`: 逐步回归筛选，基于 toad.selection.stepwise。
- `FeatureImportanceSelector`: 基于 CatBoost 的特征重要性筛选。
- `feature_bin_stats()`: 特征分箱统计表，输出各分箱的坏样本率、WOE、IV、KS 等指标。支持 overdue/dpds 多逾期标签分析。
- `feature_efficiency_analysis()`: 特征效能分析，同时输出自动分箱和分位数分箱结果。

### `scorecardpipeline/model.py` — 模型

- `ITLubberLogisticRegression`: 继承 sklearn LogisticRegression，fit 时计算标准误、z 值、p 值、VIF 等统计量，通过 `.summary()` 输出。
- `ScoreCard`: 重写 toad.ScoreCard，支持 Pipeline 模式。可导出评分卡刻度表（`.scorecard_scale()`）、分箱分数表（`.scorecard_points()`）、PMML（`.scorecard2pmml()`）。

### `scorecardpipeline/rule.py` — 规则

- `Rule`: 规则表达式类，支持 `|`/`&`/`^`/`~` 运算，用 `numexpr` 做加速求值。
- `ruleset_report()`: 批量评估规则集效果（逐条 + 汇总）。
- `sawpin_badrate_prediction_by_score()`: SWAP IN 分析，基于 base 分箱预测 test 的逾期率。支持单 target 和多逾期标签（overdue+dpds 组合）。
- `swapin_report()`: SWAPIN 报告，逐步累加置入样本，评估每步的坏账改善效果。

### `scorecardpipeline/rule_extraction.py` — 决策树规则挖掘

- `DecisionTreeRuleExtractor`: 循环训练决策树，挖掘高 LIFT 组合策略。SVG 可视化通过 dtreeviz 生成，PNG 转换依次尝试 cairosvg → svglib+reportlab，失败时打印警告。

### `scorecardpipeline/feature_selection.py` — 特征筛选器

大量 sklearn-style 筛选器：基于缺失率、众数比、IV、LIFT、方差、VIF、相关性、PSI、NullImportance、TargetPermutation 等。所有类继承 `SelectorMixin`，可用 `.plot()` 输出可视化。

### `scorecardpipeline/scorecard.py` — 评分转换器

- `StandardScoreTransformer`: 基于 PDO 公式的概率转评分转换器，支持分数裁剪、截断和 cutoff 判定。
- `BoxCoxScoreTransformer`: 基于 Box-Cox 变换的概率转评分转换器，自动学习最优变换参数。

### `scorecardpipeline/feature_engineering.py` — 特征派生

- `NumExprDerive`: 基于 numexpr 表达式的特征派生器，支持向量化计算。

### `scorecardpipeline/financial.py` — 财务计算

封装 numpy_financial，提供 `fv`、`pmt`、`nper`、`ipmt`、`ppmt`、`pv`、`rate`、`irr`、`npv`、`mirr` 等财务函数。

### `scorecardpipeline/explainability.py` — 模型解释性

- `ScorecardExplainer`: 基于 SHAP 的 WOE-LR 评分卡解释器，支持单样本解释与特征重要度汇总。
- 依赖 `shap`，安装方式：`pip install scorecardpipeline[explain]`。

### `scorecardpipeline/monitoring.py` — 模型监控

- `ModelMonitor`: 评分分布 PSI、特征 PSI、模型性能（KS/AUC）衰减监控。

### `scorecardpipeline/calibration.py` — 概率校准

- `ProbabilityCalibrator`: 支持 Platt Scaling 与 Isotonic Regression 的概率校准器。

### `scorecardpipeline/model_report.py` — 模型报告

- `QuickModelReport` / `auto_model_report()`: 生成多 Sheet 的 Excel 模型报告（目录、基本信息、模型性能、入模变量分析、稳定性分析、模型参数、部署需求）。支持 overdue/dpds 多标签。

### `scorecardpipeline/auto_report.py` — 数据测试报告

- `auto_data_testing_report()`: 三方数据或评分效果评估报告，支持多逾期标签分析，输出分箱图/KS图/直方图。

### `scorecardpipeline/excel_writer.py` — Excel 写入

- `ExcelWriter`: 基于 openpyxl，支持插入文本、数据框、图片、条件格式，读取 `template.xlsx` 作为样式模板。
- `dataframe2excel()`: 将 DataFrame 写入 Excel 并应用样式（合并列、百分比格式、条件格式等）。

### `scorecardpipeline/utils.py` — 可视化函数

- `bin_plot()`: 特征分箱可视化图，支持横向/纵向布局，显示好/坏样本堆叠柱状图、坏样本率折线、整体坏样本率参考线，以及 IV/KS/LIFT/趋势等统计指标。
- `bin_trend_plot()`: 特征分箱风险趋势图，支持按时间维度或指定维度列分组展示多面板分箱图。
- `batch_bin_trend_plot()`: 批量绘制多个特征的风险趋势图。
- `bin_overdues_plot()`: 多逾期天数分箱图，支持单目标和多逾期标签（overdue+dpds）分析。
- `ks_plot()`: KS曲线和ROC曲线。
- `hist_plot()`: 特征值分布直方图。
- `psi_plot()`: PSI稳定性分析图。
- `corr_plot()`: 特征相关性热力图。
- `distribution_plot()`: 样本时间分布图。

## 关键设计模式

- **`fit` + `transform` 范式**：所有数据处理组件均可放入 `sklearn.pipeline.Pipeline`。
- **Pipeline 超参搜索**：Step name + `__` 双下划线语法（如 `"logistic__C"`），配合 `GridSearchCV`、`Optuna` 使用。
- **多逾期标签分析**：overdue + dpd 参数组合，自动构建多级 MultiIndex 列名的分箱表，规则详情列优先。
- **numexpr 加速**：rule.py 和 feature_engineering.py 中的规则求值使用 `numexpr.evaluate()`，避免 Python 循环。
- **可选依赖降级**：SVG 转 PNG 失败时不中断流程，仅打印警告（Windows 上 CairoSVG 安装较困难）。

## 示例代码

```python
import scorecardpipeline as sc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 加载数据
data = sc.germancredit()
data['creditability'] = data['creditability'].map({"good": 0, "bad": 1})
train, test = train_test_split(data, test_size=0.3, stratify=data['creditability'])

# 构建 Pipeline
feature_pipeline = Pipeline([
    ("select", sc.FeatureSelection(target="creditability")),
    ("combiner", sc.Combiner(target="creditability", min_samples=0.2)),
    ("woe", sc.WOETransformer(target="creditability")),
])

woe_train = feature_pipeline.fit_transform(train)
model = sc.ITLubberLogisticRegression(target="creditability")
model.fit(woe_train)
print(model.summary())
```

## 参考资料

- 教程：https://scorecardpipeline.itlubber.art
- PyPI：https://pypi.org/project/scorecardpipeline
- GitHub：https://github.com/itlubber/scorecardpipeline