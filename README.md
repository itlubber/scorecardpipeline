# 评分卡pipeline建模包

<img src="https://itlubber.art/upload/scorecardpipeline.png" alt="itlubber.png" width="100%" border=0/> 

`scorecardpipeline` 封装了 `toad`、`scorecardpy`、`optbinning` 等评分卡建模相关组件，`API` 风格与 `sklearn` 高度一致，支持 `pipeline` 式端到端评分卡建模、模型报告输出、导出 `PMML` 文件、超参数搜索等

> 教程：https://itlubber.art/upload/scorecardpipeline.html
> 
> pipy包：https://pypi.org/project/scorecardpipeline/
>
> 仓库地址：https://github.com/itlubber/scorecardpipeline
> 
> 博文地址：https://itlubber.art/archives/itlubber-scorecard-end2end
> 
> 微信公共号推文：https://mp.weixin.qq.com/s/eCTp4h0fau77xOgf_V28wQ


## 交流

|  微信 |  微信公众号 |
| :---: | :----: |
| <img src="https://itlubber.art//upload/itlubber.png" alt="itlubber.png" width="50%" border=0/> | <img src="https://itlubber.art//upload/itlubber_art.png" alt="itlubber_art.png" width="50%" border=0/> |
|  itlubber  | itlubber_art |


## 引言

作为一名金融搬砖工作者，评分卡建模怎么也算是基操。本文主要对笔者日常使用的评分卡建模代码进行讲解，说明如何一步步从原始数据到最终评分卡模型以及如何解读产出的模型报告文档。

本文所有代码已全部提交至笔者GITHUB公开仓库，各位看官按需取用，用完记得顺带给个star以鼓励笔者继续开源相关工作。

本文使用笔者对toad、scorecardpy、optbinning等库进行二次封装后的代码进行实操，文中会对仓库中的部分代码细节进行说明。本文旨在对仓库评分卡建模流程进行说明，并提供一个可以直接运行的完整示例，让更多金融从业小伙伴掌握整套评分卡模型构建方法。


## 项目说明

### 仓库地址

https://github.com/itlubber/scorecardpipeline

### 代码结构

该仓库下代码主要用于提供评分卡建模相关的组件，项目结构如下：

```base
>> tree
.
├── LICENSE                         # 开源协议
├── README.md                       # 相关说明文档
├── requirements.txt                # 相关依赖包
└── setup.py                        # 打包文件
├── examples                        # 演示样例
│   └── scorecard_samples.ipynb
└── scorecardpipeline               # scorecardpipeline 包文件
    ├── excel_writer.py             # 操作 excel 的公共方法
    ├── template.xlsx               # excel 模版文件
    ├── matplot_chinese.ttf         # 中文字体
    ├── processing.py               # 数据处理相关代码
    ├── model.py                    # 模型相关代码
    └── utils.py                    # 公用方法
```

### 简要说明

+ `processing` 中提供了数据前处理相关的方法：特征筛选方法（`FeatureSelection`、`StepwiseSelection`）、变量分箱方法（`Combiner`）、变量证据权重转换方法（`WOETransformer`），方法继承`sklearn.base`中的`BaseEstimator`和`TransformerMixin`，能够支持构建`pipeline`和超参数搜索
+ `model`中提供了基于`sklearn.linear_model.LogisticRegression`实现的`ITLubberLogisticRegression`，同时重写了`toad.ScoreCard`，以支持模型相关内容的输出
+ `excel_writer` 中提供了操作 `excel` 的一系列公共方法，包含设置条件格式、设置列宽、设置数字格式、插入指定样式的内容（`insert_value2sheet`）、插入图片数据（`insert_pic2sheet`）、插入`dataframe`数据内容（`insert_df2sheet`）、保存`excel`文件（`save`）等方法

### `scorecardpipeline` 安装

+ `pipy` 安装

```bash
>> pip install scorecardpipeline -i https://pypi.Python.org/simple/

Looking in indexes: https://pypi.Python.org/simple/
Collecting scorecardpipeline
  Downloading scorecardpipeline-0.1.5-py3-none-any.whl (36 kB)
  ......
Installing collected packages: sklearn-pandas, scorecardpy, sklearn2pmml, scorecardpipeline
Successfully installed scorecardpipeline-0.1.5 scorecardpy-0.1.9.2 sklearn-pandas-2.2.0 sklearn2pmml-0.92.2
```

+ 源码编译

```bash
python setup.py sdist bdist_wheel
pip install dist/scorecardpipeline-0.1.11-py3-none-any.whl
# twine upload dist/*
```


## 评分卡建模

### 数据准备

笔者仓库下几乎所有评分卡相关项目默认使用`scorecardpy`库中提供的`germancredit`数据集进行示例，数据集中包含类别型变量、数值型变量、样本好坏标签，共`1000`条数据，数据集中不包含缺失值，在部分示例中会随机替换数据集中的内容为缺失值，模拟实际生产中的真实数据。

```python
target = "creditability"
data = sc.germancredit()
data[target] = data[target].map({"good": 0, "bad": 1})
```

### 数据集拆分

通常建模过程中会将数据集拆分成训练数据集和测试数据集集，训练数据集用于模型参数的训练，测试数据集用于评估模型好坏，且在整个建模过程中实际上只能使用一次（任何针对测试数据集调模型的，都存在模型过拟合的情况，除非你的数据不包含任何噪声数据）。而为了弥补测试数据集只能用一次用于评估模型好坏的尴尬，大多都会再拆分一个验证数据集，用于模型训练过程中判别模型是否向好的方向进行优化。

在金融建模场景中，上述三个数据集通常被称为训练数据集、测试数据集（即上述三个数据集中的验证集）、跨时间验证集（即上述三个数据集中的测试集）。同时，在金融场景中，跨时间验证集的拆分方式通常使用类似时间序列数据集的拆分方法，按照时间将数据集拆分为建模集（包含训练集和测试集）、跨时间验证集（Out of Time，OOT），以保证模型在不同时间段内的泛化能力。

本文为了简单起见，只拆分了训练集、测试集，OOT直接拷贝训练集来演示评分卡建模完整过程。

+ 数据集拆分示例


```python
train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[target])
oot = data.copy()
```

### 特征粗筛（可选）

通常当数据特征数量较多时，针对每个特征做分箱太耗时间的情况下，可以在分箱之前针对数据集进行一次简单的特征筛选工作。剔除数据集中空值或唯一值占比较高、变量之间相关性过高、变量`IV值`（`Information Value`）过低的特征。

+ 方法简介

```python
class FeatureSelection(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", empty=0.95, iv=0.02, corr=0.7, exclude=None, return_drop=True, identical=0.95, remove=None, engine="scorecardpy", target_rm=False):
        """
        ITLUBBER提供的特征筛选方法

        Args:
            target: 数据集中标签名称，默认 target
            empty: 空值率，默认 0.95, 即空值占比超过 95% 的特征会被剔除
            iv: IV值，默认 0.02，即iv值小于 0.02 时特征会被剔除
            corr: 相关性，默认 0.7，即特征之间相关性大于 0.7 时会剔除iv较小的特征
            identical: 唯一值占比，默认 0.95，即当特征的某个值占比超过 95% 时，特征会被剔除
            engine: 特征筛选使用的引擎，可选 "toad", "scorecardpy" 两种，默认 scorecardpy
            remove: 引擎使用 scorecardpy 时，可以传入需要强制删除的变量
            return_drop: 是否返回删除信息，默认 True，即默认返回删除特征信息
            target_rm: 是否剔除标签，默认 False，即不剔除
            exclude: 是否需要强制保留某些特征
        """
```

+ 使用方法


```python
# 常规使用方法
selection = FeatureSelection(target=target, engine="toad", return_drop=True, corr=0.9, iv=0.01)
train = selection.fit_transform(train)

# pipeline 使用方法
feature_pipeline = Pipeline([
        ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ......
    ])
feature_pipeline.fit(train)
```

### 特征分箱

在评分卡建模过程中，为了保证模型效果的稳定性，需要对变量进行一定的前处理。

对于数值型变量，常规的处理是使用等频分箱、等距分箱、决策树分箱、卡方分箱、最优KS分箱等方式对数值型变量进行离散化处理，以保证特征稳定的同时，能够从业务上有一定的逻辑依据和解释性。对数值型变量进行分箱还可以减少异常值对模型的影响，降低模型复杂度。

对于类别型变量，很多类别的客群属性比较相似，客户违约概率相近，将这些客群合并能够增加变量稳定性、降低模型的复杂程度和增强特征可解释性。

为了对上述两类特征进行分箱，笔者将toad和optbinning两个库的分箱功能结合，合并进了processing.Combiner，能够支持变量单调分箱、“U”型分箱以及optbinning支持的所有形式的最优特征分箱方案，同时保留缺失值单独一箱，保证了评分卡模型后期上线后缺失值不知道如何填充的尴尬。

+ 方法简介

```python
class Combiner(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", method='chi', engine="toad", empty_separate=False, min_samples=0.05, min_n_bins=2, max_n_bins=3, max_n_prebins=10, min_prebin_size=0.02, min_bin_size=0.05, max_bin_size=None, gamma=0.01, monotonic_trend="auto_asc_desc", rules={}, n_jobs=1):
        """
        特征分箱封装方法

        Args:
            target: 数据集中标签名称，默认 target
            method: 特征分箱方法，可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform", 参考 toad.Combiner & optbinning.OptimalBinning
            engine: 分箱引擎，可选 "optbinning", "toad"
            empty_separate: 是否空值单独一箱, 默认 False，推荐设置为 True
            min_samples: 最小叶子结点样本占比，参考对应文档进行设置，默认 5%
            min_n_bins: 最小分箱数，默认 2，即最小拆分2箱
            max_n_bins: 最大分像素，默认 3，即最大拆分3箱，推荐设置 3 ～ 5，不宜过多，偶尔使用 optbinning 时不起效
            max_n_prebins: 使用 optbinning 时预分箱数量
            min_prebin_size: 使用 optbinning 时预分箱叶子结点（或者每箱）样本占比，默认 2%
            min_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最小样本占比，默认 5%
            max_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最大样本占比，默认 None
            gamma: 使用 optbinning 分箱时限制过拟合的正则化参数，值越大惩罚越多，默认 0。01
            monotonic_trend: 使用 optbinning 正式分箱时的坏率策略，默认 auto，可选 "auto", "auto_heuristic", "auto_asc_desc", "ascending", "descending", "convex", "concave", "peak", "valley", "peak_heuristic", "valley_heuristic"
            rules: 自定义分箱规则，toad.Combiner 能够接收的形式
            n_jobs: 使用多进程加速的worker数量，默认单进程
        """
```

+ 使用方法


```python
combiner = Combiner(min_samples=0.2, empty_separate=True, target=target)
combiner.fit(train)
train = combiner.transform(train)

# pipeline
feature_pipeline = Pipeline([
......
        ("combiner", Combiner(target=target, min_samples=0.2)),
        ......
    ])
feature_pipeline.fit(train)

# save all bin_plot
_combiner = feature_pipeline.named_steps["combiner"]
for col in woe_train.columns:
    if col != target:
        _combiner.bin_plot(train, col, labels=True, save=f"outputs/bin_plots/train_{col}.png")
        _combiner.bin_plot(test, col, labels=True, save=f"outputs/bin_plots/test_{col}.png")
        _combiner.bin_plot(oot, col, labels=True, save=f"outputs/bin_plots/oot_{col}.png")
```

### 特征编码

特征分箱后，本质上只是将特征离散化为几箱，每一箱的值被赋予了一个标签，例如0、1、2、3、...，尽管这些标签在一定程度上也能够表征客户的风险水平，能够作为输入建立一个模型，但基于上述标签构建的模型在评估客户风险状况时比较局限，很难精准刻画客户的风险状况。

对于上述离散化后的特征，常规的处理方式有ONE-HOT编码、顺序编码、TARGET编码、COUNT编码等方式。在金融建模场景，尤其是评分卡建模时，通常使用WOE编码对离散化后的特征进行编码。WOE编码将每个分箱内的坏样本比例除以好样本比例后取对数来编码，能够反映每个分箱内客户的坏客户分布与好客户分布之间的差异以及该箱内坏好比(odds)相对于总体的坏好比之间的差异性。

+ 笔者基于toad.transform.WOETransformer重写了相关的方法，方法如下：


```python
class WOETransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", exclude=None):
        """
        WOE转换器

        Args:
            target: 数据集中标签名称，默认 target
            exclude: 不需要转换 woe 的列
        """
```

+ 使用方法

```python
# 常规使用方式
transformer = WOETransformer(target=target)
train = transformer.fit_transform(train)

# pipeline
feature_pipeline = Pipeline([
    ......
    ("transformer", WOETransformer(target=target)),
    ......
])
feature_pipeline.fit(train)
```

### 特征精筛

通常对特征`WOE`编码转换后，特征之间的相关性会上升、唯一值占比会增加、变量`IV`值会下降（同一特征分箱数越多`IV`值越大），为了避免入模特征`相关性过高`、`唯一值占比过高`、`IV过低`等问题的出现，需要在转换编码后进行特征精筛，筛选的限制条件相较分箱前的粗筛更严，以保证模型稳定性和泛化能力。

相关方法参照特征粗筛，在此不过多赘述。


### 逐步回归特征筛选

特征在进行完前面的步骤后，在正式建模前，通常还会使用逐步回归对特征做进一步的筛选。通过假设检验的方法来得到最优的入模特征组合，能够有效的解决特征之间的多重共线性。

+ 笔者基于toad.selection.stepwise重新实现了StepwiseSelection，参数与toad包中的所有参数一致，方法如下：


```python
class StepwiseSelection(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", estimator="ols", direction="both", criterion="aic", max_iter=None, return_drop=True, exclude=None, intercept=True, p_value_enter=0.2, p_remove=0.01, p_enter=0.01, target_rm=False):
        """
        逐步回归筛选方法

        Args:
            target: 数据集中标签名称，默认 target
            estimator: 预估器，默认 ols，可选 "ols", "lr", "lasso", "ridge"，通常默认即可
            direction: 逐步回归方向，默认both，可选 "forward", "backward", "both"，通常默认即可
            criterion: 评价指标，默认 aic，可选 "aic", "bic"，通常默认即可
            max_iter: 最大迭代次数，sklearn中使用的参数，默认为 None
            return_drop: 是否返回特征剔除信息，默认 True
            exclude: 强制保留的某些特征
            intercept: 是否包含截距，默认为 True
            p_value_enter: 特征进入的 p 值，用于前向筛选时决定特征是否进入模型
            p_remove: 特征剔除的 p 值，用于后向剔除时决定特征是否要剔除
            p_enter: 特征 p 值，用于判断双向逐步回归是否剔除或者准入特征
            target_rm: 是否剔除数据集中的标签，默认为 False，即剔除数据集中的标签
        """
```

+ 使用方法

```python
# 常规使用方法
stepwise = StepwiseSelection(target=target)
train = stepwise.fit_transform(train)

# pipeline
feature_pipeline = Pipeline([
        ......
        ("stepwise", StepwiseSelection(target=target, target_rm=False)),
        ......
    ])
feature_pipeline.fit(train)
```

## 逻辑回归建模

逻辑回归基本上都是使用的`sklearn.linear_model`模块里的`LogisticRegression`(偏模型)或者`statsmodels.api`提供的`Logit`(偏统计)，两者各有优点。`sklearn`库提供了可以自行调整超参数的逻辑回归模型，`statsmodels`库提供了很完整的模型摘要输出，两者类似鱼与熊掌一样，不可兼得。

笔者参考社区搜寻了很多评分卡建模相关的仓库，最终在`skorecard.liner_model`中`LogisticRegression`的实现上进行优化，增加了一些相关的统计信息输出，实现了类似`statsmodels`的模型输出的效果，同时保留了`sklearn`版`LogisticRegression`可以手工设置超参数或者基于超参数搜索框架搜寻最有超参数组合的特性。同时，也对`statsmodels`库中的`Logit`进行了一定的封装，以满足笔者评分卡建模的`pipeline`组件要求以及更详细的模型摘要信息输出和保存。

+ 方法简介

```python
class ITLubberLogisticRegression(LogisticRegression):
    """
    Extended Logistic Regression.
    Extends [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
    This class provides the following extra statistics, calculated on `.fit()` and accessible via `.summary()`:
    - `cov_matrix_`: covariance matrix for the estimated parameters.
    - `std_err_intercept_`: estimated uncertainty for the intercept
    - `std_err_coef_`: estimated uncertainty for the coefficients
    - `z_intercept_`: estimated z-statistic for the intercept
    - `z_coef_`: estimated z-statistic for the coefficients
    - `p_value_intercept_`: estimated p-value for the intercept
    - `p_value_coef_`: estimated p-value for the coefficients
    
    Example:
    ```python
    feature_pipeline = Pipeline([
        ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("combiner", Combiner(target=target, min_samples=0.2)),
        ("transform", WOETransformer(target=target)),
        ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("stepwise", StepwiseSelection(target=target)),
        # ("logistic", LogisticClassifier(target=target)),
        ("logistic", ITLubberLogisticRegression(target=target)),
    ])
    
    feature_pipeline.fit(train)
    summary = feature_pipeline.named_steps['logistic'].summary()
    ```
    
    An example output of `.summary()`:
    
    | | Coef. | Std.Err | z | P>|z| | [ 0.025 | 0.975 ] | VIF |
    |:------------------|----------:|----------:|---------:|------------:|-----------:|----------:|--------:|
    | const | -0.844037 | 0.0965117 | -8.74544 | 2.22148e-18 | -1.0332 | -0.654874 | 1.05318 |
    | duration.in.month | 0.847445 | 0.248873 | 3.40513 | 0.000661323 | 0.359654 | 1.33524 | 1.14522 |
    """

    def __init__(self, target="target", penalty="l2", calculate_stats=True, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=False, n_jobs=None, l1_ratio=None,):
        """
        Extends [sklearn.linear_model.LogisticRegression.fit()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
        
        Args:
            target (str): your dataset's target name
            calculate_stats (bool): If true, calculate statistics like standard error during fit, accessible with .summary()
        """
```

+ 使用方法

```python
# 常规使用方法
logistic = ITLubberLogisticRegression(target=target)
logistic.fit(woe_train)

logistic.plot_weights(save="outputs/logistic_train.png")
summary = logistic.summary().reset_index().rename(columns={"index": "Features"})
train_corr = logistic.corr(woe_train, save="outputs/train_corr.png")
train_report = logistic.report(woe_train)

# pipeline
feature_pipeline = Pipeline([
    ......
    ("logistic", ITLubberLogisticRegression(target=target)),
])

feature_pipeline.fit(train)
y_pred_train = feature_pipeline.predict(train.drop(columns=target))
```


### 评分卡转换

笔者重写了`toad.ScoreCard`以适配`pipeline`模式的端到端的评分卡建模，同时支持传入训练好的逻辑回归模型、输出评分排序性、评分分布、评分稳定性等。

+ 相关代码说明

```python
class ScoreCard(toad.ScoreCard, TransformerMixin):
    
    def __init__(self, target="target", pdo=60, rate=2, base_odds=35, base_score=750, combiner={}, transer=None, pretrain_lr=None, pipeline=None, **kwargs):
        """
        评分卡模型转换

        Args:
            target: 数据集中标签名称，默认 target
            pdo: odds 每增加 rate 倍时减少 pdo 分，默认 60
            rate: 倍率
            base_odds: 基础 odds，通常根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比，默认 35，即 35:1 => 0.972 => 坏样本率 2.8%
            base_score: 基础 odds 对应的分数，默认 750
            combiner: 分箱转换器，传入 pipeline 时可以为None
            transer: woe转换器，传入 pipeline 时可以为None
            pretrain_lr: 预训练好的逻辑回归模型，可以不传
            pipeline: 训练好的 pipeline，必须包含 Combiner 和 WOETransformer
            **kwargs: 其他相关参数，具体参考 toad.ScoreCard
        """
```

+ 使用方法

```python
# 使用方法
feature_pipeline = Pipeline([
        ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("combiner", Combiner(target=target, min_samples=0.2)),
        ("transformer", WOETransformer(target=target)),
        ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("stepwise", StepwiseSelection(target=target, target_rm=False)),
    ])
woe_train = feature_pipeline.fit_transform(train)

combiner = feature_pipeline.named_steps['combiner'].combiner
transformer = feature_pipeline.named_steps['transformer'].transformer

score_card = ScoreCard(target=target, combiner=combiner, transer=transformer, )
score_card.fit(woe_train)

data["score"] = score_card.transform(data)

# 评分效果
clip = 50
clip_start = max(math.ceil(train["score"].min() / clip) * clip, math.ceil(train["score"].quantile(0.01) / clip) * clip)
clip_end = min(math.ceil(train["score"].max() / clip) * clip, math.ceil(train["score"].quantile(0.99) / clip) * clip)
score_clip = [i for i in range(clip_start, clip_end, clip)]

train_score_rank = feature_bin_stats(train, "score", target=target, rules=score_clip, verbose=0, method="step", ks=True)
ks_plot(train["score"], train[target], title="Train Dataset", save="model_report/train_ksplot.png")
score_hist(train["score"], train[target], save="model_report/train_scorehist.png", bins=30, figsize=(13, 10))
```


### 超参数优化

笔者根据`sklearn.pipeline`的构造规则，重写了评分卡建模相关模块，并在此基础上加入了部分笔者认为可能会有用的方法。通过对相关模块重写后，可以支持评分卡端到端`pipeline`式的超参数搜索方案。

+ 使用方法

```python
feature_pipeline = Pipeline([
    ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
    ("combiner", Combiner(target=target, min_samples=0.2)),
    ("transformer", WOETransformer(target=target)),
    ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
    ("stepwise", StepwiseSelection(target=target, target_rm=False)),
    # ("logistic", StatsLogisticRegression(target=target)),
    ("logistic", ITLubberLogisticRegression(target=target)),
])

# pipeline 超参数搜索参数：{pipeline_step_name}__{param}
params_grid = {
    "logistic__C": [i / 1. for i in range(1, 10, 2)],
    "logistic__penalty": ["l2"],
    "logistic__class_weight": [None, "balanced"], # + [{1: i / 10.0, 0: 1 - i / 10.0} for i in range(1, 10)],
    "logistic__max_iter": [100],
    "logistic__solver": ["sag"] # ["liblinear", "sag", "lbfgs", "newton-cg"],
    "logistic__intercept": [True, False],
}

clf = GridSearchCV(feature_pipeline, params_grid, cv=5, scoring='roc_auc', verbose=-1, n_jobs=2, return_train_score=True)
clf.fit(train, train[target])

y_pred_train = clf.best_estimator_.predict(train)
print(clf.best_params_)
```

## 模型报告

### 报告说明

输出评分卡模型报告的意义在于能够对其他人呈现评分卡模型的好坏以及相关的模型信息。

过去在评分卡建模过程中必须将各种表格和图片汇总到某个文件夹内或者几个文件中，格式每次都需要进行调整和优化，很耽误时间，如果中途有改动的话有需要重新再造一遍轮子。笔者基于日常评分卡建模工作中的需求，通过`openpyxl`实现了向`excel`文件中写入制定样式的内容（文本、图片、数据表、条件格式、设定宽度、字体对齐等），并在此基础上将评分卡建模过程中需要输出的内容以设定的格式全部保存至`excel`文件中，以减少后续建模过程中的重复性工作。

+ 相关代码说明

```python
class ExcelWriter:

    def __init__(self, style_excel='报告输出模版.xlsx', style_sheet_name="初始化", fontsize=10, font='楷体', theme_color='2639E9'):
        """
        excel 文件内容写入公共方法
        
        :param style_excel: 样式模版文件，默认当前路径下的 报告输出模版.xlsx ，如果项目路径调整需要进行相应的调整
        :param style_sheet_name: 模版文件内初始样式sheet名称，默认即可
        :param fontsize: 插入excel文件中内容的字体大小，默认 10
        :param font: 插入excel文件中内容的字体，默认 楷体
        :param theme_color: 主题色，默认 2639E9，注意不包含 #
        """

    def add_conditional_formatting(self, worksheet, start_space, end_space):
        """
        设置条件格式

        :param worksheet: 当前选择设置条件格式的sheet
        :param start_space: 开始单元格位置
        :param end_space: 结束单元格位置
        """

    @staticmethod
    def set_column_width(worksheet, column, width):
        """
        调整excel列宽

        :param worksheet: 当前选择调整列宽的sheet
        :param column: 列，可以直接输入 index 或者 字母
        :param width: 设置列的宽度
        """

    @staticmethod
    def set_number_format(worksheet, space, _format):
        """
        设置数值显示格式

        :param worksheet: 当前选择调整数值显示格式的sheet
        :param space: 单元格范围
        :param _format: 显示格式，参考 openpyxl
        """

    def get_sheet_by_name(self, name):
        """
        获取sheet名称为name的工作簿，如果不存在，则从初始模版文件中拷贝一个名称为name的sheet
        
        :param name: 需要获取的工作簿名称
        """

    def insert_value2sheet(self, worksheet, insert_space, value="", style="content", auto_width=False):
        """
        向sheet中的某个单元格插入某种样式的内容

        :param worksheet: 需要插入内容的sheet
        :param insert_space: 内容插入的单元格位置，可以是 "B2" 或者 (2, 2) 任意一种形式
        :param value: 需要插入的内容
        :param style: 渲染的样式，参考 init_style 中初始设置的样式
        :param auto_width: 是否开启自动调整列宽
        """

    def insert_pic2sheet(self, worksheet, fig, insert_space, figsize=(600, 250)):
        """
        向excel中插入图片内容
        
        :param worksheet: 需要插入内容的sheet
        :param fig: 需要插入的图片路径
        :param insert_space: 插入图片的起始单元格
        :param figsize: 图片大小设置
        """

    def insert_df2sheet(self, worksheet, data, insert_space, merge_column=None, header=True, index=False, auto_width=False):
        """
        向excel文件中插入制定样式的dataframe数据

        :param worksheet: 需要插入内容的sheet
        :param data: 需要插入的dataframe
        :param insert_space: 插入内容的起始单元格位置
        :param merge_column: 需要分组显示的列，index或者列明
        :param header: 是否存储dataframe的header，暂不支持多级表头
        :param index: 是否存储dataframe的index
        :param auto_width: 是否自动调整列宽
        :return 返回插入元素最后一列之后、最后一行之后的位置
        """

    def save(self, filename):
        """
        保存excel文件
        
        :param filename: 需要保存 excel 文件的路径
        """
```

+ 使用方法

```python
writer = ExcelWriter(style_excel="./utils/报告输出模版.xlsx", theme_color="2639E9")

# 评分卡刻度
scorecard_kedu = pd.DataFrame(
    [
        ["base_odds", card.base_odds, "根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比"],
        ["base_score", card.base_score, "基础ODDS对应的分数"],
        ["rate", card.rate, "设置分数的倍率"],
        ["pdo", card.pdo, "表示分数增长PDO时，ODDS值增长到RATE倍"],
        ["B", card.offset, "补偿值，计算方式：pdo / ln(rate)"],
        ["A", card.factor, "刻度，计算方式：base_score - B * ln(base_odds)"],
    ],
    columns=["刻度项", "刻度值", "备注"],
)

worksheet = writer.get_sheet_by_name("评分卡结果")
start_row, start_col = 2, 2
end_row, end_col = writer.insert_value2sheet(worksheet, (start_row, start_col), value="评分卡刻度", style="header")
end_row, end_col = writer.insert_df2sheet(worksheet, scorecard_kedu, (end_row + 1, start_col))

end_row, end_col = writer.insert_value2sheet(worksheet, (end_row + 2, start_col), value="训练数据集评分模型效果", style="header")
ks_row = end_row
end_row, end_col = writer.insert_pic2sheet(worksheet, "model_report/train_ksplot.png", (ks_row, start_col))
end_row, end_col = writer.insert_pic2sheet(worksheet, "model_report/train_scorehist.png", (ks_row, end_col))
end_row, end_col = writer.insert_df2sheet(worksheet, train_score_rank, (end_row + 1, start_col))

for c in ["坏样本率", "LIFT值", "分档KS值"]:
    conditional_column = get_column_letter(start_col + train_score_rank.columns.get_loc(c))
    writer.add_conditional_formatting(worksheet, f'{conditional_column}{end_row - len(train_score_rank)}', f'{conditional_column}{end_row}')
```

### 汇总信息

汇总信息包含评分卡模型相关的背景说明、取样说明、数据集划分方式以及不同时点数据分布情况。

### LR拟合结果

逻辑回归拟合结果主要包含了`LR`模型的拟合情况、稳定性、模型`AUC`、`KS`等指标。

### 入模特征信息

入模特征信息包含了入模特征的数据字典、分布情况、相关性、分箱信息以及分箱图等信息，主要为了展示评分卡入模变量相关的信息。

### 评分卡结果

评分卡模型结果主要包含评分卡参数信息、变量分箱及对应分数、评分`AUC`和`KS`指标、评分分布情况、评分排序性、各个评分区间相关指标信息。

### 补充说明

后续准备有闲暇时加上`shap`、`bad case`等相关的内容，并将支持`lightgbm`、`catboost`、`xgboost`等模型的报告输出。


## 评分卡PMML转换

### 背景说明

关于评分卡上线部署方案，目前现有的大致有四种

+ 提需求给公司开发部门上线部署
+ 将评分卡模型保存为`pickle`文件提供开发部署
+ 直接使用`python`提供`api服务`部署或者定时更新
+ 评分卡转`PMML`文件部署

前几种相对简单，提供`API服务`可能有一定技术门槛，但基本上网上有现成的方案，但评分卡转换`PMML`文件部署有一定难度，网上能够参考的实现方案比较难找，本章就笔者提供的评分卡转`PMML`相关的实现进行说明。

+ 相关代码说明

```python

class ScoreCard(toad.ScoreCard, TransformerMixin):
    
    ......

    def scorecard2pmml(self, pmml: str = 'scorecard.pmml', debug: bool = False):
        """export a scorecard to pmml

        Args:
            pmml (str): io to write pmml file.
            debug (bool): If true, print information about the conversion process.
        """
```

+ 使用说明

```python
# 保存 PMML 评分卡模型
card.scorecard2pmml(pmml='scorecard.pmml', debug=True)

# 模型验证
from pypmml import Model

model = Model.fromFile("scorecard.pmml")
data["score"] = model.predict(data[model.inputNames]).values
```


## 参考资料

> https://github.com/itlubber/LogisticRegressionPipeline
>
> https://github.com/itlubber/itlubber-excel-writer
>
> https://github.com/itlubber/openpyxl-excel-style-template
>
> https://github.com/itlubber/scorecard2pmml
>
> https://pypi.org/project/pdtr/
>
> https://github.com/amphibian-dev/toad
>
> https://github.com/ShichenXie/scorecardpy
>
> https://github.com/ing-bank/skorecard
>
> http://gnpalencia.org/optbinning
>
> https://github.com/yanghaitian1/risk_control
