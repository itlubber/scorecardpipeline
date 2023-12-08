## 简介

`scorecardpipeline` 封装了 `toad`、`scorecardpy`、`optbinning` 等评分卡建模相关组件，`API` 风格与 `sklearn` 高度一致，支持 `pipeline` 式端到端评分卡建模、模型报告输出、导出 `PMML` 文件、超参数搜索等功能。

> 在线文档: [`https://scorecardpipeline.itlubber.art/`](https://scorecardpipeline.itlubber.art//)
> 
> `PIPY` 包: [`https://pypi.org/project/scorecardpipeline`](https://pypi.org/project/scorecardpipeline/)
>
> 仓库地址: [`https://github.com/itlubber/scorecardpipeline`](https://github.com/itlubber/scorecardpipeline)


## 版本说明

| **序号** | **版本号** | **发布日期** |                                  **版本说明**                                  |
| :------: | :--------: | :----------: | :----------------------------------------------------------------------------: |
|    01    |   0.1.0    |  2023.05.04  |                                     初始化                                     |
|    ..    |   ......   |    ......    |                                     ......                                     |
|    26    |   0.1.26   |  2023.11.13  | 稳定版本，新增方法[`说明文档`](https://itlubber.github.io/scorecardpipeline-docs/) |


## 环境安装

1. 安装 `python` 环境

推荐 `python 3.8.13` ，参照网上教程自行安装即可，也可直接使用系统提供的 `python3` , 版本不易过高, 最低支持 `python 3.6`，最高支持 `python 3.10`

2. 安装 `scorecardpipeline` 包

+ 在线环境安装

直接通过 `pip` 安装即可，安装后需要确认下版本是否安装正确，推荐 `-i https://test.pypi.org/simple` 指定源安装

```shell
pip install scorecardpipeline
```

+ 离线环境安装

离线环境安装需先找一个 `python` 版本和系统 (`windows`、`linux`、`mac`) 与生产一致的有网机器或者容器，在有网的环境中下载相关依赖项后拷贝到离线环境安装

```shell
# 在线环境依赖下载
pip download -d site-packages/ scorecardpipeline
# 离线环境
pip install --no-index --find-links=site-packages scorecardpipeline
```

3. 安装 `jdk 1.8+` [可选]

如果有将训练完成的评分卡模型直接导出 `PMML` 模型文件部署的需求，需要单独安装 `java` 环境

> `mac` & `windows` 参考: [`https://developer.aliyun.com/article/1082599`](https://developer.aliyun.com/article/1082599)
>
> `linux` 参考: [`https://blog.csdn.net/muzi_gao/article/details/132169159`](https://blog.csdn.net/muzi_gao/article/details/132169159)


## 使用示例

本节将就 `scorecardpipeline` 的相关功能做简要介绍，并提供一个[`简单的示例`](https://github.com/itlubber/scorecardpipeline/blob/main/examples/scorecard_samples.ipynb)，旨在让读者对如何使用 `scorecardpipeline` 进行评分卡模型构建有一个简单的印象。


### 导入依赖

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scorecardpipeline import *

# 固定随机种子以保证结果可复现
init_setting(seed=10)
```

### 数据准备

使用 `scorecardpipeline` 进行评分卡建模，需要您提前准备好一个包含目标变量的数据集，`scorecardpipeline` 提供了一个德国信贷数据集 [`germancredit`](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)，示例加载该数据集进行演示，读者可以直接替换为您本地的数据集，并修改目标变量名称为数据集中的目标变量列名

[`germancredit`](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) 数据集中包含类别型变量、数值型变量、好坏标签，共1000条数据，由于数据集中不包含缺失值，为了模拟实际生产中的真实数据，笔者将在固定随机种子的情况下随机替换数据集中的部分内容为 `np.nan`
<!-- 
<details>

  <summary>数据字典</summary> -->

 **序号** | **类型** | **特征名**                                                  | **释义**          
:------:|:------:|:--------------------------------------------------------:|:---------------:
 0      | 类别型    | creditability                                            | 客户好坏标签          
 1      | 数值型    | duration in month                                        | 月持续时间           
 2      | 数值型    | credit amount                                            | 信用额度            
 3      | 数值型    | installment rate in percentage of disposable income      | 分期付款率占可支配收入的百分比 
 4      | 数值型    | present residence since                                  | 现居住地至今          
 5      | 数值型    | age in years                                             | 年龄              
 6      | 数值型    | number of existing credits at this bank                  | 这家银行的现有信贷数量     
 7      | 数值型    | number of people being liable to provide maintenance for | 有责任为其提供维修服务的人数  
 8      | 类别型    | status of existing checking account                      | 现有支票账户的状态       
 9      | 类别型    | credit history                                           | 信用记录            
 10     | 类别型    | purpose                                                  | 目的              
 11     | 类别型    | savings account or bonds                                 | 储蓄账户/债券         
 12     | 类别型    | present employment since                                 | 至今工作至今          
 13     | 类别型    | personal status and sex                                  | 个人地位和性别         
 14     | 类别型    | other debtors or guarantors                              | 其他债务人/担保人       
 15     | 类别型    | property                                                 | 财产              
 16     | 类别型    | other installment plans                                  | 其他分期付款计划        
 17     | 类别型    | housing                                                  | 住房情况         
 18     | 类别型    | job                                                      | 工作              
 19     | 类别型    | telephone                                                | 电话              
 20     | 类别型    | foreign worker                                           | 外籍工人            

<!-- </details> -->

<br>

```python
# 加载数据集
data = sp.germancredit()

# 设置目标变量名称 & 映射目标变量值域为 {0, 1}
target = "creditability"
data[target] = data[target].map({"good": 0, "bad": 1})

# 随机替换 20% 的数据为 np.nan
for col in data.columns.drop(target):
    for i in range(len(data)):
        if np.random.rand() > 0.8:
            data[col].loc[i] = np.nan
```


### 数据拆分

通常建模过程中会将数据集拆分成训练数据集和测试数据集集，训练数据集用于模型参数的训练，测试数据集用于评估模型好坏，且在整个建模过程中实际上只能使用一次（任何针对测试数据集调模型的，都存在模型过拟合的情况，除非你的数据不包含任何噪声数据）。而为了弥补测试数据集只能用一次用于评估模型好坏的尴尬，大多都会再拆分一个验证数据集，用于模型训练过程中判别模型是否向好的方向进行优化。

在金融建模场景中，上述三个数据集通常被称为训练数据集、测试数据集（即上述三个数据集中的验证集）、跨时间验证集（即上述三个数据集中的测试集）。同时，在金融场景中，跨时间验证集的拆分方式通常使用类似时间序列数据集的拆分方法，按照时间将数据集拆分为建模集（包含训练集和测试集）、跨时间验证集（`Out of Time，OOT`），以保证模型在不同时间段内的泛化能力。

简单起见，示例只拆分训练集、测试集，OOT直接拷贝训练集来演示评分卡建模完整过程。

```python
train, test = train_test_split(data, test_size=0.3, shuffle=True, stratify=data[target])

oot = train.copy()
```


### 数据预处理

在评分卡建模标准流程中，会先针对每个变量分箱后转WOE，再筛选特征训练逻辑回归模型，训练完后转评分卡，当数据集的特征个数比较多时，可以再分箱之前先进行特征初筛，以降低分箱时的计算耗时。评分卡建模完整流程可以参照如下流程:

<div style="display: flex; justify-content: space-around;">
    <img src="https://itlubber.art/upload/scorecard-modeling-process.png" style="width: 70%; "/>
</div>


#### 特征筛选

特征筛选主要从几个方面来评估特征是否有效：

1. 单一值占比
2. 缺失值占比
3. 特征相关性
4. 特征 `IV（information value）` 值
5. 特征稳定性 `PSI（population stability index）`
6. 特征重要性
7. 方差膨胀因子 `VIF（variance inflation factor）`


`scorecardpipeline` 提供了几种常见的特征筛选方法（`1`、`2`、`3`、`4`），可以通过实例化的方式进行配置，并在训练数据集中训练完成后应用到不同数据集中。

```python
# 初始化筛选器
select = FeatureSelection(target=target, engine="toad", identical=0.95, empty=0.95, iv=0.02, corr=0.6)
# 训练数据上训练
select.fit(train)
# 应用到不同数据集
train_select = select.transform(train)
test_select = select.transform(test)
```

在 `scorecardpipeline` 中，所有筛选器都包含 `dropped` 属性，用来存放特征被剔除的原因。

```python
# 输出特征筛选信息
select.dropped
```

|    | variable                                                 | rm_reason   |
|:---:|:---------------------------------------------------------:|:------------:|
|  0 | number_of_existing_credits_at_this_bank                  | iv          |
|  1 | job                                                      | iv          |
|  2 | number_of_people_being_liable_to_provide_maintenance_for | iv          |
|  3 | telephone                                                | iv          |
|  4 | duration_in_month                                        | corr        |


#### 特征分箱


特征分箱能够提升模型的鲁棒性，特征分箱时通常对每个箱内的样本个数或占比有要求，数据中的异常值和极端值会被分到某个箱中，在转化为 `WOE` 之后能够很大程度上避免异常值或极端值对模型训练时造成偏差。同时，特征分箱转 `WOE` 之后，所有特征的值域都被缩放到了某个固定尺度下，逻辑回归模型的系数在一定程度上能够代表特征的重要程度。

+ 在 `scorecardpipeline` 中，集成了 `toad` 和 `optbinning` 两个库中提供的分箱方法，后续会陆续增加更多的分箱方法，相关内容参考: [`scorecardpipeline.Combiner`](/scorecardpipeline.html#scorecardpipeline.Combiner)


```python
# 初始化分箱器
combiner = Combiner(target=target, min_bin_size=0.2)
# 训练
combiner.fit(train_select)
# 对数据集进行分箱
train_bins = combiner.transform(train_select)
test_bins = combiner.transform(test_select)
```

+ 为了方便调整分箱，[`scorecardpipeline.Combiner`]() 提供了输出分箱图和分箱统计信息的功能，方便快速手工调整分箱。

```python
# 查看 credit_amount 信用额度 的分箱信息，并显示分箱统计信息
combiner.bin_plot(train_select, "credit_amount", result=True, desc="信用额度")
```

<div style="display: flex; justify-content: space-around;">
    <img src="https://itlubber.art/upload/sp_credit_amount.png" />
</div>


+ 当特征分箱不符合业务逻辑或者单调性不满足时，可以通过自定义规则的方式查看分箱效果。（此时不会对 `Combiner` 中的规则进行更新）

```python
# 通过 rule 字段传入自定义规则，实时查看分箱效果
combiner.bin_plot(train_select, "credit_amount", result=True, desc="信用额度", rule=[4000.0, np.nan])
```

<div style="display: flex; justify-content: space-around;">
    <img src="https://itlubber.art/upload/sp_credit_amount_rule.png" />
</div>


+ 当特征分箱调整到符合业务预期或者单调满足建模需求时，可以通过 `update` 方法更新规则 `Combiner` 中的规则。

```python
# 更新 credit_amount 的分箱规则
combiner.update({"credit_amount": [4000.0, np.nan]})
# 打印分箱规则
combiner["credit_amount"] # array([4000.,   nan])
```


#### `WOE` 转换

特征分箱后，只是将特征离散化为几箱，每一箱的值被赋予了一个标签，例如0、1、2、3、...，尽管这些标签在一定程度上也能够表征客户的风险水平，能够作为训练数据训练出一个模型，但在评估客户风险状况时很难精准刻画客户的风险状况。

通常使用 `WOE（Weight of Evidence）编码` 对分箱后的特征进行编码。`WOE编码` 将每个分箱内的坏样本占比除以好样本占比后取对数来编码，能够反映每个分箱内客户的坏客户分布与好客户分布之间的差异以及该箱内坏好比与总体的坏好比之间的差异性。


特征 `WOE（Weight of Evidence）编码` 后，有如下好处：

+ 能够将每个特征的值域放缩到同样的尺度下
+ `woe` 中包含了样本好坏信息，比原始值和分箱序号更能反应客户违约概率大小
+ `woe` 的计算公式 $woe = ln(\frac{bad_i}{bad}/\frac{good_i}{good} )$ 中引入了非线性，一定程度能够增强逻辑回归拟合非线性模型的能力


```python
# 初始化 WOE 转换器
transform = WOETransformer(target=target)
# 训练
transform.fit(train_bins)
# 转换分箱为WOE
train_woe = transform.transform(train_bins)
test_woe = transform.transform(test_bins)
```

`WOE（Weight of Evidence）编码器` 提供了方法允许使用者通过如下方式查看每个分箱对应的 `woe` 值。

<div style="display: flex; justify-content: space-around;">
    <img src="https://itlubber.art/upload/sp_woetransformer.png" />
</div>

在特征从原始值编码为 `woe` 的过程中，特征的值从连续值或者分类值映射到了有限的几个分箱对应的 `woe` 值，同时值域也统一到了一个基于样本好坏分布情况为基准的空间中，会造成特征之间的相关性增大、单一值占比上升、特征 `iv` 值较前筛时小幅度下降，所以推荐在 `WOE编码` 后，再进行一次特征精筛，以保证后续模型结果的稳定性和整体可解释性。


#### 逐步回归特征筛选

在正式训练 `逻辑回归（logistic regression，LR）模型` 前，通常会再使用逐步回归来剔除特征，能够一定程度上解决特征之间的多重共线性，以及在剔除尽可能多的特征的同时保证 `LR模型` 效果的有效性。

```python
# 初始化逐步回归特征筛选器
stepwise = StepwiseSelection(target=target)
# 训练
stepwise.fit(train_woe)
# 应用逐步回归特征筛选器
train_woe_stepwise = stepwise.transform(train_woe)
test_woe_stepwise = stepwise.transform(test_woe)
```

与 `FeatureSelection` 类似，`scorecardpipeline` 中所有特征筛选器都包含了 `dropped` 属性，可以直接输出每个特征被剔除的原因。

```python
# 逐步回归特征筛选明细信息
stepwise.dropped
```

|    | variable                    | rm_reason   |
|:--:|:---------------------------:|:-----------:|
|  0 | present_residence_since     | stepwise    |
|  1 | savings_account_and_bonds   | stepwise    |
|  2 | other_debtors_or_guarantors | stepwise    |
|  3 | property                    | stepwise    |
|  4 | other_installment_plans     | stepwise    |
|  5 | foreign_worker              | stepwise    |


### 训练 `LR` 模型

对数据集预处理完成后，数据集中的特征的值域被映射到了一个与样本好坏息息相关的空间中，特征值的大小能够一定程度上反映出客户违约概率的高低，通过 `WOE编码` 方法对原始特征进行转换时引入的非线性能够在一定程度上弥补建模时使用简单模型拟合能力不足的问题，同时，数据集经过特征筛选后，冗余信息被尽可能多的剔除，多重共线性也在一定程度上得到解决，训练模型时能够在保证模型效果的前提下提升模型的鲁棒能力。

在 `scorecardpipeline` 中，提供了基于 `sklearn` 重新实现的 `ITLubberLogisticRegression` 逻辑回归模型，能够满足丰富的超参数设置和丰富的统计信息输出。

当然，假如您想直接使用 `sklearn` 提供的 `LogisticRegression` 去完成评分卡模型构建和结果输出，也能够支持，但转换评分卡后，有部分扩展功能会受到限制，但不会影响评分卡模型转换、评分预测以及持久化存储的相关功能。

```python
# 逻辑回归模型构建
logistic = ITLubberLogisticRegression(target=target)
# 训练
logistic.fit(train_woe_stepwise)
# 预测数据集样本违约概率
y_pred_train = logistic.predict_proba(train_woe_stepwise.drop(columns=target))[:, 1]
y_pred_test = logistic.predict_proba(test_woe_stepwise.drop(columns=target))[:, 1]
```

逻辑回归模型训练完成后，能够通过 `summary` 或 `summary2` 方法输出模型训练时的各项统计信息

```python
# 数据字典或特征描述信息
feature_map = {
    "const": "截距项",
    "status_of_existing_checking_account": "现有支票账户的状态",
    "credit_history": "信用记录",
    "purpose": "目的",
    "credit_amount": "信用额度",
    "present_employment_since": "现居住地至今",
    "installment_rate_in_percentage_of_disposable_income": "分期付款率占可支配收入的百分比",
    "personal_status_and_sex": "个人地位和性别",
    "age_in_years": "年龄",
    "housing": "住房情况",
}

# summary 仅支持输出简单的统计信息，使用 summary2 可以输出有特征描述的统计信息表
logistic.summary2(feature_map=feature_map)
```

| **Features**                                        | **Describe**    | **Coef.** | **Std.Err** | **z** | **P>\|z\|** | **[ 0.025** | **0.975 ]** | **VIF** |
|:---------------------------------------------------:|:---------------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| const                                               | 截距项             | -0.8422  |  0.0940  | -8.9567  | 0.0000   | -1.0265  | -0.6579  | 1.0464   |
| status_of_existing_checking_account                 | 现有支票账户的状态       | 0.8083   |  0.1549  | 5.2191   | 0.0000   | 0.5048   | 1.1119   | 1.0565   |
| credit_history                                      | 信用记录            | 0.8427   |  0.1823  | 4.6224   | 0.0000   | 0.4854   | 1.2000   | 1.0754   |
| purpose                                             | 目的              | 1.0387   |  0.2188  | 4.7462   | 0.0000   | 0.6097   | 1.4676   | 1.0186   |
| credit_amount                                       | 信用额度            | 1.0188   |  0.2345  | 4.3456   | 0.0000   | 0.5593   | 1.4784   | 1.0220   |
| present_employment_since                            | 现居住地至今          | 0.7043   |  0.3693  | 1.9074   | 0.0565   | -0.0194  | 1.4281   | 1.0620   |
| installment_rate_in_percentage_of_disposable_income | 分期付款率占可支配收入的百分比 | 1.3074   |  0.4023  | 3.2496   | 0.0012   | 0.5189   | 2.0960   | 1.0192   |
| personal_status_and_sex                             | 个人地位和性别         | 0.8170   |  0.4786  | 1.7071   | 0.0878   | -0.1211  | 1.7551   | 1.0058   |
| age_in_years                                        | 年龄              | 0.7926   |  0.3023  | 2.6219   | 0.0087   | 0.2001   | 1.3852   | 1.0724   |
| housing                                             | 住房情况            | 0.7277   |  0.4115  | 1.7684   | 0.0770   | -0.0788  | 1.5342   | 1.0165   |

`ITLubberLogisticRegression` 逻辑回归模型还支持直接通过画图查看模型系数稳定性

```python
# 逻辑回归系数稳定情况
logistic.plot_weights(figsize=(10, 6))
```

<div style="display: flex; justify-content: space-around;">
    <img src="https://itlubber.art/upload/sp_lr_weight.png" />
</div>

同时，也提供了 `report` 方法快速查看模型在某个数据集（`woe`后的数据集）上的分类效果

```python
# 查看某个数据集的分类效果
logistic.report(train_woe_stepwise)
```

|    | desc         | precision          | recall             |   f1-score |   support |
|:--:|:------------:|:------------------:|:------------------:|:----------:|:---------:|
|  0 | 好客户       | 0.7785588752196837 | 0.9040816326530612 |   0.836638 |       490 |
|  1 | 坏客户       | 0.6412213740458015 | 0.4                |   0.492669 |       210 |
|  2 | macro avg    | 0.7098901246327426 | 0.6520408163265305 |   0.664653 |       700 |
|  3 | weighted avg | 0.7373576248675191 | 0.7528571428571429 |   0.733447 |       700 |
|  4 | accuracy     |                    |                    |   0.752857 |       700 |

+ 在模型分类报告中，$macro\ avg$ 和 $weighted\ avg$ 的计算方法如下:

$macro\ avg = \frac{metric_1 + metric_2}{2}$

$weighted\ avg = \frac{metric_1 \times \frac{support_1}{total\ count} + metric_2 \times \frac{support_2}{total\ count}}{2} = \frac{metric_1 \times support_1 + metric_2 \times support_2}{2 \times total\ count}$


### 评分卡转换

在金融领域，如果直接使用模型预测的概率作为最终的评分，可能会对业务人员造成一定程度上的理解难度，同时，[模型预测的概率其实很难代表客户真实的违约概率](https://scikit-learn.org/stable/modules/calibration.html#calibration)，不同模型预测的概率与真实违约概率之间存在不同程度的偏差。

<div style="display: flex; justify-content: space-around; ">
    <img width="80%" src="https://itlubber.art/upload/sphx_glr_plot_compare_calibration_001.png" />
</div>

<span style="display: flex; justify-content: space-around; font-size: smaller;">不同模型预测概率与真实概率之间的偏差</span>

由于模型预测概率与真实违约概率之间存在不同程度的偏差，评分卡不直接使用违约概率，而是引入了 `odds`（客户违约概率与正常概率的比值）来刻画客户的违约情况，其中 $odds = \frac{p}{1-p}$ ，$p$ 为模型预测的违约概率。

评分卡通常基于逻辑回归模型来制作，一方面是逻辑回归模型预测的违约概率与真实概率之间的偏差相比其他模型更接近，另一方面是逻辑回归模型能够很方便的转换为评分卡。

逻辑回归模型的数学公式如下:

$$
p = \frac{1}{1+e^{-\theta^Tx}}
$$

根据逻辑回归的数学公式很容易推导出入模特征与 `odds` 之间的关系:

$$
\begin{aligned}
\frac{1}{p} &= 1+e^{-\theta^Tx} \\
\frac{1}{p}-1 = \frac{1-p}{p} &= e^{-\theta^Tx} \\
ln(\frac{p}{1-p}) = ln(odds) &= \theta^Tx = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + ...... + \theta_nx_n
\end{aligned}
$$

评分卡模型是在原有模型预测的违约概率 $p$ 的基础上计算 $odds$，再根据 $ln(odds)$ 进行平移和伸缩变换得到最终评分卡分数的模型，本质上是一种将 $[0,\ 1]$ 概率空间映射到实数空间 $R$ 的一种方法。

假设评分卡模型数学形式如下：

$$
score = A - B \times ln(odds) = A - B \times ln(\frac{p}{1-p}) = A - B \times (\theta_0x_0 + \theta_1x_1 + \theta_2x_2 + ...... + \theta_nx_n)
$$

假设客户违约概率为 $p_0$ ，$base\_odds=p_0/(1-p_0)$ 时，对应的评分卡分数为 $base\_score$ ，当 $odds$ 增加 $rate$ 倍时，评分卡分数降低 $pdo$ 分，可以得到一个二元一次方程:

$$
\left\{
\begin{aligned}
base\_score &= A - B \times ln(base\_odds) \\
base\_score - pdo &= A - B \times ln(rate \times base\_odds)
\end{aligned}
\right.
$$

根据上述二元一次方程求解可以得到:

$$
\begin{aligned}
B &= pdo / ln(rate) \\
A &= base\_score + \frac{pdo}{ln(rate)} \times ln(base\_odds)
\end{aligned}
$$

> **注:** 在 `toad` 中使用的是 $woe = ln(\frac{good_i}{good}/\frac{bad_i}{bad})$ ，在上述推导中符号是反的，故而在 `toad` 库中的 `offset = base_score - factor * np.log(base_odds)`，形式上的差异并不会导致最终 `A` 和 `B` 的值，但在计算 $base\_odds$ 时需要注意使用对应的方式进行计算

当计算得到 $A$ 和 $B$ 后，概率转评分的模型也就确定了，下面我们正式进入评分转换方法。

```python
# 逻辑回归模型转评分卡
card = ScoreCard(target=target, combiner=combiner, transer=transform, pretrain_lr=logistic, base_score=50, base_odds=20, pdo=10)
# 传入 woe 数据计算评分卡参数
card.fit(train_woe_stepwise)

# 预测评分
train["score"] = card.predict(train)
test["score"] = card.predict(test)
```

在 `scorecardpipeline` 中提供了方法输出评分卡刻度

```python
# 输出评分卡刻度信息
card.scorecard_scale()
```

| 序号 | 刻度项     |   刻度值 | 备注                                                                                       |
|:--:|:----------|:--------:|:-------------------------------------------------------------------------------------------|
|  0 | base_odds  | 20       | 根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比 |
|  1 | base_score | 50       | 基础ODDS对应的分数                                                                         |
|  2 | rate       |  2       | 设置分数的倍率                                                                             |
|  3 | pdo        | 10       | 表示分数增长PDO时，ODDS值增长到RATE倍                                                      |
|  4 | B          | 14.427   | 补偿值，计算方式：pdo / ln(rate)                                                           |
|  5 | A          |  6.78072 | 刻度，计算方式：base_score - B * ln(base_odds)                                             |




### 模型报告输出


### 模型持久化存储


## 交流

<div style="display: flex; justify-content: space-around;">
    <img width="30%" alt="itlubber" src="https://itlubber.art//upload/itlubber.png"/>
    <img width="30%" alt="itlubber_art" src="https://itlubber.art//upload/itlubber_art.png"/>
</div>

<div style="display: flex; justify-content: space-around; margin-bottom: 2rem;">
    <span>微信<code>: itlubber</code></span>
    <span>订阅号<code>: itlubber_art</code></span>
</div>
