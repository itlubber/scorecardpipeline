## 简介

`scorecardpipeline` 封装了 `toad`、`scorecardpy`、`optbinning` 等评分卡建模相关组件，`API` 风格与 `sklearn` 高度一致，支持 `pipeline` 式端到端评分卡建模、模型报告输出、导出 `PMML` 文件、超参数搜索等功能。

> 在线文档: [`https://itlubber.github.io/scorecardpipeline-docs`](https://itlubber.github.io/scorecardpipeline-docs/)
> 
> `PIPY` 包: [`https://pypi.org/project/scorecardpipeline`](https://pypi.org/project/scorecardpipeline/)
>
> 仓库地址: [`https://github.com/itlubber/scorecardpipeline`](https://github.com/itlubber/scorecardpipeline)


## 交流

<div style="display: flex; justify-content: space-around;">
    <img width="30%" alt="itlubber" src="https://itlubber.art//upload/itlubber.png"/>
    <img width="30%" alt="itlubber_art" src="https://itlubber.art//upload/itlubber_art.png"/>
</div>

<div style="display: flex; justify-content: space-around; margin-bottom: 2rem;">
    <span>微信<code>: itlubber</code></span>
    <span>订阅号<code>: itlubber_art</code></span>
</div>


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

<details>

  <summary>数据字典</summary>

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
 17     | 类别型    | housing                                                  | Housing         
 18     | 类别型    | job                                                      | 工作              
 19     | 类别型    | telephone                                                | 电话              
 20     | 类别型    | foreign worker                                           | 外籍工人            

</details>

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

