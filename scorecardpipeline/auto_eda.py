# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/29 11:17
@Author  : itlubber
@Site    : itlubber.art
"""

import os
import re
import time
import traceback
import pandas as pd
from datetime import datetime, timedelta

import sweetviz as sv


def auto_eda_sweetviz(all_data, target=None, save="model_report/auto_eda.html", pairwise=True, labels=None, exclude=None, num_features=None, cat_features=None, text_features=None):
    """对数据量和特征个数较少的数据集进行自动 EDA 产出分析报告文档

    :param all_data: 需要 EDA 的数据集
    :param target: 目标变量，仅支持整数型或布尔型的目标变量
    :param labels: 当传入 target 时为需要对比的数据集名称 [true, false]，当不传入 target 时为数据集名字
    :param save: 报告存储路径，后缀使用 .html
    :param pairwise: 是否需要显示特征之间的分布情况
    :param exclude: 需要排除的特征名称
    :param num_features: 需要强制转为数值型的特征名称
    :param cat_features: 需要强制转为类别型的特征名称
    :param text_features: 需要强制转为文本的特征名称
    """
    # 配置文件初始化
    sv.config_parser.read_dict({"Layout": {"show_logo": 0}})

    # 设置跳过某些列
    feature_config = sv.FeatureConfig(skip=exclude, force_num=num_features, force_cat=cat_features, force_text=text_features)

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

    if target is None:
        report_all = sv.analyze(
            [all_data, labels] if labels else all_data
            , pairwise_analysis="on" if pairwise else "off"
            , feat_cfg=feature_config
        )
        report_all.show_html(save, open_browser=False)
    else:
        # 自动 eda 两个数据集，根据数据集中某个字段区分
        report = sv.compare_intra(
            all_data
            , all_data[target] == 1
            , labels if labels else ["坏客户", "好客户"]
            , feat_cfg=feature_config
            , pairwise_analysis="on"
        )
        report.show_html(save, open_browser=False)


if __name__ == '__main__':
    import numpy as np
    from scorecardpipeline import germancredit

    # 加载数据集
    data = germancredit()

    # 设置目标变量名称 & 映射目标变量值域为 {0, 1}
    target = "creditability"
    data[target] = data[target].map({"good": 0, "bad": 1})

    # 随机替换 20% 的数据为 np.nan
    for col in data.columns.drop(target):
        for i in range(len(data)):
            if np.random.rand() > 0.8:
                data[col].loc[i] = np.nan

    # 自动 eda 并保存文件
    auto_eda_sweetviz(data, target=target)
