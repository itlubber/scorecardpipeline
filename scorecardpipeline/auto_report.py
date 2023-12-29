# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/29 11:17
@Author  : itlubber
@Site    : itlubber.art
"""
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openpyxl.worksheet.worksheet import Worksheet

from .utils import *
from .processing import *
from .excel_writer import ExcelWriter, dataframe2excel


def auto_data_testing_report(data, features=None, target="target", date=None, data_summary_comment="", freq="M", excel_writer=None, sheet="分析报告", start_col=2, start_row=2, writer_params={}, bin_params={}, feature_map={}):
    """自动数据测试报告，用于三方数据评估或自有评分效果评估

    :param data: 需要评估的数据集，需要包含目标变量
    :param features: 需要进行分析的特征名称，支持单个字符串传入或列表传入
    :param target: 目标变量名称
    :param date: 日期列，通常为借款人申请日期或放款日期，可选字段，传入的情况下，结合字段 freq 参数输出不同时间粒度下的好坏客户分布情况
    :param freq: 结合 date 日期使用，输出需要统计的粒度，默认 M，即按月统计
    :param data_summary_comment: 数据样本概况中需要填入的备注信息，例如 "去除了历史最大逾期天数[0, dpd]内的灰客户" 等
    :param excel_writer: 需要保存的excel文件名称或写入器
    :param sheet: 需要保存的 sheet 名称，可传入已有的 worksheet 或 文字信息
    :param start_col: 开始列
    :param start_row: 开始行
    :param writer_params: excel写入器初始化参数，仅在 excel_writer 为字符串时有效
    :param bin_params: 统计分箱的参数，支持 `feature_bin_stats` 方法的参数
    :param feature_map: 特征字典，增加文档可读性使用，默认 {}
    """
    init_setting()

    if not isinstance(features, (list, tuple)):
        features = [features]

    if isinstance(excel_writer, ExcelWriter):
        writer = excel_writer
    else:
        writer = ExcelWriter(**writer_params)

    worksheet = writer.get_sheet_by_name(sheet)

    end_row, end_col = writer.insert_value2sheet(worksheet, (start_row, start_col), value="数据有效性分析报告", style="header_middle", end_space=(start_row, start_col + 17))

    if date is not None and date in data.columns:
        if data[date].dtype.name in ["str", "object"]:
            start_date = pd.to_datetime(data[date]).min().strftime("%Y-%m-%d")
            end_date = pd.to_datetime(data[date]).max().strftime("%Y-%m-%d")
        else:
            start_date = data[date].min().strftime("%Y-%m-%d")
            end_date = data[date].max().strftime("%Y-%m-%d")

        dataset_summary = pd.DataFrame(
            [
                ["回溯数据集", start_date, end_date, len(data), data[target].sum(), data[target].sum() / len(data), data_summary_comment]
            ],
            columns=["数据集", "开始时间", "结束时间", "样本总数", "坏客户数", "坏客户占比", "备注"],
        )
        end_row, end_col = dataframe2excel(dataset_summary, writer, worksheet, percent_cols=["样本占比", "坏客户占比"], start_row=end_row + 2, title="样本总体分布情况")

        distribution = distribution_plot(data, date=date, freq=freq, target=target, save=f"model_report/sample_time_distribution.png", result=True)
        end_row, end_col = writer.insert_value2sheet(worksheet, (end_row + 2, start_col), value="样本时间分布情况", style="header", end_space=(end_row + 2, start_col + len(distribution.columns) - 1))
        end_row, end_col = writer.insert_pic2sheet(worksheet, f"model_report/sample_time_distribution.png", (end_row + 1, start_col), figsize=(720, 370))
        end_row, end_col = dataframe2excel(distribution, writer, worksheet, percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率"], condition_cols=["坏样本率"], start_row=end_row)
        end_row += 2
    else:
        dataset_summary = pd.DataFrame(
            [
                ["回溯数据集", len(data), data[target].sum(), data[target].sum() / len(data), data_summary_comment]
            ],
            columns=["数据集", "样本总数", "坏客户数", "坏客户占比", "备注"],
        )
        end_row, end_col = dataframe2excel(dataset_summary, writer, worksheet, percent_cols=["样本占比", "坏客户占比"], start_row=end_row + 2, title="样本总体分布情况")
        end_row += 2

    end_row, end_col = writer.insert_value2sheet(worksheet, (end_row, start_col), value="数值类特征 OR 评分效果评估", style="header_middle", end_space=(end_row, start_col + 17))
    features_iter = tqdm(features)
    for col in features_iter:
        features_iter.set_postfix(feature=feature_map.get(col, col))
        temp = data[[col, target]]
        score_table_train = Combiner.feature_bin_stats(temp, col, desc=f"{feature_map.get(col, col)}", target=target, **bin_params)
        bin_plot(score_table_train, desc=f"{feature_map.get(col, col)}", figsize=(10, 5), anchor=0.935, save=f"model_report/feature_bins_plot_{col}.png")
        temp = temp.dropna().reset_index(drop=True)
        ks_plot(temp[col], temp[target], figsize=(10, 5), title=f"{feature_map.get(col, col)}", save=f"model_report/feature_ks_plot_{col}.png")
        hist_plot(temp[col], y_true=temp[target], figsize=(10, 6), desc=f"{feature_map.get(col, col)} 好客户 VS 坏客户", bins=30, anchor=1.11, fontsize=14, labels={0: "好客户", 1: "坏客户"}, save=f"model_report/feature_hist_plot_{col}.png")

        end_row, end_col = writer.insert_value2sheet(worksheet, (end_row + 2, start_col), value=f"数据字段: {feature_map.get(col, col)}", style="header", end_space=(end_row + 2, start_col + len(score_table_train.columns) - 1))
        ks_row = end_row + 1
        end_row, end_col = writer.insert_pic2sheet(worksheet, f"model_report/feature_bins_plot_{col}.png", (ks_row, start_col), figsize=(600, 350))
        end_row, end_col = writer.insert_pic2sheet(worksheet, f"model_report/feature_ks_plot_{col}.png", (ks_row, end_col - 1), figsize=(600, 350))
        end_row, end_col = writer.insert_pic2sheet(worksheet, f"model_report/feature_hist_plot_{col}.png", (ks_row, end_col - 1), figsize=(600, 350))

        end_row, end_col = dataframe2excel(score_table_train, writer, worksheet, percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率", "LIFT值", "累积LIFT值"], condition_cols=["坏样本率", "LIFT值"], start_row=end_row)

    if not isinstance(excel_writer, ExcelWriter) and not isinstance(sheet, Worksheet):
        writer.save(excel_writer)


if __name__ == '__main__':
    target = "creditability"
    data = germancredit()
    data[target] = data[target].map({"good": 0, "bad": 1})

    auto_data_testing_report(data
                             , features=data.select_dtypes(include="number").columns.drop(target).tolist()
                             , target=target
                             , date=None
                             , data_summary_comment=""
                             , freq="M"
                             , excel_writer="model_report/三方数据测试报告.xlsx"
                             , sheet="分析报告"
                             , start_col=2
                             , start_row=2
                             , writer_params={}
                             , bin_params={}
                             )
