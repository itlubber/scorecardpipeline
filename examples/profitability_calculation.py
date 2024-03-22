# -*- coding: utf-8 -*-
"""
@Time    : 2024/3/19 10:47
@Author  : itlubber
@Site    : itlubber.art
"""
import sys

sys.path.append("../")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from openpyxl.utils import get_column_letter, column_index_from_string

from scorecardpipeline import *


logger = init_setting(seed=8888, logger=True)


start_row, start_col = 2, 2


# 基础参数配置
irr = 0.24                          # 金融部分年化利率IRR
loan_amount = 5000                  # 单笔放款金额
periods = 12                        # 放款期数
loan_order_rate = 1.0               # 放款比例（可忽略）
rights_amount = 299                 # 权益金额
rights_discount_rate = 0.9          # 权益扣款成功率
cost_of_funds = 0.1                 # 资金成本IRR
cost_of_rights = 50                 # 权益固定成本
traffic_acquisition_cost = 0.03     # 流量成本
risk_control_cost = 55              # 每笔放款订单数据成本
operating_costs = 0.01              # 运营成本
vintage = 0.05                      # VINTAGE损失


insert_col = get_column_letter(start_col + 2)

data1 = pd.DataFrame({
    "年化利率": [irr],
    "资金成本": [cost_of_funds],
    "放款金额": [loan_amount],
    "放款期数": [periods],
    "权益金额": [rights_amount],
    "权益固定成本": [cost_of_rights],
    "权益扣款成功率": [rights_discount_rate],
    "周转率": [f"=24/({insert_col}{start_row+3}+1)"],
    "每期应还金额": [f"=({insert_col}{start_row+2}*(1+{insert_col}{start_row}/{insert_col}{start_row+7}))/{insert_col}{start_row+3}"],
    "综合年化利率(100%扣款)": [f"={insert_col}{start_row}+{insert_col}{start_row+4}/{insert_col}{start_row+2}*{insert_col}{start_row+7}"],
}).T.reset_index()
data1.index = ["基础参数(IRR口径)" if i == 0 else "" for i in range(len(data1))]

insert_row = start_row + len(data1) + 2
data2 = pd.DataFrame({
    "利息收入": [f"={insert_col}{start_row}/{insert_col}{start_row+7}*(1-{insert_col}{insert_row+8})"],
    "权益收入": [f"={insert_col}{start_row+4}*{insert_col}{start_row+6}/{insert_col}{start_row+2}"],
    "资金成本": [f"={insert_col}{start_row+1}/{insert_col}{start_row+7}"],
    "流量成本": [traffic_acquisition_cost],
    "数据成本": [f"={risk_control_cost}/{insert_col}{start_row+2}"],
    "权益成本": [f"={insert_col}{start_row+5}/{insert_col}{start_row+2}*{insert_col}{start_row+6}"],
    "运营成本": [operating_costs],
    "盈亏平衡点": [f"={insert_col}{insert_row}+{insert_col}{insert_row+1}-{insert_col}{insert_row+2}-{insert_col}{insert_row+3}-{insert_col}{insert_row+4}-{insert_col}{insert_row+5}-{insert_col}{insert_row+6}"],
    "预估资损成本": [vintage],
    "利润率": [f"={insert_col}{insert_row+7}-{insert_col}{insert_row+8}"],
}).T.reset_index()
data2.index = ["盈利测算(APR口径)" if i == 0 else "" for i in range(len(data2))]


writer = ExcelWriter()

worksheet = writer.get_sheet_by_name("盈利测算")

dataframe2excel(data1, writer, sheet_name=worksheet, header=False, index=True, start_row=start_row, start_col=start_col, merge_index=False, auto_width=False)
dataframe2excel(data2, writer, sheet_name=worksheet, header=False, index=True, start_row=insert_row, start_col=start_col, merge_index=False, percent_cols=[0])

writer.set_number_format(worksheet, f"{insert_col}{start_row}:{insert_col}{start_row+1}", "0.00%")
writer.set_number_format(worksheet, f"{insert_col}{start_row+6}", "0.00%")
writer.set_number_format(worksheet, f"{insert_col}{start_row+9}", "0.00%")
writer.set_number_format(worksheet, f"{insert_col}{start_row+7}:{insert_col}{start_row+8}", "0.00")

writer.save("./权益类产品盈利测算表.xlsx")
