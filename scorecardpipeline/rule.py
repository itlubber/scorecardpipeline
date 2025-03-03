# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/26 12:00
@Author  : itlubber
@Site    : itlubber.art
"""
import numpy as np
import numexpr as ne
from enum import Enum

import pandas as pd
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

from .processing import feature_bin_stats, Combiner
from .excel_writer import dataframe2excel


def _get_context(X, feature_names):
    return {name: X[:, i] for i, name in enumerate(feature_names)}


def _apply_expr_on_array(expr, X, feature_names):
    ctx = _get_context(X, feature_names)
    return ne.evaluate(expr, local_dict=ctx)


class RuleState(str, Enum):
    INITIALIZED = "initialized"
    APPLIED = "applied"


class RuleStateError(RuntimeError):
    pass


class RuleUnAppliedError(RuleStateError):
    pass


# 中級操作符
op_dict = {"GT": ">", "LT": "<", "EQ": "==", "ADD": "+", "GE": ">=", "LE": "<=", "SUBTRACT": "-", "MULTIPLY": "*", "DIVIDE": "/", "OR": "|", "AND": "&"}
# 数据类型 int, float-->float, 目前不支持 str
value_type_dict = {"int": float, "float": float, "string": str, "bool": bool}
# if_part, then_part, else_part
part_dict = ["if", "then", "else"]


# max_index: 数据的列数     feature_list: 列的名称
def json2expr(data, max_index, feature_list):
    if data.keys()._contains_("operator"):
        op = data.get("operator")
        params = data.get("params")
        if op == "FEATURE_INDEX":  # 取变量，一个值{判断变量，索引是否正常}
            feature = params[0].get("feature")
            if params[0].get("index") >= max_index:  # json中的索引异常: index >= 数据列数
                raise ValueError("index error")
            if feature not in feature_list:  # 变量异常:变量名不在数据的列名中
                raise ValueError("{} do not belong to the data ".format(feature))
            return feature
        elif op in op_dict:  # 两个值，递归
            value_list = [json2expr(params[0], max_index, feature_list), json2expr(params[1], max_index, feature_list)]
            return "(" + str(value_list[0]) + op_dict[op] + str(value_list[1]) + ")"
        else:  # op 不在op_dict报错
            raise TypeError("The operator: {} is invalid".format(op))

    if data.keys().__contains__("value"):
        value_type = data.get("value_type")
        value = data["value"]
        # 对取到值的类型做转换， 不在类型字典中的值报错
        if not value_type_dict.get(value_type):
            raise ValueError("Data type error!")
        return value_type_dict.get(value_type)(value)


class Rule:
    def __init__(self, expr):  # expr 既可以传递字符串，也可以传递dict
        """规则集

        :param expr: 类似 DataFrame 的 query 方法传参方式即可，目前仅支持数值型变量规则

        **参考样例**

        >>> from scorecardpipeline import *
        >>> target = "creditability"
        >>> data = germancredit()
        >>> data[target] = data[target].map({"good": 0, "bad": 1})
        >>> data = data.select_dtypes("number") # 暂不支持字符型规则
        >>> rule1 = Rule("duration_in_month < 10")
        >>> rule2 = Rule("credit_amount < 500")
        >>> rule1.report(data, target=target)
        >>> rule2.report(data, target=target)
        >>> (rule1 | rule2).report(data, target=target)
        >>> (rule1 & rule2).report(data, target=target)
        """
        self._state = RuleState.INITIALIZED
        self.expr = expr

    def __str__(self):
        return f"Rule({repr(self.expr)})"

    def __repr__(self):
        return f"Rule({repr(self.expr)})"

    def predict(self, X: DataFrame, part=""):  # dict预测对应part_dict 、字符串表达式对应"、"其他情况报错
        if not isinstance(X, DataFrame):
            raise ValueError("Rule can only predict on DataFrame.")
        
        check_array(X, dtype=None, ensure_2d=True, force_all_finite="allow-nan")
        result = X.eval(self.expr)
        
        # feature_names = X.columns.values.tolist()  # 取数据的列名
        # X = X.select_dtypes("number") # 仅支持数值型变量
        # X = check_array(X, dtype=None, ensure_2d=True, force_all_finite="allow-nan")
        # if isinstance(self.expr, dict):  # dict部分
        #     if part not in part_dict:
        #         raise TypeError("Part : {} not in ['if','then','else']".format(part))
        #     if not self.expr[part]:  # 没有返回值的情况[]
        #         return list()
        #     dict2expr = json2expr(self.expr[part], X.shape[1], feature_names)
        #     if not isinstance(dict2expr, str):  # 返回Value (类型已经做过转换),对其扩充 --> [value] * Len(X)
        #         result = [dict2expr] * len(X)
        #     else:  # 表达式在进行计算
        #         result = ne.evaluate(dict2expr, local_dict={name: X[:, i] for i, name in enumerate(feature_names)})
        #         result = result.tolist()
        #         if not isinstance(result, list):  # result 只有一个数值时，对其扩充 --> [value] * len(X)
        #             result = [result] * len(X)
        # elif isinstance(self.expr, str):  # 字符串表达式部分
        #     if part != "":
        #         raise TypeError('The part of the expression must be ""')
        #     result = ne.evaluate(self.expr, local_dict={name: X[:, i] for i, name in enumerate(feature_names)})
        # else:
        #     raise TypeError("Rule currently only supports dict and expression")

        self.result_ = result

        return result

    def report(self, datasets: pd.DataFrame, target="target", overdue=None, dpd=None, del_grey=False, desc="", filter_cols=None, prior_rules=None) -> pd.DataFrame:
        """规则效果报告表格输出

        :param datasets: 数据集，需要包含 目标变量 或 逾期天数，当不包含目标变量时，会通过逾期天数计算目标变量，同时需要传入逾期定义的DPD天数
        :param target: 目标变量名称，默认 target
        :param desc: 规则相关的描述，会出现在返回的表格当中
        :param filter_cols: 指定返回的字段列表，默认不传
        :param prior_rules: 先验规则，可以传入先验规则先筛选数据后再评估规则效果
        :param overdue: 逾期天数字段名称
        :param dpd: 逾期定义方式，逾期天数 > DPD 为 1，其他为 0，仅 overdue 字段起作用时有用
        :param del_grey: 是否删除逾期天数 (0, dpd] 的数据，仅 overdue 字段起作用时有用

        :return: pd.DataFrame，规则效果评估表
        """
        return_cols = ['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', 'LIFT值', '坏账改善']
        if desc is None or desc == "" and "指标含义" in return_cols:
            return_cols.remove("指标含义")

        rule_expr = self.expr

        def _report_one_rule(data, target, desc='', prior_rules=None):
            if prior_rules:
                prior_tables = prior_rules.report(data, target=target, desc=desc, prior_rules=None)
                prior_tables["规则分类"] = "先验规则"
                temp = data[~prior_rules.predict(data)]
                rule_result = pd.DataFrame({rule_expr: np.where(self.predict(temp), "命中", "未命中"), "target": temp[target].tolist()})
            else:
                prior_tables = pd.DataFrame(columns=return_cols)
                rule_result = pd.DataFrame({rule_expr: np.where(self.predict(data), "命中", "未命中"), "target": data[target].tolist()})

            combiner = Combiner(target=target)
            combiner.load({rule_expr: [["命中"], ["未命中"]]})
            table = feature_bin_stats(rule_result, rule_expr, combiner=combiner, desc=desc, return_cols=return_cols)

            # 准确率、精确率、召回率、F1分数
            metrics = pd.DataFrame({
                "分箱": ["命中", "未命中"],
                "准确率": [accuracy_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), accuracy_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
                "精确率": [precision_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), precision_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
                "召回率": [recall_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), recall_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
                "F1分数": [f1_score(rule_result["target"], rule_result[rule_expr].map({"命中": 1, "未命中": 0})), f1_score(rule_result["target"], rule_result[rule_expr].map({"命中": 0, "未命中": 1}))],
            })
            table = table.merge(metrics, on="分箱", how="left")

            if prior_rules:
                # prior_tables.insert(loc=0, column="规则分类", value=["先验规则"] * len(prior_tables))
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))
                table = pd.concat([prior_tables, table]) #.set_index(["规则分类"])
            else:
                table.insert(loc=0, column="规则分类", value=["验证规则"] * len(table))

            return table

        if isinstance(del_grey, bool) and del_grey:
            merge_columns = ["规则分类", "指标名称", "分箱"]
        else:
            merge_columns = ["规则分类", "指标名称", "分箱", "样本总数", "样本占比"]

        if overdue is not None:
            if not isinstance(overdue, list):
                overdue = [overdue]

            if not isinstance(dpd, list):
                dpd = [dpd]

            for i, col in enumerate(overdue):
                for j, d in enumerate(dpd):
                    _datasets = datasets.copy()
                    _datasets[f"{col}_{d}"] = (_datasets[col] > d).astype(int)

                    if isinstance(del_grey, bool) and del_grey:
                        _datasets = _datasets.query(f"({col} > {d}) | ({col} == 0)").reset_index(drop=True)

                    if "指标含义" in return_cols:
                        merge_columns.insert(0, "指标含义")

                    if i == 0 and j == 0:
                        table = _report_one_rule(_datasets, f"{col}_{d}", desc=desc, prior_rules=prior_rules) #.rename(columns={"坏账改善": f"{col} {d}+改善"})
                        table.columns = pd.MultiIndex.from_tuples([("规则详情", c) if c in merge_columns else (f"{col} DPD{d}+", c) for c in table.columns])
                    else:
                        _table = _report_one_rule(_datasets, f"{col}_{d}", desc=desc, prior_rules=prior_rules) #.rename(columns={"坏账改善": f"{col} {d}+改善"})
                        _table.columns = pd.MultiIndex.from_tuples([("规则详情", c) if c in merge_columns else (f"{col} DPD{d}+", c) for c in _table.columns])

                        # table = table.merge(_table[["规则分类", "分箱", f"{col} {d}+改善"]], on=["规则分类", "分箱"])
                        table = table.merge(_table, on=[("规则详情", c) for c in merge_columns])
        else:
            _datasets = datasets.copy()
            table = _report_one_rule(_datasets, target, desc=desc, prior_rules=prior_rules)

        if filter_cols:
            if not isinstance(filter_cols, list):
                filter_cols = [filter_cols]
            return table[[c for c in table.columns if (isinstance(c, tuple) and c[-1] in filter_cols + merge_columns) or (not isinstance(c, tuple) and c in filter_cols + merge_columns)]]

        return table

    def result(self):
        if self._state != RuleState.APPLIED:
            raise RuleUnAppliedError("Invoke `predict` to make a rule applied.")
        return self.result_

    def __eq__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        res = self.expr == other.expr
        if self._state == RuleState.INITIALIZED:
            return res
        return res and np.all(self.result() == other.result())

    # rule combinations
    def __or__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        if isinstance(self.expr, str):
            r = Rule(f"({self.expr}) | ({other.expr})")
            if self._state == RuleState.INITIALIZED:
                return r
            r.result_ = np.logical_or(self.result(), other.result())
            r._state = RuleState.APPLIED
            return r
        elif isinstance(self.expr, dict):
            self.new_dict = {}  # 汇总成新的json
            self.new_dict["name"] = str(self.expr.get("name")) + str(other.expr.get("name"))
            self.new_dict["description"] = str(self.expr.get("description")) + " || " + str(other.expr.get("description"))
            self.new_dict["output"] = self.expr.get("output")

            # if_part
            if_dict = {}
            if_dict["value_type"] = "bool"
            if_dict["operator"] = "OR"
            if_dict["params"] = list()
            if_dict["params"].append(self.expr.get("if"))
            if_dict["params"].append(other.expr.get("if"))
            self.new_dict["if"] = if_dict

            # then_part
            then_part = {}
            if not self.expr.get("then") and not other.expr.get("then"):  # 两条规则的then都为空
                then_part = {}
            elif not self.expr.get("then"):  # 一条规则的then存在
                then_part = other.expr.get("then")
            elif not other.expr.get("then"):
                then_part = self.expr.get("then")
            else:  # 两条规则的then都存在
                if self.expr.get("then").get("value_type") != other.expr.get("then").get("value_type"):
                    raise TypeError("两个规则then_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间or运算, 类型需要设置为bool类型")
                then_part["value_type"] = "bool"
                then_part["operator"] = "OR"
                then_part["params"] = list()
                then_part["params"].append(self.expr.get("then"))
                then_part["params"].append(other.expr.get("then"))
            self.new_dict["then"] = then_part

        # else_part
        else_part = {}  # self.else或者other.else存在为空的情况
        if not self.expr.get("else") and not other.expr.get("else"):
            else_part = {}
        elif not self.expr.get("else"):  # 一条规则的then存在
            else_part = other.expr.get("else")
        elif not other.expr.get("else"):
            else_part = self.expr.get("else")
        else:
            if self.expr.get("then").get("value_type") != other.expr.get("then").get("value_type"):
                raise TypeError("两个规则else part类型要一致")
            if self.expr.get("then").get("value_type") != "bool":
                raise TypeError("两个规则之间or运算, 类型需要设置为bool类型")
            else_part["value_type"] = "bool"
            else_part["operator"] = "OR"
            else_part["params"] = list()
            else_part["params"].append(self.expr.get("else"))
            else_part["params"].append(other.expr.get("else"))
        self.new_dict["else"] = else_part

        return Rule(self.new_dict)

    def __and__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        if isinstance(self.expr, str):  # 表达式
            r = Rule(f"({self.expr}) & ({other.expr})")
            if self._state == RuleState.INITIALIZED:
                return r
            r.result_ = np.logical_and(self.result(), other.result())
            r._state = RuleState.APPLIED
            return r
        elif isinstance(self.expr, dict):  # dict
            self.new_dict = {}  # 汇总成新的json
            self.new_dict["name"] = str(self.expr.get("name")) + str(other.expr.get("name"))
            self.new_dict["description"] = str(self.expr.get("description")) + " && " + str(other.expr.get("description"))
            self.new_dict["output"] = self.expr.get("output")

            # if_part
            if_dict = {}
            if_dict["value_type"] = "bool"
            if_dict["operator"] = "AND"
            if_dict["params"] = list()
            if_dict["params"].append(self.expr.get("if"))
            if_dict["params"].append(other.expr.get("if"))
            self.new_dict["if"] = if_dict

            # then_part
            then_part = {}
            if not self.expr.get("then") and not other.expr.get("then"):  # 两条规则的then都为空
                then_part = {}
            elif not self.expr.get("then"):  # 一条规则的then存在
                then_part = other.expr.get("then")
            elif not other.expr.get("then"):
                then_part = self.expr.get("then")
            else:  # 两条规则的then都存在
                if self.expr["then"].get("value_type") != other.expr["then"].get("value_type"):
                    raise TypeError("两个规则then_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间and运算, 类型需要设置为bool类型")
                then_part["value_type"] = "bool"
                then_part["operator"] = "AND"
                then_part["params"] = list()
                then_part["params"].append(self.expr.get("then"))
                then_part["params"].append(other.expr.get("then"))
            self.new_dict["then"] = then_part

            # else_part
            else_part = {}  # self.else 或者other.else 存在为空的情况
            if not self.expr.get("else") and not other.expr.get("else"):
                else_part = {}
            elif not self.expr.get("else"):  # 一条规则的then存在
                else_part = other.expr.get("else")
            elif not other.expr.get("else"):
                else_part = self.expr.get("else")
            else:
                if self.expr.get("else").get("value_type") != other.expr.get("else").get("value_type"):
                    raise TypeError("两个规则else_part类型要一致")
                if self.expr.get("then").get("value_type") != "bool":
                    raise TypeError("两个规则之间and运算, 类型需要设置为bool类型")
                else_part["value_type"] = "bool"
                else_part["operator"] = "AND"
                else_part["params"] = list()
                else_part["params"].append(self.expr.get("else"))
                else_part["params"].append(other.expr.get("else"))
            self.new_dict["else"] = else_part

            return Rule(self.new_dict)

    def __xor__(self, other):
        if not isinstance(other, Rule):
            raise TypeError(f"Input should be of type Rule, got {type(other)} instead.")
        if self._state != other._state:
            raise RuleStateError(f"Input rule should be of the same state.")
        r = Rule(f"({self.expr}) ^ ({other.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r.result_ = np.logical_xor(self.result(), other.result())
        r._state = RuleState.APPLIED
        return r

    def __mul__(self, other):
        return self.__or__(other)

    def __invert__(self):
        r = Rule(f"~({self.expr})")
        if self._state == RuleState.INITIALIZED:
            return r
        r.result_ = np.logical_not(self.result())
        r._state = RuleState.APPLIED
        return r

    @staticmethod
    def save(report, excel_writer, sheet_name=None, merge_column=None, percent_cols=None, condition_cols=None, custom_cols=None, custom_format="#,##0", color_cols=None, start_col=2, start_row=2, **kwargs):
        """保存规则结果至excel中，参数与 https://scorecardpipeline.itlubber.art/scorecardpipeline.html#scorecardpipeline.dataframe2excel 一致
        """
        if merge_column:
            merge_column = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in merge_column) or (not isinstance(c, tuple) and c in merge_column)]

        if percent_cols:
            percent_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in percent_cols) or (not isinstance(c, tuple) and c in percent_cols)]

        if condition_cols:
            condition_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in condition_cols) or (not isinstance(c, tuple) and c in condition_cols)]
        
        if custom_cols:
            custom_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in custom_cols) or (not isinstance(c, tuple) and c in custom_cols)]
        
        if color_cols:
            color_cols = [c for c in report.columns if (isinstance(c, tuple) and c[-1] in color_cols) or (not isinstance(c, tuple) and c in color_cols)]
        
        end_row, end_col = dataframe2excel(report, excel_writer, sheet_name=sheet_name, merge_column=merge_column, percent_cols=percent_cols, condition_cols=condition_cols, custom_cols=custom_cols, custom_format=custom_format, color_cols=color_cols, start_col=start_col, start_row=start_row, **kwargs)
        return end_row, end_col
