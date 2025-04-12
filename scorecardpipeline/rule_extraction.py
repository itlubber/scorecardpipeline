# -*- coding: utf-8 -*-
"""
@Time    : 2024/2/29 13:29
@Author  : itlubber
@Site    : itlubber.art
"""
import warnings
import os
import re
import graphviz
import dtreeviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from openpyxl.worksheet.worksheet import Worksheet

import category_encoders as ce
from optbinning import OptimalBinning
from sklearn.tree import DecisionTreeClassifier

from .rule import Rule
from .utils import init_setting
from .excel_writer import ExcelWriter, dataframe2excel


class DecisionTreeRuleExtractor:
    def __init__(self, target="target", labels=["positive", "negative"], feature_map={}, nan=-1., max_iter=128, writer=None, seed=None, theme_color="2639E9", decimal=4):
        """决策树自动规则挖掘工具包

        :param target: 数据集中好坏样本标签列名称，默认 target
        :param labels: 好坏样本标签名称，传入一个长度为2的列表，第0个元素为好样本标签，第1个元素为坏样本标签，默认 ["positive", "negative"]
        :param feature_map: 变量名称及其含义，在后续输出报告和策略信息时增加可读性，默认 {}
        :param nan: 在决策树策略挖掘时，默认空值填充的值，默认 -1
        :param max_iter: 最多支持在数据集上训练多少颗树模型，每次生成一棵树后，会剔除特征重要性最高的特征后，再生成树，默认 128
        :param writer: 在之前程序运行时生成的 ExcelWriter，可以支持传入一个已有的writer，后续所有内容将保存至该workbook中，默认 None
        :param seed: 随机种子，保证结果可复现使用，默认为 None
        :param theme_color: 主题色，默认 2639E9 克莱因蓝，可设置位其他颜色
        :param decimal: 精度，决策树分裂节点阈值的精度范围，默认 4，即保留4位小数
        """
        self.decimal = decimal
        self.seed = seed
        self.nan = nan
        self.target = target
        self.labels = labels
        self.theme_color = theme_color
        self.feature_map = feature_map
        self.decision_trees = []
        self.max_iter = max_iter
        self.target_enc = None
        self.feature_names = None
        self.dt_rules = pd.DataFrame()
        self.end_row = 2
        self.start_col = 2
        self.describe_columns = ["组合策略", "命中数", "命中率", "好样本数", "好样本占比", "坏样本数", "坏样本占比", "坏样本率", "LIFT值", "坏账改善", "准确率", "精确率", "召回率", "F1分数", "样本整体坏率"]

        init_setting()

        if writer:
            self.writer = writer
        else:
            self.writer = ExcelWriter(theme_color=self.theme_color)

    def encode_cat_features(self, X, y):
        cat_features = list(set(X.select_dtypes(include=[object, pd.CategoricalDtype]).columns))
        cat_features_index = [i for i, f in enumerate(X.columns) if f in cat_features]

        if len(cat_features) > 0:
            if self.target_enc is None:
                self.target_enc = ce.TargetEncoder(cols=cat_features)
                self.target_enc.fit(X[cat_features], y)
                self.target_enc.target_mapping = {}
                X_TE = X.join(self.target_enc.transform(X[cat_features]).add_suffix('_target'))
                for col in cat_features:
                    mapping = X_TE[[col, f"{col}_target"]].drop_duplicates()
                    self.target_enc.target_mapping[col] = dict(zip(mapping[col], mapping[f"{col}_target"]))
            else:
                X_TE = X.join(self.target_enc.transform(X[cat_features]).add_suffix('_target'))

            X_TE = X_TE.drop(columns=cat_features)
            return X_TE.rename(columns={f"{c}_target": c for c in cat_features})
        else:
            return X

    def get_dt_rules(self, tree):
        rules = dict()

        def recurse(node=0, parent=None):  # 搜每个节点的规则
            if node == 0 or tree.tree_.children_left[node] != -1:  # 非叶子节点,搜索每个节点的规则
                name = tree.feature_names_in_[tree.tree_.feature[node]]
                threshold = np.round(tree.tree_.threshold[node], self.decimal)
                if parent:
                    recurse(tree.tree_.children_left[node], parent & Rule("{} <= {}".format(name, threshold)))
                    recurse(tree.tree_.children_right[node], parent & Rule("{} > {}".format(name, threshold)))
                else:
                    recurse(tree.tree_.children_left[node], Rule("{} <= {}".format(name, threshold)))
                    recurse(tree.tree_.children_right[node], Rule("{} > {}".format(name, threshold)))
            else:
                rules[node] = parent

        recurse()

        return list(rules.values())

    def select_dt_rules(self, decision_tree, x, y, lift=0., max_samples=1., save=None, verbose=False, drop=False):
        rules = self.get_dt_rules(decision_tree)

        try:
            viz_model = dtreeviz.model(decision_tree,
                                       X_train=x,
                                       y_train=y,
                                       feature_names=decision_tree.feature_names_in_,
                                       target_name=self.target,
                                       class_names=self.labels,
                                       )
        except AttributeError:
            raise "请检查 dtreeviz 版本"

        rules_reports = pd.DataFrame()
        for rule in rules:
            rules_reports = pd.concat([rules_reports, rule.report(x.join(y), target=y.name).query("分箱 == '命中'")])

        rules_reports = rules_reports.rename(columns={"指标名称": "组合策略", "样本总数": "命中数", "样本占比": "命中率"}).drop(columns=["分箱"])
        rules_reports["样本整体坏率"] = round(y.mean(), self.decimal)
        rules_reports = rules_reports.query(f"LIFT值 >= {lift} & 命中率 <= {max_samples}").reset_index(drop=True)

        if len(rules_reports) > 0:
            # font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf')
            # font_manager.fontManager.addfont(font_path)
            # plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
            # plt.rcParams['axes.unicode_minus'] = False

            decision_tree_viz = viz_model.view(
                scale=1.5,
                orientation='LR',
                colors={
                    "classes": [None, None, ["#2639E9", "#F76E6C"], ["#2639E9", "#F76E6C", "#FE7715", "#FFFFFF"]],
                    "arrow": "#2639E9",
                    'text_wedge': "#F76E6C",
                    "pie": "#2639E9",
                    "tile_alpha": 1,
                    "legend_edge": "#FFFFFF",
                },
                ticks_fontsize=10,
                label_fontsize=10,
                fontname=plt.rcParams['font.family'],
            )
            if verbose:
                from IPython.display import display
                if self.feature_map is not None and len(self.feature_map) > 0:
                    display(rules_reports.replace(self.feature_map, regex=True))
                else:
                    display(rules_reports)
                display(decision_tree_viz)
            if save:
                if os.path.dirname(save) and not os.path.exists(os.path.dirname(save)):
                    os.makedirs(os.path.dirname(save))

                try:
                    decision_tree_viz.save("combine_rules_cache.svg")
                except graphviz.backend.execute.ExecutableNotFound:
                    print("请确保您已安装 graphviz 程序并且正确配置了 PATH 路径。可参考: https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft")

                try:
                    import cairosvg
                    cairosvg.svg2png(url="combine_rules_cache.svg", write_to=save, dpi=240)
                except:
                    from reportlab.graphics import renderPDF
                    from svglib.svglib import svg2rlg
                    drawing = svg2rlg("combine_rules_cache.svg")
                    renderPDF.drawToFile(drawing, save, dpi=240, fmt="PNG")

        if os.path.isfile("combine_rules_cache.svg"):
            os.remove("combine_rules_cache.svg")

        if os.path.isfile("combine_rules_cache"):
            os.remove("combine_rules_cache")

        if drop:
            if len(rules_reports) > 0:
                return rules_reports, decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(max(decision_tree.feature_importances_))], len(rules_reports)
            else:
                return rules_reports, decision_tree.feature_names_in_[list(decision_tree.feature_importances_).index(min(decision_tree.feature_importances_))], len(rules_reports)
        else:
            return rules_reports, len(rules_reports)

    def query_dt_rules(self, x, y, parsed_rules=None):
        if isinstance(parsed_rules, pd.DataFrame):
            parsed_rules = [Rule(r) for r in parsed_rules["组合策略"].unique()]

        rules_reports = pd.DataFrame()
        for rule in parsed_rules:
            rules_reports = pd.concat([rules_reports, rule.report(x.join(y), target=y.name).query("分箱 == '命中'")])

        rules_reports = rules_reports.rename(columns={"指标名称": "组合策略", "样本总数": "命中数", "样本占比": "命中率"}).drop(columns=["分箱"])
        rules_reports["样本整体坏率"] = round(y.mean(), self.decimal)

        return rules_reports

    def insert_dt_rules(self, parsed_rules, end_row, start_col, save=None, sheet=None, figsize=(500, 350)):
        if isinstance(sheet, Worksheet):
            worksheet = sheet
        else:
            worksheet = self.writer.get_sheet_by_name(sheet or "决策树组合策略挖掘")

        end_row, end_col = dataframe2excel(parsed_rules, self.writer, sheet_name=worksheet, start_row=end_row + 1, start_col=start_col, percent_cols=['好样本占比', '坏样本占比', '命中率', '坏样本率', '样本整体坏率', 'LIFT值', '坏账改善', '准确率', '精确率', '召回率', 'F1分数'], condition_cols=["坏样本率", "LIFT值"])

        if save is not None:
            end_row, end_col = self.writer.insert_pic2sheet(worksheet, save, (end_row + 1, start_col), figsize=figsize)

        return end_row, end_col

    def fit(self, x, y=None, max_depth=2, lift=0., max_samples=1., min_score=None, verbose=False, *args, **kwargs):
        """组合策略挖掘

        :param x: 包含标签的数据集
        :param max_depth: 决策树最大深度，即最多组合的特征个数，默认 2
        :param lift: 组合策略最小的lift值，默认 0.，即全部组合策略
        :param max_samples: 每条组合策略的最大样本占比，默认 1.0，即全部组合策略
        :param min_score: 决策树拟合时最小的auc，如果不满足则停止后续生成决策树
        :param verbose: 是否调试模式，仅在 jupyter 环境有效
        :param kwargs: DecisionTreeClassifier 参数，参考 https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        """
        worksheet = self.writer.get_sheet_by_name("策略详情")

        y = x[self.target]
        X_TE = self.encode_cat_features(x.drop(columns=[self.target]), y)
        X_TE = X_TE.fillna(self.nan)

        self.feature_names = list(X_TE.columns)

        for i in range(self.max_iter):
            decision_tree = DecisionTreeClassifier(max_depth=max_depth, *args, **kwargs)
            decision_tree = decision_tree.fit(X_TE, y)

            if (min_score is not None and decision_tree.score(X_TE, y) < min_score) or len(X_TE.columns) < max_depth:
                break

            try:
                parsed_rules, remove, total_rules = self.select_dt_rules(decision_tree, X_TE, y, lift=lift, max_samples=max_samples, verbose=verbose, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png", drop=True)

                if len(parsed_rules) > 0:
                    self.dt_rules = pd.concat([self.dt_rules, parsed_rules]).reset_index(drop=True)

                    if self.writer is not None:
                        if self.feature_map is not None and len(self.feature_map) > 0:
                            parsed_rules["组合策略"] = parsed_rules["组合策略"].replace(self.feature_map, regex=True)
                        self.end_row, _ = self.insert_dt_rules(parsed_rules, self.end_row, self.start_col, save=f"model_report/auto_mining_rules/combiner_rules_{i}.png", figsize=(500, 100 * total_rules), sheet=worksheet)

                X_TE = X_TE.drop(columns=remove)
                self.decision_trees.append(decision_tree)
            except:
                import traceback
                traceback.print_exc()

        if len(self.dt_rules) <= 0:
            print(f"未挖掘到有效策略, 可以考虑适当调整预设的筛选参数, 降低 lift / 提高 max_samples, 当前筛选标准为: 提取 lift >= {lift} 且 max_samples <= {max_samples} 的策略")

        return self

    def transform(self, x, y=None):
        y = x[self.target]
        X_TE = self.encode_cat_features(x.drop(columns=[self.target]), y)
        X_TE = X_TE.fillna(self.nan)
        if self.dt_rules is not None and len(self.dt_rules) > 0:
            parsed_rules = self.query_dt_rules(X_TE, y, parsed_rules=self.dt_rules)
            if self.feature_map is not None and len(self.feature_map) > 0:
                parsed_rules["组合策略"] = parsed_rules["组合策略"].replace(self.feature_map, regex=True)
            return parsed_rules
        else:
            return pd.DataFrame(columns=self.describe_columns)

    def report(self, valid=None, sheet="组合策略汇总", save=None):
        """组合策略插入excel文档

        :param valid: 验证数据集
        :param sheet: 保存组合策略的表格sheet名称
        :param save: 保存报告的文件路径

        :return: 返回每个数据集组合策略命中情况
        """
        worksheet = self.writer.get_sheet_by_name(sheet or "决策树组合策略挖掘")

        if sheet:
            self.writer.workbook.move_sheet(sheet, -1)

        parsed_rules_train = self.dt_rules.copy()

        if self.feature_map is not None and len(self.feature_map) > 0:
            parsed_rules_train["组合策略"] = parsed_rules_train["组合策略"].replace(self.feature_map, regex=True)

        self.end_row, _ = self.writer.insert_value2sheet(worksheet, (2 if sheet else self.end_row + 2, self.start_col), value="组合策略: 训练集", style="header_middle", end_space=(2 if sheet else self.end_row + 2, self.start_col + len(parsed_rules_train.columns) - 1))
        self.end_row, _ = self.insert_dt_rules(parsed_rules_train, self.end_row, self.start_col, sheet=worksheet)
        outputs = (parsed_rules_train,)

        if valid is not None:
            if isinstance(valid, pd.DataFrame) and len(valid) > 0:
                parsed_rules_val = self.transform(valid)
                self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value="组合策略: 验证集", style="header_middle", end_space=(self.end_row + 2, self.start_col + len(parsed_rules_val.columns) - 1))
                self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col, sheet=worksheet)
                outputs = outputs + (parsed_rules_val,)

            elif isinstance(valid, (list, tuple)):
                for i, dataset in enumerate(valid):
                    if isinstance(dataset, pd.DataFrame) and len(dataset) > 0:
                        parsed_rules_val = self.transform(dataset)
                        self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value=f"组合策略: 验证集 {i + 1}", style="header_middle", end_space=(self.end_row + 2, self.start_col + len(parsed_rules_val.columns) - 1))
                        self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col, sheet=worksheet)
                        outputs = outputs + (parsed_rules_val,)

            elif isinstance(valid, dict):
                for k, dataset in valid.items():
                    if isinstance(dataset, pd.DataFrame) and len(dataset) > 0:
                        parsed_rules_val = self.transform(dataset)
                        self.end_row, _ = self.writer.insert_value2sheet(worksheet, (self.end_row + 2, self.start_col), value=f"组合策略: {k}", style="header_middle", end_space=(self.end_row + 2, self.start_col + len(parsed_rules_val.columns) - 1))
                        self.end_row, _ = self.insert_dt_rules(parsed_rules_val, self.end_row, self.start_col, sheet=worksheet)
                        outputs = outputs + (parsed_rules_val,)

        if save:
            self.writer.save(save)

        return outputs
