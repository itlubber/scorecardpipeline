# -*- coding: utf-8 -*-
"""
@Time    : 2023/05/21 16:23
@Author  : itlubber
@Site    : itlubber.art
"""

import os
import math
import numpy as np
import pandas as pd
import scorecardpy as sc
import matplotlib.pyplot as plt
import toad
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report
from sklearn.utils._array_api import get_namespace
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from .utils import *
from .processing import *


class ITLubberLogisticRegression(LogisticRegression):

    def __init__(self, target="target", penalty="l2", calculate_stats=True, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=False, n_jobs=None, l1_ratio=None, ):
        """ITLubberLogisticRegression，继承 sklearn.linear_model.LogisticRegression 方法，增加了统计性描述相关的内容输出，核心实现逻辑参考：https://github.com/ing-bank/skorecard/blob/main/skorecard/linear_model/linear_model.py#L11

        :param target: 数据集中标签名称，默认 target
        :param calculate_stats: 是否在训练模型时记录模型统计信息，默认 True，可以通过 summary 方法输出相关统计信息
        :param tol: 停止求解的标准，float类型，默认为1e-4
        :param C: 正则化系数λ的倒数，float类型，默认为1.0，必须是正浮点型数，值越小惩罚越大
        :param fit_intercept: 是否存在截距或偏差，bool类型，默认为True
        :param class_weight: 类型权重参数，默认 None，支持传入 dict or balanced，当设置 balanced 时，权重计算方式：n_samples / (n_classes * np.bincount(y))
        :param solver: 求解器设置，默认 lbfgs。对于小型数据集来说，选择 liblinear 更好；对于大型数据集来说，saga 或者 sag 会更快一些。对于多类问题我们只能使用 newton-cg、sag、saga、lbfgs。对于正则化来说，newton-cg、lbfgs 和 sag 只能用于L2正则化(因为这些优化算法都需要损失函数的一阶或者二阶连续导数， 因此无法用于没有连续导数的L1正则化)；而 liblinear，saga 则可处理L1正则化。newton-cg 是牛顿家族中的共轭梯度法，lbfgs 是一种拟牛顿法，sag 则是随机平均梯度下降法，saga 是随机优化算法，liblinear 是坐标轴下降法。
        :param penalty: 惩罚项，默认 l2，可选 l1、l2，solver 为 newton-cg、sag 和 lbfgs 时只支持L2，L1假设的是模型的参数满足拉普拉斯分布，L2假设的模型参数满足高斯分布
        :param intercept_scaling: 仅在 solver 选择 liblinear 并且 fit_intercept 设置为 True 的时候才有用
        :param dual: 对偶或原始方法，bool类型，默认为False，对偶方法只用在求解线性多核(liblinear)的L2惩罚项上。当样本数量>样本特征的时候，dual通常设置为False
        :param random_state: 随机数种子，int类型，可选参数，默认为无，仅在 solver 为 sag 和 liblinear 时有用
        :param max_iter: 算法收敛最大迭代次数，int类型，默认 100。只在 solver 为 newton-cg、sag 和 lbfgs 时有用
        :param multi_class: 分类方法参数选择，默认 auto，可选 ovr、multinomial，如果分类问题是二分类问题，那么这两个参数的效果是一样的，主要体现在多分类问题上
        :param verbose: 日志级别，当 solver 为 liblinear、lbfgs 时设置为任意正数显示详细计算过程
        :param warm_start: 热启动参数，bool类型，表示是否使用上次的模型结果作为初始化，默认为 False
        :param n_jobs: 并行运算数量，默认为1，如果设置为-1，则表示将电脑的cpu全部用上
        :param l1_ratio: 弹性网络参数，其中0 <= l1_ratio <=1，仅当 penalty 为 elasticnet 时有效

        **参考样例**

        >>> feature_pipeline = Pipeline([
        >>>     ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
        >>>     ("combiner", Combiner(target=target, min_samples=0.2)),
        >>>     ("transform", WOETransformer(target=target)),
        >>>     ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
        >>>     ("stepwise", StepwiseSelection(target=target)),
        >>>     # ("logistic", LogisticClassifier(target=target)),
        >>>     ("logistic", ITLubberLogisticRegression(target=target)),
        >>> ])
        >>> feature_pipeline.fit(train)
        >>> summary = feature_pipeline.named_steps['logistic'].summary()
        >>> summary
                                                            Coef.  Std.Err       z  P>|z|  [ 0.025  0.975 ]    VIF
        const                                               -0.8511   0.0991 -8.5920 0.0000  -1.0452  -0.6569 1.0600
        credit_history                                       0.8594   0.1912  4.4954 0.0000   0.4847   1.2341 1.0794
        age_in_years                                         0.6176   0.2936  2.1032 0.0354   0.0421   1.1932 1.0955
        savings_account_and_bonds                            0.8842   0.2408  3.6717 0.0002   0.4122   1.3563 1.0331
        credit_amount                                        0.7027   0.2530  2.7771 0.0055   0.2068   1.1987 1.1587
        status_of_existing_checking_account                  0.6891   0.1607  4.2870 0.0000   0.3740   1.0042 1.0842
        personal_status_and_sex                              0.8785   0.5051  1.7391 0.0820  -0.1116   1.8685 1.0113
        purpose                                              1.1370   0.2328  4.8844 0.0000   0.6807   1.5932 1.0282
        present_employment_since                             0.7746   0.3247  2.3855 0.0171   0.1382   1.4110 1.0891
        installment_rate_in_percentage_of_disposable_income  1.3785   0.3434  4.0144 0.0001   0.7055   2.0515 1.0300
        duration_in_month                                    0.9310   0.1986  4.6876 0.0000   0.5417   1.3202 1.1636
        other_installment_plans                              0.8521   0.3459  2.4637 0.0138   0.1742   1.5301 1.0117
        housing                                              0.8251   0.4346  1.8983 0.0577  -0.0268   1.6770 1.0205
        """
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio, )
        self.target = target
        self.calculate_stats = calculate_stats

    def fit(self, x, sample_weight=None, **kwargs):
        """逻辑回归训练方法

        :param x: 训练数据集，需包含目标变量
        :param sample_weight: 样本权重，参考：https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit
        :param kwargs: 其他逻辑回归模型训练参数
        :return: ITLubberLogisticRegression，训练完成的逻辑回归模型
        """
        y = x[self.target]
        x = x.drop(columns=[self.target])

        if not self.calculate_stats:
            return super().fit(x, y, sample_weight=sample_weight, **kwargs)

        x = self.convert_sparse_matrix(x)

        if isinstance(x, pd.DataFrame):
            self.names_ = ["const"] + [f for f in x.columns]
        else:
            self.names_ = ["const"] + [f"x{i}" for i in range(x.shape[1])]

        lr = super().fit(x, y, sample_weight=sample_weight, **kwargs)

        predProbs = self.predict_proba(x)

        # Design matrix -- add column of 1's at the beginning of your x matrix
        if lr.fit_intercept:
            x_design = np.hstack([np.ones((x.shape[0], 1)), x])
        else:
            x_design = x

        self.vif = [variance_inflation_factor(np.matrix(x_design), i) for i in range(x_design.shape[-1])]
        p = np.product(predProbs, axis=1)
        self.cov_matrix_ = np.linalg.inv((x_design * p[..., np.newaxis]).T @ x_design)
        std_err = np.sqrt(np.diag(self.cov_matrix_)).reshape(1, -1)

        # In case fit_intercept is set to True, then in the std_error array
        # Index 0 corresponds to the intercept, from index 1 onwards it relates to the coefficients
        # If fit intercept is False, then all the values are related to the coefficients
        if lr.fit_intercept:

            self.std_err_intercept_ = std_err[:, 0]
            self.std_err_coef_ = std_err[:, 1:][0]

            self.z_intercept_ = self.intercept_ / self.std_err_intercept_

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = scipy.stats.norm.sf(abs(self.z_intercept_)) * 2

        else:
            self.std_err_intercept_ = np.array([np.nan])
            self.std_err_coef_ = std_err[0]

            self.z_intercept_ = np.array([np.nan])

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = np.array([np.nan])

        self.z_coef_ = self.coef_ / self.std_err_coef_
        self.p_val_coef_ = scipy.stats.norm.sf(abs(self.z_coef_)) * 2

        return self

    def decision_function(self, x):
        """决策函数

        :param x: 需要预测的数据集，可以包含目标变量，会根据列名进行判断，如果包含会删除相关特征
        :return: np.ndarray，预测结果
        """
        check_is_fitted(self)

        if isinstance(x, pd.DataFrame) and self.target in x.columns:
            x = x.drop(columns=self.target)

        xp, _ = get_namespace(x)
        x = self._validate_data(x, accept_sparse="csr", reset=False)
        scores = safe_sparse_dot(x, self.coef_.T, dense_output=True) + self.intercept_
        return xp.reshape(scores, (-1,)) if scores.shape[1] == 1 else scores

    def corr(self, data, save=None, annot=True):
        """数据集的特征相关性图

        :param data: 需要画特征相关性图的数据集
        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param annot: 是否在图中显示相关性的数值，默认 True
        """
        corr_plot(data.drop(columns=[self.target]), save=save, annot=annot)

    def report(self, data):
        """逻辑回归模型报告

        :param data: 需要评估的数据集
        :return: pd.DataFrame，模型报告，包含准确率、F1等指标，参考：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        """
        report_dict = classification_report(data[self.target], self.predict(data.drop(columns=self.target)), output_dict=True, target_names=["好客户", "坏客户"])
        accuracy = report_dict.pop("accuracy")
        _report = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "desc"})
        _report.loc[len(_report)] = ['accuracy', '', '', accuracy, len(data)]
        return _report

    def summary(self):
        """
        :return: pd.DataFrame，逻辑回归模型统计信息

        - `Coef.`: 逻辑回归入模特征系数
        - `Std.Err`: 标准误差
        - `z`: Z检验统计量
        - `P>|z|`: P值
        - `[ 0.025`: 置信区间下界
        - `0.975 ]`: 置信区间上界
        - `VIF`: 膨胀方差因子

        **参考样例**

        >>> summary = logistic.summary()
        >>> summary
                                                            Coef.  Std.Err       z  P>|z|  [ 0.025  0.975 ]    VIF
        const                                               -0.8511   0.0991 -8.5920 0.0000  -1.0452  -0.6569 1.0600
        credit_history                                       0.8594   0.1912  4.4954 0.0000   0.4847   1.2341 1.0794
        age_in_years                                         0.6176   0.2936  2.1032 0.0354   0.0421   1.1932 1.0955
        savings_account_and_bonds                            0.8842   0.2408  3.6717 0.0002   0.4122   1.3563 1.0331
        credit_amount                                        0.7027   0.2530  2.7771 0.0055   0.2068   1.1987 1.1587
        status_of_existing_checking_account                  0.6891   0.1607  4.2870 0.0000   0.3740   1.0042 1.0842
        personal_status_and_sex                              0.8785   0.5051  1.7391 0.0820  -0.1116   1.8685 1.0113
        purpose                                              1.1370   0.2328  4.8844 0.0000   0.6807   1.5932 1.0282
        present_employment_since                             0.7746   0.3247  2.3855 0.0171   0.1382   1.4110 1.0891
        installment_rate_in_percentage_of_disposable_income  1.3785   0.3434  4.0144 0.0001   0.7055   2.0515 1.0300
        duration_in_month                                    0.9310   0.1986  4.6876 0.0000   0.5417   1.3202 1.1636
        other_installment_plans                              0.8521   0.3459  2.4637 0.0138   0.1742   1.5301 1.0117
        housing                                              0.8251   0.4346  1.8983 0.0577  -0.0268   1.6770 1.0205
        """
        check_is_fitted(self)

        if not hasattr(self, "std_err_coef_"):
            msg = "Summary statistics were not calculated on .fit(). Options to fix:\n"
            msg += "\t- Re-fit using .fit(X, y, calculate_stats=True)\n"
            msg += "\t- Re-inititialize using LogisticRegression(calculate_stats=True)"
            raise AssertionError(msg)

        data = {
            "Coef.": (self.intercept_.tolist() + self.coef_.tolist()[0]),
            "Std.Err": (self.std_err_intercept_.tolist() + self.std_err_coef_.tolist()),
            "z": (self.z_intercept_.tolist() + self.z_coef_.tolist()[0]),
            "P>|z|": (self.p_val_intercept_.tolist() + self.p_val_coef_.tolist()[0]),
        }

        stats = pd.DataFrame(data, index=self.names_)
        stats["[ 0.025"] = stats["Coef."] - 1.96 * stats["Std.Err"]
        stats["0.975 ]"] = stats["Coef."] + 1.96 * stats["Std.Err"]

        stats["VIF"] = self.vif

        return stats

    def summary2(self, feature_map={}):
        """summary 的基础上，支持传入数据字典，输出带有特征释义的统计信息表

        :param feature_map: 数据字典，默认 {}

        :return: pd.DataFrame，逻辑回归模型统计信息
        """
        stats = self.summary().reset_index().rename(columns={"index": "Features"})

        if feature_map is not None and len(feature_map) > 0:
            stats.insert(loc=1, column="Describe", value=[feature_map.get(c, "") for c in stats["Features"]])

        return stats

    @staticmethod
    def convert_sparse_matrix(x):
        """稀疏特征优化"""
        if scipy.sparse.issparse(x):
            return x.toarray()
        else:
            return x

    def plot_weights(self, save=None, figsize=(15, 8), fontsize=14, color=["#2639E9", "#F76E6C", "#FE7715"]):
        """逻辑回归模型系数误差图

        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param figsize: 图片大小，默认 (15, 8)
        :param fontsize: 字体大小，默认 14
        :param color: 图片主题颜色，默认即可

        :return: Figure
        """
        summary = self.summary()

        x = summary["Coef."]
        y = summary.index
        lower_error = summary["Coef."] - summary["[ 0.025"]
        upper_error = summary["0.975 ]"] - summary["Coef."]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.errorbar(x, y, xerr=[lower_error, upper_error], fmt="o", ecolor=color[0], elinewidth=2, capthick=2, capsize=4, ms=6, mfc=color[0], mec=color[0])
        # ax.tick_params(axis='x', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        # ax.tick_params(axis='y', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        ax.axvline(0, color=color[0], linestyle='--', ymax=len(y), alpha=0.5)
        ax.spines['top'].set_color(color[0])
        ax.spines['bottom'].set_color(color[0])
        ax.spines['right'].set_color(color[0])
        ax.spines['left'].set_color(color[0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title("Regression Meta Analysis - Weight Plot\n", fontsize=fontsize, fontweight="bold")
        ax.set_xlabel("Weight Estimates", fontsize=fontsize, weight="bold")
        ax.set_ylabel("Variable", fontsize=fontsize, weight="bold")

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save), exist_ok=True)

            plt.savefig(save, dpi=240, format="png", bbox_inches="tight")

        return fig


class ScoreCard(toad.ScoreCard, TransformerMixin):

    def __init__(self, target="target", pdo=60, rate=2, base_odds=35, base_score=750, combiner={}, transer=None, pretrain_lr=None, pipeline=None, **kwargs):
        """评分卡模型

        :param target: 数据集中标签名称，默认 target
        :param pdo: odds 每增加 rate 倍时减少 pdo 分，默认 60
        :param rate: 倍率
        :param base_odds: 基础 odds，通常根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比，默认 35，即 35:1 => 0.972 => 坏样本率 2.8%
        :param base_score: 基础 odds 对应的分数，默认 750
        :param combiner: 分箱转换器，传入 pipeline 时可以为None
        :param transer: woe转换器，传入 pipeline 时可以为None
        :param pretrain_lr: 预训练好的逻辑回归模型，可以不传
        :param pipeline: 训练好的 pipeline，必须包含 Combiner 和 WOETransformer
        :param kwargs: 其他相关参数，具体参考 toad.ScoreCard
        """
        if pipeline:
            combiner = self.class_steps(pipeline, Combiner)[0]
            transer = self.class_steps(pipeline, WOETransformer)[0]

            if self.class_steps(pipeline, (ITLubberLogisticRegression, LogisticRegression)):
                pretrain_lr = self.class_steps(pipeline, (ITLubberLogisticRegression, LogisticRegression))[0]

        super().__init__(
            combiner=combiner.combiner if isinstance(combiner, Combiner) else combiner, transer=transer.transformer if isinstance(transer, WOETransformer) else transer,
            pdo=pdo, rate=rate, base_odds=base_odds, base_score=base_score, **kwargs
        )

        self.target = target
        self.pipeline = pipeline
        self.pretrain_lr = pretrain_lr

    def fit(self, x):
        """评分卡模型训练方法

        :param x: 转换为 WOE 后的训练数据，需包含目标变量

        :return: ScoreCard，训练好的评分卡模型
        """
        y = x[self.target]

        if self.pretrain_lr:
            x = x[self.pretrain_lr.feature_names_in_]
        else:
            x = x.drop(columns=[self.target])

        self._feature_names = x.columns.tolist()

        for f in self.features_:
            if f not in self.transer:
                raise Exception('column \'{f}\' is not in transer'.format(f=f))

        if self.pretrain_lr:
            self.model = self.pretrain_lr
        else:
            self.model.fit(x, y)

        self.rules = self._generate_rules()

        sub_score = self.woe_to_score(x)
        self.base_effect = pd.Series(np.median(sub_score, axis=0), index=self.features_)

        return self

    def transform(self, x):
        """评分转换方法

        :param x: 需要预测模型评分的原始数据，非 woe 转换后的数据

        :return: 预测的评分分数
        """
        return self.predict(x)

    def _check_rules(self, combiner, transer):
        """评分卡特征分箱校验方法

        :param combiner: 特征分箱器
        :param transer: WOE转换器
        :return: bool，是否通过检验
        """
        for col in self.features_:
            if col not in combiner:
                raise Exception('column \'{col}\' is not in combiner'.format(col=col))

            if col not in transer:
                raise Exception('column \'{col}\' is not in transer'.format(col=col))

            l_c = len(combiner[col])
            l_t = len(transer[col]['woe'])

            if l_c == 0:
                continue

            if np.issubdtype(combiner[col].dtype, np.number):
                if l_c != l_t - 1:
                    if np.isnan(combiner[col]).sum() > 0:
                        combiner.update({col: combiner[col][:-1]})
                    else:
                        raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col=col, l_t=l_t, l_c=l_c + 1))
            else:
                if l_c != l_t:
                    if sum([sum([1 for b in r if b in ("nan", "None")]) for r in combiner[col]]) > 0:
                        combiner.update({col: [[np.nan if b == "nan" else (None if b == "None" else b)  for b in r] for r in combiner[col]]})
                        self._check_rules(combiner, transer)
                    else:
                        raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col=col, l_t=l_t, l_c=l_c))

        return True

    @staticmethod
    def score_clip(score, clip=50):
        """传入评分分数，根据评分分布情况，返回评分等距分箱规则

        :param score: 评分数据
        :param clip: 区间间隔

        :return: list，评分分箱规则
        """
        clip_start = max(math.ceil(score.min() / clip) * clip, math.ceil(score.quantile(0.01) / clip) * clip)
        clip_end = min(math.ceil(score.max() / clip) * clip, math.ceil(score.quantile(0.99) / clip) * clip)
        return [i for i in range(clip_start, clip_end, clip)]

    def scorecard_scale(self):
        """输出评分卡基准信息，包含 base_odds、base_score、rate、pdo、A、B

        :return: pd.DataFrame，评分卡基准信息
        """
        scorecard_kedu = pd.DataFrame(
            [
                ["base_odds", self.base_odds, "根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比"],
                ["base_score", self.base_score, "基础ODDS对应的分数"],
                ["rate", self.rate, "设置分数的倍率"],
                ["pdo", self.pdo, "表示分数增长PDO时，ODDS值增长到RATE倍"],
                ["B", self.factor, "补偿值，计算方式：pdo / ln(rate)"],
                ["A", self.offset, "刻度，计算方式：base_score - B * ln(base_odds)"],
            ],
            columns=["刻度项", "刻度值", "备注"],
        )
        return scorecard_kedu

    @classmethod
    def format_bins(self, bins, index=False, ellipsis=None, decimal=4):
        """分箱转换为标签

        :param bins: 分箱
        :param index: 是否需要索引
        :param ellipsis: 字符显示最大长度

        :return: ndarray: 分箱标签
        """
        if len(bins) == 0:
            return ["全部样本"]

        if isinstance(bins, list): bins = np.array(bins)
        EMPTYBINS = len(bins) if not isinstance(bins[0], (set, list, np.ndarray)) else -1

        l = []
        if not isinstance(bins[0], (set, list, np.ndarray)):
            has_empty = len(bins) > 0 and pd.isnull(bins[-1])
            if has_empty: bins = bins[:-1]
            sp_l = ["负无穷"] + [round_float(b, decimal=decimal) for b in bins] + ["正无穷"]
            for i in range(len(sp_l) - 1): l.append('[' + str(sp_l[i]) + ' , ' + str(sp_l[i + 1]) + ')')
            if has_empty: l.append('缺失值')
        else:
            for keys in bins:
                keys_update = set()
                for key in keys:
                    if pd.isnull(key) or key == "nan":
                        keys_update.add("缺失值")
                    elif key.strip() == "":
                        keys_update.add("空字符串")
                    else:
                        keys_update.add(key)
                label = ','.join(keys_update)

                if ellipsis is not None:
                    label = label[:ellipsis] + '..' if len(label) > ellipsis else label

                l.append(label)

        if index:
            l = ["{:02}.{}".format(i if b != '缺失值' else EMPTYBINS, b) for i, b in enumerate(l)]

        return np.array(l)

    def scorecard_points(self, feature_map={}):
        """输出评分卡分箱信息及其对应的分数

        :param feature_map: 数据字典，默认 {}，传入入模特征的数据字典，输出信息中将增加一列 变量含义
        :return: pd.DataFrame，评分卡分箱信息
        """
        card_points = self.export(to_frame=True).rename(columns={"name": "变量名称", "value": "变量分箱", "score": "对应分数"})

        if feature_map is not None and len(feature_map) > 0:
            card_points.insert(loc=1, column="变量含义", value=[feature_map.get(c, "") for c in card_points["变量名称"]])

        return card_points

    def scorecard2pmml(self, pmml: str = 'scorecard.pmml', debug: bool = False):
        """转换评分卡模型为本地 PMML 文件，使用本功能需要提前在环境中安装 jdk 1.8+ 以及 sklearn2pmml 库

        :param pmml: 保存 PMML 模型文件的路径
        :param debug: bool，是否开启调试模式，默认 False，当设置为 True 时，会返回评分卡 pipeline，同时显示转换细节

        :return: sklearn.pipeline.Pipeline，当设置 debug 为 True 时，返回评分卡 pipeline
        """
        from sklearn_pandas import DataFrameMapper
        from sklearn.linear_model import LinearRegression
        from sklearn2pmml import sklearn2pmml, PMMLPipeline
        from sklearn2pmml.preprocessing import LookupTransformer, ExpressionTransformer

        mapper = []
        samples = {}
        for var, rule in self.rules.items():
            end_string = ''
            expression_string = ''
            total_bins = len(rule['scores'])
            if isinstance(rule['bins'][0], (np.ndarray, list)):
                default_value = 0.
                mapping = {}
                for bins, score in zip(rule['bins'], rule['scores'].tolist()):
                    for _bin in bins:
                        if pd.isnull(_bin) or _bin == 'nan':
                            default_value = float(score)
                        else:
                            mapping[_bin] = float(score)

                mapper.append((
                    [var],
                    LookupTransformer(mapping=mapping, default_value=default_value),
                ))
                samples[var] = [list(mapping.keys())[i] for i in np.random.randint(0, len(mapping), 20)]
            else:
                has_empty = len(rule['bins']) > 0 and pd.isnull(rule['bins'][-1])

                if has_empty:
                    score_empty = rule['scores'][-1]
                    total_bins -= 1
                    bin_scores = rule['scores'][:-1]
                    bin_vars = rule['bins'][:-1]
                    expression_string += f'{score_empty} if pandas.isnull(X[0]) '
                else:
                    bin_scores = rule['scores']
                    bin_vars = rule['bins']

                for i in range(len(bin_scores)):
                    if i == 0:
                        _expression_string = f'{bin_scores[i]}'
                    elif i == total_bins - 1:
                        _expression_string += f' if X[0] < {bin_vars[i - 1]} else {bin_scores[i]}'
                    else:
                        _expression_string += f' if X[0] < {bin_vars[i - 1]} else ({bin_scores[i]} '
                        end_string += ')'

                _expression_string += end_string

                if has_empty:
                    expression_string += f'else ({_expression_string})' if _expression_string.count('else') > 0 else _expression_string
                else:
                    expression_string += _expression_string

                mapper.append((
                    [var],
                    ExpressionTransformer(expression_string),
                ))
                samples[var] = np.random.random(20) * 100

        scorecard_mapper = DataFrameMapper(mapper, df_out=True)

        pipeline = PMMLPipeline([
            ('preprocessing', scorecard_mapper),
            ('scorecard', LinearRegression(fit_intercept=False)),
        ])

        pipeline.fit(pd.DataFrame(samples), pd.Series(np.random.randint(0, 2, 20), name='score'))

        pipeline.named_steps['scorecard'].coef_ = np.ones(len(scorecard_mapper.features))

        try:
            sklearn2pmml(pipeline, pmml, with_repr=True, debug=debug)
        except:
            import traceback
            print(traceback.format_exc())
            return pipeline

        if debug:
            return pipeline

    @staticmethod
    def KS_bucket(y_pred, y_true, bucket=10, method="quantile"):
        """用于评估评分卡排序性的方法

        :param y_pred: 模型预测结果，传入评分卡预测的评分或LR预测的概率
        :param y_true: 样本好坏标签
        :param bucket: 分箱数量，默认 10
        :param method: 分箱方法，支持 chi、dt、quantile、step、kmeans，默认 quantile

        :return: 评分卡分箱后的统计信息，推荐直接使用 feature_bin_stats 方法
        """
        return toad.metrics.KS_bucket(y_pred, y_true, bucket=bucket, method=method)

    @staticmethod
    def KS(y_pred, y_true):
        """计算 KS 指标

        :param y_pred: 模型预测结果，传入评分卡预测的评分或LR预测的概率
        :param y_true: 样本好坏标签

        :return: float，KS 指标
        """
        return toad.metrics.KS(y_pred, y_true)

    @staticmethod
    def AUC(y_pred, y_true):
        """计算 AUC 指标

        :param y_pred: 模型预测结果，传入评分卡预测的评分或LR预测的概率
        :param y_true: 样本好坏标签

        :return: float，AUC 指标
        """
        return toad.metrics.AUC(y_pred, y_true)

    @staticmethod
    def perf_eva(y_pred, y_true, title="", plot_type=["ks", "roc"], save=None, figsize=(14, 6)):
        """评分卡效果评估方法

        :param y_pred: 模型预测结果，传入评分卡预测的评分或LR预测的概率
        :param y_true: 样本好坏标签
        :param title: 图像标题
        :param plot_type: 画图的类型，可选 ks、auc、lift、pr
        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param figsize: 图像尺寸大小，传入一个tuple，默认 （14， 6）

        :return: dict，包含 ks、auc、gini、figure
        """
        # plt.figure(figsize=figsize)
        rt = sc.perf_eva(y_true, y_pred, title=title, plot_type=plot_type, show_plot=True)

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save))

            rt["pic"].savefig(save, dpi=240, format="png", bbox_inches="tight")

        return rt

    @staticmethod
    def ks_plot(score, y_true, title="", fontsize=14, figsize=(16, 8), save=None, **kwargs):
        """数值特征 KS曲线 & ROC曲线
        
        :param score: 数值特征，通常为评分卡分数
        :param y_true: 标签值
        :param title: 图像标题
        :param fontsize: 字体大小，默认 14
        :param figsize: 图像大小，默认 (16, 8)
        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param kwargs: 其他参数，参考：scorecardpipeline.utils.hist_plot
        """
        ks_plot(score, y_true, title=title, fontsize=fontsize, figsize=figsize, save=save, **kwargs)

    @staticmethod
    def PSI(y_pred_train, y_pred_oot):
        """计算两个数据集评分或预测结果的 PSI

        :param y_pred_train: 基准数据集的数值特征，通常为评分卡分数
        :param y_pred_oot: 对照数据集的数值特征
        :return: float，PSI 指标值
        """
        return toad.metrics.PSI(y_pred_train, y_pred_oot)

    @staticmethod
    def perf_psi(y_pred_train, y_pred_oot, y_true_train, y_true_oot, keys=["train", "test"], x_limits=None, x_tick_break=50, show_plot=True, return_distr_dat=False):
        """scorecardpy 的 perf_psi 方法，基于两个数据集的画 PSI 图

        :param y_pred_train: 基准数据集的数值特征，通常为评分卡分数
        :param y_pred_oot: 对照数据集的数值特征
        :param y_true_train: 基准数据集的真实标签
        :param y_true_oot: 基准数据集的真实标签
        :param keys: 基准数据集和对照数据集的名称
        :param x_limits: x 轴的区间，默认为 None
        :param x_tick_break: 评分区间步长
        :param show_plot: 是否显示图像，默认 True
        :param return_distr_dat: 是否返回分布数据

        :return: dict，PSI 指标 & 图片
        """
        return sc.perf_psi(
            score={keys[0]: y_pred_train, keys[1]: y_pred_oot},
            label={keys[0]: y_true_train, keys[1]: y_true_oot},
            x_limits=x_limits,
            x_tick_break=x_tick_break,
            show_plot=show_plot,
            return_distr_dat=return_distr_dat,
        )

    @staticmethod
    def score_hist(score, y_true, figsize=(15, 10), bins=20, save=None, **kwargs):
        """数值特征分布图

        :param score: 数值特征，通常为评分卡分数
        :param y_true: 标签值
        :param figsize: 图像大小，默认 (15, 10)
        :param bins: 分箱数量大小，默认 30
        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param kwargs: scorecardpipeline.utils.hist_plot 方法的其他参数
        """
        hist_plot(score, y_true, figsize=figsize, bins=bins, save=save, **kwargs)

    def _format_rule(self, rule, decimal=4, **kwargs):
        """分箱区间精度调整

        :param rule: 分箱信息
        :param decimal: 精度
        :return: dict，评分卡分箱及分数
        """
        bins = self.format_bins(rule['bins'])
        scores = np.around(rule['scores'], decimals=decimal).tolist()

        return dict(zip(bins, scores))

    @staticmethod
    def class_steps(pipeline, query):
        """根据 query 查询 pipeline 中对应的 step

        :param pipeline: sklearn.pipeline.Pipeline，训练后的数据预处理 pipeline
        :param query: 需要查询的类，可以从 pipeline 中查找 WOETransformer 和 Combiner

        :return: list，对应的组件
        """
        return [v for k, v in pipeline.named_steps.items() if isinstance(v, query)]

    def feature_bin_stats(self, data, feature, rules={}, method='step', max_n_bins=10, desc="评分卡分数", ks=False, **kwargs):
        """评估评分卡排序性的方法，可以输出各分数区间的各项指标

        :param data: 需要查看的数据集
        :param feature: 数值性特征名称，通常为预测的概率或评分卡分数
        :param rules: 自定义的区间划分规则
        :param method: 分箱方法
        :param max_n_bins: 最大分箱数
        :param desc: 特征描述
        :param ks: 是否统计 KS 指标并输出相关统计信息
        :param kwargs: Combiner.feature_bin_stats 方法的其他参数

        :return: pd.DataFrame，评分各区间的统计信息
        """
        return Combiner.feature_bin_stats(data, feature, target=self.target, method=method, max_n_bins=max_n_bins, desc=desc, ks=ks, **kwargs)
