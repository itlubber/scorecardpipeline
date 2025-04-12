# -*- coding: utf-8 -*-
"""
@Time    : 2023/05/21 16:23
@Author  : itlubber
@Site    : itlubber.art
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from sklearn.base import BaseEstimator, TransformerMixin

import toad
import scorecardpy as sc
from joblib import Parallel, delayed
from optbinning import OptimalBinning
from toad.plot import proportion_plot, badrate_plot

from .utils import *


def drop_identical(frame, threshold=0.95, return_drop=False, exclude=None, target=None):
    """
    剔除数据集中单一值占比过高的特征

    :param frame: 需要进行特征单一值占比过高筛选的数据集
    :param threshold: 单一值占比阈值，超过阈值剔除特征
    :param return_drop: 是否返回特征剔除信息，默认 False
    :param exclude: 是否排除某些特征，不进行单一值占比筛选，默认为 None
    :param target: 数据集中的目标变量列名，默认为 None，即数据集中不包含 target

    :return:
        + 筛选后的数据集: pd.DataFrame，剔除单一值占比过高特征的数据集
        + 剔除的特征: list / np.ndarray，当 return_drop 设置为 True 时，返回被剔除的单一值占比过高的特征列表
    """
    cols = frame.columns.copy()

    if target:
        cols = cols.drop(target)

    if exclude:
        cols = cols.drop(exclude)

    if threshold <= 1:
        threshold = len(frame) * threshold

    drop_list = []
    for col in cols:
        n = frame[col].value_counts().max()

        if n > threshold:
            drop_list.append(col)

    if return_drop:
        return frame.drop(columns=drop_list), np.array(drop_list)

    return frame.drop(columns=drop_list)


def drop_corr(frame, target=None, threshold=0.7, by='IV', return_drop=False, exclude=None):
    """
    剔除数据集中特征相关性过高的特征

    :param frame: 需要进行特征相关性过高筛选的数据集
    :param target: 数据集中的目标变量列名，默认为 None
    :param threshold: 相关性阈值，超过阈值剔除特征
    :param by: 剔除指标的依据，两个特征相关性超过阈值时，保留指标更大的特征，默认根据 IV 进行判断
    :param return_drop: 是否返回特征剔除信息，默认 False
    :param exclude: 是否排除某些特征，不进行单一值占比筛选，默认为 None

    :return:
        + 筛选后的数据集: pd.DataFrame，剔除特征相关性过高的数据集
        + 剔除的特征: list，当 return_drop 设置为 True 时，返回被剔除的相关性过高的特征列表
    """
    if not isinstance(by, (str, pd.Series)):
        by = pd.Series(by, index=frame.columns)

    cols = frame.columns.copy()

    if exclude is not None:
        exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        cols = cols.drop(exclude)

    f, t = toad.utils.split_target(frame[cols], target)
    corr = f.select_dtypes("number").corr().abs()

    drops = []
    ix, cn = np.where(np.triu(corr.values, 1) > threshold)

    if len(ix):
        graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])
        uni, counts = np.unique(graph, return_counts=True)
        weights = np.zeros(len(corr.index))

        if isinstance(by, pd.Series):
            weights = by[corr.index].values
        elif by.upper() == 'IV':
            for ix in uni:
                weights[ix] = toad.IV(frame[corr.index[ix]], target=t)

        while True:
            nodes = uni[np.argwhere(counts == np.amax(counts))].flatten()
            n = nodes[np.argsort(weights[nodes])[0]]
            i, c = np.where(graph == n)
            pairs = graph[(i, 1 - c)]

            if weights[pairs].sum() > weights[n]:
                dro = [n]
            else:
                dro = pairs.tolist()
            drops += dro

            di, _ = np.where(np.isin(graph, dro))
            graph = np.delete(graph, di, axis=0)

            if len(graph) <= 0:
                break

            uni, counts = np.unique(graph, return_counts=True)

    drop_list = corr.index[drops].values
    
    if return_drop:
        return frame.drop(columns=drop_list), drop_list

    return frame.drop(columns=drop_list)


def select(frame, target='target', empty=0.95, iv=0.02, corr=0.7, identical=0.95, return_drop=False, exclude=None):
    """
    根据缺失率、IV指标、相关性、单一值占比等进行特征筛选，返回特征剔除后的数据集和剔除的特征信息

    :param frame: 需要进行特征筛选的数据集
    :param target: 数据集中的目标变量列名，默认为 target
    :param empty: 缺失率阈值，超过阈值剔除特征
    :param iv: IV 阈值，低于阈值剔除特征
    :param corr: 相关性阈值，超过阈值剔除 IV 较小的特征
    :param identical: 单一值占比阈值，超过阈值剔除特征
    :param return_drop: 是否返回剔除特征信息，默认 False
    :param exclude: 是否排除某些特征，不进行特征筛选，默认为 None

    :return:
        + 筛选后的数据集: pd.DataFrame，特征筛选后的数据集
        + 剔除的特征信息: dict，当 return_drop 设置为 True 时，返回被剔除特征信息
    """
    empty_drop = iv_drop = corr_drop = identical_drop = iv_list = None

    if empty:
        frame, empty_drop = toad.selection.drop_empty(frame, threshold=empty, return_drop=True, exclude=exclude)
    
    if iv:
        frame, iv_drop, iv_list = toad.selection.drop_iv(frame, target=target, threshold=iv, return_drop=True, return_iv=True, exclude=exclude)

    if corr:
        weights = iv_list if iv else 'IV'
        frame, corr_drop = drop_corr(frame, target=target, threshold=corr, by=weights, return_drop=True, exclude=exclude)

    if identical:
        frame, identical_drop = drop_identical(frame, threshold=identical, return_drop=True, exclude=exclude, target=target)

    if return_drop:
        drop_info = {
            'empty': empty_drop,
            'iv': iv_drop,
            'corr': corr_drop,
            'identical': identical_drop,
        }
        return frame, drop_info

    return frame


class FeatureSelection(TransformerMixin, BaseEstimator):

    def __init__(self, target="target", empty=0.95, iv=0.02, corr=0.7, exclude=None, return_drop=True, identical=0.95, remove=None, engine="scorecardpy", target_rm=False):
        """特征筛选方法

        :param target: 数据集中标签名称，默认 target
        :param empty: 空值率，默认 0.95, 即空值占比超过 95% 的特征会被剔除
        :param iv: IV值，默认 0.02，即iv值小于 0.02 时特征会被剔除
        :param corr: 相关性，默认 0.7，即特征之间相关性大于 0.7 时会剔除iv较小的特征
        :param identical: 单一值占比，默认 0.95，即当特征的某个值占比超过 95% 时，特征会被剔除
        :param exclude: 是否需要强制保留某些特征
        :param return_drop: 是否返回删除信息，默认 True，即默认返回删除特征信息
        :param remove: 引擎使用 scorecardpy 时，可以传入需要强制删除的变量
        :param engine: 特征筛选使用的引擎，可选 "toad", "scorecardpy" 两种，默认 scorecardpy
        :param target_rm: 是否剔除标签，默认 False，即不剔除
        """
        self.engine = engine
        self.target = target
        self.empty = empty
        self.identical = identical
        self.iv = iv
        self.corr = corr
        self.exclude = exclude
        self.remove = remove
        self.return_drop = return_drop
        self.target_rm = target_rm
        self.select_columns = None
        self.dropped = None

    def fit(self, x, y=None):
        """训练特征筛选方法

        :param x: 数据集，需要包含目标变量

        :return: 训练后的 FeatureSelection
        """
        if self.engine == "toad":
            selected = select(x, target=self.target, empty=self.empty, identical=self.identical, iv=self.iv, corr=self.corr, exclude=self.exclude, return_drop=self.return_drop)
        else:
            selected = sc.var_filter(x, y=self.target, iv_limit=self.iv, missing_limit=self.empty, identical_limit=self.identical, var_rm=self.remove, var_kp=self.exclude, return_rm_reason=self.return_drop)

        if self.return_drop and isinstance(selected, dict):
            self.dropped = selected["rm"]
            self.select_columns = list(selected["dt"].columns)
        elif self.return_drop and isinstance(selected, (tuple, list)):
            self.dropped = pd.DataFrame([(feature, reason) for reason, features in selected[1].items() for feature in features], columns=["variable", "rm_reason"])
            self.select_columns = list(selected[0].columns)
        else:
            self.select_columns = list(selected.columns)

        if self.target_rm and self.target in self.select_columns:
            self.select_columns.remove(self.target)

        return self

    def transform(self, x, y=None):
        """特征筛选转换器

        :param x: 需要进行特征筛选的数据集

        :return: pd.DataFrame，特征筛选后的数据集
        """
        return x[[col for col in self.select_columns if col in x.columns]]


class StepwiseSelection(TransformerMixin, BaseEstimator):

    def __init__(self, target="target", estimator="ols", direction="both", criterion="aic", max_iter=None, return_drop=True, exclude=None, intercept=True, p_value_enter=0.2, p_remove=0.01, p_enter=0.01, target_rm=False):
        """逐步回归特征筛选方法

        :param target: 数据集中标签名称，默认 target
        :param estimator: 预估器，默认 ols，可选 "ols", "lr", "lasso", "ridge"，通常默认即可
        :param direction: 逐步回归方向，默认both，可选 "forward", "backward", "both"，通常默认即可
        :param criterion: 评价指标，默认 aic，可选 "aic", "bic", "ks", "auc"，通常默认即可
        :param max_iter: 最大迭代次数，sklearn中使用的参数，默认为 None
        :param return_drop: 是否返回特征剔除信息，默认 True
        :param exclude: 强制保留的某些特征
        :param intercept: 是否包含截距，默认为 True
        :param p_value_enter: 特征进入的 p 值，用于前向筛选时决定特征是否进入模型
        :param p_remove: 特征剔除的 p 值，用于后向剔除时决定特征是否要剔除
        :param p_enter: 特征 p 值，用于判断双向逐步回归是否剔除或者准入特征
        :param target_rm: 是否剔除数据集中的标签，默认为 False，即剔除数据集中的标签
        """
        self.target = target
        self.intercept = intercept
        self.p_value_enter = p_value_enter
        self.p_remove = p_remove
        self.p_enter = p_enter
        self.estimator = estimator
        self.direction = direction
        self.criterion = criterion
        self.max_iter = max_iter
        self.return_drop = return_drop
        self.target_rm = target_rm
        self.exclude = exclude
        self.select_columns = None
        self.dropped = None

    def fit(self, x, y=None):
        """训练逐步回归特征筛选方法

        :param x: 数据集，需要包含目标变量

        :return: 训练后的 StepwiseSelection
        """
        selected = toad.selection.stepwise(x, target=self.target, estimator=self.estimator, direction=self.direction, criterion=self.criterion, exclude=self.exclude, intercept=self.intercept, p_value_enter=self.p_value_enter,
                                           p_remove=self.p_remove, p_enter=self.p_enter, return_drop=self.return_drop)
        if self.return_drop:
            self.dropped = pd.DataFrame([(col, "stepwise") for col in selected[1]], columns=["variable", "rm_reason"])
            selected = selected[0]

        self.select_columns = list(selected.columns)

        if self.target_rm and self.target in self.select_columns:
            self.select_columns.remove(self.target)

        return self

    def transform(self, x, y=None):
        """逐步回归特征筛选转换器

        :param x: 需要进行特征筛选的数据集

        :return: pd.DataFrame，特征筛选后的数据集
        """
        return x[[col for col in self.select_columns if col in x.columns]]


class FeatureImportanceSelector(BaseEstimator, TransformerMixin):

    def __init__(self, top_k=126, target="target", selector="catboost", params=None, max_iv=None):
        """基于特征重要性的特征筛选方法

        :param top_k: 依据特征重要性进行排序，筛选最重要的 top_k 个特征
        :param target: 数据集中标签名称，默认 target
        :param selector: 特征选择器，目前只支持 catboost ，可以支持数据集中包含字符串的数据
        :param params: selector 的参数，不传使用默认参数
        :param max_iv: 是否需要删除 IV 过高的特征，建议设置为 1.0
        """
        self.target = target
        self.top_k = top_k
        self.max_iv = max_iv
        self.selector = selector
        self.params = params
        self.feature_names_ = None
        self.high_iv_feature_names_ = None
        self.low_importance_feature_names_ = None
        self.select_columns = None
        self.dropped = None

    def fit(self, x, y=None):
        """特征重要性筛选器训练

        :param x: 数据集，需要包含目标变量

        :return: 训练后的 FeatureImportanceSelector
        """
        x = x.copy()

        if self.max_iv is not None:
            self.high_iv_feature_names_ = list(toad.quality(x, target=self.target, cpu_cores=-1, iv_only=True).query(f"iv > {self.max_iv}").index)
            x = x[[c for c in x.columns if c not in self.high_iv_feature_names_]]

        X = x.drop(columns=self.target)
        Y = x[self.target]

        self.feature_names_ = list(X.columns)
        cat_features_index = [i for i in range(len(self.feature_names_)) if self.feature_names_[i] not in X.select_dtypes("number").columns]

        if self.selector == "catboost":
            self.catboost_selector(x=X, y=Y, cat_features=cat_features_index)
        else:
            pass

        return self

    def transform(self, x, y=None):
        """特征重要性筛选器转换方法

        :param x: 需要进行特征筛选的数据集

        :return: pd.DataFrame，特征筛选后的数据集
        """
        return x[self.select_columns + [self.target]]

    def catboost_selector(self, x, y, cat_features=None):
        """基于 `CatBoost` 的特征重要性筛选器

        :param x: 需要进行特征重要性筛选的数据集，不包含目标变量
        :param y: 数据集中对应的目标变量值
        :param cat_features: 类别型特征的索引
        """
        from catboost import Pool, cv, metrics, CatBoostClassifier

        cat_data = Pool(data=x, label=y, cat_features=cat_features)

        if self.params is None:
            self.params = {
                "iterations": 256,
                "objective": "CrossEntropy",
                "eval_metric": "AUC",
                "learning_rate": 1e-2,
                "colsample_bylevel": 0.1,
                "depth": 4,
                "boosting_type": "Ordered",
                "bootstrap_type": "Bernoulli",
                "subsample": 0.8,
                "random_seed": 1024,
                "early_stopping_rounds": 10,
                "verbose": 0,
            }

        cat_model = CatBoostClassifier(**self.params)
        cat_model.fit(cat_data, eval_set=[cat_data])

        self.select_columns = [name for score, name in sorted(zip(cat_model.feature_importances_, cat_model.feature_names_), reverse=True)][:self.top_k]
        self.low_importance_feature_names_ = [c for c in x.columns if c not in self.select_columns]


class Combiner(TransformerMixin, BaseEstimator):

    def __init__(self, target="target", method='chi', empty_separate=True, min_n_bins=2, max_n_bins=None, max_n_prebins=20, min_prebin_size=0.02, min_bin_size=0.05, max_bin_size=None, gamma=0.01, monotonic_trend="auto_asc_desc", adj_rules={}, n_jobs=1, **kwargs):
        """特征分箱封装方法

        :param target: 数据集中标签名称，默认 target
        :param method: 特征分箱方法，可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform", 参考 toad.Combiner: https://github.com/amphibian-dev/toad/blob/master/toad/transform.py#L178-L355 & optbinning.OptimalBinning: https://gnpalencia.org/optbinning/
        :param empty_separate: 是否空值单独一箱, 默认 True
        :param min_n_bins: 最小分箱数，默认 2，即最小拆分2箱
        :param max_n_bins: 最大分箱数，默认 None，即不限制拆分箱数，推荐设置 3 ～ 5，不宜过多，偶尔使用 optbinning 时不起效
        :param max_n_prebins: 使用 optbinning 时预分箱数量
        :param min_prebin_size: 使用 optbinning 时预分箱叶子结点（或者每箱）样本占比，默认 2%
        :param min_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最小样本占比，默认 5%
        :param max_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最大样本占比，默认 None
        :param gamma: 使用 optbinning 分箱时限制过拟合的正则化参数，值越大惩罚越多，默认 0.01
        :param monotonic_trend: 使用 optbinning 正式分箱时的坏率策略，默认 auto，可选 "auto", "auto_heuristic", "auto_asc_desc", "ascending", "descending", "convex", "concave", "peak", "valley", "peak_heuristic", "valley_heuristic"
        :param adj_rules: 自定义分箱规则，toad.Combiner 能够接收的形式
        :param n_jobs: 使用多进程加速的worker数量，默认单进程
        """
        self.combiner = toad.transform.Combiner()
        self.method = method
        self.empty_separate = empty_separate
        self.target = target
        self.max_n_bins = max_n_bins
        self.min_n_bins = min_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.gamma = gamma
        self.monotonic_trend = monotonic_trend
        self.adj_rules = adj_rules
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def update(self, rules):
        """更新 Combiner 中特征的分箱规则

        :param rules: dict，需要更新规则，格式如下：{特征名称: 分箱规则}
        """
        self.combiner.update(rules)

        # 检查规则内容
        for feature in rules.keys():
            self.check_rules(feature=feature)

    @staticmethod
    def optbinning_bins(feature, data=None, target="target", min_n_bins=2, max_n_bins=3, max_n_prebins=10, min_prebin_size=0.02, min_bin_size=0.05, max_bin_size=None, gamma=0.01, monotonic_trend="auto_asc_desc", **kwargs):
        """基于 optbinning.OptimalBinning 的特征分箱方法，使用 optbinning.OptimalBinning 分箱失败时，使用 toad.transform.Combiner 的卡方分箱处理

        :param feature: 需要进行分箱的特征名称
        :param data: 训练数据集
        :param target: 数据集中标签名称，默认 target
        :param min_n_bins: 最小分箱数，默认 2，即最小拆分2箱
        :param max_n_bins: 最大分箱数，默认 None，即不限制拆分箱数，推荐设置 3 ～ 5，不宜过多，偶尔不起效
        :param max_n_prebins: 使用 optbinning 时预分箱数量
        :param min_prebin_size: 使用 optbinning 时预分箱叶子结点（或者每箱）样本占比，默认 2%
        :param min_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最小样本占比，默认 5%
        :param max_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最大样本占比，默认 None
        :param gamma: 使用 optbinning 分箱时限制过拟合的正则化参数，值越大惩罚越多，默认 0.01
        :param monotonic_trend: 使用 optbinning 正式分箱时的坏率策略，默认 auto，可选 "auto", "auto_heuristic", "auto_asc_desc", "ascending", "descending", "convex", "concave", "peak", "valley", "peak_heuristic", "valley_heuristic"
        """
        data = data[[feature, target]].copy()
        if data[feature].dropna().nunique() <= min_n_bins:
            splits = []
            for v in data[feature].dropna().unique():
                splits.append(v)

            if str(data[feature].dtypes) in ["object", "string", "category"]:
                rule = {feature: [[s] for s in splits]}
                rule[feature].append([[np.nan]])
            else:
                rule = {feature: sorted(splits) + [np.nan]}
        else:
            try:
                y = data[target]
                if str(data[feature].dtypes) in ["object", "string", "category"]:
                    dtype = "categorical"
                    x = data[feature].astype("category").values
                else:
                    dtype = "numerical"
                    x = data[feature].values

                _combiner = OptimalBinning(feature, dtype=dtype, min_n_bins=min_n_bins, max_n_bins=max_n_bins, max_n_prebins=max_n_prebins, min_prebin_size=min_prebin_size, min_bin_size=min_bin_size, max_bin_size=max_bin_size, monotonic_trend=monotonic_trend, gamma=gamma, **kwargs).fit(x, y)
                if _combiner.status == "OPTIMAL":
                    rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.splits] + [[None] if dtype == "categorical" else np.nan]}
                else:
                    raise Exception("optimalBinning error")

            except Exception as e:
                _combiner = toad.transform.Combiner()
                _combiner.fit(data[[feature, target]].dropna(), target, method="chi", min_samples=min_bin_size, n_bins=max_n_bins, empty_separate=False)
                rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.export()[feature]] + [[None] if dtype == "categorical" else np.nan]}

        return rule

    def fit(self, x: pd.DataFrame, y=None):
        """特征分箱训练

        :param x: 需要分箱的数据集，需要包含目标变量

        :return: Combiner，训练完成的分箱器
        """
        x = x.copy()

        # 处理数据集中分类变量包含 np.nan，toad 分箱后被转为 'nan' 字符串的问题
        cat_cols = list(x.drop(columns=self.target).select_dtypes(exclude="number").columns)
        # x[cat_cols] = x[cat_cols].replace(np.nan, None)

        if self.method in ["cart", "mdlp", "uniform"]:
            feature_optbinning_bins = partial(self.optbinning_bins, data=x, target=self.target, min_n_bins=self.min_n_bins, max_n_bins=self.max_n_bins, max_n_prebins=self.max_n_prebins, min_prebin_size=self.min_prebin_size, min_bin_size=self.min_bin_size, max_bin_size=self.max_bin_size, gamma=self.gamma, monotonic_trend=self.monotonic_trend, **self.kwargs)
            if self.n_jobs is not None:
                rules = Parallel(n_jobs=self.n_jobs)(delayed(feature_optbinning_bins)(feature) for feature in x.columns.drop(self.target))
                [self.combiner.update(r) for r in rules]
                # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                #     [executor.submit(feature_optbinning_bins(feature)) for feature in x.columns.drop(self.target)]
            else:
                for feature in x.drop(columns=[self.target]):
                    rule = feature_optbinning_bins(feature)
                    self.combiner.update(rule)
        else:
            if self.method in ["step", "quantile"]:
                self.combiner.fit(x, y=self.target, method=self.method, n_bins=self.max_n_bins, empty_separate=self.empty_separate, **self.kwargs)
            else:
                self.combiner.fit(x, y=self.target, method=self.method, min_samples=self.min_bin_size, n_bins=self.max_n_bins, empty_separate=self.empty_separate, **self.kwargs)

        if self.adj_rules is not None and len(self.adj_rules) > 0:
            self.update(self.adj_rules)

        # 检查类别变量空值是否被转为字符串，如果转为了字符串，强制转回空值，同时检查分箱顺序并调整为正确顺序
        self.check_rules()

        return self

    def check_rules(self, feature=None):
        """检查类别变量空值是否被转为字符串，如果转为了字符串，强制转回空值，同时检查分箱顺序并调整为正确顺序"""
        for col in self.combiner.rules.keys():
            if feature is not None and col != feature:
                continue

            _rule = self.combiner[col]

            if len(_rule) > 0 and not np.issubdtype(_rule.dtype, np.number) and isinstance(_rule[0], (list, tuple)):
                if sum([sum([1 for b in r if b in ("nan", "None")]) for r in _rule]) > 0:
                    _rule = [[np.nan if b == "nan" else (None if b == "None" else b) for b in r] for r in _rule]
                    if [np.nan] in _rule:
                        _rule.remove([np.nan])
                        _rule.append([np.nan])
                    if [None] in _rule:
                        _rule.remove([None])
                        _rule.append([None])

                    self.combiner.update({col: _rule})

    def transform(self, x, y=None, labels=False):
        """特征分箱转换方法

        :param x: 需要进行分箱转换的数据集
        :param labels: 进行分箱转换时是否转换为分箱信息，默认 False，即转换为分箱索引

        :return: pd.DataFrame，分箱转换后的数据集
        """
        return self.combiner.transform(x, labels=labels)

    def export(self, to_json=None):
        """特征分箱器导出 json 保存

        :param to_json: json 文件的路径

        :return: dict，特征分箱信息
        """
        return self.combiner.export(to_json=to_json)

    def load(self, from_json):
        """特征分箱器加载离线保存的 json 文件

        :param from_json: json 文件的路径

        :return: Combiner，特征分箱器
        """
        self.combiner.load(from_json)
        return self

    @classmethod
    def feature_bin_stats(cls, data, feature, target="target", rules=None, method='step', desc="", combiner=None, ks=True, max_n_bins=None, min_bin_size=None, max_bin_size=None, greater_is_better="auto", empty_separate=True, return_cols=None, return_rules=False, verbose=0, **kwargs):
        """特征分箱统计表，汇总统计特征每个分箱的各项指标信息

        :param data: 需要查看分箱统计表的数据集
        :param feature: 需要查看的分箱统计表的特征名称
        :param target: 数据集中标签名称，默认 target
        :param rules: 根据自定义的规则查看特征分箱统计表，支持 list（单个特征分箱规则） 或 dict（多个特征分箱规则） 格式传入
        :param combiner: 提前训练好的特征分箱器，优先级小于 rules
        :param method: 特征分箱方法，当传入 rules 或 combiner 时失效，可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform", 参考 toad.Combiner: https://github.com/amphibian-dev/toad/blob/master/toad/transform.py#L178-L355 & optbinning.OptimalBinning: https://gnpalencia.org/optbinning/
        :param desc: 特征描述信息，大部分时候用于传入特征对应的中文名称或者释义
        :param ks: 是否统计 KS 信息
        :param max_n_bins: 最大分箱数，默认 None，即不限制拆分箱数，推荐设置 3 ～ 5，不宜过多，偶尔使用 optbinning 时不起效
        :param min_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最小样本占比，默认 5%
        :param max_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最大样本占比，默认 None
        :param empty_separate: 是否空值单独一箱, 默认 False，推荐设置为 True
        :param return_cols: list，指定返回部分特征分箱统计表的列，默认 None
        :param return_rules: 是否返回特征分箱信息，默认 False
        :param greater_is_better: 是否越大越好，默认 ""auto", 根据最后两箱的 lift 指标自动推断是否越大越好, 可选 True、False、auto
        :param kwargs: scorecardpipeline.processing.Combiner 的其他参数

        :return:
            + 特征分箱统计表: pd.DataFrame
            + 特征分箱信息: list，当参数 return_rules 为 True 时返回
        """
        if combiner is None:
            if method not in ["chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform"]:
                raise 'method is the one of ["chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform"]'

            _combiner = cls(
                target=target
                , method=method
                , empty_separate=empty_separate
                , min_n_bins=2
                , max_n_bins=max_n_bins
                , min_bin_size=min_bin_size
                , max_bin_size=max_bin_size
                , **kwargs
            )
            _combiner.fit(data[[feature, target]])

        else:
            _combiner = deepcopy(combiner)

        if rules is not None and len(rules) > 0:
            if isinstance(rules, (list, np.ndarray)):
                _combiner.update({feature: rules})
            else:
                _combiner.update(rules)

        feature_bin_dict = feature_bins(_combiner[feature])

        df_bin = _combiner.transform(data[[feature, target]], labels=False)
        table = df_bin[[feature, target]].groupby([feature, target]).agg(len).unstack()
        table.columns.name = None
        table = table.rename(columns={0: '好样本数', 1: '坏样本数'}).fillna(0)
        if "好样本数" not in table.columns:
            table["好样本数"] = 0
        if "坏样本数" not in table.columns:
            table["坏样本数"] = 0

        table["指标名称"] = feature
        table["指标含义"] = desc
        table = table.reset_index().rename(columns={feature: "分箱"})

        table['样本总数'] = table['好样本数'] + table['坏样本数']
        table['样本占比'] = table['样本总数'] / table['样本总数'].sum()
        table['好样本占比'] = table['好样本数'] / table['好样本数'].sum()
        table['坏样本占比'] = table['坏样本数'] / table['坏样本数'].sum()
        table['坏样本率'] = table['坏样本数'] / table['样本总数']

        table = table.fillna(0.)

        table['分档WOE值'] = table.apply(lambda x: np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)), axis=1)
        table['分档IV值'] = table.apply(lambda x: (x['好样本占比'] - x['坏样本占比']) * np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)), axis=1)

        table = table.replace(np.inf, 0).replace(-np.inf, 0)

        table["LIFT值"] = table['坏样本率'] / (table["坏样本数"].sum() / table["样本总数"].sum())
        table["坏账改善"] = (table["坏样本数"].sum() / table["样本总数"].sum() - (table["坏样本数"].sum() - table["坏样本数"]) / (table["样本总数"].sum() - table["样本总数"])) / (table["坏样本数"].sum() / table["样本总数"].sum())

        def reverse_series(series):
            return series.reindex(series.index[::-1])

        if greater_is_better == "auto":
            if table[table["分箱"] != "缺失值"]["LIFT值"].iloc[-1] > table["LIFT值"].iloc[0]:
                table["累积LIFT值"] = (reverse_series(table['坏样本数']).cumsum() / reverse_series(table['样本总数']).cumsum()) / (table["坏样本数"].sum() / table["样本总数"].sum())
                table["累积坏账改善"] = (table["坏样本数"].sum() / table["样本总数"].sum() - (table["坏样本数"].sum() - reverse_series(table['坏样本数']).cumsum()) / (table["样本总数"].sum() - reverse_series(table['样本总数']).cumsum())) / (table["坏样本数"].sum() / table["样本总数"].sum())
                if ks:
                    table = table.sort_values("分箱")
                    table["累积好样本数"] = reverse_series(table["好样本数"]).cumsum()
                    table["累积坏样本数"] = reverse_series(table["坏样本数"]).cumsum()
                    table["分档KS值"] = table["累积坏样本数"] / table['坏样本数'].sum() - table["累积好样本数"] / table['好样本数'].sum()
            else:
                table["累积LIFT值"] = (table['坏样本数'].cumsum() / table['样本总数'].cumsum()) / (table["坏样本数"].sum() / table["样本总数"].sum())
                table["累积坏账改善"] = (table["坏样本数"].sum() / table["样本总数"].sum() - (table["坏样本数"].sum() - table['坏样本数'].cumsum()) / (table["样本总数"].sum() - table['样本总数'].cumsum())) / (table["坏样本数"].sum() / table["样本总数"].sum())
                if ks:
                    table = table.sort_values("分箱")
                    table["累积好样本数"] = table["好样本数"].cumsum()
                    table["累积坏样本数"] = table["坏样本数"].cumsum()
                    table["分档KS值"] = table["累积坏样本数"] / table['坏样本数'].sum() - table["累积好样本数"] / table['好样本数'].sum()
        elif greater_is_better is False:
            table["累积LIFT值"] = (reverse_series(table['坏样本数']).cumsum() / reverse_series(table['样本总数']).cumsum()) / (table["坏样本数"].sum() / table["样本总数"].sum())
            table["累积坏账改善"] = (table["坏样本数"].sum() / table["样本总数"].sum() - (table["坏样本数"].sum() - reverse_series(table['坏样本数']).cumsum()) / (table["样本总数"].sum() - reverse_series(table['样本总数']).cumsum())) / (table["坏样本数"].sum() / table["样本总数"].sum())
            if ks:
                table = table.sort_values("分箱")
                table["累积好样本数"] = reverse_series(table["好样本数"]).cumsum()
                table["累积坏样本数"] = reverse_series(table["坏样本数"]).cumsum()
                table["分档KS值"] = table["累积坏样本数"] / table['坏样本数'].sum() - table["累积好样本数"] / table['好样本数'].sum()
        else:
            table["累积LIFT值"] = (table['坏样本数'].cumsum() / table['样本总数'].cumsum()) / (table["坏样本数"].sum() / table["样本总数"].sum())
            table["累积坏账改善"] = (table["坏样本数"].sum() / table["样本总数"].sum() - (table["坏样本数"].sum() - table['坏样本数'].cumsum()) / (table["样本总数"].sum() - table['样本总数'].cumsum())) / (table["坏样本数"].sum() / table["样本总数"].sum())
            if ks:
                table = table.sort_values("分箱")
                table["累积好样本数"] = table["好样本数"].cumsum()
                table["累积坏样本数"] = table["坏样本数"].cumsum()
                table["分档KS值"] = table["累积坏样本数"] / table['坏样本数'].sum() - table["累积好样本数"] / table['好样本数'].sum()

        table["分箱"] = table["分箱"].map(feature_bin_dict)
        table = table.set_index(['指标名称', '指标含义', '分箱']).reindex([(feature, desc, b) for b in feature_bin_dict.values()]).fillna(0).reset_index()
        table['指标IV值'] = table['分档IV值'].sum()

        if return_cols:
            table = table[[c for c in return_cols if c in table.columns]]
        elif ks:
            table = table[['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '坏账改善', '累积LIFT值', '累积坏账改善', '累积好样本数', '累积坏样本数', '分档KS值']]
        else:
            table = table[['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '坏账改善', '累积LIFT值', '累积坏账改善']]

        if return_rules:
            return table, list(_combiner[feature])
        else:
            return table

    def bin_plot(self, data, x, rule={}, desc="", result=False, save=None, **kwargs):
        """特征分箱图

        :param data: 需要查看分箱图的数据集
        :param x: 需要查看的分箱图的特征名称
        :param rule: 自定义的特征分箱规则，不会修改已训练好的特征分箱信息
        :param desc: 特征描述信息，大部分时候用于传入特征对应的中文名称或者释义
        :param result: 是否返回特征分箱统计表，默认 False
        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        :param kwargs: scorecardpipeline.utils.bin_plot 方法其他的参数，参考：http://localhost:63342/scorecardpipeline/docs/build/html/scorecardpipeline.html#scorecardpipeline.utils.bin_plot
        :return: pd.DataFrame，特征分箱统计表，当 result 参数为 True 时返回
        """
        feature_table = self.feature_bin_stats(data, x, target=self.target, rules=rule, desc=desc, combiner=self.combiner, ks=True)
        bin_plot(feature_table, desc=desc, save=save, **kwargs)

        if result:
            return feature_table

    def proportion_plot(self, data, x, transform=False, labels=False, keys=None):
        """数据集中特征的分布情况

        :param data: 需要查看样本分布的数据集
        :param x: 需要查看样本分布的特征名称
        :param transform: 是否进行分箱转换，默认 False，当特征为数值型变量时推荐转换分箱后在查看数据分布
        :param labels: 进行分箱转换时是否转换为分箱信息，默认 False，即转换为分箱索引
        :param keys: 根据某个 key 划分数据集查看数据分布情况，默认 None
        """
        if transform:
            x = self.combiner.transform(x, labels=labels)

        proportion_plot(x, keys=keys)

    def corr_plot(self, data, transform=False, figure_size=(20, 15), save=None):
        """特征相关图

        :param data: 需要查看特征相关性的数据集
        :param transform: 是否进行分箱转换，默认 False
        :param figure_size: 图像大小，默认 (20, 15)
        :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
        """
        if transform:
            data = self.combiner.transform(data, labels=False)

        corr_plot(data, figure_size=figure_size, save=save)

    def badrate_plot(self, data, date_column, feature, labels=True):
        """查看不同时间段的分箱是否平稳，线敞口随时间变化而增大为优，代表了特征在更新的时间区分度更强。线之前没有交叉为优，代表分箱稳定

        :param data: 需要查看分箱平稳情况的数据集，需包含时间列
        :param feature: 需要查看分箱平稳性的特征名称
        :param date_column: 数据集中的日期列名称
        :param labels: 进行分箱转换时是否转换为分箱信息，默认 True，即转换为分箱
        """
        badrate_plot(self.combiner.transform(data[[date_column, feature, self.target]], labels=labels), target=self.target, x=date_column, by=feature)

    @property
    def rules(self):
        """dict，特征分箱明细信息"""
        return self.combiner._rules

    @rules.setter
    def rules(self, value):
        """设置分箱信息

        :param value: 特征分箱
        """
        self.combiner._rules = value

    def __len__(self):
        """返回特征分箱器特征的个数

        :return: int，分箱器中特征的个数
        """
        return len(self.combiner._rules.keys())

    def __contains__(self, key):
        """查看某个特征是否在分箱器中

        :param key: 特征名称
        :return: bool，是否在分箱器中
        """
        return key in self.combiner._rules

    def __getitem__(self, key):
        """获取某个特征的分箱信息

        :param key: 特征名称
        :return: list，特征分箱信息
        """
        return self.combiner._rules[key]

    def __setitem__(self, key, value):
        """设置某个特征的分箱信息

        :param key: 特征名称
        :param value: 分箱规则
        """
        self.combiner._rules[key] = value

    def __iter__(self):
        """分箱规则的迭代器

        :return: iter，迭代器
        """
        return iter(self.combiner._rules)


def feature_bin_stats(data, feature, target="target", overdue=None, dpd=None, rules=None, method='step', desc="", combiner=None, ks=True, max_n_bins=None, min_bin_size=None, max_bin_size=None, greater_is_better="auto", empty_separate=True, return_cols=None, return_rules=False, del_grey=False, verbose=0, **kwargs):
    """特征分箱统计表，汇总统计特征每个分箱的各项指标信息

    :param data: 需要查看分箱统计表的数据集
    :param feature: 需要查看的分箱统计表的特征名称
    :param target: 数据集中标签名称，默认 target
    :param overdue: 逾期天数字段名称, 当传入 overdue 时，会忽略 target 参数
    :param dpd: 逾期定义方式，逾期天数 > DPD 为 1，其他为 0，仅 overdue 字段起作用时有用
    :param del_grey: 是否删除逾期天数 (0, dpd] 的数据，仅 overdue 字段起作用时有用
    :param rules: 根据自定义的规则查看特征分箱统计表，支持 list（单个特征分箱规则） 或 dict（多个特征分箱规则） 格式传入
    :param combiner: 提前训练好的特征分箱器，优先级小于 rules
    :param method: 特征分箱方法，当传入 rules 或 combiner 时失效，可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform", 参考 toad.Combiner: https://github.com/amphibian-dev/toad/blob/master/toad/transform.py#L178-L355 & optbinning.OptimalBinning: https://gnpalencia.org/optbinning/
    :param desc: 特征描述信息，大部分时候用于传入特征对应的中文名称或者释义
    :param ks: 是否统计 KS 信息
    :param max_n_bins: 最大分箱数，默认 None，即不限制拆分箱数，推荐设置 3 ～ 5，不宜过多，偶尔使用 optbinning 时不起效
    :param min_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最小样本占比，默认 5%
    :param max_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最大样本占比，默认 None
    :param empty_separate: 是否空值单独一箱, 默认 False，推荐设置为 True
    :param return_cols: list，指定返回部分特征分箱统计表的列，默认 None
    :param return_rules: 是否返回特征分箱信息，默认 False
    :param greater_is_better: 是否越大越好，默认 ""auto", 根据最后两箱的 lift 指标自动推断是否越大越好, 可选 True、False、auto
    :param kwargs: scorecardpipeline.processing.Combiner 的其他参数

    :return:
        + 特征分箱统计表: pd.DataFrame
        + 特征分箱信息: list，当参数 return_rules 为 True 时返回

    """
    if overdue and dpd is None:
        raise ValueError("传入 overdue 参数时必须同时传入 dpd")

    drop_empty = True if not empty_separate and data[feature].isnull().sum() <= 0 else False

    if overdue is None:
        table, rule = Combiner.feature_bin_stats(data, feature, target=target, rules=rules, method=method, desc=desc, combiner=combiner, ks=ks, max_n_bins=max_n_bins, min_bin_size=min_bin_size, max_bin_size=max_bin_size, greater_is_better=greater_is_better, empty_separate=empty_separate, return_cols=return_cols, return_rules=True, verbose=verbose, **kwargs)

        if drop_empty:
            table = table.iloc[:-1]
            rule = rule[:-1]

        if return_rules:
            return table, rule
        else:
            return table

    if not isinstance(overdue, list):
        overdue = [overdue]

    if not isinstance(dpd, list):
        dpd = [dpd]

    if isinstance(del_grey, bool) and del_grey:
        merge_columns = ["指标名称", "指标含义", "分箱"]
    else:
        merge_columns = ["指标名称", "指标含义", "分箱", "样本总数", "样本占比"]

    table = pd.DataFrame()
    for i, col in enumerate(overdue):
        for j, d in enumerate(dpd):
            target = f"{col} {d}+"
            _datasets = data[[feature] + overdue].copy()
            _datasets[target] = (_datasets[col] > d).astype(int)

            if isinstance(del_grey, bool) and del_grey:
                _datasets = _datasets.query(f"({col} > {d}) | ({col} == 0)").reset_index(drop=True)

            if i == 0 and j == 0:
                if combiner is None:
                    if rules is not None and len(rules) > 0:
                        if isinstance(rules, (list, np.ndarray)):
                            rules = {feature: rules}

                    combiner = Combiner(target=target, adj_rules=rules, method=method, empty_separate=empty_separate, min_n_bins=2, max_n_bins=max_n_bins, min_bin_size=min_bin_size, max_bin_size=max_bin_size, **kwargs)
                    combiner.fit(_datasets)

                table, rule = Combiner.feature_bin_stats(_datasets, feature, target=target, method=method, desc=desc, combiner=combiner, ks=ks, max_n_bins=max_n_bins, min_bin_size=min_bin_size, max_bin_size=max_bin_size, greater_is_better=greater_is_better, empty_separate=empty_separate, return_rules=True, verbose=verbose, **kwargs)
                table.columns = pd.MultiIndex.from_tuples([("分箱详情", c) if c in merge_columns else (target, c) for c in table.columns])
            else:
                _table = Combiner.feature_bin_stats(_datasets, feature, target=target, method=method, desc=desc, combiner=combiner, ks=ks, max_n_bins=max_n_bins, min_bin_size=min_bin_size, max_bin_size=max_bin_size, greater_is_better=greater_is_better, empty_separate=empty_separate, verbose=verbose, **kwargs)
                _table.columns = pd.MultiIndex.from_tuples([("分箱详情", c) if c in merge_columns else (target, c) for c in _table.columns])

                table = table.merge(_table, on=[("分箱详情", c) for c in merge_columns])

    if drop_empty:
        table = table.iloc[:-1]
        rule = rule[:-1]

    if return_cols is not None:
        if not isinstance(return_cols, list):
            return_cols = [return_cols]

        table = table[[c for c in table.columns if (isinstance(c, tuple) and c[-1] in return_cols + merge_columns) or (not isinstance(c, tuple) and c in return_cols + merge_columns)]]

    if return_rules:
        return (table, rule)
    else:
        return table


class WOETransformer(TransformerMixin, BaseEstimator):

    def __init__(self, target="target", exclude=None):
        """WOE转换器

        :param target: 数据集中标签名称，默认 target
        :param exclude: 不需要转换 woe 的列
        """
        self.target = target
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []
        self.transformer = toad.transform.WOETransformer()

    def fit(self, x, y=None):
        """WOE转换器训练

        :param x: Combiner 转换后的数据（label 为 False），需要包含目标变量
        :return: WOETransformer，训练完成的WOE转换器
        """
        self.transformer.fit(x.drop(columns=self.exclude + [self.target]), x[self.target])
        return self

    def transform(self, x, y=None):
        """特征WOE转换方法

        :param x: 需要进行WOE转换的数据集

        :return: pd.DataFrame，WOE转换后的数据集
        """
        return self.transformer.transform(x)
    
    def export(self, to_json=None):
        """特征分箱器导出 json 保存

        :param to_json: json 文件的路径

        :return: dict，特征分箱信息
        """
        return self.transformer.export(to_json=to_json)

    def load(self, from_json):
        """特征分箱器加载离线保存的 json 文件

        :param from_json: json 文件的路径

        :return: Combiner，特征分箱器
        """
        self.transformer.load(from_json)
        return self

    @property
    def rules(self):
        """dict，特征 WOE 明细信息"""
        return self.transformer._rules

    @rules.setter
    def rules(self, value):
        self.transformer._rules = value

    def __len__(self):
        return len(self.transformer._rules.keys())

    def __contains__(self, key):
        return key in self.transformer._rules

    def __getitem__(self, key):
        return self.transformer._rules[key]

    def __setitem__(self, key, value):
        self.transformer._rules[key] = value

    def __iter__(self):
        return iter(self.transformer._rules)


def feature_efficiency_analysis(data, feature, overdue=["MOB1"], dpd=[7, 3, 0], greate_is_better=True, verbose=True, ks=False, **kwargs):
    auto_feature_tables = feature_bin_stats(
        data
        , feature
        , min_bin_size=0.01
        , overdue=overdue, dpd=dpd
        , method="mdlp"
        , max_n_bins=10
        , del_grey=False
        , desc=f"样本数 {len(data)} 坏样本率 {round((data[overdue[0]] > dpd[0]).mean(), 4)}"
        # , greate_is_better=greate_is_better
        , return_cols=["坏样本数", "坏样本率", "LIFT值", "累积LIFT值", "分档KS值"]
        , **kwargs
    )

    quantile_feature_tables = feature_bin_stats(
        data
        , feature, rules=data[feature].quantile([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8] if greate_is_better else [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.98, 0.99]).unique().tolist() + [np.nan]
        , method="mdlp", overdue=overdue, dpd=dpd, max_n_bins=10, greate_is_better=greate_is_better, min_bin_size=0.01, return_cols=["坏样本数", "坏样本率", "LIFT值", "累积LIFT值", "分档KS值"]
        , **kwargs
    )

    if ks:
        ks_plot(data.dropna(subset=[feature])[feature], (data.dropna(subset=[feature])[overdue[0]] > dpd[0]).astype(int), figsize=(10, 6))

    if verbose:
        from IPython.display import display
        display(auto_feature_tables)
        display(quantile_feature_tables)
    else:
        return auto_feature_tables, quantile_feature_tables
