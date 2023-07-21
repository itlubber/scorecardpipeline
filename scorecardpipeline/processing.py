# -*- coding: utf-8 -*-
"""
@Time    : 2023/05/21 16:23
@Author  : itlubber
@Site    : itlubber.art
"""

import numpy as np
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from sklearn.base import BaseEstimator, TransformerMixin

import toad
import scorecardpy as sc
from optbinning import OptimalBinning
from toad.plot import proportion_plot, badrate_plot

from .utils import *


def drop_identical(frame, threshold = 0.95, return_drop = False, exclude = None, target = None):
    """drop columns by identical
    Args:
        frame (DataFrame): dataframe that will be used
        threshold (number): drop the features whose identical num is greater than threshold. if threshold is float, it will be use as percentage
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped
        target (str): target's name in dataframe
    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    cols = frame.columns.copy()
    
    if target is not None:
        cols.drop(target)

    if exclude is not None:
        cols = cols.drop(exclude)

    if threshold < 1:
        threshold = len(frame) * threshold

    drop_list = []
    for col in cols:
        n = frame[col].value_counts().max()
        
        if n > threshold:
            drop_list.append(col)

    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (np.array(drop_list),)

    return toad.utils.unpack_tuple(res)


def select(frame, target = 'target', empty = 0.9, iv = 0.02, corr = 0.7,
            identical=0.95, return_drop = False, exclude = None):
    """select features by rate of empty, iv and correlation
    Args:
        frame (DataFrame)
        target (str): target's name in dataframe
        empty (number): drop the features which empty num is greater than threshold. if threshold is less than `1`, it will be use as percentage
        identical (number): drop the features which identical num is greater than threshold. if threshold is less than `1`, it will be use as percentage
        iv (float): drop the features whose IV is less than threshold
        corr (float): drop features that has the smallest IV in each groups which correlation is greater than threshold
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature name that will not be dropped
    Returns:
        DataFrame: selected dataframe
        dict: list of dropped feature names in each step
    """
    empty_drop, iv_drop, corr_drop, identical_drop = None, None, None, None

    if empty is not False:
        frame, empty_drop = toad.selection.drop_empty(frame, threshold = empty, return_drop = True, exclude = exclude)
        
    if identical is not False:
        frame, identical_drop = drop_identical(frame, threshold = identical, return_drop = True, exclude = exclude, target = target)

    if iv is not False:
        frame, iv_drop, iv_list = toad.selection.drop_iv(frame, target = target, threshold = iv, return_drop = True, return_iv = True, exclude = exclude)

    if corr is not False:
        weights = 'IV'

        if iv is not False:
            weights = iv_list

        frame, corr_drop = toad.selection.drop_corr(frame, target = target, threshold = corr, by = weights, return_drop = True, exclude = exclude)

    res = (frame,)
    if return_drop:
        d = {
            'empty': empty_drop,
            'identical': identical_drop,
            'iv': iv_drop,
            'corr': corr_drop,
        }
        res += (d,)

    return toad.utils.unpack_tuple(res)


class FeatureSelection(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", empty=0.95, iv=0.02, corr=0.7, exclude=None, return_drop=True, identical=0.95, remove=None, engine="scorecardpy", target_rm=False):
        """ITLUBBER提供的特征筛选方法

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
        return x[[col for col in self.select_columns if col in x.columns]]


class StepwiseSelection(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", estimator="ols", direction="both", criterion="aic", max_iter=None, return_drop=True, exclude=None, intercept=True, p_value_enter=0.2, p_remove=0.01, p_enter=0.01, target_rm=False):
        """逐步回归筛选方法

        Args:
            target: 数据集中标签名称，默认 target
            estimator: 预估器，默认 ols，可选 "ols", "lr", "lasso", "ridge"，通常默认即可
            direction: 逐步回归方向，默认both，可选 "forward", "backward", "both"，通常默认即可
            criterion: 评价指标，默认 aic，可选 "aic", "bic", "ks", "auc"，通常默认即可
            max_iter: 最大迭代次数，sklearn中使用的参数，默认为 None
            return_drop: 是否返回特征剔除信息，默认 True
            exclude: 强制保留的某些特征
            intercept: 是否包含截距，默认为 True
            p_value_enter: 特征进入的 p 值，用于前向筛选时决定特征是否进入模型
            p_remove: 特征剔除的 p 值，用于后向剔除时决定特征是否要剔除
            p_enter: 特征 p 值，用于判断双向逐步回归是否剔除或者准入特征
            target_rm: 是否剔除数据集中的标签，默认为 False，即剔除数据集中的标签
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
        return x[[col for col in self.select_columns if col in x.columns]]


class FeatureImportanceSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, top_k=126, target="target", selector="catboost", params=None, max_iv=None):
        """基于特征重要性的特征筛选方法
        
        Args:
            target: 数据集中标签名称，默认 target
            top_k: 依据特征重要性进行排序，筛选最重要的 top_k 个特征
            max_iv: 是否需要删除 IV 过高的特征，建议设置为 1.0
            selector: 特征选择器，目前只支持 catboost ，可以支持数据集中包含字符串的数据
            params: selector 的参数，不传使用默认参数
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
        return x[self.select_columns + [self.target]]
        
        
    def catboost_selector(self, x, y, cat_features=None):
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
    
    def __init__(self, target="target", method='chi', empty_separate=False, min_n_bins=2, max_n_bins=None, max_n_prebins=20, min_prebin_size=0.02, min_bin_size=0.05, max_bin_size=None, gamma=0.01, monotonic_trend="auto_asc_desc", adj_rules={}, n_jobs=1):
        """特征分箱封装方法

        Args:
            target: 数据集中标签名称，默认 target
            method: 特征分箱方法，可选 "chi", "dt", "quantile", "step", "kmeans", "cart", "mdlp", "uniform", 参考 toad.Combiner & optbinning.OptimalBinning
            empty_separate: 是否空值单独一箱, 默认 False，推荐设置为 True
            min_n_bins: 最小分箱数，默认 2，即最小拆分2箱
            max_n_bins: 最大分箱数，默认 None，即不限制拆分箱数，推荐设置 3 ～ 5，不宜过多，偶尔使用 optbinning 时不起效
            max_n_prebins: 使用 optbinning 时预分箱数量
            min_prebin_size: 使用 optbinning 时预分箱叶子结点（或者每箱）样本占比，默认 2%
            min_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最小样本占比，默认 5%
            max_bin_size: 使用 optbinning 正式分箱叶子结点（或者每箱）最大样本占比，默认 None
            gamma: 使用 optbinning 分箱时限制过拟合的正则化参数，值越大惩罚越多，默认 0。01
            monotonic_trend: 使用 optbinning 正式分箱时的坏率策略，默认 auto，可选 "auto", "auto_heuristic", "auto_asc_desc", "ascending", "descending", "convex", "concave", "peak", "valley", "peak_heuristic", "valley_heuristic"
            adj_rules: 自定义分箱规则，toad.Combiner 能够接收的形式
            n_jobs: 使用多进程加速的worker数量，默认单进程
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
        
    def update(self, rules):
        self.combiner.update(rules)
    
    def optbinning_bins(self, feature, data=None, target="target", min_n_bins=2, max_n_bins=3, max_n_prebins=10, min_prebin_size=0.02, min_bin_size=0.05, max_bin_size=None, gamma=0.01, monotonic_trend="auto_asc_desc"):
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

                _combiner = OptimalBinning(feature, dtype=dtype, min_n_bins=min_n_bins, max_n_bins=max_n_bins, max_n_prebins=max_n_prebins, min_prebin_size=min_prebin_size, min_bin_size=min_bin_size, max_bin_size=max_bin_size, monotonic_trend=monotonic_trend, gamma=gamma).fit(x, y)
                if _combiner.status == "OPTIMAL":
                    rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.splits] + [[np.nan] if dtype == "categorical" else np.nan]}
                else:
                    raise Exception("optimalBinning error")
            
            except Exception as e:
                _combiner = toad.transform.Combiner()
                _combiner.fit(data[[feature, target]].dropna(), target, method="chi", min_samples=self.min_bin_size, n_bins=self.max_n_bins, empty_separate=False)
                rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.export()[feature]] + [[np.nan] if dtype == "categorical" else np.nan]}
        
        self.combiner.update(rule)
    
    def fit(self, x, y=None):
        if self.method in ["cart", "mdlp", "uniform"]:
            feature_optbinning_bins = partial(self.optbinning_bins, data=x, target=self.target, min_n_bins=self.min_n_bins, max_n_bins=self.max_n_bins, max_n_prebins=self.max_n_prebins, min_prebin_size=self.min_prebin_size, min_bin_size=self.min_bin_size, max_bin_size=self.max_bin_size, gamma=self.gamma, monotonic_trend=self.monotonic_trend)
            if self.n_jobs > 1:
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    [executor.submit(feature_optbinning_bins(feature)) for feature in x.columns.drop(self.target)]
            else:
                for feature in x.drop(columns=[self.target]):
                    feature_optbinning_bins(feature)
        else:
            if self.method in ["step", "quantile"]:
                self.combiner.fit(x, y=self.target, method=self.method, n_bins=self.max_n_bins, empty_separate=self.empty_separate)
            else:
                self.combiner.fit(x, y=self.target, method=self.method, min_samples=self.min_bin_size, n_bins=self.max_n_bins, empty_separate=self.empty_separate)
        
        self.update(self.adj_rules)
        
        return self
    
    def transform(self, x, y=None, labels=False):
        return self.combiner.transform(x, labels=labels)

    def export(self, to_json=None):
        return self.combiner.export(to_json=to_json)
    
    def load(self, from_json=None):
        self.combiner.load(from_json=from_json)
        return self
    
    def bin_plot(self, data, x, rule={}, desc="", result=False, save=None, **kwargs):
        feature_table = feature_bin_stats(data, x, target=self.target, rules=rule, desc=desc, combiner=self.combiner, ks=True)
        bin_plot(feature_table, desc=desc, save=save, **kwargs)
        
        if result:
            return feature_table
    
    def proportion_plot(self, x, transform=False, labels=False):
        if transform:
            x = self.combiner.transform(x, labels=labels)
        proportion_plot(x)
    
    def corr_plot(self, data, transform=False, figure_size=(20, 15), save=None):
        if transform:
            data = self.combiner.transform(data, labels=False)
        
        corr_plot(data, figure_size=figure_size, save=save)
    
    def badrate_plot(self, data, date_column, feature, labels=True):
        badrate_plot(self.combiner.transform(data[[date_column, feature, self.target]], labels=labels), target=self.target, x=date_column, by=feature)
    
    @property
    def rules(self):
        return self.combiner._rules
    
    @rules.setter
    def rules(self, value):
        self.combiner._rules = value
    
    def __len__(self):
        return len(self.combiner._rules.keys())
    
    def __contains__(self, key):
        return key in self.combiner._rules
    
    def __getitem__(self, key):
        return self.combiner._rules[key]
    
    def __setitem__(self, key, value):
        self.combiner._rules[key] = value

    def __iter__(self):
        return iter(self.combiner._rules)


class WOETransformer(TransformerMixin, BaseEstimator):
    
    def __init__(self, target="target", exclude=None):
        """WOE转换器

        Args:
            target: 数据集中标签名称，默认 target
            exclude: 不需要转换 woe 的列
        """
        self.target = target
        self.exclude = exclude if isinstance(exclude, list) else [exclude] if exclude else []
        self.transformer = toad.transform.WOETransformer()
        
    def fit(self, x, y=None):
        self.transformer.fit(x.drop(columns=self.exclude + [self.target]), x[self.target])
        return self

    def transform(self, x, y=None):
        return self.transformer.transform(x)
    
    @property
    def rules(self):
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
