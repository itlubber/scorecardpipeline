# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/8 14:06
@Author  : itlubber
@Site    : itlubber.art
"""

from functools import partial
from abc import ABCMeta, abstractmethod

import math
import numpy as np
import pandas as pd
from copy import deepcopy
from joblib import Parallel, delayed

from sklearn.utils import _safe_indexing
from sklearn.utils._encode import _unique
from sklearn.utils._mask import _get_mask
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils.sparsefuncs import mean_variance_axis, min_max_axis
from sklearn.utils.validation import check_is_fitted, check_array, indexable
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.feature_selection import RFECV, RFE, SelectFromModel, SelectKBest
from sklearn.feature_selection._from_model import _calculate_threshold, _get_feature_importances
# from statsmodels.stats.outliers_influence import variance_inflation_factor

from .processing import Combiner


class SelectorMixin(BaseEstimator, TransformerMixin):

    def transform(self, x):
        check_is_fitted(self, "select_columns")
        return x[[col for col in self.select_columns if col in x.columns]]
    
    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.select_columns


class TypeSelector(SelectorMixin):
    def __init__(self, dtype_include=None, dtype_exclude=None, exclude=None):
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude
        self.exclude = exclude

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        if not hasattr(x, 'iloc'):
            raise ValueError("make_column_selector can only be applied to pandas dataframes")
        
        self.n_features_in_ = x.shape[1]
        
        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        if self.dtype_include is not None or self.dtype_exclude is not None:
            cols = x.select_dtypes(include=self.dtype_include, exclude=self.dtype_exclude).columns
        else:
            cols = x.columns
        
        self.scores_ = x.dtypes
        self.select_columns = list(set(cols.tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))
        
        self.dropped = pd.DataFrame([(col, f"data type or name not match") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


class RegexSelector(SelectorMixin):
    def __init__(self, pattern=None, exclude=None):
        self.pattern = pattern
        self.exclude = exclude

        if self.pattern is None:
            raise ValueError("pattern must be a regular expression.")

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        if not hasattr(x, 'iloc'):
            raise ValueError("make_column_selector can only be applied to pandas dataframes")

        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        self.scores_ = x.columns.str.contains(self.pattern, regex=True).astype(int)
        self.select_columns = list(set(x.columns[self.scores_ == 1].tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"feature name not match {self.pattern}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


def value_ratio(x, value):
    if isinstance(x, pd.DataFrame):
        return np.mean(_get_mask(x.values, value), axis=0)

    return np.mean(_get_mask(x, value), axis=0)


def mode_ratio(x, dropna=True):
    if isinstance(x, (list, np.ndarray)):
        x = pd.Series(x)

    summary = x.value_counts(dropna=dropna)
    return (summary.index[0], summary.iloc[0] / sum(summary)) if len(summary) > 0 else (np.nan, 1.0)


class NanSelector(SelectorMixin):

    def __init__(self, threshold=0.95, missing_values=np.nan, exclude=None, **kwargs):
        self.exclude = exclude
        self.threshold = threshold
        self.missing_values = missing_values
        self.dropped = None
        self.select_columns = None
        self.scores_ = None
        self.n_features_in_ = None
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None):
        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        self.scores_ = pd.Series(value_ratio(x, self.missing_values), index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ < self.threshold]).index.tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"nan ratio >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


class ModeSelector(SelectorMixin):

    def __init__(self, threshold=0.95, exclude=None, dropna=True, n_jobs=None, **kwargs):
        self.dropna = dropna
        self.exclude = exclude
        self.threshold = threshold
        self.dropped = None
        self.select_columns = None
        self.scores_ = None
        self.n_features_in_ = None
        self.kwargs = kwargs
        self.n_jobs = n_jobs

    def fit(self, x: pd.DataFrame, y=None):
        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

            x = x.drop(columns=[c for c in self.exclude if c in x.columns])

        self.scores_ = pd.DataFrame(Parallel(n_jobs=self.n_jobs)(delayed(mode_ratio)(x[c], self.dropna) for c in x.columns), columns=["Mode", "Ratio"], index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_["Ratio"] < self.threshold]).index.tolist()))
        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"mode ratio >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


class CardinalitySelector(SelectorMixin):
    """Feature selection via categorical feature's cardinality.

    Examples
    -----------
    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import CardinalitySelector
    >>> x = pd.DataFrame({"f2": ["F", "м", "F"], "f3": ["M1", "M2", "м3"]})
    >>> cs = CardinalitySelector(threshold=2)
    >>> cs.fit_transform(x)
    """
    def __init__(self, threshold=10, exclude=None, dropna=True):
        self.exclude = exclude
        self.threshold = threshold
        self.dropna = dropna

    def fit(self, x, y=None, **fit_params):
        self.n_features_in_ = x.shape[1]

        if self.exclude:
            if not isinstance(self.exclude, (list, tuple, np.ndarray)):
                self.exclude = [self.exclude]

        self.scores_ = pd.Series(x.nunique(axis=0, dropna=self.dropna).values, index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ < self.threshold]).index.tolist()))

        if self.exclude:
            self.select_columns = list(set(self.select_columns + self.exclude))

        self.dropped = pd.DataFrame([(col, f"cardinality >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


def IV(x, y, regularization=1.0):
    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization
    uniques = np.unique(x)
    n_cats = len(uniques)
    event_rates = np.zeros(n_cats, dtype=np.float64)
    nonevent_rates = np.zeros(n_cats, dtype=np.float64)
    for i, cat in enumerate(uniques):
        mask = x == cat
        event_rates[i] = np.count_nonzero(mask & event_mask) + regularization
        nonevent_rates[i] = np.count_nonzero(mask & nonevent_mask) + regularization

    # Ignore unique values. This helps to prevent overfitting on id-like columns.
    bad_pos = (event_rates + nonevent_rates) == (2 * regularization + 1)
    event_rates /= event_tot
    nonevent_rates /= nonevent_tot
    ivs = (event_rates - nonevent_rates) * np.log(event_rates / nonevent_rates)
    ivs[bad_pos] = 0.
    return np.sum(ivs).item()


def _IV(x, y, regularization=1.0, n_jobs=None):
    x = check_array(x, dtype=None, force_all_finite=True, ensure_2d=True)
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(le.classes_) != 2:
        raise ValueError("Only support binary label for computing information value!")
    _, n_features = x.shape
    iv_values = Parallel(n_jobs=n_jobs)(delayed(IV)(x[:, i], y, regularization=regularization) for i in range(n_features))
    return np.asarray(iv_values, dtype=np.float64)


class InformationValueSelector(SelectorMixin):

    def __init__(self, threshold=0.02, target="target", regularization=1.0, methods=None, n_jobs=None, **kwargs):
        self.dropped = None
        self.select_columns = None
        self.scores_ = None
        self.n_features_in_ = None
        self.combiner = None
        self.threshold = threshold
        self.target = target
        self.regularization = regularization
        self.n_jobs = n_jobs
        self.methods = methods
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None):
        if y is None:
            if self.target not in x.columns:
                raise ValueError(f"需要传入 y 或者 x 中包含 {self.target}.")
            y = x[self.target]
            x = x.drop(columns=self.target)

        self.n_features_in_ = x.shape[1]
        
        if self.methods:
            temp = x.copy()
            temp[self.target] = y
            self.combiner = Combiner(target=self.target, method=self.methods, **self.kwargs)
            self.combiner.fit(temp)
            xt = self.combiner.transform(x)
        else:
            xt = x.copy()
        
        self.scores_ = pd.Series(_IV(xt, y, regularization=self.regularization, n_jobs=self.n_jobs), index=xt.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + [self.target]))
        self.dropped = pd.DataFrame([(col, f"IV <= {self.threshold}") for col in xt.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


def LIFT(y_pred, y_true):
    """Calculate lift according to label data.

    Parameters
    -----------
    y_true : array-like
    y_pred : array-like

    Returns
    -----------
    lift : float

    Examples
    -----------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1])
    >>> LIFT(y_true, y_pred) # (5 / 7) / (6 / 9)
    1.0714285714285716
    """
    _y_true = column_or_1d(y_true)
    base_bad_rate = np.average(y_true)

    score = []
    for v in np.unique(y_pred):
        if pd.isnull(v):
            _y_pred = column_or_1d(y_pred.isnull())
        else:
            _y_pred = column_or_1d(y_pred == v)
        hit_bad_rate = np.count_nonzero((_y_true == 1) & (_y_pred == 1)) / np.count_nonzero(_y_pred)
        score.append(hit_bad_rate / base_bad_rate)

    return np.nanmax(score)


class LiftSelector(SelectorMixin):
    """Feature selection via lift score.

    Parameters
    -----------
    threshold : float or str (default=3.0)
        Feature which has a lift score greater than `threshold` will be kept.
    n_jobs : int or None, (default=None)
        Number of parallel.
    
    Attributes
    -----------
    threshold_: float
        The threshold value used for feature selection.
    scores_ : array-like of shape (n_features,)
        Lift scores of features.
    """
    def __init__(self, target="target", threshold=3.0, n_jobs=None, methods=None, **kwargs):
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.target = target
        self.methods = methods
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        if y is None:
            if self.target not in x.columns:
                raise ValueError(f"需要传入 y 或者 x 中包含 {self.target}.")
            y = x[self.target]
            x = x.drop(columns=self.target)

        self.n_features_in_ = x.shape[1]
        
        if self.methods:
            temp = x.copy()
            temp[self.target] = y
            self.combiner = Combiner(target=self.target, method=self.methods, **self.kwargs)
            self.combiner.fit(temp)
            xt = self.combiner.transform(x)
        else:
            xt = x.copy()

        self.scores_ = pd.Series(Parallel(n_jobs=self.n_jobs)(delayed(LIFT)(x[c], y) for c in xt.columns), index=xt.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + [self.target]))
        self.dropped = pd.DataFrame([(col, f"LIFT < {self.threshold}") for col in xt.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


class VarianceSelector(SelectorMixin):
    """Feature selector that removes all low-variance features."""

    def __init__(self, threshold=0.0, exclude=None):
        self.threshold = threshold
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []

    def fit(self, x, y=None):
        self.n_features_in_ = x.shape[1]
        
        if hasattr(x, "toarray"):  # sparse matrix
            _, scores = mean_variance_axis(x, axis=0)
            if self.threshold == 0:
                mins, maxes = min_max_axis(x, axis=0)
                peak_to_peaks = maxes - mins
        else:
            scores = np.nanvar(x, axis=0)
            if self.threshold == 0:
                peak_to_peaks = np.ptp(x, axis=0)

        if self.threshold == 0:
            # Use peak-to-peak to avoid numeric precision issues for constant features
            compare_arr = np.array([scores, peak_to_peaks])
            scores = np.nanmin(compare_arr, axis=0)

        if np.all(~np.isfinite(scores) | (scores <= self.threshold)):
            msg = "No feature in x meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (x contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        self.scores_ = pd.Series(scores, index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ > self.threshold]).index.tolist() + self.exclude))
        self.dropped = pd.DataFrame([(col, f"Variance <= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        
        return self


def VIF(x, n_jobs=None, missing=-1):
    columns = x.columns
    x = x.fillna(missing).values
    lr = partial(lambda x, y: LinearRegression(fit_intercept=False).fit(x, y).predict(x))
    y_pred = Parallel(n_jobs=n_jobs)(delayed(lr)(x[:, np.arange(x.shape[1]) != i], x[:, i]) for i in range(x.shape[1]))
    vif = [np.sum(x[:, i] ** 2) / np.sum((y_pred[i] - x[:, i]) ** 2) for i in range(x.shape[1])]

    return pd.Series(vif, index=columns)


class VIFSelector(SelectorMixin):

    def __init__(self, threshold=4.0, exclude=None, missing=-1, n_jobs=None):
        """VIF越高，多重共线性的影响越严重, 在金融风险中我们使用经验法则:若VIF>4，则我们认为存在多重共线性, 计算比较消耗资源, 如果数据维度较大的情况下, 尽量不要使用

        :param exclude: 数据集中需要强制保留的变量
        :param threshold: 阈值, VIF 大于 threshold 即剔除该特征
        :param missing: 缺失值默认填充 -1
        :param n_jobs: 线程数
        """
        self.threshold = threshold
        self.missing = missing
        self.n_jobs = n_jobs
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []

    def fit(self, x: pd.DataFrame, y=None):
        if self.exclude:
            x = x.drop(columns=self.exclude)
        
        self.n_features_in_ = x.shape[1]
        
        # vif = partial(variance_inflation_factor, np.matrix(x.fillna(self.missing)))
        # self.scores_ = pd.Series(Parallel(n_jobs=None)(delayed(vif)(i) for i in range(x.shape[1])), index=x.columns)
        self.scores_ = VIF(x, missing=self.missing, n_jobs=self.n_jobs)
        
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ > self.threshold]).index.tolist() + self.exclude))
        self.dropped = pd.DataFrame([(col, f"VIF > {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])

        return self


class CorrSelector(SelectorMixin):
    def __init__(self, threshold=0.7, method="pearson", weights=None, exclude=None, **kwargs):
        self.threshold = threshold
        self.method = method
        self.weights = weights
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None):
        if self.exclude:
            x = x.drop(columns=self.exclude)

        self.n_features_in_ = x.shape[1]

        if self.weights is None:
            self.weights = pd.Series(np.zeros(self.n_features_in_), index=x.columns)
        elif not isinstance(self.weights, pd.Series):
            self.weights = pd.Series(self.weights, index=x.columns)
            x = x[sorted(x.columns, key=self.weights.sort_values())]

        corr = x.corr(method=self.method, **self.kwargs)
        self.scores_ = corr
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)

        # corr_matrix = self.scores_.values
        # mask = np.full(self.n_features_in_, True, dtype=bool)
        # for i in range(self.n_features_in_):
        #     if not mask[i]:
        #         continue
        #     for j in range(i + 1, self.n_features_in_):
        #         if not mask[j]:
        #             continue
        #         if abs(corr_matrix[i, j]) < self.threshold:
        #             continue
        #         mask[j] = False
        #
        # self.select_columns = list(set([c for i, c in enumerate(x.columns) if mask[i]] + self.exclude))

        drops = []
        ix, cn = np.where(np.triu(corr.values, 1) > self.threshold)
        weights = self.weights.values

        if len(ix):
            graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])
            uni, counts = np.unique(graph, return_counts=True)

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

        self.dropped = pd.DataFrame([(col, f"corr > {self.threshold}") for col in corr.index[drops].values], columns=["variable", "rm_reason"])
        self.select_columns = list(set([c for c in x.columns if c not in corr.index[drops].values] + self.exclude))

        return self


def _psi_score(expected, actual):
    n_expected = len(expected)
    n_actual = len(actual)

    psi = []
    for value in _unique(expected):
        expected_cnt = np.count_nonzero(expected == value)
        actual_cnt = np.count_nonzero(actual == value)
        expected_cnt = expected_cnt if expected_cnt else 1.
        actual_cnt = actual_cnt if actual_cnt else 1.
        expected_rate = expected_cnt / n_expected
        actual_rate = actual_cnt / n_actual
        psi.append((actual_rate - expected_rate) * np.log(actual_rate / expected_rate))
    
    return sum(psi)


def PSI(train, test, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs'):
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_psi_score)(train[:, i], test[:,i]) for i in range(len(train.columns)))
    return scores


class PSISelector(SelectorMixin):

    def __init__(self, threshold=0.1, cv=None, method=None, exclude=None, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs', **kwargs):
        self.threshold = threshold
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        if exclude is not None:
            self.exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]
        else:
            self.exclude = []
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None, groups=None):
        if self.method is not None:
            temp = x.copy()
            if y is not None:
                if self.kwargs and "target" in self.kwargs and self.kwargs["target"] not in temp.columns:
                    temp[self.kwargs["target"]] = y
                elif "target" not in temp.columns:
                    temp["target"] = y

            self.combiner = Combiner(method=self.method, **self.kwargs).fit(temp)
            x = self.combiner.transform(x)

        if self.exclude:
            x = x.drop(columns=self.exclude)

        self.n_features_in_ = x.shape[1]
        x, groups = indexable(x, groups)
        cv = check_cv(self.cv)
        n_jobs = self.n_jobs
        verbose = self.verbose
        pre_dispatch = self.pre_dispatch

        cv_scores = []
        for train, test in cv.split(x, y, groups):
            scores = PSI(_safe_indexing(x, train), _safe_indexing(x, test), n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
            cv_scores.append(scores)

        self.scores_ = pd.Series(np.mean(cv_scores, axis=0), index=x.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + self.exclude))
        self.dropped = pd.DataFrame([(col, f"PSI >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])

        return self


class NullImportanceSelector(SelectorMixin):
    
    def __init__(self, estimator, target="target", threshold=1.0, norm_order=1, importance_getter='auto', cv=3, n_runs=5, **kwargs):
        self.estimator = estimator
        self.threshold = threshold
        self.norm_order = norm_order
        self.importance_getter = importance_getter
        self.cv = cv
        self.n_runs = n_runs
        self.target = target
    
    @staticmethod
    def _feature_score_v1(actual_importances, null_importances):
        # 未进行特征shuffle的特征重要性除以shuffle以后的0.75分位数作为score
        actual_importance = actual_importances.mean()
        return np.log(1e-10 + actual_importance / (1. + np.percentile(null_importances, 75)))
    
    @staticmethod
    def _feature_score_v2(actual_importances, null_importances):
        # shuffle之后特征重要性低于实际target对应特征的重要性0.25分位数的次数百分比
        return np.count_nonzero(null_importances < np.percentile(actual_importances, 25)) / null_importances.shape[0]

    def fit(self, x: pd.DataFrame, y=None):
        if self.target in x.columns:
            y = x[self.target]
            x = x.drop(columns=self.target)

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        
        n_splits = cv.get_n_splits()
        n_runs = self.n_runs
        getter = self.importance_getter
        norm_order = self.norm_order
        
        # 计算shuffle之后的特征重要性
        estimator = deepcopy(self.estimator)
        n_samples, n_features = x.shape
        null_importances = np.zeros((n_features, n_splits * n_runs))
        idx = np.arange(n_samples)
        for run in range(n_runs):
            np.random.shuffle(idx)
            y_shuffled = y[idx]

            for fold_, (train_idx, valid_idx) in enumerate(cv.split(y_shuffled, y_shuffled)):
                estimator.fit(x.loc[train_idx], y_shuffled.loc[train_idx])
                null_importance = _get_feature_importances(estimator, getter, transform_func=None, norm_order=norm_order)
                null_importances[:, n_splits * run + fold_] = null_importance
        
        # 计算未shuffle的特征重要性
        estimator = clone(self.estimator)
        actual_importances = np.zeros((n_features, n_splits * n_runs))
        for run in range(n_runs):
            for fold_, (train_idx, valid_idx) in enumerate(cv.split(y, y)):
                estimator.fit(x.loc[train_idx], y.loc[train_idx])
                actual_importance = _get_feature_importances(estimator, getter, transform_func=None, norm_order=norm_order)
                actual_importances[:, n_splits * run + fold_] = actual_importance

        self.null_importances = null_importances
        self.actual_importances_ = actual_importances
        
        scores = np.zeros(n_features)
        for i in range(n_features):
            scores[i] = self._feature_score_v2(actual_importances[i, :], null_importances[i, :])

        self.scores_ = pd.Series(scores, index=x.columns)
        self.threshold = _calculate_threshold(self.estimator, scores, self.threshold)
        
        if self.threshold > 1.0:
            self.select_columns = list(set(self.scores_.sort_values(ascending=False).iloc[:math.floor(self.threshold)].index.tolist() + [self.target]))
            self.dropped = pd.DataFrame([(col, f"nullimportance not top {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        else:
            self.select_columns = list(set((self.scores_[self.scores_ > self.threshold]).index.tolist() + [self.target]))
            self.dropped = pd.DataFrame([(col, f"nullimportance <= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])


class TargetPermutationSelector(SelectorMixin):
    pass


class BorutaSelector(SelectorMixin):
    pass


class ExhaustiveSelector(SelectorMixin):
    pass


class MICSelector(SelectorMixin):
    pass


class FeatureImportanceSelector(SelectorMixin):
    pass


class StabilitySelector(SelectorMixin):
    pass
