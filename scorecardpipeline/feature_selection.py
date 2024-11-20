# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/8 14:06
@Author  : itlubber
@Site    : itlubber.art
"""

import operator
import sys
import types
from copy import deepcopy
from functools import reduce
from itertools import chain, combinations
from functools import partial
from abc import ABCMeta, abstractmethod

import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scipy.stats import sem
from scipy.stats._continuous_distns import t
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection._validation import cross_val_score, _score
from sklearn.utils._encode import _unique
from sklearn.utils._mask import _get_mask
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.utils import _safe_indexing, check_X_y
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils.sparsefuncs import mean_variance_axis, min_max_axis
from sklearn.utils.validation import check_is_fitted, check_array, indexable, column_or_1d
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier, MetaEstimatorMixin
from sklearn.feature_selection import RFECV, RFE, SelectFromModel, SelectKBest, GenericUnivariateSelect
from sklearn.feature_selection._from_model import _calculate_threshold, _get_feature_importances
# from statsmodels.stats.outliers_influence import variance_inflation_factor

from .processing import Combiner


class SelectorMixin(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.select_columns = None
        self.scores_ = None
        self.dropped = None
        self.n_features_in_ = None

    def transform(self, x):
        check_is_fitted(self, "select_columns")
        return x[[col for col in self.select_columns if col in x.columns]]
    
    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.select_columns

    def fit(self, x, y=None):
        pass


class TypeSelector(SelectorMixin):

    def __init__(self, dtype_include=None, dtype_exclude=None, exclude=None):
        super().__init__()
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
        super().__init__()
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


class NullSelector(SelectorMixin):

    def __init__(self, threshold=0.95, missing_values=np.nan, exclude=None, **kwargs):
        super().__init__()
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
        super().__init__()
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

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_selection import CardinalitySelector
    >>> x = pd.DataFrame({"f2": ["F", "м", "F"], "f3": ["M1", "M2", "м3"]})
    >>> cs = CardinalitySelector(threshold=2)
    >>> cs.fit_transform(x)
    """
    def __init__(self, threshold=10, exclude=None, dropna=True):
        super().__init__()
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
    uniques = np.unique(x)
    n_cats = len(uniques)

    if n_cats <= 1:
        return 0.0

    event_mask = y == 1
    nonevent_mask = y != 1
    event_tot = np.count_nonzero(event_mask) + 2 * regularization
    nonevent_tot = np.count_nonzero(nonevent_mask) + 2 * regularization

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

    def __init__(self, threshold=0.02, target="target", regularization=1.0, methods=None, n_jobs=None, combiner=None, **kwargs):
        super().__init__()
        self.dropped = None
        self.select_columns = None
        self.scores_ = None
        self.n_features_in_ = None
        self.combiner = combiner
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

        if self.combiner:
            xt = self.combiner.transform(x)
        elif self.methods:
            temp = x.copy()
            temp[self.target] = y
            self.combiner = Combiner(target=self.target, method=self.methods, n_jobs=self.n_jobs, **self.kwargs)
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

    **参考样例**

    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1])
    >>> LIFT(y_true, y_pred) # (5 / 7) / (6 / 9)
    1.0714285714285716
    """
    if len(np.unique(y_pred)) <= 1:
        return 1.0

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

    **属性字段**

    :param threshold_: float. The threshold value used for feature selection.
    :param scores_ : array-like of shape (n_features,). Lift scores of features.
    :param select_columns : array-like
    :param dropped : DataFrame
    """
    def __init__(self, target="target", threshold=3.0, n_jobs=None, methods=None, combiner=None, **kwargs):
        """
        :param target: target
        :param threshold: float or str (default=3.0). Feature which has a lift score greater than `threshold` will be kept.
        :param n_jobs: int or None, (default=None). Number of parallel.
        :param combiner: Combiner
        :param methods: Combiner's methods
        """
        super().__init__()
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.target = target
        self.methods = methods
        self.combiner = combiner
        self.kwargs = kwargs

    def fit(self, x: pd.DataFrame, y=None, **fit_params):
        if y is None:
            if self.target not in x.columns:
                raise ValueError(f"需要传入 y 或者 x 中包含 {self.target}.")
            y = x[self.target]
            x = x.drop(columns=self.target)

        self.n_features_in_ = x.shape[1]

        if self.combiner:
            xt = self.combiner.transform(x)
        elif self.methods:
            temp = x.copy()
            temp[self.target] = y
            self.combiner = Combiner(target=self.target, method=self.methods, n_jobs=self.n_jobs, **self.kwargs)
            self.combiner.fit(temp)
            xt = self.combiner.transform(x)
        else:
            xt = x.copy()

        # _lift = {}
        # for c in tqdm(xt.columns):
        #     _lift[c] = LIFT(xt[c], y)
        # self.scores_ = pd.Series(_lift)
        
        self.scores_ = pd.Series(Parallel(n_jobs=self.n_jobs)(delayed(LIFT)(xt[c], y) for c in xt.columns), index=xt.columns)
        self.threshold = _calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + [self.target]))
        self.dropped = pd.DataFrame([(col, f"LIFT < {self.threshold}") for col in xt.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self


class VarianceSelector(SelectorMixin):
    """Feature selector that removes all low-variance features."""

    def __init__(self, threshold=0.0, exclude=None):
        super().__init__()
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
            if x.shape[0] == 1:
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
        super().__init__()
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
        self.select_columns = list(set((self.scores_[self.scores_ < self.threshold]).index.tolist() + self.exclude))
        self.dropped = pd.DataFrame([(col, f"VIF >= {self.threshold}") for col in x.columns if col not in self.select_columns], columns=["variable", "rm_reason"])

        return self


class CorrSelector(SelectorMixin):
    def __init__(self, threshold=0.7, method="pearson", weights=None, exclude=None, **kwargs):
        super().__init__()
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
        
        _weight = pd.Series(np.zeros(self.n_features_in_), index=x.columns)
        
        if self.weights is not None:
            if isinstance(self.weights, pd.Series):
                _weight_columns = list(set(self.weights.index) & set(x.columns))
                _weight.loc[_weight_columns] = self.weights[_weight_columns]
            else:
                _weight = pd.Series(self.weights, index=x.columns)

        self.weights = _weight
        x = x[sorted(x.columns, key=lambda c: self.weights.loc[c], reverse=True)]

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
        super().__init__()
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

            self.combiner = Combiner(method=self.method, n_jobs=self.n_jobs, **self.kwargs).fit(temp)
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
        super().__init__()
        self.estimator = estimator
        self.threshold = threshold
        self.norm_order = norm_order
        self.importance_getter = importance_getter
        self.cv = cv
        self.n_runs = n_runs
        self.target = target
    
    @staticmethod
    def _feature_score_v0(actual_importances, null_importances):
        return actual_importances.mean(axis=1) / null_importances.mean(axis=1)
    
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
            np.random.shuffle(idx)
            y_shuffled = y[idx]
            x_shuffled = x[idx]
            
            for fold_, (train_idx, valid_idx) in enumerate(cv.split(y_shuffled, y_shuffled)):
                estimator.fit(x_shuffled.loc[train_idx], y_shuffled.loc[train_idx])
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


class TargetPermutationSelector(NullImportanceSelector):
    
    def __init__(self, estimator, target="target", threshold=1.0, norm_order=1, importance_getter='auto', cv=3, n_runs=5, **kwargs):
        super().__init__(estimator, target=target, threshold=threshold, norm_order=norm_order, importance_getter=importance_getter, cv=cv, n_runs=n_runs, **kwargs)


class ExhaustiveSelector(SelectorMixin, MetaEstimatorMixin):
    """Exhaustive Feature Selection for Classification and Regression.

    **属性字段**

    :param subset_info_: list of dicts. A list of dictionary with the following keys: 'support_mask', mask array of the selected features 'cv_scores', cross validate scores
    :param support_mask_: array-like of booleans. Array of final chosen features
    :param best_idx_: array-like, shape = [n_predictions]. Feature Indices of the selected feature subsets.
    :param best_score_: float. Cross validation average score of the selected subset.
    :param best_feature_indices_: array-like, shape = (n_features,), Feature indices of the selected feature subsets.

    **参考样例**
    
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> from scorecardpipeline.feature_selection import ExhaustiveSelector
    >>> X, y = load_iris(return_X_y=True, as_frame=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> efs = ExhaustiveSelector(knn, min_features=1, max_features=4, cv=3)
    >>> efs.fit(X, y)
    ExhaustiveFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3), max_features=4)
    >>> efs.best_score_
    0.9733333333333333
    >>> efs.best_idx_
    12
    """
    def __init__(self, estimator, min_features=1, max_features=1, scoring="accuracy", cv=3, verbose=0, n_jobs=None, pre_dispatch='2*n_jobs'):
        """
        :param estimator: scikit-learn classifier or regressor
        :param min_features: int (default: 1). Minimum number of features to select
        :param max_features: int (default: 1). Maximum number of features to select
        :param verbose: bool (default: True). Prints progress as the number of epochs to stdout.
        :param scoring: str, (default='_passthrough_scorer'). Scoring metric in faccuracy, f1, precision, recall, roc_auc) for classifiers, {'mean_absolute_error', 'mean_squared_error', 'median_absolute_error', 'r2'} for regressors, or a callable object or function with signature ``scorer(estimator, X, y)``.
        :param cv: int (default: 5). Scikit-learn cross-validation generator or `int`, If estimator is a classifier (or y consists of integer class labels), stratified k-fold is performed, and regular k-fold cross-validation otherwise. No cross-validation if cv is None, False, or 0.
        :param n_jobs: int (default: 1). The number of CPUs to use for evaluating different feature subsets in parallel. -1 means 'all CPUs'.
        :param pre_dispatch: int, or string (default: '2*n_jobs'). Controls the number of jobs that get dispatched during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
        """
        super().__init__()
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
    
    def _validate_params(self, x, y):
        check_X_y(x, y, estimator=self.estimator)
        _, n_features = x.shape
        if not isinstance(self.min_features, int) or (self.max_features > n_features or self.max_features < 1):
            raise AttributeError("max_features must be smaller than %d and larger than 0" % (n_features + 1))
        if not isinstance(self.min_features, int) or (self.min_features > n_features or self.min_features < 1):
            raise AttributeError("min_features must be smaller than %d and larger than 0" % (n_features + 1))
        
        if self.max_features < self.min_features:
            raise AttributeError("min_features must be less equal than max_features")
        return x, y
    
    @staticmethod
    def _calc_score(estimator, x, y, indices, groups=None, scoring=None, cv=None, **fit_params):
        _, n_features = x.shape
        mask = np.in1d(np.arange(n_features), indices)
        x = x[:, mask]
        
        if cv is None:
            try:
                estimator.fit(x, y, **fit_params)
            except:
                scores = np.nan
            else:
                scores = _score(estimator, x, y, scoring)
            
            scores = np.asarray([scores], dtype=np.float64)
        else:
            scores = cross_val_score(estimator, x, y, groups=groups, cv=cv, scoring=scoring, n_jobs=None, pre_dispatch='2*n_jobs', error_score=np.nan, fit_params=fit_params)
        
        return mask, scores

    @staticmethod
    def ncr(n, r):
        """Return the number of combinations of length r from n items.

        :param n: int, Total number of items
        :param r: int, Number of items to select from n
        :return: Number of combinations, integer
        """
        r = min(r, n - r)
        if r == 0:
            return 1
        numerator = reduce(operator.mul, range(n, n - r, -1))
        denominator = reduce(operator.mul, range(1, r + 1))
        return numerator // denominator

    @staticmethod
    def _calc_confidence(scores, confidence=0.95):
        std_err = sem(scores)
        bound = std_err * t._ppf((1 + confidence) / 2.0, len(scores))
        return bound, std_err

    def fit(self, X, y, groups=None, **fit_params):
        """Perform feature selection and learn model from training data.

        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples, ), Target values.
        :param groups: array-like of shape (n_samples,), Group labels for the samples used while splitting the dataset into train/test set. Passed to the fit method of the cross-validator.
        :param fit_params: dict, Parameters to pass to the fit method of classifier
        :return: ExhaustiveFeatureSelector
        """
        X, y = self._validate_params(X, y)
        _, n_features = X.shape
        min_features, max_features = self.min_features, self.max_features
        candidates = chain.from_iterable(combinations(range(n_features), r=i) for i in range(min_features, max_features + 1))
        # chain has no __len__ method
        n_combinations = sum(self.ncr(n=n_features, r=i) for i in range(min_features, max_features + 1))

        estimator = self.estimator
        scoring = check_scoring(estimator, self.scoring)
        cv = self.cv
        n_jobs = self.n_jobs
        pre_dispatch = self.pre_dispatch
        parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        work = enumerate(parallel(delayed(self._calc_score)(clone(estimator), X, y, c, groups=groups, scoring=scoring, cv=cv, **fit_params) for c in candidates))
        
        subset_info = []
        append_subset_info = subset_info.append
        try:
            for iteration, (mask, cv_scores) in work:
                avg_score = np.nanmean(cv_scores).item()
                append_subset_info({"support_mask": mask, "cv_scores": cv_scores, "avg_score": avg_score})
                if self.verbose:
                    print("Feature set: %d/%d, avg score: %.3f" % (iteration + 1, n_combinations, avg_score))
        except KeyboardInterrupt:
            print("Stopping early due to keyboard interrupt...")
        finally:
            max_score = float("-inf")
            best_idx, best_info = -1, {}
            for i, info in enumerate(subset_info):
                if info["avg_score"] > max_score:
                    max_score = info["avg_score"]
                    best_idx, best_info = i, info
            score = max_score
            mask = best_info["support_mask"]
            self.subset_info_ = subset_info
            self.support_mask_ = mask
            self.best_idx_ = best_idx
            self.best_score_ = score
            self.best_feature_indices_ = np.where(mask)[0]
            return self
    
    def _get_support_mask(self):
        check_is_fitted(self, "support_mask_")
        return self.support_mask_


class BorutaSelector(SelectorMixin):

    def __init__(self):
        # 对原始特征进行复制一份，并且将其按行进行随机打乱，称为Shadow Feature。将Shadow Feature与原始特征Real Feature进行横向拼接在一起，使用某种模型（随机森林、GBDT）进行计算特征重要性。将Shadow Feature中重要性最高的值为基准，删除Real Feature中重要性低于其的特征。多重复几个迭代。（一般来说随机生成的特征效果不如原始的，因此可以以Shadow Feature的特征重要性作为基准来判断Real Feature的好坏）
        super().__init__()


class MICSelector(SelectorMixin):
    pass


class FeatureImportanceSelector(SelectorMixin):
    pass


class StabilitySelector(SelectorMixin):
    pass


class REFSelector(SelectorMixin):
    pass


class SequentialFeatureSelector(SelectorMixin):
    pass


# class SelectFromModel(SelectorMixin):
#     pass
