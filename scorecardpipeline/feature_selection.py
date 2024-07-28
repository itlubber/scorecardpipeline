# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/8 14:06
@Author  : itlubber
@Site    : itlubber.art
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from abc import ABCMeta, abstractmethod
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._mask import _get_mask
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV, RFE, SelectFromModel, SelectKBest
from sklearn.model_selection import StratifiedKFold, GroupKFold

from .processing import Combiner


class SelectorMixin(BaseEstimator, TransformerMixin):

    def transform(self, x):
        check_is_fitted(self, "select_columns")
        return x[[col for col in self.select_columns if col in x.columns]]
    
    def __call__(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.select_columns

    @staticmethod
    def _calculate_threshold(estimator, importances, threshold):
        if threshold is None:
            # determine default from estimator
            est_name = estimator.__class__.__name__
            is_l1_penalized = hasattr(estimator, "penalty") and estimator.penalty == "l1"
            is_lasso = "Lasso" in est_name
            is_elasticnet_l1_penalized = "ElasticNet" in est_name and (
                (hasattr(estimator, "l1_ratio_") and np.isclose(estimator.l1_ratio_, 1.0))
                or (hasattr(estimator, "l1_ratio") and np.isclose(estimator.l1_ratio, 1.0))
            )
            if is_l1_penalized or is_lasso or is_elasticnet_l1_penalized:
                # the natural default threshold is 0 when l1 penalty was used
                threshold = 1e-5
            else:
                threshold = "mean"

        if isinstance(threshold, str):
            if "*" in threshold:
                scale, reference = threshold.split("*")
                scale = float(scale.strip())
                reference = reference.strip()

                if reference == "median":
                    reference = np.median(importances)
                elif reference == "mean":
                    reference = np.mean(importances)
                else:
                    raise ValueError("Unknown reference: " + reference)

                threshold = scale * reference

            elif threshold == "median":
                threshold = np.median(importances)

            elif threshold == "mean":
                threshold = np.mean(importances)

            else:
                raise ValueError("Expected threshold='mean' or threshold='median' got %s" % threshold)

        else:
            threshold = float(threshold)

        return threshold


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
        self.threshold = self._calculate_threshold(self, self.scores_, self.threshold)
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
        self.threshold = self._calculate_threshold(self, self.scores_, self.threshold)
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
        self.threshold = self._calculate_threshold(self, self.scores_, self.threshold)
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
        self.threshold = self._calculate_threshold(self, self.scores_, self.threshold)
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
        self.threshold = self._calculate_threshold(self, self.scores_, self.threshold)
        self.select_columns = list(set((self.scores_[self.scores_ >= self.threshold]).index.tolist() + [self.target]))
        self.dropped = pd.DataFrame([(col, f"LIFT < {self.threshold}") for col in xt.columns if col not in self.select_columns], columns=["variable", "rm_reason"])
        return self
