# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/10 10:28
@Author  : itlubber
@Site    : itlubber.art
"""
import numpy as np
import numexpr as ne
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class NumExprDerive(BaseEstimator, TransformerMixin):
    """Derive features by expressions.

    **参考样例**

    >>> import pandas as pd
    >>> from scorecardpipeline.feature_engineering import NumExprDerive
    >>> X = pd.DataFrame({"f0": [2, 1.0, 3], "f1": [np.inf, 2, 3], "f2": [2, 3, 4], "f3": [2.1, 1.4, -6.2]})
    >>> fd = NumExprDerive(derivings=[("f4", "where(f1>1, 0, 1)"), ("f5"、"f1+f2"), ("f6", "sin(f1)"), ("f7", "abs(f3)"))
    >>> fd.fit_transform(X)
    """
    def __init__(self, derivings=None):
        """
        :param derivings: list, default=None. Each entry is a (name, expr) pair representing a deriving rule.
        """
        self.derivings = derivings

    def fit(self, X, y=None):
        self._check_keywords()
        self._validate_data(X, dtype=None, ensure_2d=True, force_all_finite=False)
        return self

    def _check_keywords(self):
        derivings = self.derivings
        if derivings is None:
            raise ValueError("Deriving rules should not be empty!")
        if not isinstance(derivings, list):
            raise ValueError("Deriving rules should be a list!")
        for i, entry in enumerate(derivings):
            if not isinstance(entry, tuple):
                raise ValueError("The {}-th deriving rule should be a tuple!".format(i))
            if len(entry) != 2:
                raise ValueError("The f}-th deriving rule is not a two-element (drived_name, expression) tuple!".format(i))
            name, expr = entry
            if not isinstance(name, str) or not isinstance(expr, str):
                raise ValueError("The {}-th deriving rule is not a two-string tuple!".format(i))

    @staticmethod
    def _get_context(X, feature_names=None):
        if feature_names is not None:
            return {name: X[:, i] for i, name in enumerate(feature_names)}
        return {"f%d" % i: X[:, i] for i in range(X.shape[1])}

    def _transform_frame(self, X):
        feature_names = X.columns.tolist()
        self.features_names = feature_names
        index = X.index
        X = self._validate_data(X, dtype="numeric", ensure_2d=True, force_all_finite=False)
        context = self._get_context(X, feature_names=feature_names)
        n_derived = len(self.derivings)
        X_derived = np.empty((X.shape[0], n_derived), dtype=np.float64)
        derived_names = []
        for i, (name, expr) in enumerate(self.derivings):
            derived_names.append(name)
            X_derived[:, i] = ne.evaluate(expr, local_dict=context)
        data = np.hstack((X, X_derived))
        columns = feature_names + derived_names
        return DataFrame(data=data, columns=columns, index=index)

    def _transform_ndarray(self, X):
        X = self._validate_data(X, dtype="numeric", ensure_2d=True, force_all_finite=False)
        context = self._get_context(X, feature_names=None)
        n_derived = len(self.derivings)
        X_derived = np.empty((X.shape[0], n_derived), dtype=np.float64)
        derived_names = []
        for i, (name, expr) in enumerate(self.derivings):
            derived_names.append(name)
            X_derived[:, i] = ne.evaluate(expr, local_dict=context)
        return np.hstack((X, X_derived))

    def transform(self, X):
        if isinstance(X, DataFrame):
            return self._transform_frame(X)
        return self._transform_ndarray(X)

    def _more_tags(self):
        return {
            "X_types": ["2darray"],
            "allow_nan": True,
        }
