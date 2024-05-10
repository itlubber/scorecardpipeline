# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/15 16:52
@Author  : itlubber
@Site    : itlubber.art
"""
import math
from abc import abstractmethod
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class BaseScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, down_lmt=300, up_lmt=1000, greater_is_better=True, cutoff=None):
        self.down_lmt = down_lmt
        self.up_lmt = up_lmt
        self.greater_is_better = greater_is_better
        self.cutoff = cutoff

    @abstractmethod
    def predict(self, x):
        pass

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


class StandardScoreTransformer(BaseScoreTransformer):
    """Stretch the predicted probability to a normal distributed score."""

    def __init__(self, base_score=660, pdo=75, rate=2, bad_rate=0.15, down_lmt=300, up_lmt=1000, greater_is_better=True, cutoff=None):
        super().__init__(down_lmt=down_lmt, up_lmt=up_lmt, greater_is_better=greater_is_better, cutoff=cutoff)
        self.base_score = base_score
        self.pdo = pdo
        self.rate = rate
        self.bad_rate = bad_rate

    def fit(self, X, y=None, **fit_params):
        self._validate_data(X, reset=True, accept_sparse=False, dtype="numeric", copy=False, force_all_finite=True)

        base_score, down_lmt, up_lmt = self.base_score, self.down_lmt, self.up_lmt
        if not down_lmt <= base_score <= up_lmt:
            raise ValueError("base_score should be greater than {} and less than {}!".format(down_lmt, up_lmt))

        bad_rate = self.bad_rate
        if not 0.0 <= bad_rate <= 1.0:
            raise ValueError("bad rate should be greater than e and less than 1!")

        base_odds = bad_rate / (1. - bad_rate)
        if self.greater_is_better:
            B = self.pdo / np.log(self.rate)
        else:
            B = -self.pdo / np.log(self.rate)

        A = base_score + B + np.log(base_odds)

        self.A_ = A
        self.B_ = B
        self.base_odds = base_odds
        return self

    def scorecard_scale(self):
        """输出评分卡基准信息，包含 base_odds、base_score、rate、pdo、A、B

        :return: pd.DataFrame，评分卡基准信息
        """
        scorecard_kedu = pd.DataFrame(
            [
                ["base_odds", self.base_odds, "根据业务经验设置的基础比率（违约概率/正常概率），估算方法：坏客户占比 / (1 - 样本坏客户占比)"],
                ["base_score", self.base_score, "基础ODDS对应的分数"],
                ["rate", self.rate, "设置分数的倍率"],
                ["pdo", self.pdo, "表示分数增长PDO时，ODDS值增长到RATE倍"],
                ["B", self.A_, "补偿值，计算方式：pdo / ln(rate)"],
                ["A", self.B_, "刻度，计算方式：base_score - B * ln(base_odds)"],
            ],
            columns=["刻度项", "刻度值", "备注"],
        )
        return scorecard_kedu

    def _transform(self, X):
        check_is_fitted(self, ["A_", "B_"])
        Xt = self._validate_data(X, reset=False, accept_sparse=False, dtype="numeric", copy=True, force_all_finite=True)
        # if not np.all((0 <= Xt) & (Xt <= 1)):
        #     raise ValueError ("Input should be probabilities between 0 and 1.")
        A, B = self.A_, self.B_
        down_lmt, up_lmt = self.down_lmt, self.up_lmt
        points = A - B * np.log(Xt / (1.0 - Xt))
        points = np.clip(points, down_lmt, up_lmt)
        return points

    def transform(self, X):
        data = self._transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, columns=columns, index=index)
        return data

    def predict(self, X):
        scores = np.ravel(self._transform(X))
        if self.cutoff is None:
            cutoff = self._transform([[0.5]])[0][0]
        elif not self.down_lmt < self.cutoff < self.up_lmt:
            raise ValueError("Cutoff point should be within down_lmt and up_lmt!")
        else:
            cutoff = self.cutoff

        if self.greater_is_better:
            return (scores < cutoff).astype(np.int)
        else:
            return (scores > cutoff).astype(np.int)

    def _inverse_transform(self, X):
        check_is_fitted(self, ["A_", "B_"])
        Xt = check_array(X, accept_sparse=False, dtype="numeric", copy=True, force_all_finite=True)
        down_lmt, up_lmt = self.down_lmt, self.up_lmt
        if not np.all(np.logical_and((down_lmt <= Xt), (Xt <= up_lmt))):
            raise ValueError("Input should be points between {} and {}".format(down_lmt, up_lmt))
        A, B = self.A_, self.B_
        probs = 1.0 - 1.0 / (np.exp((A - Xt) / B) + 1.0)
        return probs

    def inverse_transform(self, X):
        data = self._inverse_transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, columns=columns, index=index)
        return data

    def _more_tags(self):
        return {
            "allow_nan": False,
        }


class NPRoundStandardScoreTransformer(StandardScoreTransformer):

    def __init__(self, base_score=660, pdo=75, bad_rate=0.15, down_lmt=300, up_lmt=1000, round_decimals=0, greater_is_better=True, cutoff=None):
        self.round_decimals = round_decimals
        super(NPRoundStandardScoreTransformer, self).__init__(base_score=base_score, pdo=pdo, bad_rate=bad_rate, down_lmt=down_lmt, up_lmt=up_lmt,
                                                              greater_is_better=greater_is_better, cutoff=cutoff)

    def _transform(self, X):
        points = super()._transform(X)
        decimals = self.round_decimals
        points = np.round(points, decimals=decimals)
        return points


class RoundStandardScoreTransformer(StandardScoreTransformer):
    """Stretch the predicted probability to a normal distributed score."""

    def __init__(self, base_score=660, pdo=75, bad_rate=0.15, down_lmt=300, up_lmt=1000, round_decimals=0, greater_is_better=True, cutoff=None):
        self.round_decimals = round_decimals
        super(RoundStandardScoreTransformer, self).__init__(base_score=base_score, pdo=pdo, bad_rate=bad_rate, down_lmt=down_lmt, up_lmt=up_lmt,
                                                            greater_is_better=greater_is_better, cutoff=cutoff)

    def _transform(self, X):
        points = super()._transform(X)
        decimals = self.round_decimals
        points = np.array([[round(x[0], decimals)] for x in points])
        return points


class BoxCoxScoreTransformer(BaseScoreTransformer):
    def __init__(self, down_lmt=300, up_lmt=1000, greater_is_better=True, cutoff=None):
        super().__init__(down_lmt=down_lmt, up_lmt=up_lmt, greater_is_better=greater_is_better, cutoff=cutoff)

    @staticmethod
    def _box_cox_optimize(x):
        """Find and return optimal lambda parameter of the Box-Cox transform by MLE, for observed data x.

        We here use scipy builtins which uses the brent optimizer.
        """
        # the computation of Lambda is influenced by NaNs so we need to get rid of them
        _, lmbda = stats.boxcox(x, lmbda=None)
        return lmbda

    def fit(self, X, y=None, **fit_params):
        X = check_array(X, accept_sparse=False, dtype="numeric", copy=True, force_all_finite=True)
        if np.min(X) <= 0 or np.max(X) >= 1:
            raise ValueError("The Box-Cox score transformation can only be applied to strictly positive probabilities")
        if self.greater_is_better:
            self.lambdas_ = np.array([self._box_cox_optimize(1.0 - col) for col in X.T])
        else:
            self.lambdas_ = np.array([self._box_cox_optimize(col) for col in X.T])
        for i, lmbda in enumerate(self.lambdas_):
            X[:, i] = stats.boxcox(X[:, i], lmbda)
        self.scaler_ = MinMaxScaler(feature_range=(self.down_lmt, self.up_lmt)).fit(X)
        return self

    def _transform(self, X):
        check_is_fitted(self, ["lambdas_", "scaler_"])
        X = check_array(X, accept_sparse=False, dtype="numeric", copy=True, force_all_finite=True)
        if np.min(X) < 0 or np.max(X) > 1:
            raise ValueError("The Box-Cox score transformation can only be applied to strictly positive probabilities")
        if self.greater_is_better:
            X = 1.0 - X
        for i, lmbda in enumerate(self.lambdas_):
            X[:, i] = stats.boxcox(X[:, i], lmbda)
        return self.scaler_.transform(X)

    def transform(self, X):
        data = self._transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, index=index, columns=columns)
        return data

    def predict(self, X):
        scores = np.ravel(self._transform(X))
        if self.cutoff is None:
            lmbda = self.lambdas_[0]
            if lmbda != 0:
                p = (0.5 ** lmbda - 1) / lmbda
            else:
                p = np.log(0.5)
            scaler = self.scaler_
            p *= scaler.scale_
            p += scaler.min_
            if scaler.clip:
                if p < scaler.feature_range[0]:
                    p = scaler.feature_range[0]
                elif p > scaler.feature_range[1]:
                    p = scaler.feature_range[1]
            cutoff = p
        elif not self.down_lmt < self.cutoff < self.up_lmt:
            raise ValueError("Cutoff point should be within 'down_lmt' and 'up_lmt'!")
        else:
            cutoff = self.cutoff
        if self.greater_is_better:
            return (scores < cutoff).astype(np.int)
        else:
            return (scores > cutoff).astype(np.int)

    def _inverse_transform(self, X):
        check_is_fitted(self, ["lambdas_", "scaler_"])
        X = check_array(X, accept_sparse=False, dtype="numeric", copy=True, force_all_finite=True)
        if np.min(X) < self.down_lmt or np.max(X) > self.up_lmt:
            raise ValueError("The Box-Cox score inverse transformation can only be applied to strictly bounded scores")
        X_inv = self.scaler_.inverse_transform(X)
        for i, lmbda in enumerate(self.lambdas_):
            X_inv[:, i] = self._box_cox_inverse_tranform(X_inv[:, i], lmbda)
        if self.greater_is_better:
            X_inv = 1.0 - X_inv
        return X_inv

    def inverse_transform(self, X):
        data = self._inverse_transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, index=index, columns=columns)
        return data

    @staticmethod
    def _box_cox_inverse_tranform(x, lmbda):
        """Return inverse-transformed input x following Box-Cox inverse transform with parameter lambda"""
        if lmbda == 0:
            x_inv = np.exp(x)
        else:
            x_inv = (x * lmbda + 1) ** (1 / lmbda)

        return x_inv


if __name__ == '__main__':
    import sys

    sys.path.append("../")
    from scorecardpipeline import *

    import h2o
    h2o.init()

    test_select = h2o.H2OFrame(load_pickle("/Users/lubberit/Desktop/workspace/scorecardpipeline/examples/model_report/h2o_model/test_select.pkl"))

    model_path = '/Users/lubberit/Desktop/workspace/scorecardpipeline/examples/model_report/h2o_model/StackedEnsemble_BestOfFamily_1_AutoML_1_20240415_162619'
    best_model = h2o.load_model(model_path)

    # score_transform = StandardScoreTransformer(base_score=400, pdo=50, bad_rate=test_select["target"].mean()[0], greater_is_better=True)
    score_transform = BoxCoxScoreTransformer(greater_is_better=False)
    y_pred = best_model.predict(test_select).as_data_frame()[["p1"]]
    score_transform.fit(y_pred)

    print(best_model.predict(test_select))
    score = score_transform.transform(y_pred)
    print(score)
    print(score_transform.inverse_transform(score))
    # print(score_transform.scorecard_scale())
