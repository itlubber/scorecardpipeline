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


def _check_array_compat(X, accept_sparse=False, dtype="numeric", copy=False, force_all_finite=True):
    """兼容 sklearn 1.5/1.6 的 check_array 包装。"""
    import inspect

    sig = inspect.signature(check_array)
    all_finite_param = "ensure_all_finite" if "ensure_all_finite" in sig.parameters else "force_all_finite"
    kwargs = {"accept_sparse": accept_sparse, "dtype": dtype, "copy": copy, all_finite_param: force_all_finite}
    return check_array(X, **kwargs)


class BaseScoreTransformer(BaseEstimator, TransformerMixin):
    def _validate_data(self, X, reset=False, accept_sparse=False, dtype="numeric", copy=False, force_all_finite=True):
        """兼容不同 sklearn 版本的输入校验。"""
        return _check_array_compat(
            X, accept_sparse=accept_sparse, dtype=dtype, copy=copy, force_all_finite=force_all_finite
        )

    def __init__(self, down_lmt=300, up_lmt=1000, greater_is_better=True, cutoff=None):
        """评分转换器基类，将模型预测概率转换为标准评分

        所有评分转换器均继承自此类，提供分数裁剪、截断和 cutoff 判定等通用功能。
        子类需要实现 ``predict`` 方法和 ``_transform`` 方法。

        :param down_lmt: 分数下限，默认 300
        :param up_lmt: 分数上限，默认 1000
        :param greater_is_better: 分数越高是否代表客户越优质，默认 True。
            True 表示分数越高客户越优质（低风险），False 表示分数越低客户越优质
        :param cutoff: 决策截断点，默认为 None（自动以 0.5 概率对应的分数作为 cutoff）
        """
        self.down_lmt = down_lmt
        self.up_lmt = up_lmt
        self.greater_is_better = greater_is_better
        self.cutoff = cutoff
        self.fitted_ = False

    def __sklearn_is_fitted__(self):
        """sklearn 检查是否已拟合"""
        return self.fitted_

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
    """将模型预测概率转换为标准分布评分的转换器

    基于 PDO（Points to Double the Odds）公式，将模型的预测概率映射到指定区间的评分。
    评分越高代表客户越优质（低违约风险），分数范围受 down_lmt / up_lmt 控制。

    **评分公式**::

        score = A - sgn * B * ln(odds)
        其中 odds = p / (1 - p)，p 为预测违约概率
        sgn = -1 (greater_is_better=True) 或 sgn = 1 (greater_is_better=False)
        A = base_score + sgn * B * ln(base_odds)
        B = pdo / ln(rate)

    **参考样例**

    >>> import numpy as np
    >>> from scorecardpipeline.scorecard import StandardScoreTransformer
    >>> # 生成模拟的违约概率
    >>> proba = np.array([[0.05], [0.15], [0.30], [0.50]])
    >>> transformer = StandardScoreTransformer(base_score=660, pdo=75, rate=2, bad_rate=0.15)
    >>> transformer.fit(proba)
    >>> scores = transformer.transform(proba)
    >>> transformer.scorecard_scale()
    >>> transformer.predict(proba)  # 二分类决策（基于 cutoff）
    >>> # 分数反推概率
    >>> transformer.inverse_transform(scores)
    """

    def __init__(
        self,
        base_score=660,
        pdo=75,
        rate=2,
        bad_rate=0.15,
        down_lmt=300,
        up_lmt=1000,
        greater_is_better=True,
        cutoff=None,
    ):
        """标准评分转换器

        :param base_score: 基础分数，当 bad_rate 对应的 odds 时的评分，默认 660
        :param pdo: Points to Double the Odds，odds 每增长 rate 倍分数增长的绝对值，默认 75
        :param rate: odds 增长的倍率，默认 2，即 odds 每翻倍，分数增长 pdo 分
        :param bad_rate: 基准违约率，用于计算 base_odds，默认 0.15，即 base_odds = 0.15/0.85
        :param down_lmt: 分数下限，默认 300
        :param up_lmt: 分数上限，默认 1000
        :param greater_is_better: 分数越高是否代表客户越优质，默认 True
        :param cutoff: 决策截断点，默认为 None（自动以 0.5 概率对应的分数作为 cutoff）
        """
        super().__init__(down_lmt=down_lmt, up_lmt=up_lmt, greater_is_better=greater_is_better, cutoff=cutoff)
        self.base_score = base_score
        self.pdo = pdo
        self.rate = rate
        self.bad_rate = bad_rate

    def fit(self, X, y=None, **fit_params):
        """训练标准评分转换器，计算评分公式参数 A、B

        :param X: 训练数据，通常为模型预测的违约概率（shape: [n_samples, n_features]）
        :param y: 目标变量，此处不使用，仅为 sklearn 接口兼容性
        :param fit_params: 其他拟合参数
        :return: self，训练完成的 StandardScoreTransformer
        """
        self._validate_data(X, reset=True, accept_sparse=False, dtype="numeric", copy=False, force_all_finite=True)

        base_score, down_lmt, up_lmt = self.base_score, self.down_lmt, self.up_lmt
        if not down_lmt <= base_score <= up_lmt:
            raise ValueError(f"base_score should be greater than {down_lmt} and less than {up_lmt}!")

        bad_rate = self.bad_rate
        if not 0.0 <= bad_rate <= 1.0:
            raise ValueError("bad rate should be greater than e and less than 1!")

        base_odds = bad_rate / (1.0 - bad_rate)
        B = self.pdo / np.log(self.rate)
        if self.greater_is_better:
            sgn = -1
        else:
            sgn = 1
        A = base_score + sgn * B * np.log(base_odds)

        self.A_ = A
        self.B_ = B
        self.sgn_ = sgn
        self.base_odds = base_odds
        self.fitted_ = True
        return self

    def scorecard_scale(self):
        """输出评分卡基准信息，包含 base_odds、base_score、rate、pdo、A、B

        :return: pd.DataFrame，评分卡基准信息
        """
        scorecard_kedu = pd.DataFrame(
            [
                [
                    "base_odds",
                    self.base_odds,
                    "根据业务经验设置的基础比率（违约概率/正常概率），估算方法：坏客户占比 / (1 - 样本坏客户占比)",
                ],
                ["base_score", self.base_score, "基础ODDS对应的分数"],
                ["rate", self.rate, "设置分数的倍率"],
                ["pdo", self.pdo, "表示分数增长PDO时，ODDS值增长到RATE倍"],
                ["B", self.B_, "刻度，计算方式：pdo / ln(rate)"],
                [
                    "A",
                    self.A_,
                    "补偿值，计算方式：base_score - sgn * B * ln(base_odds)，其中 greater_is_better=True 时 sgn=-1，False 时 sgn=1",
                ],
                [
                    "score",
                    f"{self.A_:.4f} {'+' if self.sgn_ == -1 else '-'} {self.B_:.4f} * ln(odds)",
                    f"评分公式：greater_is_better={self.greater_is_better_}，分数越高客户越{'优质' if self.greater_is_better_ else '劣质'}",
                ],
            ],
            columns=["刻度项", "刻度值", "备注"],
        )
        return scorecard_kedu

    def _transform(self, X):
        """评分转换内部方法，将概率转换为分数

        :param X: 概率数组（已校验）
        :return: 分数数组
        """
        check_is_fitted(self, ["A_", "B_", "sgn_"])
        Xt = self._validate_data(X, reset=False, accept_sparse=False, dtype="numeric", copy=True, force_all_finite=True)
        # if not np.all((0 <= Xt) & (Xt <= 1)):
        #     raise ValueError ("Input should be probabilities between 0 and 1.")
        A, B, sgn = self.A_, self.B_, self.sgn_
        down_lmt, up_lmt = self.down_lmt, self.up_lmt
        points = A - sgn * B * np.log(Xt / (1.0 - Xt))
        points = np.clip(points, down_lmt, up_lmt)
        return points

    def transform(self, X):
        """将概率转换为分数

        :param X: 概率数组，元素值应在 (0, 1) 区间
        :return: 分数数组，与输入形状相同
        """
        data = self._transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, columns=columns, index=index)
        return data

    def predict(self, X):
        """基于 cutoff 阈值进行二分类决策

        :param X: 概率数组
        :return: np.ndarray，二分类标签（0 或 1）
        """
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
        """分数反推概率的内部方法

        :param X: 分数数组（已校验）
        :return: 概率数组
        """
        check_is_fitted(self, ["A_", "B_", "sgn_"])
        Xt = _check_array_compat(X, accept_sparse=False, dtype="numeric", copy=True, force_all_finite=True)
        down_lmt, up_lmt = self.down_lmt, self.up_lmt
        if not np.all(np.logical_and((down_lmt <= Xt), (Xt <= up_lmt))):
            raise ValueError(f"Input should be points between {down_lmt} and {up_lmt}")
        A, B, sgn = self.A_, self.B_, self.sgn_
        probs = 1.0 - 1.0 / (np.exp((A - Xt) / (sgn * B)) + 1.0)
        return probs

    def inverse_transform(self, X):
        """将分数反推为概率

        :param X: 分数数组，元素值应在 [down_lmt, up_lmt] 区间
        :return: 概率数组
        """
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
    """标准评分转换器，输出非四舍五入的浮点分数

    继承自 ``StandardScoreTransformer``，区别在于 ``_transform`` 使用
    ``np.round`` 进行截断处理，而非 Python 原生的 round 函数。
    其他行为与 ``StandardScoreTransformer`` 完全一致。

    **参考样例**

    >>> import numpy as np
    >>> from scorecardpipeline.scorecard import NPRoundStandardScoreTransformer
    >>> proba = np.array([[0.05], [0.15], [0.30], [0.50]])
    >>> transformer = NPRoundStandardScoreTransformer(base_score=660, pdo=75, bad_rate=0.15)
    >>> transformer.fit(proba)
    >>> transformer.transform(proba)
    """

    def __init__(
        self,
        base_score=660,
        pdo=75,
        bad_rate=0.15,
        down_lmt=300,
        up_lmt=1000,
        round_decimals=0,
        greater_is_better=True,
        cutoff=None,
    ):
        """标准评分转换器（非四舍五入版）

        :param base_score: 基础分数，默认 660
        :param pdo: Points to Double the Odds，默认 75
        :param bad_rate: 基准违约率，默认 0.15
        :param down_lmt: 分数下限，默认 300
        :param up_lmt: 分数上限，默认 1000
        :param round_decimals: 小数位数，使用 np.round 进行截断，默认 0（即整数）
        :param greater_is_better: 分数越高是否代表客户越优质，默认 True
        :param cutoff: 决策截断点，默认 None
        """
        self.round_decimals = round_decimals
        super().__init__(
            base_score=base_score,
            pdo=pdo,
            bad_rate=bad_rate,
            down_lmt=down_lmt,
            up_lmt=up_lmt,
            greater_is_better=greater_is_better,
            cutoff=cutoff,
        )

    def _transform(self, X):
        """评分转换，使用 np.round 截断

        :param X: 概率数组
        :return: 截断后的分数数组
        """
        points = super()._transform(X)
        decimals = self.round_decimals
        points = np.round(points, decimals=decimals)
        return points


class RoundStandardScoreTransformer(StandardScoreTransformer):
    """标准评分转换器，输出四舍五入的整数分数

    继承自 ``StandardScoreTransformer``，区别在于 ``_transform`` 使用
    Python 原生 ``round`` 函数进行四舍五入，而非 ``np.round`` 截断。
    其他行为与 ``StandardScoreTransformer`` 完全一致。

    **参考样例**

    >>> import numpy as np
    >>> from scorecardpipeline.scorecard import RoundStandardScoreTransformer
    >>> proba = np.array([[0.05], [0.15], [0.30], [0.50]])
    >>> transformer = RoundStandardScoreTransformer(base_score=660, pdo=75, bad_rate=0.15, round_decimals=0)
    >>> transformer.fit(proba)
    >>> transformer.transform(proba)  # 输出整数评分
    """

    def __init__(
        self,
        base_score=660,
        pdo=75,
        bad_rate=0.15,
        down_lmt=300,
        up_lmt=1000,
        round_decimals=0,
        greater_is_better=True,
        cutoff=None,
    ):
        """标准评分转换器（四舍五入版）

        :param base_score: 基础分数，默认 660
        :param pdo: Points to Double the Odds，默认 75
        :param bad_rate: 基准违约率，默认 0.15
        :param down_lmt: 分数下限，默认 300
        :param up_lmt: 分数上限，默认 1000
        :param round_decimals: 小数位数，使用 round 进行四舍五入，默认 0（即整数）
        :param greater_is_better: 分数越高是否代表客户越优质，默认 True
        :param cutoff: 决策截断点，默认 None
        """
        self.round_decimals = round_decimals
        super().__init__(
            base_score=base_score,
            pdo=pdo,
            bad_rate=bad_rate,
            down_lmt=down_lmt,
            up_lmt=up_lmt,
            greater_is_better=greater_is_better,
            cutoff=cutoff,
        )

    def _transform(self, X):
        """评分转换，使用 round 进行四舍五入

        :param X: 概率数组
        :return: 四舍五入后的分数数组
        """
        points = super()._transform(X)
        decimals = self.round_decimals
        points = np.array([[round(x[0], decimals)] for x in points])
        return points


class BoxCoxScoreTransformer(BaseScoreTransformer):
    """基于 Box-Cox 变换的概率转评分转换器

    使用 Box-Cox 变换将概率分布转换为正态分布，再通过 MinMaxScaler 缩放到指定分数区间。
    与 StandardScoreTransformer 不同，Box-Cox 变换无需预设 bad_rate，可自动学习最优变换参数。

    **评分公式**::

        x' = boxcox(x, lambda)  # 自动学习的 lambda 参数
        score = scaler(x')      # MinMaxScaler 缩放到 [down_lmt, up_lmt]

    **参考样例**

    >>> import numpy as np
    >>> from scorecardpipeline.scorecard import BoxCoxScoreTransformer
    >>> proba = np.array([[0.05], [0.15], [0.30], [0.50], [0.80]])
    >>> transformer = BoxCoxScoreTransformer(down_lmt=300, up_lmt=1000, greater_is_better=True)
    >>> transformer.fit(proba)
    >>> scores = transformer.transform(proba)
    >>> transformer.predict(proba)  # 二分类决策
    >>> transformer.inverse_transform(scores)  # 分数反推概率
    """

    def __init__(self, down_lmt=300, up_lmt=1000, greater_is_better=True, cutoff=None):
        """Box-Cox 评分转换器

        :param down_lmt: 分数下限，默认 300
        :param up_lmt: 分数上限，默认 1000
        :param greater_is_better: 分数越高是否代表客户越优质，默认 True
        :param cutoff: 决策截断点，默认 None
        """
        super().__init__(down_lmt=down_lmt, up_lmt=up_lmt, greater_is_better=greater_is_better, cutoff=cutoff)

    @staticmethod
    def _box_cox_optimize(x):
        """使用 MLE 找到 Box-Cox 变换的最优 lambda 参数

        使用 scipy.stats.boxcox 的 brent 优化器求解。
        注意：NaN 值会影响结果，调用前需确保数据中无 NaN。

        :param x: 一维数组，元素值必须严格大于 0
        :return: float，最优 lambda 值
        """
        # the computation of Lambda is influenced by NaNs so we need to get rid of them
        _, lmbda = stats.boxcox(x, lmbda=None)
        return lmbda

    def fit(self, X, y=None, **fit_params):
        """训练 Box-Cox 评分转换器，学习每列的最优 lambda 参数

        :param X: 训练数据，违约概率数组，元素值须严格在 (0, 1) 区间
        :param y: 目标变量，此处不使用，仅为 sklearn 接口兼容性
        :param fit_params: 其他拟合参数
        :return: self，训练完成的 BoxCoxScoreTransformer
        """
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
        self.fitted_ = True
        return self

    def _transform(self, X):
        """评分转换内部方法，将概率通过 Box-Cox 变换转换为分数

        :param X: 概率数组（已校验，元素值须在 (0, 1) 区间）
        :return: 分数数组
        """
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
        """将概率转换为分数

        :param X: 概率数组，元素值须严格在 (0, 1) 区间
        :return: 分数数组，与输入形状相同
        """
        data = self._transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, index=index, columns=columns)
        return data

    def predict(self, X):
        """基于 cutoff 阈值进行二分类决策

        :param X: 概率数组
        :return: np.ndarray，二分类标签（0 或 1）
        """
        scores = np.ravel(self._transform(X))
        if self.cutoff is None:
            lmbda = self.lambdas_[0]
            if lmbda != 0:
                p = (0.5**lmbda - 1) / lmbda
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
        """分数反推概率的内部方法

        :param X: 分数数组（已校验，元素值须在 [down_lmt, up_lmt] 区间）
        :return: 概率数组
        """
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
        """将分数反推为概率

        :param X: 分数数组，元素值须在 [down_lmt, up_lmt] 区间
        :return: 概率数组
        """
        data = self._inverse_transform(X)
        if isinstance(X, DataFrame):
            columns = X.columns
            index = X.index
            return DataFrame(data=data, index=index, columns=columns)
        return data

    @staticmethod
    def _box_cox_inverse_tranform(x, lmbda):
        """Box-Cox 逆变换

        :param x: 变换后的值
        :param lmbda: Box-Cox 变换参数
        :return: 原始值
        """
        if lmbda == 0:
            x_inv = np.exp(x)
        else:
            x_inv = (x * lmbda + 1) ** (1 / lmbda)

        return x_inv


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    import h2o

    from scorecardpipeline import *

    h2o.init()

    test_select = h2o.H2OFrame(
        load_pickle(
            "/Users/lubberit/Desktop/workspace/scorecardpipeline/examples/model_report/h2o_model/test_select.pkl"
        )
    )

    model_path = "/Users/lubberit/Desktop/workspace/scorecardpipeline/examples/model_report/h2o_model/StackedEnsemble_BestOfFamily_1_AutoML_1_20240415_162619"
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
