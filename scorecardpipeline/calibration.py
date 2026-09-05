# -*- coding: utf-8 -*-
"""评分卡概率校准模块。

提供 Platt Scaling 和 Isotonic Regression 两种经典校准方法，
用于缓解评分卡模型预测概率与真实概率之间的系统性偏差。
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator(BaseEstimator, TransformerMixin):
    """概率校准器，支持 Platt Scaling 和 Isotonic Regression。

    评分卡模型训练完成后，预测概率可能存在系统性偏差，
    使用 ProbabilityCalibrator 可以校正预测概率，使其更接近真实概率。

    Parameters
    ----------
    method : str, default="platt"
        校准方法，可选 ``"platt"``、``"isotonic"``。

    Examples
    --------
    >>> from scorecardpipeline import ProbabilityCalibrator
    >>> # train_proba: 训练集模型预测正类概率 (n_samples,)
    >>> # y_train: 训练集真实标签
    >>> calibrator = ProbabilityCalibrator(method="isotonic")
    >>> calibrator.fit(train_proba, y_train)
    >>> calibrated = calibrator.transform(test_proba)
    >>>
    >>> # 也支持 fit_transform 一步到位
    >>> calibrated = calibrator.fit_transform(train_proba, y_train)
    >>>
    >>> # 输入为 (n_samples, 2) 概率矩阵时同样支持
    >>> calibrated_matrix = calibrator.transform(test_proba_matrix)
    """

    def __init__(self, method="platt"):
        self.method = method
        self.calibrator_ = None

    def fit(self, proba: np.ndarray, y: np.ndarray):
        """训练校准器。

        :param proba: 模型预测的正类概率，shape ``(n_samples,)``。
        :param y: 真实标签。
        :return: self
        """
        proba = np.asarray(proba).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if self.method == "platt":
            self.calibrator_ = LogisticRegression()
            self.calibrator_.fit(proba.reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(proba, y)
        else:
            raise ValueError("method must be 'platt' or 'isotonic'")
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """校准概率。

        :param proba: 待校准概率，shape ``(n_samples,)`` 或 ``(n_samples, 2)``。
        :return: 校准后的概率，shape 与输入一致。
        """
        if self.calibrator_ is None:
            raise RuntimeError("请先调用 fit 方法。")
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            # 输入为二分类概率矩阵，取正类概率校准后再还原
            pos_proba = proba[:, 1]
            calibrated = self.transform(pos_proba)
            return np.column_stack([1 - calibrated, calibrated])
        proba = proba.reshape(-1)
        if self.method == "platt":
            calibrated = self.calibrator_.predict_proba(proba.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator_.transform(proba)
        return calibrated

    def fit_transform(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        """训练并校准概率。"""
        self.fit(proba, y)
        return self.transform(proba)
