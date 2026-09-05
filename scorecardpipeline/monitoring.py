# -*- coding: utf-8 -*-
"""模型监控与漂移检测模块。

提供评分卡模型上线后的稳定性监控能力，包括：
- 评分分布 PSI 监控
- 特征 PSI / CSI 监控
- 模型性能（KS/AUC）衰减监控
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .processing import Combiner


def _psi(expected_pct: pd.Series, actual_pct: pd.Series) -> float:
    """计算 PSI，要求两个 Series 的索引一致且和为 1。"""
    expected_pct = expected_pct.copy()
    actual_pct = actual_pct.copy()
    expected_pct[expected_pct == 0] = 1e-10
    actual_pct[actual_pct == 0] = 1e-10
    diff = actual_pct - expected_pct
    return np.sum(diff * np.log(actual_pct / expected_pct))


class ModelMonitor:
    """评分卡模型监控器。

    支持评分分布 PSI、特征 PSI、模型性能（KS/AUC）衰减监控，
    帮助识别模型上线后的稳定性问题。

    Parameters
    ----------
    combiner : Combiner, optional
        训练好的分箱器，用于特征 PSI 计算。如不提供，则基于 reference 数据自动分箱。
    score_bins : int, optional
        评分分箱数量，默认 10。

    Examples
    --------
    >>> from scorecardpipeline import ModelMonitor
    >>> monitor = ModelMonitor(score_bins=10)
    >>> monitor.fit_reference(train, score_train, y_true=y_train)
    >>> # 评分分布 PSI
    >>> psi = monitor.score_psi(score_test)
    >>> # 特征 PSI
    >>> feature_psi = monitor.feature_psi(test)
    >>> # 性能衰减
    >>> metrics = monitor.performance_decay(score_test, y_test)
    >>> # 完整监控报告
    >>> report = monitor.monitor_report(test, score_test, y_test)
    """

    def __init__(self, combiner: Optional[Combiner] = None, score_bins: int = 10):
        self.combiner = combiner
        self.score_bins = score_bins
        self.reference_ = None
        self.reference_score_ = None
        self.feature_bins_ = {}
        self.score_breaks_ = None

    def _discretize_score(self, score: np.ndarray) -> pd.Series:
        """将评分离散化为分箱索引。"""
        if self.score_breaks_ is None:
            raise ValueError("请先调用 fit_reference 建立基准。")
        return pd.cut(score, bins=self.score_breaks_, labels=False, include_lowest=True)

    def fit_reference(self, X: pd.DataFrame, score: np.ndarray, y_true: Optional[np.ndarray] = None):
        """建立监控基准。

        :param X: 基准样本特征 DataFrame。
        :param score: 基准样本的模型评分（预测概率或标准评分均可）。
        :param y_true: 基准样本的真实标签，可选。
        """
        self.reference_ = X.copy()
        self.reference_score_ = np.asarray(score)
        if y_true is not None:
            self.reference_y_true_ = np.asarray(y_true)

        # 评分分箱边界
        self.score_breaks_ = np.linspace(score.min(), score.max(), self.score_bins + 1)

        # 特征分箱边界（数值型用等频，类别型用众数）
        numeric_cols = X.select_dtypes(include="number").columns
        for col in numeric_cols:
            self.feature_bins_[col] = pd.qcut(X[col], q=10, duplicates="drop").cat.categories

    def _compute_distribution(self, values: pd.Series, bins) -> pd.Series:
        """计算分箱占比。"""
        if pd.api.types.is_categorical_dtype(values) or values.dtype == object:
            pct = values.value_counts(normalize=True).sort_index()
        else:
            pct = pd.cut(values, bins=bins).value_counts(normalize=True).sort_index()
        return pct

    def score_psi(self, score: np.ndarray) -> float:
        """计算评分分布 PSI。"""
        reference_pct = (
            pd.Series(self._discretize_score(self.reference_score_)).value_counts(normalize=True).sort_index()
        )
        actual_pct = pd.Series(self._discretize_score(np.asarray(score))).value_counts(normalize=True).sort_index()
        # 补齐缺失分箱
        all_bins = reference_pct.index.union(actual_pct.index)
        reference_pct = reference_pct.reindex(all_bins, fill_value=0)
        actual_pct = actual_pct.reindex(all_bins, fill_value=0)
        return _psi(reference_pct, actual_pct)

    def feature_psi(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算各特征的 PSI。

        :return: DataFrame，列为 feature 和 psi。
        """
        if self.reference_ is None:
            raise ValueError("请先调用 fit_reference 建立基准。")
        results = []
        for col in X.select_dtypes(include="number").columns:
            if col not in self.reference_:
                continue
            ref_pct = self._compute_distribution(self.reference_[col], self.feature_bins_.get(col))
            act_pct = self._compute_distribution(X[col], self.feature_bins_.get(col))
            all_bins = ref_pct.index.union(act_pct.index)
            ref_pct = ref_pct.reindex(all_bins, fill_value=0)
            act_pct = act_pct.reindex(all_bins, fill_value=0)
            results.append({"feature": col, "psi": _psi(ref_pct, act_pct)})
        return pd.DataFrame(results)

    def performance_decay(self, score: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """对比当前数据与基准数据的模型性能差异。

        :return: dict，包含当前 KS、AUC 及与基准的差异。
        """
        from toad.metrics import AUC, KS

        metrics = {
            "current_ks": KS(score, y_true),
            "current_auc": AUC(score, y_true),
        }
        if hasattr(self, "reference_y_true_"):
            metrics["reference_ks"] = KS(self.reference_score_, self.reference_y_true_)
            metrics["reference_auc"] = AUC(self.reference_score_, self.reference_y_true_)
            metrics["ks_decay"] = metrics["reference_ks"] - metrics["current_ks"]
            metrics["auc_decay"] = metrics["reference_auc"] - metrics["current_auc"]
        return metrics

    def monitor_report(self, X: pd.DataFrame, score: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict:
        """生成完整监控报告。

        :return: dict，包含 score_psi、feature_psi、performance_decay。
        """
        report = {
            "score_psi": self.score_psi(score),
            "feature_psi": self.feature_psi(X),
        }
        if y_true is not None:
            report["performance_decay"] = self.performance_decay(score, y_true)
        return report
