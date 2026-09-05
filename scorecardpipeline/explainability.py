# -*- coding: utf-8 -*-
"""评分卡模型解释性模块。

支持对 WOE-LR 评分卡模型进行 SHAP 解释，并将 WOE 空间的归因映射回
原始特征空间，输出单样本评分解释。
"""

import numpy as np
import pandas as pd

try:
    import shap
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    shap = None
    LogisticRegression = None  # type: ignore

from .processing import Combiner, WOETransformer


class ScorecardExplainer:
    """评分卡模型解释器。

    针对 ``Combiner`` + ``WOETransformer`` + ``LogisticRegression`` 的
    经典评分卡流程，提供基于 SHAP 的特征归因。

    Parameters
    ----------
    model : sklearn-like estimator
        训练好的逻辑回归模型（如 ``ITLubberLogisticRegression``）。
    combiner : Combiner
        训练好的分箱器。
    woe_transformer : WOETransformer
        训练好的 WOE 转换器。
    feature_names : list[str], optional
        原始特征名，默认从 ``combiner`` 推导。

    Examples
    --------
    >>> from scorecardpipeline import ScorecardExplainer, ITLubberLogisticRegression
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ("combiner", Combiner(target="creditability")),
    ...     ("woe", WOETransformer(target="creditability")),
    ... ])
    >>> woe_train = pipeline.fit_transform(train)
    >>> model = ITLubberLogisticRegression(target="creditability")
    >>> model.fit(woe_train)
    >>> explainer = ScorecardExplainer(
    ...     model=model,
    ...     combiner=pipeline.named_steps["combiner"],
    ...     woe_transformer=pipeline.named_steps["woe"],
    ... )
    >>> # 批量 SHAP 解释
    >>> shap_df = explainer.explain(test)
    >>> # 单样本解释
    >>> sample_explanation = explainer.explain_sample(test, index=0)
    >>> # 特征重要度汇总
    >>> importance = explainer.summary(test)
    """

    def __init__(self, model, combiner: Combiner, woe_transformer: WOETransformer, feature_names=None):
        if shap is None:  # pragma: no cover
            raise ImportError(
                "ScorecardExplainer requires 'shap'. " "Install with: pip install scorecardpipeline[explain]"
            )
        self.model = model
        self.combiner = combiner
        self.woe_transformer = woe_transformer
        self.feature_names = feature_names or list(combiner.combiner.rules.keys())

    def _get_woe_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """将原始特征转换为 WOE 特征。"""
        binned = self.combiner.transform(X)
        woe = self.woe_transformer.transform(binned)
        # 仅保留建模时使用的 WOE 特征
        woe_features = [c for c in self.feature_names if c in woe.columns and c != "target"]
        return woe[woe_features]

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """计算样本的 SHAP 解释值。

        :param X: 原始特征 DataFrame（需包含建模特征）。
        :return: DataFrame，列为原始特征，值为每个样本对该特征的 SHAP 贡献。
        """
        woe_X = self._get_woe_features(X)
        explainer = shap.Explainer(self.model, woe_X.values)
        shap_values = explainer(woe_X.values)
        return pd.DataFrame(shap_values.values, columns=woe_X.columns, index=woe_X.index)

    def explain_sample(self, X: pd.DataFrame, index: int = 0) -> pd.DataFrame:
        """解释单个样本，输出每个原始特征的 SHAP 贡献。

        :param X: 原始特征 DataFrame。
        :param index: 样本索引。
        :return: DataFrame，包含 feature、shap_value、feature_value 三列。
        """
        shap_df = self.explain(X)
        sample_shap = shap_df.iloc[index]
        sample_raw = X.iloc[index]
        result = pd.DataFrame(
            {
                "feature": sample_shap.index,
                "shap_value": sample_shap.values,
                "feature_value": [sample_raw.get(c, np.nan) for c in sample_shap.index],
            }
        )
        return result.sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)

    def summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """汇总特征重要度（按 SHAP 绝对值均值排序）。"""
        shap_df = self.explain(X)
        importance = shap_df.abs().mean().sort_values(ascending=False)
        return pd.DataFrame(
            {
                "feature": importance.index,
                "mean_abs_shap": importance.values,
            }
        ).reset_index(drop=True)
