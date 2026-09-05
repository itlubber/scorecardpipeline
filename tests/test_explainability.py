# -*- coding: utf-8 -*-
"""Tests for scorecardpipeline.explainability."""

import pandas as pd
import pytest

from scorecardpipeline.explainability import ScorecardExplainer
from scorecardpipeline.model import ITLubberLogisticRegression

try:
    import shap  # noqa: F401

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


@pytest.mark.skipif(not HAS_SHAP, reason="shap not installed")
class TestScorecardExplainer:
    @pytest.fixture
    def trained_explainer(self, woe_pipeline, train_test_data):
        train, _ = train_test_data
        woe_train = woe_pipeline.transform(train)
        model = ITLubberLogisticRegression(target="creditability")
        model.fit(woe_train)
        return ScorecardExplainer(
            model=model,
            combiner=woe_pipeline.named_steps["combiner"],
            woe_transformer=woe_pipeline.named_steps["woe"],
        )

    def test_explain_returns_dataframe(self, trained_explainer, train_test_data):
        _, test = train_test_data
        shap_df = trained_explainer.explain(test)
        assert isinstance(shap_df, pd.DataFrame)
        assert shap_df.shape[0] == test.shape[0]

    def test_explain_sample(self, trained_explainer, train_test_data):
        _, test = train_test_data
        sample = trained_explainer.explain_sample(test, index=0)
        assert set(sample.columns) == {"feature", "shap_value", "feature_value"}
        assert len(sample) > 0

    def test_summary(self, trained_explainer, train_test_data):
        _, test = train_test_data
        summary = trained_explainer.summary(test)
        assert "feature" in summary.columns
        assert "mean_abs_shap" in summary.columns
