# -*- coding: utf-8 -*-
"""Tests for scorecardpipeline.monitoring."""

import numpy as np
import pandas as pd
import pytest

from scorecardpipeline.monitoring import ModelMonitor


class TestModelMonitor:
    @pytest.fixture
    def reference_data(self):
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature_a": np.random.normal(0, 1, 500),
                "feature_b": np.random.normal(5, 2, 500),
            }
        )

    @pytest.fixture
    def current_data(self):
        np.random.seed(43)
        return pd.DataFrame(
            {
                "feature_a": np.random.normal(0.5, 1.2, 300),
                "feature_b": np.random.normal(5, 2, 300),
            }
        )

    def test_score_psi(self, reference_data, current_data):
        monitor = ModelMonitor(score_bins=10)
        ref_score = reference_data["feature_a"] * 2 + reference_data["feature_b"]
        cur_score = current_data["feature_a"] * 2 + current_data["feature_b"]
        monitor.fit_reference(reference_data, ref_score)
        psi = monitor.score_psi(cur_score)
        assert psi >= 0

    def test_feature_psi(self, reference_data, current_data):
        monitor = ModelMonitor(score_bins=10)
        ref_score = reference_data["feature_a"] * 2 + reference_data["feature_b"]
        monitor.fit_reference(reference_data, ref_score)
        feature_psi = monitor.feature_psi(current_data)
        assert "feature" in feature_psi.columns
        assert "psi" in feature_psi.columns
        assert len(feature_psi) == 2

    def test_performance_decay(self, reference_data, current_data):
        monitor = ModelMonitor(score_bins=10)
        ref_score = reference_data["feature_a"] * 2 + reference_data["feature_b"]
        cur_score = current_data["feature_a"] * 2 + current_data["feature_b"]
        y_true = (ref_score > ref_score.median()).astype(int).values
        monitor.fit_reference(reference_data, ref_score, y_true=y_true)
        cur_y_true = (cur_score > cur_score.median()).astype(int).values
        metrics = monitor.performance_decay(cur_score, cur_y_true)
        assert "current_ks" in metrics
        assert "current_auc" in metrics
