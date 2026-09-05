# -*- coding: utf-8 -*-
"""Regression tests for scorecardpipeline.model components."""

import numpy as np

from scorecardpipeline.model import ITLubberLogisticRegression


class TestITLubberLogisticRegression:
    def test_summary_shape(self, woe_pipeline, train_test_data):
        train, _ = train_test_data
        woe_train = woe_pipeline.transform(train)
        model = ITLubberLogisticRegression(target="creditability")
        model.fit(woe_train)
        summary = model.summary()
        assert "Coef." in summary.columns
        assert "VIF" in summary.columns
        assert len(summary) == woe_train.shape[1]

    def test_predict_proba(self, woe_pipeline, train_test_data):
        train, test = train_test_data
        woe_train = woe_pipeline.transform(train)
        woe_test = woe_pipeline.transform(test)
        model = ITLubberLogisticRegression(target="creditability")
        model.fit(woe_train)
        proba = model.predict_proba(woe_test)
        assert proba.shape == (woe_test.shape[0], 2)
        assert np.all((proba >= 0) & (proba <= 1))
