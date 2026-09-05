# -*- coding: utf-8 -*-
"""Regression tests for scorecardpipeline.scorecard components."""

import numpy as np

from scorecardpipeline.scorecard import (
    RoundStandardScoreTransformer,
    StandardScoreTransformer,
)


class TestStandardScoreTransformer:
    def test_pdo_formula(self):
        proba = np.array([[0.1], [0.5], [0.9]])
        transformer = StandardScoreTransformer(base_score=660, pdo=75, rate=2, bad_rate=0.15)
        transformer.fit(proba)
        scores = transformer.transform(proba)
        assert scores.shape == proba.shape
        assert np.all(scores >= transformer.down_lmt)
        assert np.all(scores <= transformer.up_lmt)

    def test_inverse_transform(self):
        # 选择不会被 clip 到边界的概率，保证反推一致性
        proba = np.array([[0.15], [0.3], [0.6]])
        transformer = StandardScoreTransformer(base_score=660, pdo=75, rate=2, bad_rate=0.15)
        transformer.fit(proba)
        scores = transformer.transform(proba)
        recovered = transformer.inverse_transform(scores)
        np.testing.assert_allclose(recovered, proba, rtol=1e-5)

    def test_rounded_score_transformer(self):
        proba = np.array([[0.1], [0.5], [0.9]])
        transformer = RoundStandardScoreTransformer(base_score=660, pdo=75, bad_rate=0.15)
        transformer.fit(proba)
        scores = transformer.transform(proba)
        assert scores.dtype.kind in "iu" or np.all(scores == scores.round())
