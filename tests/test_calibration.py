# -*- coding: utf-8 -*-
"""Tests for scorecardpipeline.calibration."""

import numpy as np
import pytest

from scorecardpipeline.calibration import ProbabilityCalibrator


class TestProbabilityCalibrator:
    @pytest.fixture
    def binary_data(self):
        np.random.seed(42)
        proba = np.random.uniform(0, 1, 200)
        y = (proba > 0.5).astype(int)
        return proba, y

    def test_platt_scaling(self, binary_data):
        proba, y = binary_data
        calibrator = ProbabilityCalibrator(method="platt")
        calibrated = calibrator.fit_transform(proba, y)
        assert calibrated.shape == proba.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_isotonic_scaling(self, binary_data):
        proba, y = binary_data
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrated = calibrator.fit_transform(proba, y)
        assert calibrated.shape == proba.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))

    def test_two_column_input(self, binary_data):
        proba, y = binary_data
        proba_matrix = np.column_stack([1 - proba, proba])
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(proba, y)
        calibrated = calibrator.transform(proba_matrix)
        assert calibrated.shape == proba_matrix.shape
        assert np.all((calibrated >= 0) & (calibrated <= 1))
