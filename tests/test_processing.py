# -*- coding: utf-8 -*-
"""Regression tests for scorecardpipeline.processing components."""

import numpy as np

from scorecardpipeline.processing import (
    Combiner,
    FeatureSelection,
    StepwiseSelection,
)


class TestFeatureSelection:
    def test_feature_selection_keeps_target(self, train_test_data):
        train, _ = train_test_data
        selector = FeatureSelection(target="creditability", engine="scorecardpy", iv=0.02)
        selected = selector.fit_transform(train)
        assert "creditability" in selected.columns
        assert selected.shape[1] <= train.shape[1]

    def test_feature_selection_return_drop(self, train_test_data):
        train, _ = train_test_data
        selector = FeatureSelection(target="creditability", engine="scorecardpy", iv=0.02, return_drop=True)
        selector.fit(train)
        assert selector.dropped is not None
        assert selector.select_columns is not None


class TestCombiner:
    def test_combiner_fit_transform(self, train_test_data):
        train, _ = train_test_data
        combiner = Combiner(target="creditability", method="chi", min_bin_size=0.05, max_n_bins=4)
        binned = combiner.fit_transform(train)
        assert binned.shape[0] == train.shape[0]
        assert "creditability" in binned.columns

    def test_combiner_export_load(self, train_test_data, tmp_path):
        train, _ = train_test_data
        combiner = Combiner(target="creditability", method="chi", min_bin_size=0.05, max_n_bins=4)
        combiner.fit(train)
        rules_path = tmp_path / "rules.json"
        combiner.export(to_json=str(rules_path))
        assert rules_path.exists()

        new_combiner = Combiner(target="creditability")
        new_combiner.load(str(rules_path))
        assert new_combiner.fitted_

    def test_optbinning_method(self, train_test_data):
        train, _ = train_test_data
        combiner = Combiner(target="creditability", method="cart", min_bin_size=0.05, max_n_bins=4)
        binned = combiner.fit_transform(train)
        assert binned.shape[0] == train.shape[0]


class TestWOETransformer:
    def test_woe_transformer(self, woe_pipeline, train_test_data):
        train, _ = train_test_data
        woe_train = woe_pipeline.transform(train)
        assert "creditability" in woe_train.columns
        # WOE values should not contain infinities after clipping
        assert not np.isinf(woe_train.select_dtypes(include=np.number).values).any()


class TestStepwiseSelection:
    def test_stepwise_selection(self, woe_pipeline, train_test_data):
        train, _ = train_test_data
        woe_train = woe_pipeline.transform(train)
        selector = StepwiseSelection(target="creditability", direction="both", criterion="aic")
        selected = selector.fit_transform(woe_train)
        assert "creditability" in selected.columns
        assert selected.shape[1] <= woe_train.shape[1]
