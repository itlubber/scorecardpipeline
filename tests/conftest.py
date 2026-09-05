# -*- coding: utf-8 -*-
"""Pytest configuration and shared fixtures for scorecardpipeline."""

import pytest
from sklearn.model_selection import train_test_split

from scorecardpipeline import Pipeline
from scorecardpipeline.processing import Combiner, FeatureSelection, WOETransformer
from scorecardpipeline.utils import germancredit


@pytest.fixture(scope="session")
def raw_data():
    """Load the German credit dataset with a 0/1 target."""
    data = germancredit()
    data["creditability"] = data["creditability"].map({"good": 0, "bad": 1})
    return data


@pytest.fixture(scope="session")
def train_test_data(raw_data):
    """Split the German credit dataset into train and test sets."""
    train, test = train_test_split(
        raw_data,
        test_size=0.3,
        random_state=42,
        stratify=raw_data["creditability"],
    )
    return train.copy(), test.copy()


@pytest.fixture(scope="session")
def woe_pipeline(train_test_data):
    """A minimal sklearn Pipeline that ends with WOE-encoded train/test data."""
    train, _ = train_test_data
    pipeline = Pipeline(
        [
            ("select", FeatureSelection(target="creditability", engine="scorecardpy", iv=0.02)),
            ("combiner", Combiner(target="creditability", method="chi", min_bin_size=0.05, max_n_bins=4)),
            ("woe", WOETransformer(target="creditability")),
        ]
    )
    pipeline.fit(train)
    return pipeline
