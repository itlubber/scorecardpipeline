# -*- coding: utf-8 -*-
"""Regression tests for scorecardpipeline.rule components."""

import pandas as pd
import pytest

from scorecardpipeline.rule import Rule, ruleset_report


class TestRule:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "age": [20, 30, 40, 50],
                "income": [3000, 5000, 8000, 12000],
                "target": [0, 1, 0, 1],
            }
        )

    def test_rule_and_or(self, sample_df):
        rule = Rule("age > 25") & Rule("income > 4000")
        result = rule.predict(sample_df)
        assert result.tolist() == [False, True, True, True]

    def test_rule_not(self, sample_df):
        rule = ~Rule("age > 30")
        result = rule.predict(sample_df)
        assert result.tolist() == [True, True, False, False]

    def test_ruleset_report(self, sample_df):
        rules = [Rule("age > 30"), Rule("income > 5000")]
        report = ruleset_report(sample_df, rules, target="target")
        assert "样本总数" in report.columns
        assert "LIFT值" in report.columns
