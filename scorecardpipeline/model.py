# -*- coding: utf-8 -*-
"""
@Time    : 2023/05/21 16:23
@Author  : itlubber
@Site    : itlubber.art
"""

import os
import math
import numpy as np
import pandas as pd
import scorecardpy as sc
import matplotlib.pyplot as plt
import toad
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from .utils import *
from .processing import *


class ITLubberLogisticRegression(LogisticRegression):
    """
    Extended Logistic Regression.
    Extends [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
    This class provides the following extra statistics, calculated on `.fit()` and accessible via `.summary()`:
    - `cov_matrix_`: covariance matrix for the estimated parameters.
    - `std_err_intercept_`: estimated uncertainty for the intercept
    - `std_err_coef_`: estimated uncertainty for the coefficients
    - `z_intercept_`: estimated z-statistic for the intercept
    - `z_coef_`: estimated z-statistic for the coefficients
    - `p_value_intercept_`: estimated p-value for the intercept
    - `p_value_coef_`: estimated p-value for the coefficients
    
    Example:
    ```python
    feature_pipeline = Pipeline([
        ("preprocessing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("combiner", Combiner(target=target, min_samples=0.2)),
        ("transform", WOETransformer(target=target)),
        ("processing_select", FeatureSelection(target=target, engine="scorecardpy")),
        ("stepwise", StepwiseSelection(target=target)),
        # ("logistic", LogisticClassifier(target=target)),
        ("logistic", ITLubberLogisticRegression(target=target)),
    ])
    
    feature_pipeline.fit(train)
    summary = feature_pipeline.named_steps['logistic'].summary()
    ```
    
    An example output of `.summary()`:
    
    |                   |     Coef. |   Std.Err |        z |       P>|z| |    [ 0.025 |   0.975 ] |     VIF |
    |:------------------|----------:|----------:|---------:|------------:|-----------:|----------:|--------:|
    | const             | -0.844037 | 0.0965117 | -8.74544 | 2.22148e-18 | -1.0332    | -0.654874 | 1.05318 |
    | duration.in.month |  0.847445 | 0.248873  |  3.40513 | 0.000661323 |  0.359654  |  1.33524  | 1.14522 |
    """

    def __init__(self, target="target", penalty="l2", calculate_stats=True, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=False, n_jobs=None, l1_ratio=None,):
        """
        Extends [sklearn.linear_model.LogisticRegression.fit()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

        Args:
            target (str): your dataset's target name
            calculate_stats (bool): If true, calculate statistics like standard error during fit, accessible with .summary()
        """
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio,)
        self.target = target
        self.calculate_stats = calculate_stats

    def fit(self, x, sample_weight=None, **kwargs):
        y = x[self.target]
        x = x.drop(columns=[self.target])
        
        if not self.calculate_stats:
            return super().fit(x, y, sample_weight=sample_weight, **kwargs)

        x = self.convert_sparse_matrix(x)
        
        if isinstance(x, pd.DataFrame):
            self.names_ = ["const"] + [f for f in x.columns]
        else:
            self.names_ = ["const"] + [f"x{i}" for i in range(x.shape[1])]

        lr = super().fit(x, y, sample_weight=sample_weight, **kwargs)

        predProbs = self.predict_proba(x)

        # Design matrix -- add column of 1's at the beginning of your x matrix
        if lr.fit_intercept:
            x_design = np.hstack([np.ones((x.shape[0], 1)), x])
        else:
            x_design = x

        self.vif = [variance_inflation_factor(np.matrix(x_design), i) for i in range(x_design.shape[-1])]
        p = np.product(predProbs, axis=1)
        self.cov_matrix_ = np.linalg.inv((x_design * p[..., np.newaxis]).T @ x_design)
        std_err = np.sqrt(np.diag(self.cov_matrix_)).reshape(1, -1)

        # In case fit_intercept is set to True, then in the std_error array
        # Index 0 corresponds to the intercept, from index 1 onwards it relates to the coefficients
        # If fit intercept is False, then all the values are related to the coefficients
        if lr.fit_intercept:

            self.std_err_intercept_ = std_err[:, 0]
            self.std_err_coef_ = std_err[:, 1:][0]

            self.z_intercept_ = self.intercept_ / self.std_err_intercept_

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = scipy.stats.norm.sf(abs(self.z_intercept_)) * 2

        else:
            self.std_err_intercept_ = np.array([np.nan])
            self.std_err_coef_ = std_err[0]

            self.z_intercept_ = np.array([np.nan])

            # Get p-values under the gaussian assumption
            self.p_val_intercept_ = np.array([np.nan])

        self.z_coef_ = self.coef_ / self.std_err_coef_
        self.p_val_coef_ = scipy.stats.norm.sf(abs(self.z_coef_)) * 2

        return self
    
    def corr(self, data, save=None, annot=True):
        corr_plot(data.drop(columns=[self.target]), save=save, annot=annot)

    def report(self, data):
        report_dict = classification_report(data[self.target], self.predict(data.drop(columns=self.target)), output_dict=True, target_names=["好客户", "坏客户"])
        accuracy = report_dict.pop("accuracy")
        _report = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "desc"})
        _report.loc[len(_report)] = ['accuracy', '', '', accuracy, len(data)]
        return _report

    def summary(self):
        """
        Puts the summary statistics of the fit() function into a pandas DataFrame.
        Returns:
            data (pandas DataFrame): The statistics dataframe, indexed by the column name
        """
        check_is_fitted(self)

        if not hasattr(self, "std_err_coef_"):
            msg = "Summary statistics were not calculated on .fit(). Options to fix:\n"
            msg += "\t- Re-fit using .fit(X, y, calculate_stats=True)\n"
            msg += "\t- Re-inititialize using LogisticRegression(calculate_stats=True)"
            raise AssertionError(msg)

        data = {
            "Coef.": (self.intercept_.tolist() + self.coef_.tolist()[0]),
            "Std.Err": (self.std_err_intercept_.tolist() + self.std_err_coef_.tolist()),
            "z": (self.z_intercept_.tolist() + self.z_coef_.tolist()[0]),
            "P>|z|": (self.p_val_intercept_.tolist() + self.p_val_coef_.tolist()[0]),
        }
        
        stats = pd.DataFrame(data, index=self.names_)
        stats["[ 0.025"] = stats["Coef."] - 1.96 * stats["Std.Err"]
        stats["0.975 ]"] = stats["Coef."] + 1.96 * stats["Std.Err"]
        
        stats["VIF"] = self.vif
        
        return stats
    
    def summary2(self, feature_map={}):
        stats = self.summary().reset_index().rename(columns={"index": "Features"})
        
        if feature_map is not None and len(feature_map) > 0:
            stats.insert(loc=1, column="Describe", value=[feature_map.get(c, "") for c in stats["Features"]])
        
        return stats
    
    @staticmethod
    def convert_sparse_matrix(x):
        if scipy.sparse.issparse(x):
            return x.toarray()
        else:
            return x
    
    def plot_weights(self, save=None, figsize=(15, 8), fontsize=14, color=["#2639E9", "#F76E6C", "#FE7715"]):
        summary = self.summary()
        
        x = summary["Coef."]
        y = summary.index
        lower_error = summary["Coef."] - summary["[ 0.025"]
        upper_error = summary["0.975 ]"] - summary["Coef."]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.errorbar(x, y, xerr=[lower_error, upper_error], fmt="o", ecolor=color[0], elinewidth=2, capthick=2, capsize=4, ms=6, mfc=color[0], mec=color[0])
        # ax.tick_params(axis='x', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        # ax.tick_params(axis='y', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
        ax.axvline(0, color=color[0], linestyle='--', ymax=len(y), alpha=0.5)
        ax.spines['top'].set_color(color[0])
        ax.spines['bottom'].set_color(color[0])
        ax.spines['right'].set_color(color[0])
        ax.spines['left'].set_color(color[0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title("Regression Meta Analysis - Weight Plot\n", fontsize=fontsize, fontweight="bold")
        ax.set_xlabel("Weight Estimates", fontsize=fontsize, weight="bold")
        ax.set_ylabel("Variable", fontsize=fontsize, weight="bold")
        
        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save), exist_ok=True)
                
            plt.savefig(save, dpi=240, format="png", bbox_inches="tight")

        return fig


class ScoreCard(toad.ScoreCard, TransformerMixin):
    
    def __init__(self, target="target", pdo=60, rate=2, base_odds=35, base_score=750, combiner={}, transer=None, pretrain_lr=None, pipeline=None, **kwargs):
        """
        评分卡模型转换

        Args:
            target: 数据集中标签名称，默认 target
            pdo: odds 每增加 rate 倍时减少 pdo 分，默认 60
            rate: 倍率
            base_odds: 基础 odds，通常根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比，默认 35，即 35:1 => 0.972 => 坏样本率 2.8%
            base_score: 基础 odds 对应的分数，默认 750
            combiner: 分箱转换器，传入 pipeline 时可以为None
            transer: woe转换器，传入 pipeline 时可以为None
            pretrain_lr: 预训练好的逻辑回归模型，可以不传
            pipeline: 训练好的 pipeline，必须包含 Combiner 和 WOETransformer
            **kwargs: 其他相关参数，具体参考 toad.ScoreCard
        """
        if pipeline:
            combiner = self.class_steps(pipeline, Combiner)[0]
            transer = self.class_steps(pipeline, WOETransformer)[0]
            
            if self.class_steps(pipeline, (ITLubberLogisticRegression, LogisticRegression)):
                pretrain_lr = self.class_steps(pipeline, (ITLubberLogisticRegression, LogisticRegression))[0]
            
        super().__init__(
                            combiner=combiner.combiner if isinstance(combiner, Combiner) else combiner, transer=transer.transformer if isinstance(transer, WOETransformer) else transer, 
                            pdo=pdo, rate=rate, base_odds=base_odds, base_score=base_score, **kwargs
                        )
        
        self.target = target
        self.pipeline = pipeline
        self.pretrain_lr = pretrain_lr
        
    def fit(self, x):
        y = x[self.target]
        
        if self.pretrain_lr:
            x = x[self.pretrain_lr.feature_names_in_]
        else:
            x = x.drop(columns=[self.target])
        
        self._feature_names = x.columns.tolist()

        for f in self.features_:
            if f not in self.transer:
                raise Exception('column \'{f}\' is not in transer'.format(f = f))

        if self.pretrain_lr:
            self.model = self.pretrain_lr
        else:
            self.model.fit(x, y)
        
        self.rules = self._generate_rules()

        sub_score = self.woe_to_score(x)
        self.base_effect = pd.Series(np.median(sub_score, axis=0), index = self.features_)

        return self
    
    def transform(self, x):
        return self.predict(x)
    
    def _check_rules(self, combiner, transer):
        for col in self.features_:
            if col not in combiner:
                raise Exception('column \'{col}\' is not in combiner'.format(col = col))
            
            if col not in transer:
                raise Exception('column \'{col}\' is not in transer'.format(col = col))

            l_c = len(combiner[col])
            l_t = len(transer[col]['woe'])

            if l_c == 0:
                continue

            if np.issubdtype(combiner[col].dtype, np.number):
                if l_c != l_t - 1:
                    if np.isnan(combiner[col]).sum() > 0:
                        combiner.update({col: combiner[col][:-1]})
                    else:
                        raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col = col, l_t = l_t, l_c = l_c + 1))
            else:
                if l_c != l_t:
                    raise Exception('column \'{col}\' is not matched, assert {l_t} bins but given {l_c}'.format(col = col, l_t = l_t, l_c = l_c))

        return True
    
    @staticmethod
    def score_clip(score, clip=50):
        clip_start = max(math.ceil(score.min() / clip) * clip, math.ceil(score.quantile(0.01) / clip) * clip)
        clip_end = min(math.ceil(score.max() / clip) * clip, math.ceil(score.quantile(0.99) / clip) * clip)
        return [i for i in range(clip_start, clip_end, clip)]
    
    def scorecard_scale(self):
        scorecard_kedu = pd.DataFrame(
            [
                ["base_odds", self.base_odds, "根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比"],
                ["base_score", self.base_score, "基础ODDS对应的分数"],
                ["rate", self.rate, "设置分数的倍率"],
                ["pdo", self.pdo, "表示分数增长PDO时，ODDS值增长到RATE倍"],
                ["B", self.offset, "补偿值，计算方式：pdo / ln(rate)"],
                ["A", self.factor, "刻度，计算方式：base_score - B * ln(base_odds)"],
            ],
            columns=["刻度项", "刻度值", "备注"],
        )
        return scorecard_kedu
    
    def scorecard_points(self, feature_map={}):
        card_points = self.export(to_frame=True).rename(columns={"name": "变量名称", "value": "变量分箱", "score": "对应分数"})
        
        if feature_map is not None and len(feature_map) > 0:
            card_points.insert(loc=1, column="变量含义", value=[feature_map.get(c, "") for c in card_points["变量名称"]])
        
        return card_points
    
    def scorecard2pmml(self, pmml: str = 'scorecard.pmml', debug: bool = False):
        """export a scorecard to pmml

        Args:
            pmml (str): io to write pmml file.
            debug (bool): If true, print information about the conversion process.
        """
        from sklearn_pandas import DataFrameMapper
        from sklearn.linear_model import LinearRegression
        from sklearn2pmml import sklearn2pmml, PMMLPipeline
        from sklearn2pmml.preprocessing import LookupTransformer, ExpressionTransformer

        mapper = []
        samples = {}
        for var, rule in self.rules.items():
            end_string = ''
            expression_string = ''
            total_bins = len(rule['bins'])
            if isinstance(rule['bins'][0], (np.ndarray, list)):
                default_value = 0.
                mapping = {}
                for bins, score  in zip(rule['bins'], rule['scores'].tolist()):
                    for _bin in bins:
                        if _bin == 'nan':
                            default_value = float(score)

                        mapping[_bin] = float(score)

                mapper.append((
                    [var],
                    LookupTransformer(mapping=mapping, default_value=default_value),
                ))
                samples[var] = [list(mapping.keys())[i] for i in np.random.randint(0, len(mapping), 20)]
            else:
                has_empty = len(rule['bins']) > 0 and pd.isnull(rule['bins'][-1])

                if has_empty:
                    score_empty = rule['scores'][-1]
                    total_bins -= 1
                    iter = enumerate(zip(rule['bins'][:-1], rule['scores'][:-1]), start=1)
                else:
                    iter = enumerate(zip(rule['bins'], rule['scores']), start=1)

                if has_empty:
                    expression_string += f'{score_empty} if pandas.isnull(X[0])'

                for i, (bin_var, score) in iter:
                    if i == 1 and not has_empty:
                        expression_string += f'{score} if X[0] < {bin_var}'
                    elif i == total_bins:
                        expression_string += f' else {score}'
                    else:
                        expression_string += f' else ({score} if X[0] < {bin_var}'
                        end_string += ')'

                expression_string += end_string

                mapper.append((
                    [var],
                    ExpressionTransformer(expression_string),
                ))
                samples[var] = np.random.random(20) * 100

        scorecard_mapper = DataFrameMapper(mapper, df_out=True)

        pipeline = PMMLPipeline([
            ('preprocessing', scorecard_mapper),
            ('scorecard', LinearRegression(fit_intercept=False)),
        ])

        pipeline.named_steps['scorecard'].fit(
            pd.DataFrame(
                np.random.randint(0, 100, (100, len(scorecard_mapper.features))),
                columns=[m[0][0] for m in scorecard_mapper.features]
            ),
            pd.Series(np.random.randint(0, 2, 100), name='score')
        )

        pipeline.named_steps['scorecard'].coef_ = np.ones(len(scorecard_mapper.features))

        sklearn2pmml(pipeline, pmml, with_repr=True, debug=debug)
    
    @staticmethod
    def KS_bucket(y_pred, y_true, bucket=10, method="quantile"):
        return toad.metrics.KS_bucket(y_pred, y_true, bucket=bucket, method=method)
    
    @staticmethod
    def KS(y_pred, y_true):
        return toad.metrics.KS(y_pred, y_true)
    
    @staticmethod
    def AUC(y_pred, y_true):
        return toad.metrics.AUC(y_pred, y_true)
    
    @staticmethod
    def perf_eva(y_pred, y_true, title="", plot_type=["ks", "roc"], save=None, figsize=(14, 6)):
        # plt.figure(figsize=figsize)
        rt = sc.perf_eva(y_true, y_pred, title=title, plot_type=plot_type, show_plot=True)

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save))
            
            rt["pic"].savefig(save, dpi=240, format="png", bbox_inches="tight")
        
        return rt
    
    @staticmethod
    def ks_plot(score, y_true, title="", fontsize=14, figsize=(16, 8), save=None, **kwargs):
        ks_plot(score, y_true, title=title, fontsize=fontsize, figsize=figsize, save=save, **kwargs)
    
    @staticmethod
    def PSI(y_pred_train, y_pred_oot):
        return toad.metrics.PSI(y_pred_train, y_pred_oot)
    
    @staticmethod
    def perf_psi(y_pred_train, y_pred_oot, y_true_train, y_true_oot, keys=["train", "test"], x_limits=None, x_tick_break=50, show_plot=True, return_distr_dat=False):
        return sc.perf_psi(
            score = {keys[0]: y_pred_train, keys[1]: y_pred_oot},
            label = {keys[0]: y_true_train, keys[1]: y_true_oot},
            x_limits = x_limits,
            x_tick_break = x_tick_break,
            show_plot = show_plot,
            return_distr_dat = return_distr_dat,
        )
    
    @staticmethod
    def score_hist(score, y_true, figsize=(15, 10), bins=20, save=None, **kwargs):
        hist_plot(score, y_true, figsize=figsize, bins=bins, save=save, **kwargs)
    
    def _format_rule(self, rule, decimal = 4, **kwargs):
        bins = self.format_bins(rule['bins'])
        scores = np.around(rule['scores'], decimals = decimal).tolist()
        
        return dict(zip(bins, scores))
    
    @staticmethod
    def class_steps(pipeline, query):
        return [v for k, v in pipeline.named_steps.items() if isinstance(v, query)]
    
    def feature_bin_stats(self, data, feature, rules={}, method='step', max_n_bins=10, desc="评分卡分数", ks=False, **kwargs):
        return feature_bin_stats(data, feature, target=self.target, method=method, max_n_bins=max_n_bins, desc=desc, ks=ks, **kwargs)
