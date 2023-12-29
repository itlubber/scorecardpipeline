# -*- coding: utf-8 -*-
"""
@Time    : 2023/2/15 17:55
@Author  : itlubber
@Site    : itlubber.art
"""
from toad.metrics import KS, AUC, F1, PSI
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter, column_index_from_string
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union

from .logger import init_logger
from .utils import *
from .processing import FeatureSelection, FeatureImportanceSelector, StepwiseSelection, Combiner, WOETransformer
from .model import ITLubberLogisticRegression, ScoreCard
from .excel_writer import ExcelWriter, dataframe2excel
from .auto_eda import auto_eda_sweetviz
from .auto_report import auto_data_testing_report


feature_bin_stats = Combiner.feature_bin_stats


__version__ = "0.1.29"
__all__ = (
    "__version__"
    , "FeatureSelection", "FeatureImportanceSelector", "StepwiseSelection", "Combiner", "WOETransformer"
    , "ITLubberLogisticRegression", "ScoreCard"
    , "Pipeline", "KS", "AUC", "PSI", "F1", "FeatureUnion", "make_pipeline", "make_union"
    , "init_logger", "init_setting", "load_pickle", "save_pickle", "germancredit"
    , "ColorScaleRule", "get_column_letter", "column_index_from_string", "seed_everything"
    , "feature_bins", "feature_bin_stats", "extract_feature_bin", "inverse_feature_bins"
    , "bin_plot", "corr_plot", "ks_plot", "hist_plot", "psi_plot", "csi_plot", "dataframe_plot", "distribution_plot"
    , "ExcelWriter", "dataframe2excel", "auto_eda_sweetviz", "auto_data_testing_report",
)
