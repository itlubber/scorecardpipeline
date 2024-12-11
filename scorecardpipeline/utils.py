# -*- coding: utf-8 -*-
"""
@Time    : 2023/05/21 16:23
@Author  : itlubber
@Site    : itlubber.art
"""
import warnings

warnings.filterwarnings("ignore")

import os
import re
import six
import pickle
import random
import joblib
import warnings
import numpy as np
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter, FuncFormatter

import toad
import seaborn as sns
from joblib import Parallel, delayed
from optbinning import OptimalBinning
from sklearn.metrics import roc_curve, auc, roc_auc_score

from .logger import init_logger


def seed_everything(seed: int, freeze_torch=False):
    """
    固定当前环境随机种子，以保证后续实验可重复

    :param seed: 随机种子
    :param freeze_torch: 是否固定 pytorch 的随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if freeze_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def init_setting(font_path=None, seed=None, freeze_torch=False, logger=False, **kwargs):
    """
    初始化环境配置，去除警告信息、修改 pandas 默认配置、固定随机种子、日志记录

    :param seed: 随机种子，默认为 None
    :param freeze_torch: 是否固定 pytorch 环境
    :param font_path: 画图时图像使用的字体，支持系统字体名称、本地字体文件路径，默认为 scorecardppeline 提供的中文字体
    :param logger: 是否需要初始化日志器，默认为 False ，当参数为 True 时返回 logger
    :param kwargs: 日志初始化传入的相关参数

    :return: 当 logger 为 True 时返回 logging.Logger
    """
    warnings.filterwarnings("ignore")

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option("display.max_colwidth", 300)
    pd.set_option('expand_frame_repr', False)

    if "seaborn-ticks" in plt.style.available:
        plt.style.use('seaborn-ticks')
    else:
        plt.style.use('seaborn-v0_8-ticks')

    if font_path is not None and font_path.lower() in [font.fname.lower() for font in font_manager.fontManager.ttflist]:
        plt.rcParams['font.family'] = font_path
    else:
        font_path = font_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf')

        if not os.path.isfile(font_path):
            import wget
            font_path = wget.download(
                "https://itlubber.art/upload/matplot_chinese.ttf"
                , os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf')
            )

        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()

    plt.rcParams['axes.unicode_minus'] = False

    if seed:
        seed_everything(seed, freeze_torch=freeze_torch)

    if logger:
        return init_logger(**kwargs)


def load_pickle(file, engine="joblib"):
    """
    导入 pickle 文件

    :param file: pickle 文件路径

    :return: pickle 文件的内容
    """
    if engine == "joblib":
        return joblib.load(file)
    elif engine == "dill":
        import dill
        with open(file, "rb") as f:
            return dill.load(f)
    elif engine == "pickle":
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"engine 目前只支持 [joblib, dill, pickle], 不支持 {engine}")


def save_pickle(obj, file, engine="joblib"):
    """
    保持数据至 pickle 文件

    :param obj: 需要保存的数据
    :param file: 文件路径
    """
    if engine == "joblib":
        return joblib.dump(obj, file)
    elif engine == "dill":
        import dill
        with open(file, "wb") as f:
            return dill.dump(obj, f)
    elif engine == "pickle":
        with open(file, "wb") as f:
            return pickle.dump(obj, f)
    else:
        raise ValueError(f"engine 目前只支持 [joblib, dill, pickle], 不支持 {engine}")


def feature_describe(data, feature=None, percentiles=None, missing=None, cardinality=None):
    if feature and feature not in data.columns:
        raise ValueError(f"feature {feature} must in columns.")

    if cardinality and cardinality < 1:
        raise ValueError(f"cardinality must grater 1")

    if percentiles is None:
        percentiles = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]

    if feature:
        series = data[feature]
    else:
        series = data.copy()

    if missing:
        series = series.replace(missing, np.nan)

    if (cardinality and series.nunique() <= cardinality) or not pd.api.types.is_numeric_dtype(series):
        describe = {
            "样本数": len(series),
            "非空数": len(series) - series.isnull().sum(),
            "查得率": 1 - series.isnull().mean(),
        }
        describe.update((series.replace(np.nan, '缺失值').value_counts(dropna=False) / len(series)).to_dict())
        return pd.Series(describe, name=feature)
    else:
        describe = {
            "样本数": len(series),
            "非空数": len(series) - series.isnull().sum(),
            "查得率": 1 - series.isnull().mean(),
            "最小值": series.min(),
            "平均值": series.mean(),
            "最大值": series.max(),
            # "众数": series.mode()[0],
        }
        quantile = series.quantile(percentiles)
        quantile.index = [f"{int(i * 100)}%" for i in percentiles]
        describe.update(quantile.to_dict())
        return pd.Series(describe, name=feature).reindex(['样本数', '非空数', '查得率', '最小值', '平均值'] + [f"{int(i * 100)}%" for i in percentiles] + ['最大值'])


def groupby_feature_describe(data, by=None, n_jobs=-1, **kwargs):
    if not isinstance(by, (tuple, list, np.ndarray)):
        by = [by]

    describe = pd.DataFrame()

    def __feature_describe(group, _p, f, **kwargs):
        temp = feature_describe(group[f], **kwargs)
        temp.index = pd.MultiIndex.from_product([[f], temp.index])
        temp = pd.DataFrame(temp, columns=[_p])
        return temp

    def _feature_describe(group, _p, _by=None, **kwargs):
        if len(_p) <= 1:
            _p = _p[0]

        __describe = partial(lambda f: __feature_describe(group, _p, f, **kwargs))
        return _p, pd.concat(Parallel(n_jobs=n_jobs)(delayed(__describe)(f) for f in group.columns if f not in _by))[_p]

    for info in Parallel(n_jobs=n_jobs)(delayed(_feature_describe)(group, p, _by=by, **kwargs) for p, group in data.groupby(by=by)):
        describe[info[0]] = info[1]

    if len(by) > 1:
        describe.columns = pd.MultiIndex.from_tuples(describe.columns)

    describe.index.names = ["特征名称", "统计指标"]

    return describe


def germancredit():
    """
    加载德国信贷数据集 German Credit Data

    数据来源：https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

    :return: pd.DataFrame
    """
    from pandas.api.types import CategoricalDtype

    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'germancredit.csv'))

    cate_levels = {
        "status_of_existing_checking_account": ['... < 0 DM', '0 <= ... < 200 DM', '... >= 200 DM / salary assignments for at least 1 year', 'no checking account'],
        "credit_history": ["no credits taken/ all credits paid back duly", "all credits at this bank paid back duly", "existing credits paid back duly till now", "delay in paying off in the past", "critical account/ other credits existing (not at this bank)"],
        "savings_account_and_bonds": ["... < 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM", "... >= 1000 DM", "unknown/ no savings account"],
        "present_employment_since": ["unemployed", "... < 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years", "... >= 7 years"],
        "personal_status_and_sex": ["male : divorced/separated", "female : divorced/separated/married", "male : single", "male : married/widowed", "female : single"],
        "other_debtors_or_guarantors": ["none", "co-applicant", "guarantor"],
        "property": ["real estate", "building society savings agreement/ life insurance", "car or other, not in attribute Savings account/bonds", "unknown / no property"],
        "other_installment_plans": ["bank", "stores", "none"],
        "housing": ["rent", "own", "for free"],
        "job": ["unemployed/ unskilled - non-resident", "unskilled - resident", "skilled employee / official", "management/ self-employed/ highly qualified employee/ officer"],
        "telephone": ["none", "yes, registered under the customers name"],
        "foreign_worker": ["yes", "no"]}

    def cate_type(levels):
        return CategoricalDtype(categories=levels, ordered=True)

    for i in cate_levels.keys():
        data[i] = data[i].astype(cate_type(cate_levels[i]))

    return data


def round_float(num, decimal=4):
    """
    调整数值分箱的上下界小数点精度，如未超出精度保持原样输出

    :param num: 分箱的上界或者下界
    :param decimal: 小数点保留的精度

    :return: 精度调整后的数值
    """
    if ~pd.isnull(num) and isinstance(num, float):
        return float(str(num).split(".")[0] + "." + str(num).split(".")[1][:decimal])
    else:
        return num


def feature_bins(bins, decimal=4):
    """
    根据 Combiner 的规则生成分箱区间，并生成区间对应的索引

    :param bins: Combiner 的规则
    :param decimal: 区间上下界需要保留的精度，默认小数点后4位

    :return: dict ，key 为区间的索引，value 为区间
    """
    if len(bins) == 0:
        return {0: "全部样本"}
    if isinstance(bins, list): bins = np.array(bins)
    EMPTYBINS = len(bins) if not isinstance(bins[0], (set, list, np.ndarray)) else -1

    l = []
    if not isinstance(bins[0], (set, list, np.ndarray)):
        has_empty = len(bins) > 0 and pd.isnull(bins[-1])
        if has_empty: bins = bins[:-1]
        sp_l = ["负无穷"] + [round_float(b, decimal=decimal) for b in bins] + ["正无穷"]
        for i in range(len(sp_l) - 1): l.append('[' + str(sp_l[i]) + ' , ' + str(sp_l[i + 1]) + ')')
        if has_empty: l.append('缺失值')
    else:
        for keys in bins:
            keys_update = set()
            for key in keys:
                if pd.isnull(key) or (isinstance(key, str) and key == "nan"):
                    keys_update.add("缺失值")
                elif isinstance(key, str) and key.strip() == "":
                    keys_update.add("空字符串")
                else:
                    keys_update.add(key)
            label = ','.join(keys_update)
            l.append(label)

    return {i if b != "缺失值" else EMPTYBINS: b for i, b in enumerate(l)}


def extract_feature_bin(bin_var):
    """
    根据单个区间提取的分箱的上下界

    :param bin_var: 区间字符串

    :return: list or tuple
    """
    pattern = re.compile(r"^(\[|\()(-inf|负无穷|[-+]?\d+(\.\d+)?)(\s*,\s*|\s*~\s*)(inf|正无穷|[-+]?\d+(\.\d+)?)?(\]|\))$")
    match = pattern.match(bin_var)

    if match:
        start = -np.inf if match.group(2) in ["-inf", "负无穷"] else float(match.group(2))
        end = np.inf if match.group(5) in ["inf", "正无穷"] else float(match.group(5))
        return start, end
    else:
        return [np.nan if b == "缺失值" else ("" if b == "空字符串" else b) for b in bin_var.split("%,%" if "%,%" in bin_var else ",")]


def inverse_feature_bins(feature_table, bin_col="分箱"):
    """
    根据变量分箱表得到 Combiner 的规则

    :param feature_table: 变量分箱表
    :param bin_col: 变量分箱表中分箱对应的列名，默认 分箱

    :return: list
    """
    if isinstance(feature_table, pd.DataFrame):
        bin_vars = feature_table[bin_col].tolist()
    elif isinstance(feature_table, pd.Series):
        bin_vars = feature_table.tolist()
    else:
        bin_vars = feature_table

    has_empty = ("缺失值" in bin_vars) | (np.nan in bin_vars)

    bin_vars = [b for b in bin_vars if b not in ["缺失值", np.nan]]
    extract_bin_vars = [extract_feature_bin(bin_var) for bin_var in bin_vars]

    if isinstance(extract_bin_vars[0], tuple):
        inverse_bin_vars = sorted({s for b in extract_bin_vars for s in b})[1:-1]
        inverse_bin_vars += [np.nan] if has_empty else []
    else:
        inverse_bin_vars = extract_bin_vars
        inverse_bin_vars += [[np.nan]] if has_empty else []

    return inverse_bin_vars


def bin_plot(feature_table, desc="", figsize=(10, 6), colors=["#2639E9", "#F76E6C", "#FE7715"], save=None, anchor=0.935, max_len=35, fontdict={"color": "#000000"}, hatch=True, ending="分箱图"):
    """简单策略挖掘：特征分箱图

    :param feature_table: 特征分箱的统计信息表，由 feature_bin_stats 运行得到
    :param desc: 特征中文含义或者其他相关信息
    :param figsize: 图像尺寸大小，传入一个tuple，默认 （10， 6）
    :param colors: 图片主题颜色，默认即可
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param anchor: 图例在图中的位置，通常 0.95 左右，根据图片标题与图例之间的空隙自行调整即可
    :param max_len: 分箱显示的最大长度，防止分类变量分箱过多文本过长导致图像显示区域很小，默认最长 35 个字符
    :param fontdict: 柱状图上的文字内容格式设置，参考 https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
    :param hatch: 柱状图是否显示斜杠，默认显示
    :param ending: 分箱图标题显示的后缀，标题格式为: f'{desc}{ending}'
    
    :return: Figure
    """
    feature_table = feature_table.copy()

    feature_table["分箱"] = feature_table["分箱"].apply(lambda x: x if not pd.isnull(x) and re.match("^\[.*\)$", x) else (str(x)[:max_len] + ".." if len(str(x)) > max_len else str(x)))

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.barh(feature_table['分箱'], feature_table['好样本数'], color=colors[0], label='好样本', hatch="/" if hatch else None)
    ax1.barh(feature_table['分箱'], feature_table['坏样本数'], left=feature_table['好样本数'], color=colors[1], label='坏样本', hatch="\\" if hatch else None)
    ax1.set_xlabel('样本数')

    ax2 = ax1.twiny()
    ax2.plot(feature_table['坏样本率'], feature_table['分箱'], colors[2], label='坏样本率', linestyle='-.')
    ax2.set_xlabel('坏样本率: 坏样本数 / 样本总数')
    ax2.set_xlim(xmin=0.)

    for i, rate in enumerate(feature_table['坏样本率']):
        ax2.scatter(rate, i, color=colors[2])

    if fontdict and fontdict.get("color"):
        for i, v in feature_table[['样本总数', '好样本数', '坏样本数', '坏样本率']].iterrows():
            ax1.text(v['样本总数'] / 2, i + len(feature_table) / 60, f"{int(v['好样本数'])}:{int(v['坏样本数'])}:{v['坏样本率']:.2%}", fontdict=fontdict)

    ax1.invert_yaxis()
    ax2.xaxis.set_major_formatter(PercentFormatter(1, decimals=0, is_latex=True))
    # ax2.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'{}%'.format(round(x * 100, 4))))

    fig.suptitle(f'{desc}{ending}\n\n')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

    plt.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def corr_plot(data, figure_size=(16, 8), fontsize=16, mask=False, save=None, annot=True, max_len=35, linewidths=0.1, fmt='.2f', step=2 * 5 + 1, linecolor='white', **kwargs):
    """
    特征相关图

    :param data: 原始数据
    :param figure_size: 图片大小，默认 (16, 8)
    :param fontsize: 字体大小，默认 16
    :param mask: 是否只显示下三角部分内容，默认 False
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param annot: 是否在图中显示相关性的数值，默认 True
    :param max_len: 特征显示的最大长度，防止特征名称过长导致图像区域非常小，默认 35，可以传 None 表示不限制
    :param fmt: 数值显示格式，当 annot 为 True 时该参数生效，默认显示两位小数点
    :param step: 色阶的步数，以 0 为中心，默认 2（以0为中心对称） * 5（划分五个色阶） + 1（0一档单独显示）= 11
    :param linewidths: 相关图之间的线条宽度，默认 0.1 ，如果设置为 None 则不现实线条
    :param linecolor: 线的颜色，当 linewidths 大于 0 时生效，默认为 white
    :param kwargs: sns.heatmap 函数其他参数，参考：https://seaborn.pydata.org/generated/seaborn.heatmap.html
    :return: Figure
    """
    if max_len is None:
        corr = data.corr()
    else:
        corr = data.rename(columns={c: c if len(str(c)) <= max_len else f"{str(c)[:max_len]}..." for c in data.columns}).corr()

    corr_mask = np.zeros_like(corr, dtype=bool)
    corr_mask[np.triu_indices_from(corr_mask)] = True

    fig, ax = plt.subplots(figsize=figure_size)
    map_plot = sns.heatmap(corr
                           , cmap=sns.diverging_palette(267, 267, n=step, s=100, l=40)
                           , vmax=1
                           , vmin=-1
                           , center=0
                           , square=True
                           , linewidths=linewidths
                           , annot=annot
                           , fmt=fmt
                           , linecolor=linecolor
                           , robust=True
                           , cbar=True
                           , ax=ax
                           , mask=corr_mask if mask else None
                           , **kwargs
                           )

    map_plot.tick_params(axis='x', labelrotation=270, labelsize=fontsize)
    map_plot.tick_params(axis='y', labelrotation=0, labelsize=fontsize)

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def ks_plot(score, target, title="", fontsize=14, figsize=(16, 8), save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], anchor=0.945):
    """
    数值特征 KS曲线 & ROC曲线

    :param score: 数值特征，通常为评分卡分数
    :param target: 标签值
    :param title: 图像标题
    :param fontsize: 字体大小，默认 14
    :param figsize: 图像大小，默认 (16, 8)
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param colors: 图片主题颜色，默认即可
    :param anchor: 图例显示的位置，默认 0.945，根据实际显示情况进行调整即可，0.95 附近小范围调整
    :return: Figure
    """
    auc_value = roc_auc_score(target, score)

    if auc_value < 0.5:
        warnings.warn('评分AUC指标小于50%, 推断数据值越大, 正样本率越高, 将数据值转为负数后进行绘图')
        score = -score
        auc_value = 1 - auc_value

    # if np.mean(score) < 0 or np.mean(score) > 1:
    #     warnings.warn('Since the average of pred is not in [0,1], it is treated as credit score but not probability.')
    #     score = -score

    df = pd.DataFrame({'label': target, 'pred': score})

    def n0(x):
        return sum(x == 0)

    def n1(x):
        return sum(x == 1)

    df_ks = df.sort_values('pred', ascending=False).reset_index(drop=True) \
        .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / len(df.index)))) \
        .groupby('group')['label'].agg([n0, n1]) \
        .reset_index().rename(columns={'n0': 'good', 'n1': 'bad'}) \
        .assign(
        group=lambda x: (x.index + 1) / len(x.index),
        cumgood=lambda x: np.cumsum(x.good) / sum(x.good),
        cumbad=lambda x: np.cumsum(x.bad) / sum(x.bad)
    ).assign(ks=lambda x: abs(x.cumbad - x.cumgood))

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # KS曲线
    dfks = df_ks.loc[lambda x: x.ks == max(x.ks)].sort_values('group').iloc[0]

    ax[0].plot(df_ks.group, df_ks.ks, color=colors[0], label="KS曲线")
    ax[0].plot(df_ks.group, df_ks.cumgood, color=colors[1], label="累积好客户占比")
    ax[0].plot(df_ks.group, df_ks.cumbad, color=colors[2], label="累积坏客户占比")
    ax[0].fill_between(df_ks.group, df_ks.cumbad, df_ks.cumgood, color=colors[0], alpha=0.25)

    ax[0].plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
    ax[0].text(dfks['group'], dfks['ks'], f"KS: {round(dfks['ks'], 4)} at: {dfks.group:.2%}", horizontalalignment='center', fontsize=fontsize)

    ax[0].spines['top'].set_color(colors[0])
    ax[0].spines['bottom'].set_color(colors[0])
    ax[0].spines['right'].set_color(colors[0])
    ax[0].spines['left'].set_color(colors[0])
    ax[0].set_xlabel('% of Population', fontsize=fontsize)
    ax[0].set_ylabel('% of Total Bad / Good', fontsize=fontsize)

    ax[0].set_xlim((0, 1))
    ax[0].set_ylim((0, 1))

    handles1, labels1 = ax[0].get_legend_handles_labels()

    # ax[0].legend(loc='upper center', ncol=len(labels1), bbox_to_anchor=(0.5, 1.1), frameon=False)

    # ROC 曲线
    fpr, tpr, thresholds = roc_curve(target, score)

    ax[1].plot(fpr, tpr, color=colors[0], label="ROC Curve")
    ax[1].stackplot(fpr, tpr, color=colors[0], alpha=0.25)
    ax[1].plot([0, 1], [0, 1], color=colors[1], lw=2, linestyle=':')
    # ax[1].tick_params(axis='x', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
    # ax[1].tick_params(axis='y', labelrotation=0, grid_color="#FFFFFF", labelsize=fontsize)
    ax[1].text(0.5, 0.5, f"AUC: {auc_value:.4f}", fontsize=fontsize, horizontalalignment="center", transform=ax[1].transAxes)

    ax[1].spines['top'].set_color(colors[0])
    ax[1].spines['bottom'].set_color(colors[0])
    ax[1].spines['right'].set_color(colors[0])
    ax[1].spines['left'].set_color(colors[0])
    ax[1].set_xlabel("False Positive Rate", fontsize=fontsize)
    ax[1].set_ylabel('True Positive Rate', fontsize=fontsize)

    ax[1].set_xlim((0, 1))
    ax[1].set_ylim((0, 1))

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")

    handles2, labels2 = ax[1].get_legend_handles_labels()

    if title: title += " "
    fig.suptitle(f"{title}K-S & ROC CURVE\n", fontsize=fontsize, fontweight="bold")

    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

    plt.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        plt.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def hist_plot(score, y_true=None, figsize=(15, 10), bins=30, save=None, labels=["好样本", "坏样本"], desc="", anchor=1.11, fontsize=14, kde=False, **kwargs):
    """
    数值特征分布图

    :param score: 数值特征，通常为评分卡分数
    :param y_true: 标签值
    :param figsize: 图像大小，默认 (15, 10)
    :param bins: 分箱数量大小，默认 30
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param labels: 字典或列表，图例显示的分类名称，默认 ["好样本", "坏样本"]，按照目标变量顺序对应即可，从0开始
    :param anchor: 图例显示的位置，默认 1.1，根据实际显示情况进行调整即可，1.1 附近小范围调整
    :param fontsize: 字体大小，默认 14
    :param kwargs: sns.histplot 函数其他参数，参考：https://seaborn.pydata.org/generated/seaborn.histplot.html

    :return: Figure
    """
    target_unique = 1 if y_true is None else len(np.unique(y_true))

    if y_true is not None:
        if isinstance(labels, dict):
            y_true = y_true.map(labels)
            hue_order = list(labels.values())
        else:
            y_true = y_true.map({i: v for i, v in enumerate(labels)})
            hue_order = labels
    else:
        y_true = None
        hue_order = None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    palette = sns.diverging_palette(340, 267, n=target_unique, s=100, l=40)

    sns.histplot(
        x=score, hue=y_true, element="step", stat="probability", bins=bins, common_bins=True, common_norm=True, palette=palette, hue_order=hue_order[::-1], ax=ax, kde=kde, **kwargs
    )

    # sns.despine()

    ax.spines['top'].set_color("#2639E9")
    ax.spines['bottom'].set_color("#2639E9")
    ax.spines['right'].set_color("#2639E9")
    ax.spines['left'].set_color("#2639E9")

    ax.set_xlabel("值域范围", fontsize=fontsize)
    ax.set_ylabel("样本占比", fontsize=fontsize)

    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.set_title(f"{desc + ' ' if desc else '特征'}分布情况\n\n", fontsize=fontsize)

    if y_true is not None:
        ax.legend([t for t in hue_order for _ in range(2)] if kde else hue_order, loc='upper center', ncol=target_unique * 2 if kde else target_unique, bbox_to_anchor=(0.5, anchor), frameon=False, fontsize=fontsize)

    fig.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        plt.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def psi_plot(expected, actual, labels=["预期", "实际"], desc="", save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True, max_len=None, hatch=True):
    """
    特征 PSI 图

    :param expected: 期望分布情况，传入需要验证的特征分箱表
    :param actual: 实际分布情况，传入需要参照的特征分箱表
    :param labels: 期望分布和实际分布的名称，默认 ["预期", "实际"]
    :param desc: 标题前缀显示的名称，默认为空，推荐传入特征名称或评分卡名字
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param colors: 图片主题颜色，默认即可
    :param figsize: 图像大小，默认 (15, 8)
    :param anchor: 图例显示的位置，默认 0.94，根据实际显示情况进行调整即可，0.94 附近小范围调整
    :param width: 预期分布与实际分布柱状图之间的间隔，默认 0.35
    :param result: 是否返回 PSI 统计表，默认 False
    :param plot: 是否画 PSI图，默认 True
    :param max_len: 特征显示的最大长度，防止特征名称过长导致图像区域非常小，默认 None 表示不限制
    :param hatch: 是否显示柱状图上的斜线，默认为 True

    :return: 当 result 为 True 时，返回 pd.DataFrame
    """
    expected = expected.rename(columns={"样本总数": f"{labels[0]}样本数", "样本占比": f"{labels[0]}样本占比", "坏样本率": f"{labels[0]}坏样本率"})
    actual = actual.rename(columns={"样本总数": f"{labels[1]}样本数", "样本占比": f"{labels[1]}样本占比", "坏样本率": f"{labels[1]}坏样本率"})
    df_psi = expected.merge(actual, on="分箱", how="outer").replace(np.nan, 0)
    df_psi[f"{labels[1]}% - {labels[0]}%"] = df_psi[f"{labels[1]}样本占比"] - df_psi[f"{labels[0]}样本占比"]
    df_psi[f"ln({labels[1]}% / {labels[0]}%)"] = np.log(df_psi[f"{labels[1]}样本占比"] / df_psi[f"{labels[0]}样本占比"])
    df_psi["分档PSI值"] = (df_psi[f"{labels[1]}% - {labels[0]}%"] * df_psi[f"ln({labels[1]}% / {labels[0]}%)"])
    df_psi = df_psi.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
    df_psi["总体PSI值"] = df_psi["分档PSI值"].sum()
    df_psi["指标名称"] = desc

    if plot:
        x = df_psi['分箱'].apply(lambda l: l if max_len is None or len(str(l)) < max_len else f"{str(l)[:max_len]}...")
        x_indexes = np.arange(len(x))
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.bar(x_indexes - width / 2, df_psi[f'{labels[0]}样本占比'], width, label=f'{labels[0]}样本占比', color=colors[0], hatch="/" if hatch else None)
        ax1.bar(x_indexes + width / 2, df_psi[f'{labels[1]}样本占比'], width, label=f'{labels[1]}样本占比', color=colors[1], hatch="\\" if hatch else None)

        ax1.set_ylabel('样本占比: 分箱内样本数 / 样本总数')
        ax1.set_xticks(x_indexes)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelrotation=90)

        ax2 = ax1.twinx()
        ax2.plot(x, df_psi[f"{labels[0]}坏样本率"], color=colors[0], label=f"{labels[0]}坏样本率", linestyle=(5, (10, 3)))
        ax2.plot(x, df_psi[f"{labels[1]}坏样本率"], color=colors[1], label=f"{labels[1]}坏样本率", linestyle=(5, (10, 3)))

        ax2.scatter(x, df_psi[f"{labels[0]}坏样本率"], marker=".")
        ax2.scatter(x, df_psi[f"{labels[1]}坏样本率"], marker=".")

        ax2.set_ylabel('坏样本率: 坏样本数 / 样本总数')

        fig.suptitle(f"{desc + ' ' if desc else ''}{labels[0]} vs {labels[1]} 群体稳定性指数(PSI): {df_psi['分档PSI值'].sum():.4f}\n\n")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save), exist_ok=True)

            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    if result:
        return df_psi[["指标名称", "分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比", f"{labels[0]}坏样本率", f"{labels[1]}样本数", f"{labels[1]}样本占比", f"{labels[1]}坏样本率", f"{labels[1]}% - {labels[0]}%", f"ln({labels[1]}% / {labels[0]}%)", "分档PSI值", "总体PSI值"]]


def csi_plot(expected, actual, score_bins, labels=["预期", "实际"], desc="", save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True, max_len=None, hatch=True):
    """
    特征 CSI 图

    :param expected: 期望分布情况，传入需要验证的特征分箱表
    :param actual: 实际分布情况，传入需要参照的特征分箱表
    :param score_bins: 逻辑回归模型评分表
    :param labels: 期望分布和实际分布的名称，默认 ["预期", "实际"]
    :param desc: 标题前缀显示的名称，默认为空，推荐传入特征名称或评分卡名字
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param colors: 图片主题颜色，默认即可
    :param figsize: 图像大小，默认 (15, 8)
    :param anchor: 图例显示的位置，默认 0.94，根据实际显示情况进行调整即可，0.94 附近小范围调整
    :param width: 预期分布与实际分布柱状图之间的间隔，默认 0.35
    :param result: 是否返回 CSI 统计表，默认 False
    :param plot: 是否画 CSI图，默认 True
    :param max_len: 特征显示的最大长度，防止特征名称过长导致图像区域非常小，默认 None 表示不限制
    :param hatch: 是否显示柱状图上的斜线，默认为 True
    :return: 当 result 为 True 时，返回 pd.DataFrame
    """
    expected = expected.rename(columns={"样本总数": f"{labels[0]}样本数", "样本占比": f"{labels[0]}样本占比", "坏样本率": f"{labels[0]}坏样本率"})
    actual = actual.rename(columns={"样本总数": f"{labels[1]}样本数", "样本占比": f"{labels[1]}样本占比", "坏样本率": f"{labels[1]}坏样本率"})
    df_csi = expected.merge(actual, on="分箱", how="outer").replace(np.nan, 0)
    df_csi[f"{labels[1]}% - {labels[0]}%"] = df_csi[f"{labels[1]}样本占比"] - df_csi[f"{labels[0]}样本占比"]
    df_csi = df_csi.merge(pd.DataFrame({"分箱": feature_bins(score_bins["bins"]).values(), "对应分数": score_bins["scores"]}), on="分箱", how="left").replace(np.nan, 0)
    df_csi["分档CSI值"] = (df_csi[f"{labels[1]}% - {labels[0]}%"] * df_csi["对应分数"])
    df_csi = df_csi.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
    df_csi["总体CSI值"] = df_csi["分档CSI值"].sum()
    df_csi["指标名称"] = desc

    if plot:
        x = df_csi['分箱'].apply(lambda l: str(l) if pd.isnull(l) or len(str(l)) < max_len else f"{str(l)[:max_len]}...")
        x_indexes = np.arange(len(x))
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.bar(x_indexes - width / 2, df_csi[f'{labels[0]}样本占比'], width, label=f'{labels[0]}样本占比', color=colors[0], hatch="/" if hatch else None)
        ax1.bar(x_indexes + width / 2, df_csi[f'{labels[1]}样本占比'], width, label=f'{labels[1]}样本占比', color=colors[1], hatch="\\" if hatch else None)

        ax1.set_ylabel('样本占比: 分箱内样本数 / 样本总数')
        ax1.set_xticks(x_indexes)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelrotation=90)

        ax2 = ax1.twinx()
        ax2.plot(x, df_csi[f"{labels[0]}坏样本率"], color=colors[0], label=f"{labels[0]}坏样本率", linestyle=(5, (10, 3)))
        ax2.plot(x, df_csi[f"{labels[1]}坏样本率"], color=colors[1], label=f"{labels[1]}坏样本率", linestyle=(5, (10, 3)))

        ax2.scatter(x, df_csi[f"{labels[0]}坏样本率"], marker=".")
        ax2.scatter(x, df_csi[f"{labels[1]}坏样本率"], marker=".")

        ax2.set_ylabel('坏样本率: 坏样本数 / 样本总数')

        fig.suptitle(f"{desc + ' ' if desc else ''}{labels[0]} vs {labels[1]} 特征稳定性指标(CSI): {df_csi['分档CSI值'].sum():.4f}\n\n")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                os.makedirs(os.path.dirname(save), exist_ok=True)

            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    if result:
        return df_csi[["指标名称", "分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比", f"{labels[0]}坏样本率", f"{labels[1]}样本数", f"{labels[1]}样本占比", f"{labels[1]}坏样本率", f"{labels[1]}% - {labels[0]}%", "对应分数", "分档CSI值", "总体CSI值"]]


def dataframe_plot(df, row_height=0.4, font_size=14, header_color='#2639E9', row_colors=['#dae3f3', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, ax=None, save=None, **kwargs):
    """
    将 dataframe 转换为图片，推荐行和列都不多的数据集使用该方法

    :param df: 需要画图的 dataframe 数据
    :param row_height: 行高，默认 0.4
    :param font_size: 字体大小，默认 14
    :param header_color: 标题颜色，默认 #2639E9
    :param row_colors: 行颜色，默认 ['#dae3f3', 'w']，交替使用两种颜色
    :param edge_color: 表格边框颜色，默认白色
    :param bbox: 边的显示情况，[左，右，上，下]，即仅显示上下两条边框
    :param header_columns: 标题行数，默认仅有一个标题行，即 0
    :param ax: 如果需要在某张画布的子图中显示，那么传入对应的 ax 即可
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param kwargs: plt.table 相关的参数，参考：https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.table.html

    :return: Figure
    """
    data = df.copy()
    for col in data.select_dtypes('datetime'):
        data[col] = data[col].dt.strftime("%Y-%m-%d")

    for col in data.select_dtypes('float'):
        data[col] = data[col].apply(lambda x: np.nan if pd.isnull(x) else round(x, 4))

    cols_width = [max(data[col].apply(lambda x: len(str(x).encode())).max(), len(str(col).encode())) / 8. for col in data.columns]

    if ax is None:
        size = (sum(cols_width), (len(data) + 1) * row_height)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, colWidths=cols_width, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    fig.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def distribution_plot(data, date="date", target="target", save=None, figsize=(10, 6), colors=["#2639E9", "#F76E6C", "#FE7715"], freq="M", anchor=0.94, result=False, hatch=True):
    """
    样本时间分布图

    :param data: 数据集
    :param date: 日期列名称，如果格式非日期，会尝试自动转为日期格式，默认 date，替换为数据中对应的日期列（如申请时间、授信时间、放款时间等）
    :param target: 数据集中标签列的名称，默认 target
    :param save: 图片保存的地址，如果传入路径中有文件夹不存在，会新建相关文件夹，默认 None
    :param figsize: 图像大小，默认 (10, 6)
    :param colors: 图片主题颜色，默认即可
    :param freq: 汇总统计的日期格式，按年、季度、月、周、日等统计，参考：https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    :param anchor: 图例显示的位置，默认 0.94，根据实际显示情况进行调整即可，0.94 附近小范围调整
    :param result: 是否返回分布表，默认 False
    :param hatch: 是否显示柱状图上的斜线，默认为 True
    :return:
    """
    df = data.copy()

    if 'time' not in str(df[date].dtype):
        df[date] = pd.to_datetime(df[date])

    temp = df.set_index(date).assign(
        好样本=lambda x: (x[target] == 0).astype(int),
        坏样本=lambda x: (x[target] == 1).astype(int),
    ).resample(freq).agg({"好样本": sum, "坏样本": sum})

    temp.index = [i.strftime("%Y-%m-%d") for i in temp.index]

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    temp.plot(kind='bar', stacked=True, ax=ax1, color=colors[:2], hatch="/" if hatch else None, legend=False)
    ax1.tick_params(axis='x', labelrotation=-90)
    ax1.set(xlabel=None)
    ax1.set_ylabel('样本数')
    ax1.set_title('不同时点数据集样本分布情况\n\n')

    ax2 = plt.twinx()
    (temp["坏样本"] / temp.sum(axis=1)).plot(ax=ax2, color=colors[-1], style="--", linewidth=2, label="坏样本率")
    # sns.despine()

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

    fig.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    if result:
        temp = temp.reset_index().rename(columns={date: "日期", "index": "日期", 0: "好样本", 1: "坏样本"})
        temp["样本总数"] = temp["坏样本"] + temp["好样本"]
        temp["样本占比"] = temp["样本总数"] / temp["样本总数"].sum()
        temp["好样本占比"] = temp["好样本"] / temp["好样本"].sum()
        temp["坏样本占比"] = temp["坏样本"] / temp["坏样本"].sum()
        temp["坏样本率"] = temp["坏样本"] / temp["样本总数"]

        return temp[["日期", "样本总数", "样本占比", "好样本", "好样本占比", "坏样本", "坏样本占比", "坏样本率"]]


def sample_lift_transformer(df, rule, target='target', sample_rate=0.7):
    """采取好坏样本 sample_rate:1 的抽样方式时，计算抽样样本和原始样本上的 lift 指标

    :param df: 原始数据，需全部为数值型变量
    :param rule: Rule
    :param target: 目标变量名称
    :param sample_rate: 好样本采样比例

    :return:
        lift_sam: float, 抽样样本上拒绝人群的lift
        lift_ori: float, 原始样本上拒绝人群的lift
    """
    rj_df = df[rule.predict(df)]
    ps_df = df[~rule.predict(df)]

    # 拒绝样本好坏样本数
    rj = len(rj_df)
    bad_rj = rj_df[target].sum()
    good_rj = rj - bad_rj

    # 通过样本好坏样本数
    ps = len(ps_df)
    bad_ps = ps_df[target].sum()
    good_ps = ps - bad_ps

    # 抽样样本上的lift
    lift_sam = (bad_rj / rj) / ((bad_rj + bad_ps) / (rj + ps))

    # 原始样本上的lift
    lift_ori = bad_rj / (bad_rj + bad_ps) * (1 + (sample_rate * bad_ps + good_ps) / (sample_rate * bad_rj + good_rj))

    return lift_sam, lift_ori


def tasks_executor(tasks, n_jobs=-1, pool="thread"):
    """多进程或多线程任务执行

    :param tasks: 任务
    :param n_jobs: 线城池或进程池数量
    :param pool: 类型，默认 thread 线城池,
    """
    if len(tasks) <= 0:
        raise ValueError("执行任务数必须大于0")

    if pool == "joblib":
        pass

    from concurrent.futures import wait, ALL_COMPLETED
    if pool == "thread":
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else 1)
    elif pool == "process":
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else 1)

    _tasks = []
    for task in tasks:
        executor.submit(task)

    wait(_tasks, return_when=ALL_COMPLETED)

    return [t.result() for t in _tasks]
