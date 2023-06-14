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
import random
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import toad
from optbinning import OptimalBinning

from .logger import init_logger


def seed_everything(seed: int, freeze_torch=False):
    """
    固定随机种子
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
    warnings.filterwarnings("ignore")
    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('display.max_colwidth', 300)
    plt.style.use('seaborn-ticks')
    if font_path:
        if not os.path.isfile(font_path):
            import wget
            font_path = wget.download("https://itlubber.art/upload/matplot_chinese.ttf", 'matplot_chinese.ttf')
    else:
        font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matplot_chinese.ttf')
    
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    
    # for font in font_manager.fontManager.ttflist:
    #     if "hei" in font.fname.split("/")[-1].lower():
    #         plt.rcParams['font.family'] = font.name
    #         break
    
    plt.rcParams['axes.unicode_minus'] = False
    
    if seed:
        seed_everything(seed, freeze_torch=freeze_torch)
    
    if logger:
        return init_logger(**kwargs)


def load_pickle(file):
    return joblib.load(file)


def save_pickle(obj, file):
    joblib.dump(obj, file)
    
    
def germancredit():
    '''
    German Credit Data
    ------
    Credit data that classifies debtors described by a set of attributes as good or bad credit risks. See source link below for detailed information.
    [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))
    '''
    from pandas.api.types import CategoricalDtype
    
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'germancredit.csv'))
    
    cate_levels = {
            "status_of_existing_checking_account": ['... < 0 DM', '0 <= ... < 200 DM', '... >= 200 DM / salary assignments for at least 1 year', 'no checking account'], 
            "credit_history": ["no credits taken/ all credits paid back duly", "all credits at this bank paid back duly", "existing credits paid back duly till now", "delay in paying off in the past", "critical account/ other credits existing (not at this bank)"], 
            "savings_account_and_bonds": ["... < 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM", "... >= 1000 DM", "unknown/ no savings account"],
            "present_employment_since": ["unemployed", "... < 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years", "... >= 7 years"], 
            "personal_status_and_sex": ["male : divorced/separated", "female : divorced/separated/married", "male : single", "male : married/widowed", "female : single"], 
            "other_debtors_or_guarantors": ["none", "co-applicant", "guarantor"], 
            "property": ["real estate",  "building society savings agreement/ life insurance",  "car or other, not in attribute Savings account/bonds",  "unknown / no property"],
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


def round_float(num, decimal = 4):
    if ~pd.isnull(num) and isinstance(num, float):
        return float(str(num).split(".")[0] + "." + str(num).split(".")[1][:decimal])
    else:
        return num
    

def feature_bins(bins, decimal = 4):
    if len(bins) == 0:
        return {0: "全部样本"}
    if isinstance(bins, list): bins = np.array(bins)
    EMPTYBINS = len(bins) if not isinstance(bins[0], (set, list, np.ndarray)) else -1
    
    l = []
    if not isinstance(bins[0], (set, list, np.ndarray)):
        has_empty = len(bins) > 0 and pd.isnull(bins[-1])
        if has_empty: bins = bins[:-1]
        sp_l = ["负无穷"] + [round_float(b, decimal=decimal) for b in bins] + ["正无穷"]
        for i in range(len(sp_l) - 1): l.append('['+str(sp_l[i])+' , '+str(sp_l[i+1])+')')
        if has_empty: l.append('缺失值')
    else:
        for keys in bins:
            keys_update = set()
            for key in keys:
                if pd.isnull(key) or key == "nan":
                    keys_update.add("缺失值")
                elif key.strip() == "":
                    keys_update.add("空字符串")
                else:
                    keys_update.add(key)
            label = ','.join(keys_update)
            l.append(label)

    return {i if b != "缺失值" else EMPTYBINS: b for i, b in enumerate(l)}


def feature_bin_stats(data, feature, target="target", rules={}, method='step', max_n_bins=None, min_bin_size=None, clip_v=None, desc="", verbose=0, combiner=None, ks=True, **kwargs):
    if combiner is None:
        if method not in ['dt', 'chi', 'quantile', 'step', 'kmeans', 'cart']:
            raise "method is the one of ['dt', 'chi', 'quantile', 'step', 'kmeans', 'cart']"
        
        combiner = toad.transform.Combiner()
        
        if method in ["cart", "mdlp", "uniform"]:
            try:
                y = data[target]
                if str(data[feature].dtypes) in ["object", "string", "category"]:
                    dtype = "categorical"
                    x = data[feature].astype("category").values
                else:
                    dtype = "numerical"
                    x = data[feature].values

                _combiner = OptimalBinning(feature, dtype=dtype, max_n_bins=max_n_bins, **kwargs).fit(x, y)
                if _combiner.status == "OPTIMAL":
                    rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.splits] + [[np.nan] if dtype == "categorical" else np.nan]}
                else:
                    raise Exception("optimalBinning error")
            
            except Exception as e:
                _combiner = toad.transform.Combiner()
                _combiner.fit(data[[feature, target]].dropna(), target, method="chi", min_samples=min_bin_size, n_bins=max_n_bins, **kwargs)
                rule = {feature: [s.tolist() if isinstance(s, np.ndarray) else s for s in _combiner.export()[feature]] + [[np.nan] if dtype == "categorical" else np.nan]}
        
            combiner.update(rule)
        else:
            if method in ["step", "quantile"]:
                combiner.fit(data[[feature, target]], y=target, method=method, n_bins=max_n_bins, **kwargs)
            else:
                combiner.fit(data[[feature, target]], y=target, method=method, min_samples=min_bin_size, n_bins=max_n_bins, **kwargs)
    
    if len(rules) > 0:
        if isinstance(rules, (list, np.ndarray)):
            combiner.update({feature: rules})
        else:
            combiner.update(rules)

    feature_bin_dict = feature_bins(np.array(combiner[feature]))
    
    df_bin = combiner.transform(data[[feature, target]], labels=False)
    
    table = df_bin[[feature, target]].groupby([feature, target]).agg(len).unstack()
    table.columns.name = None
    table = table.rename(columns = {0 : '好样本数', 1 : '坏样本数'}).fillna(0)
    if "好样本数" not in table.columns:
        table["好样本数"] = 0
    if "坏样本数" not in table.columns:
        table["坏样本数"] = 0
    
    table["指标名称"] = feature
    table["指标含义"] = desc
    table = table.reset_index().rename(columns={feature: "分箱"})

    table['样本总数'] = table['好样本数'] + table['坏样本数']
    table['样本占比'] = table['样本总数'] / table['样本总数'].sum()
    table['好样本占比'] = table['好样本数'] / table['好样本数'].sum()
    table['坏样本占比'] = table['坏样本数'] / table['坏样本数'].sum()
    table['坏样本率'] = table['坏样本数'] / table['样本总数']
    
    table = table.fillna(0.)
    
    table['分档WOE值'] = table.apply(lambda x : np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)),axis=1)
    table['分档IV值'] = table.apply(lambda x : (x['好样本占比'] - x['坏样本占比']) * np.log(x['好样本占比'] / (x['坏样本占比'] + 1e-6)), axis=1)
    
    table = table.replace(np.inf, 0).replace(-np.inf, 0)
    
    table['指标IV值'] = table['分档IV值'].sum()
    
    table["LIFT值"] = table['坏样本率'] / (table["坏样本数"].sum() / table["样本总数"].sum())
    table["累积LIFT值"] = (table['坏样本数'].cumsum() / table['样本总数'].cumsum()) / (table["坏样本数"].sum() / table["样本总数"].sum())
    
    if ks:
        table = table.sort_values("分箱")
        table["累积好样本数"] = table["好样本数"].cumsum()
        table["累积坏样本数"] = table["坏样本数"].cumsum()
        table["分档KS值"] = table["累积坏样本数"] / table['坏样本数'].sum() - table["累积好样本数"] / table['好样本数'].sum()
    
    table["分箱"] = table["分箱"].map(feature_bin_dict)
    table = table.set_index(['指标名称', '指标含义', '分箱']).reindex([(feature, desc, b) for b in feature_bin_dict.values()]).fillna(0).reset_index()
    
    if ks:
        return table[['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '累积LIFT值', '累积好样本数', '累积坏样本数', '分档KS值']]
    else:
        return table[['指标名称', "指标含义", '分箱', '样本总数', '样本占比', '好样本数', '好样本占比', '坏样本数', '坏样本占比', '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '累积LIFT值']]


def bin_plot(feature_table, desc="", figsize=(10, 6), colors=["#2639E9", "#F76E6C", "#FE7715"], save=None, anchor=0.945, max_len=35):
    """简单策略挖掘：特征分箱图

    :param feature_table: 特征分箱的统计信息表，由 feature_bin_stats 运行得到
    :param desc: 特征中文含义或者其他相关信息
    :param figsize: 图像尺寸大小，传入一个tuple，默认 （10， 6）
    :param colors: 图片主题颜色，默认即可
    :param save: 图片保存路径

    :return Figure
    """
    feature_table = feature_table.copy()

    feature_table["分箱"] = feature_table["分箱"].apply(lambda x: x if not pd.isnull(x) and re.match("^\[.*\)$", x) else (str(x)[:max_len] + ".." if len(str(x)) > max_len else str(x)))

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.barh(feature_table['分箱'], feature_table['好样本数'], color=colors[0], label='好样本', hatch="/")
    ax1.barh(feature_table['分箱'], feature_table['坏样本数'], left=feature_table['好样本数'], color=colors[1], label='坏样本', hatch="\\")
    ax1.set_xlabel('样本数')

    ax2 = ax1.twiny()
    ax2.plot(feature_table['坏样本率'], feature_table['分箱'], colors[2], label='坏样本率', linestyle='-.')
    ax2.set_xlabel('坏样本率: 坏样本数 / 样本总数')

    for i, rate in enumerate(feature_table['坏样本率']):
        ax2.scatter(rate, i, color=colors[2])

    for i, v in feature_table[['样本总数', '好样本数', '坏样本数', '坏样本率']].iterrows():
        ax1.text(v['样本总数'] / 2, i + len(feature_table) / 60, f"{int(v['好样本数'])}:{int(v['坏样本数'])}:{v['坏样本率']:.2%}")

    ax1.invert_yaxis()

    fig.suptitle(f'{desc}分箱图\n\n')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

    plt.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def corr_plot(data, figure_size=(16, 8),  fontsize=14, mask=False, save=None, annot=True, max_len=35):
    if max_len is None:
        corr = data.corr()
    else:
        corr = data.rename(columns={c: c if len(str(c)) <= max_len else f"{str(c)[:max_len]}..." for c in data.columns}).corr()
    
    corr_mask = np.zeros_like(corr, dtype = np.bool)
    corr_mask[np.triu_indices_from(corr_mask)] = True

    map_plot = toad.tadpole.tadpole.heatmap(
        corr,
        mask = corr_mask if mask else None,
        cmap = sns.diverging_palette(267, 267, n=10, s=100, l=40),
        vmax = 1,
        vmin = -1,
        center = 0,
        square = True,
        linewidths = .1,
        annot = annot,
        fmt = '.2f',
        figure_size = figure_size,
    )

    map_plot.tick_params(axis='x', labelrotation=270, labelsize=fontsize)
    map_plot.tick_params(axis='y', labelrotation=0, labelsize=fontsize)
    
    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        
        plt.savefig(save, dpi=240, format="png", bbox_inches="tight")
    
    return map_plot


def ks_plot(score, target, title="", fontsize=14, figsize=(16, 8), save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], anchor=0.945):
        if np.mean(score) < 0 or np.mean(score) > 1:
            warnings.warn('Since the average of pred is not in [0,1], it is treated as predicted score but not probability.')
            score = -score

        df = pd.DataFrame({'label': target, 'pred': score})
        def n0(x): return sum(x==0)
        def n1(x): return sum(x==1)
        df_ks = df.sort_values('pred', ascending=False).reset_index(drop=True) \
            .assign(group=lambda x: np.ceil((x.index+1)/(len(x.index)/len(df.index)))) \
            .groupby('group')['label'].agg([n0, n1]) \
            .reset_index().rename(columns={'n0':'good','n1':'bad'}) \
            .assign(
                group=lambda x: (x.index+1)/len(x.index),
                cumgood=lambda x: np.cumsum(x.good)/sum(x.good), 
                cumbad=lambda x: np.cumsum(x.bad)/sum(x.bad)
            ).assign(ks=lambda x:abs(x.cumbad-x.cumgood))

        fig, ax = plt.subplots(1, 2, figsize = figsize)

        # KS曲线
        dfks = df_ks.loc[lambda x: x.ks==max(x.ks)].sort_values('group').iloc[0]

        ax[0].plot(df_ks.group, df_ks.ks, color=colors[0], label="KS曲线")
        ax[0].plot(df_ks.group, df_ks.cumgood, color=colors[1], label="累积好客户占比")
        ax[0].plot(df_ks.group, df_ks.cumbad, color=colors[2], label="累积坏客户占比")
        ax[0].fill_between(df_ks.group, df_ks.cumbad, df_ks.cumgood, color=colors[0], alpha=0.25)

        ax[0].plot([dfks['group'], dfks['group']], [0, dfks['ks']], 'r--')
        ax[0].text(dfks['group'], dfks['ks'], f"KS: {round(dfks['ks'],4)} at: {dfks.group:.2%}", horizontalalignment='center', fontsize=fontsize)

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
        auc_value = toad.metrics.AUC(score, target)

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


def hist_plot(score, y_true=None, figsize=(15, 10), bins=30, save=None, labels=["坏样本", "好样本"], anchor=1.1, fontsize=14, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    palette = sns.diverging_palette(340, 267, n=2, s=100, l=40)

    sns.histplot(
                x=score, hue=y_true.replace({i: v for i, v in enumerate(labels)}) if y_true is not None else y_true, element="step", stat="probability", bins=bins, common_bins=True, common_norm=True, palette=palette, ax=ax, **kwargs
            )

    sns.despine()
    
    ax.spines['top'].set_color("#2639E9")
    ax.spines['bottom'].set_color("#2639E9")
    ax.spines['right'].set_color("#2639E9")
    ax.spines['left'].set_color("#2639E9")

    ax.set_xlabel("评分分布", fontsize=fontsize)
    ax.set_ylabel("样本占比", fontsize=fontsize)
    
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    if y_true is not None:
        ax.legend(labels[:y_true.nunique()], loc='upper center', ncol=y_true.nunique(), bbox_to_anchor=(0.5, anchor), frameon=False, fontsize=fontsize)
    
    fig.tight_layout()

    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        
        plt.savefig(save, dpi=240, format="png", bbox_inches="tight")
    
    return fig


def psi_plot(expected, actual, labels=["预期", "实际"], save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True, max_len=None):
    expected = expected.rename(columns={"样本总数": f"{labels[0]}样本数", "样本占比": f"{labels[0]}样本占比", "坏样本率": f"{labels[0]}坏样本率"})
    actual = actual.rename(columns={"样本总数": f"{labels[1]}样本数", "样本占比": f"{labels[1]}样本占比", "坏样本率": f"{labels[1]}坏样本率"})
    df_psi = expected.merge(actual, on="分箱", how="outer").replace(np.nan, 0)
    df_psi[f"{labels[1]}% - {labels[0]}%"] = df_psi[f"{labels[1]}样本占比"] - df_psi[f"{labels[0]}样本占比"]
    df_psi[f"ln({labels[1]}% / {labels[0]}%)"] = np.log(df_psi[f"{labels[1]}样本占比"] / df_psi[f"{labels[0]}样本占比"])
    df_psi["分档PSI值"] = (df_psi[f"{labels[1]}% - {labels[0]}%"] * df_psi[f"ln({labels[1]}% / {labels[0]}%)"])
    df_psi = df_psi.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
    df_psi["总体PSI值"] = df_psi["分档PSI值"].sum()
    
    if plot:
        x = df_psi['分箱'].apply(lambda l: l if max_len is None else f"{str(l)[:max_len]}...")
        x_indexes = np.arange(len(x))
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.bar(x_indexes - width / 2, df_psi[f'{labels[0]}样本占比'], width, label=f'{labels[0]}样本占比', color=colors[0], hatch="/")
        ax1.bar(x_indexes + width / 2, df_psi[f'{labels[1]}样本占比'], width, label=f'{labels[1]}样本占比', color=colors[1], hatch="\\")

        ax1.set_ylabel('样本占比: 分箱内样本数 / 样本总数')
        ax1.set_xticks(x_indexes)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelrotation=90)

        ax2 = ax1.twinx()
        ax2.plot(df_psi["分箱"], df_psi[f"{labels[0]}坏样本率"], color=colors[0], label=f"{labels[0]}坏样本率", linestyle=(5, (10, 3)))
        ax2.plot(df_psi["分箱"], df_psi[f"{labels[1]}坏样本率"], color=colors[1], label=f"{labels[1]}坏样本率", linestyle=(5, (10, 3)))

        ax2.scatter(df_psi["分箱"], df_psi[f"{labels[0]}坏样本率"], marker=".")
        ax2.scatter(df_psi["分箱"], df_psi[f"{labels[1]}坏样本率"], marker=".")

        ax2.set_ylabel('坏样本率: 坏样本数 / 样本总数')
        
        fig.suptitle(f"{labels[0]} vs {labels[1]} 群体稳定性指数(PSI): {df_psi['分档PSI值'].sum():.4f}\n\n")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                    os.makedirs(os.path.dirname(save), exist_ok=True)

            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    if result:
        return df_psi[["分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比", f"{labels[0]}坏样本率", f"{labels[1]}样本数", f"{labels[1]}样本占比", f"{labels[1]}坏样本率", f"{labels[1]}% - {labels[0]}%", f"ln({labels[1]}% / {labels[0]}%)", "分档PSI值", "总体PSI值"]]


def csi_plot(expected, actual, score_bins, labels=["预期", "实际"], save=None, colors=["#2639E9", "#F76E6C", "#FE7715"], figsize=(15, 8), anchor=0.94, width=0.35, result=False, plot=True, max_len=None):
    expected = expected.rename(columns={"样本总数": f"{labels[0]}样本数", "样本占比": f"{labels[0]}样本占比", "坏样本率": f"{labels[0]}坏样本率"})
    actual = actual.rename(columns={"样本总数": f"{labels[1]}样本数", "样本占比": f"{labels[1]}样本占比", "坏样本率": f"{labels[1]}坏样本率"})
    df_csi = expected.merge(actual, on="分箱", how="outer").replace(np.nan, 0)
    df_csi[f"{labels[1]}% - {labels[0]}%"] = df_csi[f"{labels[1]}样本占比"] - df_csi[f"{labels[0]}样本占比"]
    df_csi = df_csi.merge(pd.DataFrame({"分箱": feature_bins(score_bins["bins"]).values(), "对应分数": score_bins["scores"]}), on="分箱", how="left").replace(np.nan, 0)
    df_csi["分档CSI值"] = (df_csi[f"{labels[1]}% - {labels[0]}%"] * df_csi["对应分数"])
    df_csi = df_csi.fillna(0).replace(np.inf, 0).replace(-np.inf, 0)
    df_csi["总体CSI值"] = df_csi["分档CSI值"].sum()
    
    if plot:
        x = df_csi['分箱'].apply(lambda l: l if max_len is None else f"{str(l)[:max_len]}...")
        x_indexes = np.arange(len(x))
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.bar(x_indexes - width / 2, df_csi[f'{labels[0]}样本占比'], width, label=f'{labels[0]}样本占比', color=colors[0], hatch="/")
        ax1.bar(x_indexes + width / 2, df_csi[f'{labels[1]}样本占比'], width, label=f'{labels[1]}样本占比', color=colors[1], hatch="\\")

        ax1.set_ylabel('样本占比: 分箱内样本数 / 样本总数')
        ax1.set_xticks(x_indexes)
        ax1.set_xticklabels(x)
        ax1.tick_params(axis='x', labelrotation=90)

        ax2 = ax1.twinx()
        ax2.plot(df_csi["分箱"], df_csi[f"{labels[0]}坏样本率"], color=colors[0], label=f"{labels[0]}坏样本率", linestyle=(5, (10, 3)))
        ax2.plot(df_csi["分箱"], df_csi[f"{labels[1]}坏样本率"], color=colors[1], label=f"{labels[1]}坏样本率", linestyle=(5, (10, 3)))

        ax2.scatter(df_csi["分箱"], df_csi[f"{labels[0]}坏样本率"], marker=".")
        ax2.scatter(df_csi["分箱"], df_csi[f"{labels[1]}坏样本率"], marker=".")

        ax2.set_ylabel('坏样本率: 坏样本数 / 样本总数')
        
        fig.suptitle(f"{labels[0]} vs {labels[1]} 特征稳定性指标(CSI): {df_csi['分档CSI值'].sum():.4f}\n\n")

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', ncol=len(labels1 + labels2), bbox_to_anchor=(0.5, anchor), frameon=False)

        fig.tight_layout()

        if save:
            if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
                    os.makedirs(os.path.dirname(save), exist_ok=True)

            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
    
    if result:
        return df_csi[["分箱", f"{labels[0]}样本数", f"{labels[0]}样本占比", f"{labels[0]}坏样本率", f"{labels[1]}样本数", f"{labels[1]}样本占比", f"{labels[1]}坏样本率", f"{labels[1]}% - {labels[0]}%", "对应分数", "分档CSI值", "总体CSI值"]]


def dataframe_plot(df, row_height=0.4, font_size=14, header_color='#2639E9', row_colors=['#dae3f3', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, ax=None, save=None, **kwargs):
    data = df.copy()
    for col in data.select_dtypes('datetime'):
        data[col] = data[col].dt.strftime("%Y-%m-%d")

    for col in data.select_dtypes('float'):
        data[col] = data[col].apply(lambda x: np.nan if pd.isnull(x) else round(x, 4))

    cols_width = [max(data[col].apply(lambda x:len(str(x).encode())).max(), len(str(col).encode())) / 8. for col in data.columns]

    if ax is None:
        size = (sum(cols_width), (len(data) + 1) * row_height)
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, colWidths=cols_width, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])

    fig.tight_layout()
    
    if save:
        if os.path.dirname(save) != "" and not os.path.exists(os.path.dirname(save)):
            os.makedirs(os.path.dirname(save))

        fig.savefig(save, dpi=240, format="png", bbox_inches="tight")

    return fig


def distribution_plot(df, date="date", target="target", save=None, figsize=(10, 6), colors=["#2639E9", "#F76E6C", "#FE7715"], freq="M", anchor=0.94, result=False):
    temp = df.set_index(date).assign(
        好样本=lambda x: (x[target] == 0).astype(int),
        坏样本=lambda x: (x[target] == 1).astype(int),
    ).resample(freq).agg({"好样本": sum, "坏样本": sum})
    
    temp.index = [i.strftime("%Y-%m-%d") for i in temp.index]

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    temp.plot(kind='bar', stacked=True, ax=ax1, color=colors[:2], hatch="/", legend=False)
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
