# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/18 10:15
@Author  : itlubber
@Site    : itlubber.art
"""
import copy
import datetime
import numpy as np
import pandas as pd

min_num = 40
hit_num = 30
sub_div_bin = 0.1
sample_type_target = {'Total': ['fpd_30_act', 'mob3_dpd_30_act']}
target_ripe = {'fpd_30_act': ['agr_fpd_30'], 'mob3_dpd_30_act': ['agr_mob3_dpd_30']}
target_del_col = {'fpd_30_act': ['agr_fpd_30', 'agr_mob3_dpd_30', 'mob3_dpd_30_act'], 'mob3_dpd_30_act': ['agr_mob3_dpd_30', 'agr_fpd_30', 'fpd_30_act']}
target_min_rate = {'fpd_30_act': [0.015], 'mob3_dpd_30_act': [0.02]}
sample_type_lift = {'Total': {'fpd_30_act': 1.8, 'mob3_dpd_30_act': 3}}


def cv(x):
    """计算一组数的离散系数"""
    m = np.mean(x)
    if m == 0:
        return None
    else:
        return np.std(x) / np.mean(x)


def cv_type(x):
    """基于离散系数判断数据稳定性"""
    if x < 0.15:
        return '非常稳定'
    elif x < 0.4:
        return '相对稳定'
    elif x < 0.75:
        return '不稳定'
    else:
        return '极不稳定'


def get_trend(data, method=np.mean, key='', flag=['高', '低']):
    """判断数据的趋势

    :param data: 需要判断的数据,list类型或者dataframe类型
    :param flag: ['高','低'] 或者 ['快','慢']
    :param method: 次数对应求和,速度对应平均
    :param key:默认为None,如果传入数据为dataframe,则需要指定key,但是必须从小到大排好顺序

    :return: 不同趋势
    """
    if len(data) == 0: return '无数据'

    if isinstance(data, pd.DataFrame):
        if key != None:
            data = list(data[data[key].notnull()][key])
        else:
            raise '没有指定需要判断趋势的列名'

    if len(data) == 0:
        return '无数据'
    elif len(data) == 1:
        return '无趋势'
    elif len(data) > 4:
        n1 = len(data) // 4
        n2 = len(data) % 4
        k1 = n1
        k2 = 2 * n1 + (1 if n2 >= 3 else 0)
        k3 = k2 + n1 + (1 if n2 >= 2 else 0)
        x = [method(data[0:k1]), method(data[k1:k2]), method(data[k2:k3]), method(data[k3:])]
    else:
        x = data

    cmax = max(x)
    cmin = min(x)
    cv = np.std(x) / np.mean(x) if np.mean(x) != 0 else 0

    if cv < 0.15:
        return '平稳'
    elif len(x) == 2:
        if cmax == x[0]:
            return '%s > %s' % (flag[0], flag[1])
        else:
            return '%s > %s' % (flag[1], flag[0])
    elif len(x) == 3:
        if cmax == x[1]:
            return '%s > %s > %s' % (flag[1], flag[0], flag[1])
        elif cmin == x[1]:
            return '%s > %s > %s' % (flag[0], flag[1], flag[0])
        elif cmax == x[0] and cmin == x[2]:
            return '%s > %s' % (flag[0], flag[1])
        else:
            return '%s > %s' % (flag[1], flag[0])
    else:
        if cmax == x[1] or cmax == x[2]:
            return '%s > %s > %s' % (flag[1], flag[0], flag[1])
        elif cmin == x[1] or cmin == x[2]:
            return '%s > %s > %s' % (flag[0], flag[1], flag[0])
        elif cmax == x[0] and cmin == x[3]:
            return '%s > %s' % (flag[0], flag[1])
        else:
            return '%s > %s' % (flag[1], flag[0])


def calculate_var_lift(data, data_sub, hit_flag, target, cut_point, direction):
    """计算lift

    data: 全量样本
    data_sub: 满足条件的样本
    hit_flag: 触碰标签
    target: string 需要计算lift的列名
    """
    if '>' in direction:
        data_hit = data_sub[data_sub[hit_flag] > cut_point] if len(data_sub) > 0 else data_sub
        bad = len(data_hit[data_hit[target] == 1]) if len(data_hit) > 0 else np.nan
        hit_bad_rate = bad / len(data_hit) if len(data_hit) > 0 else np.nan
    else:
        data_hit = data_sub[data_sub[hit_flag] <= cut_point] if len(data_sub) > 0 else data_sub
        bad = len(data_hit[data_hit[target] == 1]) if len(data_hit) > 0 else np.nan
        hit_bad_rate = bad / len(data_hit) if len(data_hit) > 0 else np.nan
    bad = len(data[data[target] == 1])
    allhit_bad_rate = bad / len(data) if len(data) > 0 else np.nan
    return hit_bad_rate / allhit_bad_rate if allhit_bad_rate > 0 else np.nan


def calculate_var_odds(data, data_sub, hit_flag, target, cut_point, direction):
    """计算odds ratio
    
    data: DataFrame 成熟的样本子集的补集(未触碰所有样本)
    data_sub : 成熟的样本子集（如某个省份的数据集）
    hit_flag: 触碰标签
    target: string 需要计算odds ratio的列名
    """
    if '>' in direction:
        data_hit = data_sub[data_sub[hit_flag] > cut_point] if len(data_sub) > 0 else data_sub
        bad = len(data_hit[data_hit[target] == 1]) if len(data_hit) > 0 else np.nan
        good = len(data_hit[data_hit[target] != 1]) if len(data_hit) > 0 else np.nan
        hit_bad_good_rate = bad / good if good > 0 else np.nan
        # data_nohit = data[data[hit_flag] <= cut_point]
        bad_nohit = len(data[data[target] == 1])
        good_nohit = len(data[data[target] != 1])
        nohit_bad_good_rate = bad_nohit / good_nohit if good_nohit > 0 else np.nan
    else:
        data_hit = data_sub[data_sub[hit_flag] <= cut_point] if len(data_sub) > 0 else data_sub
        bad = len(data_hit[data_hit[target] == 1]) if len(data_hit) > 0 else np.nan
        good = len(data_hit[data_hit[target] != 1]) if len(data_hit) > 0 else np.nan
        hit_bad_good_rate = bad / good if good > 0 else np.nan
        # data_nohit = data[data[hit_flag] > cut_point]
        bad_nohit = len(data[data[target] == 1])
        good_nohit = len(data[data[target] != 1])
        nohit_bad_good_rate = bad_nohit / good_nohit if good_nohit > 0 else np.nan
    return hit_bad_good_rate / nohit_bad_good_rate if nohit_bad_good_rate > 0 else np.nan


# 1.7按月对规则进行泛化，分析规则触碰情况和风险表现情况
def get_month_odds(data, rule_name, rule_type, cut_point, target, rule_limit, rule_all, use_credit_flag, circle_mth, mth, direction, var):
    """
    :param data: 数据框
    :param rule_name: 规则名
    :param rule_type: 规则类型
    :param cut_point: 切点，cut-off
    :param target: 目标字段
    :param rule_limit: 泛化样本类型
    :param rule_all: 线上已有规则触碰情况对应的字段列名
    :param use_credit_flag: 是否用信列
    :param circle_mth: 申请月对应的列
    :param mth: 要泛化的具体月
    :param direction: 变量取值方向
    :param var: 泛化变量名
    :return:  按月泛化分析结果
    """
    data = copy.deepcopy(data)
    data = data[data[circle_mth] == mth]
    dic = {}
    dic['报告日期'] = datetime.datetime.now().strftime('%Y-%m-%d')
    if rule_limit == 'Total':
        len_prod = len(data['product_name'].unique())
        if len_prod >= 3:
            dic['产品名称'] = str(data['product_name'].values[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data['product_name'].unique()[0]) + '&' + str(data['product_name'].unique()[1])
        else:
            dic['产品名称'] = str(data['product_name'].unique()[0])
    elif rule_limit == '其他':
        data_prod = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]['product_name'].unique()
        len_prod = len(data_prod)
        if len_prod >= 3:
            dic['产品名称'] = str(data_prod[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data_prod[0]) + '&' + str(data_prod[1])
        else:
            dic['产品名称'] = str(data_prod[0])
    elif rule_limit == sample_type_col[rule_limit][1][0]:
        data_prod = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]['product_name'].unique()
        len_prod = len(data_prod)
        if len_prod >= 3:
            dic['产品名称'] = str(data_prod[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data_prod[0]) + '&' + str(data_prod[1])
        else:
            dic['产品名称'] = str(data_prod[0])
    dic['样本类型'] = rule_limit
    dic['风险类型'] = '短期风险' if target == 'fpd_30_act' else '中长期风险'
    dic['目标字段'] = target
    dic['规则名称'] = rule_name
    dic['规则类型'] = rule_type
    dic['申请月'] = mth
    dic['申请量'] = len(data)
    dic['通过量'] = len(data[data[rule_all] == 0])
    dic['通过率'] = dic['通过量'] / dic['申请量'] if dic['申请量'] > 0 else None
    dic['用信量'] = len(data[data[use_credit_flag] == 1])
    dic['用信率'] = dic['用信量'] / dic['通过量'] if dic['通过量'] > 0 else None
    dic['置前触碰量'] = len(data[data[rule_all] == 1])
    ## 统计逾期成熟样本量
    tmp15 = data[data['agr_fpd_15'] == 1]
    tmp = data[data[target_ripe[target][0]] == 1]
    if '>' in direction:
        if rule_limit == 'Total':
            dic['策略触碰量'] = len(data[data[var] > cut_point])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point)])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point)])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | (data[var] > cut_point)])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['整体成熟量'] = len(tmp)
            dic['额外触碰成熟量'] = len(tmp[(tmp[rule_all] == 0) & (tmp[var] > cut_point)])  # len(tmp[tmp[var] > cut_point])
            dic['整体成熟坏样本量'] = len(tmp[tmp[target] == 1])
            dic['额外触碰成熟坏样本量'] = len(tmp[(tmp[rule_all] == 0) & (tmp[target] == 1) & (tmp[var] > cut_point)])
            dic['fpd15整体成熟量'] = len(tmp15)
            dic['fpd15额外触碰成熟量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] > cut_point)])
            dic['fpd15整体成熟坏样本量'] = len(tmp15[tmp15['fpd_15_act'] == 1])
            dic['fpd15额外触碰成熟坏样本量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] > cut_point) & (tmp15['fpd_15_act'] == 1)])
            dic['整体逾期率'] = len(tmp[tmp[target] == 1]) / dic['整体成熟量'] if dic['整体成熟量'] > 0 else None
            dic['额外触碰逾期率'] = dic['额外触碰成熟坏样本量'] / dic['额外触碰成熟量'] if dic['额外触碰成熟量'] > 0 else None
            dic['置后触碰整体逾期率'] = (dic['整体成熟坏样本量'] - dic['额外触碰成熟坏样本量']) / (dic['整体成熟量'] - dic['额外触碰成熟量']) if (dic['整体成熟量'] - dic['额外触碰成熟量']) > 0 else None
            dic['逾期率下降值'] = dic['整体逾期率'] - dic['置后触碰整体逾期率'] if dic['置后触碰整体逾期率'] != None else None
            dic['逾期率下降幅度'] = (dic['整体逾期率'] - dic['置后触碰整体逾期率']) / dic['整体逾期率'] if (dic['整体逾期率'] != None and dic['置后触碰整体逾期率'] != None and
                                                                                                            dic['整体逾期率'] != 0) else None
            tmp_sub = tmp
            tmp_nohit = tmp[(tmp[var] <= cut_point)]
            tmp_sub15 = tmp15
            tmp_nohit15 = tmp15[tmp15[var] <= cut_point]
            dic['额外触碰Odds'] = calculate_var_odds(data=tmp_nohit, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰Lift'] = calculate_var_lift(data=tmp, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Odds'] = calculate_var_odds(data=tmp_nohit15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Lift'] = calculate_var_lift(data=tmp15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == '其他':
            dic['策略触碰量'] = len(data[(data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['整体成熟量'] = len(tmp)
            dic['额外触碰成熟量'] = len(tmp[(tmp[rule_all] == 0) & (tmp[var] > cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['整体成熟坏样本量'] = len(tmp[(tmp[target] == 1)])
            dic['额外触碰成熟坏样本量'] = len(tmp[(tmp[rule_all] == 0) & (tmp[target] == 1) & (tmp[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟量'] = len(tmp15)
            dic['fpd15额外触碰成熟量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] > cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟坏样本量'] = len(tmp15[tmp15['fpd_15_act'] == 1])
            dic['fpd15额外触碰成熟坏样本量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] > cut_point) &
                                                         (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])) & (tmp15['fpd_15_act'] == 1)])
            dic['整体逾期率'] = len(tmp[tmp[target] == 1]) / dic['整体成熟量'] if dic['整体成熟量'] > 0 else None
            dic['额外触碰逾期率'] = dic['额外触碰成熟坏样本量'] / dic['额外触碰成熟量'] if dic['额外触碰成熟量'] > 0 else None
            dic['置后触碰整体逾期率'] = (dic['整体成熟坏样本量'] - dic['额外触碰成熟坏样本量']) / (dic['整体成熟量'] - dic['额外触碰成熟量']) if (dic['整体成熟量'] - dic['额外触碰成熟量']) > 0 else None
            dic['逾期率下降值'] = dic['整体逾期率'] - dic['置后触碰整体逾期率'] if dic['置后触碰整体逾期率'] != None else None
            dic['逾期率下降幅度'] = (dic['整体逾期率'] - dic['置后触碰整体逾期率']) / dic['整体逾期率'] if (dic['整体逾期率'] != None and dic['置后触碰整体逾期率'] != None and
                                                                                                            dic['整体逾期率'] != 0) else None
            tmp_sub = tmp[tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]
            tmp_nohit = tmp[(((tmp[var] <= cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))) |
                             (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))]
            tmp_sub15 = tmp15[tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]
            tmp_nohit15 = tmp15[(((tmp15[var] <= cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))) |
                                 (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))]
            dic['额外触碰Odds'] = calculate_var_odds(data=tmp_nohit, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰Lift'] = calculate_var_lift(data=tmp, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Odds'] = calculate_var_odds(data=tmp_nohit15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Lift'] = calculate_var_lift(data=tmp15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            dic['策略触碰量'] = len(data[(data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['整体成熟量'] = len(tmp)
            dic['额外触碰成熟量'] = len(tmp[(tmp[var] > cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['整体成熟坏样本量'] = len(tmp[(tmp[target] == 1)])
            dic['额外触碰成熟坏样本量'] = len(tmp[(tmp[rule_all] == 0) & (tmp[target] == 1) & (tmp[var] > cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟量'] = len(tmp15)
            dic['fpd15额外触碰成熟量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] > cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟坏样本量'] = len(tmp15[tmp15['fpd_15_act'] == 1])
            dic['fpd15额外触碰成熟坏样本量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] > cut_point) &
                                                         (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])) & (tmp15['fpd_15_act'] == 1)])
            dic['整体逾期率'] = len(tmp[tmp[target] == 1]) / dic['整体成熟量'] if dic['整体成熟量'] > 0 else None
            dic['额外触碰逾期率'] = dic['额外触碰成熟坏样本量'] / dic['额外触碰成熟量'] if dic['额外触碰成熟量'] > 0 else None
            dic['置后触碰整体逾期率'] = (dic['整体成熟坏样本量'] - dic['额外触碰成熟坏样本量']) / (dic['整体成熟量'] - dic['额外触碰成熟量']) if (dic['整体成熟量'] - dic['额外触碰成熟量']) > 0 else None
            dic['逾期率下降值'] = dic['整体逾期率'] - dic['置后触碰整体逾期率'] if dic['置后触碰整体逾期率'] != None else None
            dic['逾期率下降幅度'] = (dic['整体逾期率'] - dic['置后触碰整体逾期率']) / dic['整体逾期率'] if (dic['整体逾期率'] != None and dic['置后触碰整体逾期率'] != None and
                                                                                                            dic['整体逾期率'] != 0) else None
            tmp_sub = tmp[tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]
            tmp_nohit = tmp[(((tmp[var] <= cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))) | (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))]
            tmp_sub15 = tmp15[tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]
            tmp_nohit15 = tmp15[(((tmp15[var] <= cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))) |
                                 (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))]
            dic['额外触碰Odds'] = calculate_var_odds(data=tmp_nohit, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰Lift'] = calculate_var_lift(data=tmp, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Odds'] = calculate_var_odds(data=tmp_nohit15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Lift'] = calculate_var_lift(data=tmp15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
    else:
        if rule_limit == 'Total':
            dic['策略触碰量'] = len(data[data[var] <= cut_point])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point)])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point)])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | (data[var] <= cut_point)])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['整体成熟量'] = len(tmp)
            dic['额外触碰成熟量'] = len(tmp[tmp[var] <= cut_point])
            dic['整体成熟坏样本量'] = len(tmp[tmp[target] == 1])
            dic['额外触碰成熟坏样本量'] = len(tmp[(tmp[rule_all] == 0) & (tmp[target] == 1) & (tmp[var] <= cut_point)])
            dic['fpd15整体成熟量'] = len(tmp15)
            dic['fpd15额外触碰成熟量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] <= cut_point)])
            dic['fpd15整体成熟坏样本量'] = len(tmp15[tmp15['fpd_15_act'] == 1])
            dic['fpd15额外触碰成熟坏样本量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] <= cut_point) & (tmp15['fpd_15_act'] == 1)])
            dic['整体逾期率'] = len(tmp[tmp[target] == 1]) / dic['整体成熟量'] if dic['整体成熟量'] > 0 else None
            dic['额外触碰逾期率'] = dic['额外触碰成熟坏样本量'] / dic['额外触碰成熟量'] if dic['额外触碰成熟量'] > 0 else None
            dic['置后触碰整体逾期率'] = (dic['整体成熟坏样本量'] - dic['额外触碰成熟坏样本量']) / (dic['整体成熟量'] - dic['额外触碰成熟量']) if (dic['整体成熟量'] - dic['额外触碰成熟量']) > 0 else None
            dic['逾期率下降值'] = dic['整体逾期率'] - dic['置后触碰整体逾期率'] if dic['置后触碰整体逾期率'] != None else None
            dic['逾期率下降幅度'] = (dic['整体逾期率'] - dic['置后触碰整体逾期率']) / dic['整体逾期率'] if (dic['整体逾期率'] != None and dic['置后触碰整体逾期率'] != None and
                                                                                                            dic['整体逾期率'] != 0) else None
            tmp_sub = tmp
            tmp_nohit = tmp[(tmp[var] > cut_point)]
            tmp_sub15 = tmp15
            tmp_nohit15 = tmp15[tmp15[var] > cut_point]
            dic['额外触碰Odds'] = calculate_var_odds(data=tmp_nohit, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰Lift'] = calculate_var_lift(data=tmp, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Odds'] = calculate_var_odds(data=tmp_nohit15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Lift'] = calculate_var_lift(data=tmp15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == '其他':
            dic['策略触碰量'] = len(data[(data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['整体成熟量'] = len(tmp)
            dic['额外触碰成熟量'] = len(tmp[(tmp[var] <= cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['整体成熟坏样本量'] = len(tmp[(tmp[target] == 1)])
            dic['额外触碰成熟坏样本量'] = len(tmp[(tmp[target] == 1) & (tmp[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟量'] = len(tmp15)
            dic['fpd15额外触碰成熟量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] <= cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟坏样本量'] = len(tmp15[tmp15['fpd_15_act'] == 1])
            dic['fpd15额外触碰成熟坏样本量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] <= cut_point) &
                                                         (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])) & (tmp15['fpd_15_act'] == 1)])
            dic['整体逾期率'] = len(tmp[tmp[target] == 1]) / dic['整体成熟量'] if dic['整体成熟量'] > 0 else None
            dic['额外触碰逾期率'] = dic['额外触碰成熟坏样本量'] / dic['额外触碰成熟量'] if dic['额外触碰成熟量'] > 0 else None
            dic['置后触碰整体逾期率'] = (dic['整体成熟坏样本量'] - dic['额外触碰成熟坏样本量']) / (dic['整体成熟量'] - dic['额外触碰成熟量']) if (dic['整体成熟量'] - dic['额外触碰成熟量']) > 0 else None
            dic['逾期率下降值'] = dic['整体逾期率'] - dic['置后触碰整体逾期率'] if dic['置后触碰整体逾期率'] != None else None
            dic['逾期率下降幅度'] = (dic['整体逾期率'] - dic['置后触碰整体逾期率']) / dic['整体逾期率'] if (dic['整体逾期率'] != None and dic['置后触碰整体逾期率'] != None and
                                                                                                            dic['整体逾期率'] != 0) else None
            tmp_sub = tmp[tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]
            tmp_nohit = tmp[(((tmp[var] > cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))) |
                             (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))]
            tmp_sub15 = tmp15[tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]
            tmp_nohit15 = tmp15[(((tmp15[var] > cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))) |
                                 (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))]
            dic['额外触碰Odds'] = calculate_var_odds(data=tmp_nohit, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰Lift'] = calculate_var_lift(data=tmp, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Odds'] = calculate_var_odds(data=tmp_nohit15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Lift'] = calculate_var_lift(data=tmp15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            dic['策略触碰量'] = len(data[(data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['整体成熟量'] = len(tmp)
            dic['额外触碰成熟量'] = len(tmp[(tmp[var] <= cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['整体成熟坏样本量'] = len(tmp[(tmp[target] == 1)])
            dic['额外触碰成熟坏样本量'] = len(tmp[(tmp[target] == 1) & (tmp[var] <= cut_point) & (data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟量'] = len(tmp15)
            dic['fpd15额外触碰成熟量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] <= cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['fpd15整体成熟坏样本量'] = len(tmp15[tmp15['fpd_15_act'] == 1])
            dic['fpd15额外触碰成熟坏样本量'] = len(tmp15[(tmp15[rule_all] == 0) & (tmp15[var] <= cut_point) &
                                                         (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])) & (tmp15['fpd_15_act'] == 1)])
            dic['整体逾期率'] = len(tmp[tmp[target] == 1]) / dic['整体成熟量'] if dic['整体成熟量'] > 0 else None
            dic['额外触碰逾期率'] = dic['额外触碰成熟坏样本量'] / dic['额外触碰成熟量'] if dic['额外触碰成熟量'] > 0 else None
            dic['置后触碰整体逾期率'] = (dic['整体成熟坏样本量'] - dic['额外触碰成熟坏样本量']) / (dic['整体成熟量'] - dic['额外触碰成熟量']) if (dic['整体成熟量'] - dic['额外触碰成熟量']) > 0 else None
            dic['逾期率下降值'] = dic['整体逾期率'] - dic['置后触碰整体逾期率'] if dic['置后触碰整体逾期率'] != None else None
            dic['逾期率下降幅度'] = (dic['整体逾期率'] - dic['置后触碰整体逾期率']) / dic['整体逾期率'] if (dic['整体逾期率'] != None and dic['置后触碰整体逾期率'] != None and
                                                                                                            dic['整体逾期率'] != 0) else None
            tmp_sub = tmp[tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]
            tmp_nohit = tmp[(((tmp[var] > cut_point) & (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))) |
                             (tmp[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))]
            tmp_sub15 = tmp15[tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]
            tmp_nohit15 = tmp15[(((tmp15[var] > cut_point) & (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))) |
                                 (tmp15[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))]
            dic['额外触碰Odds'] = calculate_var_odds(data=tmp_nohit, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰Lift'] = calculate_var_lift(data=tmp, data_sub=tmp_sub, hit_flag=var, target=target, cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Odds'] = calculate_var_odds(data=tmp_nohit15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            dic['额外触碰fpd15_Lift'] = calculate_var_lift(data=tmp15, data_sub=tmp_sub15, hit_flag=var, target='fpd_15_act', cut_point=cut_point, direction=direction)
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
    return df_ls


# 1.8按周对规则进行泛化，分析规则触碰情况
def get_weeks_hit(data, rule_name, rule_type, cut_point, target, rule_limit, rule_all, use_credit_flag, circle_week, week, direction, var):
    """
    :param data:  数据框
    :param rule_name: 规则名
    :param rule_type: 规则类型
    :param cut_point: 切点，cut-off
    :param target: 目标字段
    :param rule_limit: 泛化样本类型
    :param rule_all: 线上已有规则触碰情况对应的字段列名
    :param use_credit_flag: 是否用信列
    :param circle_week: 申请周对应的列
    :param week: 要泛化的具体周
    :param direction: 变量取值方向
    :param var: 泛化变量名
    :return:  按周泛化分析结果
    """
    data = copy.deepcopy(data)
    data = data[data[circle_week] == week]
    dic = {}
    dic['报告日期'] = datetime.datetime.now().strftime('%Y-%m-%d')
    if rule_limit == 'Total':
        len_prod = len(data['product_name'].unique())
        if len_prod >= 3:
            dic['产品名称'] = str(data['product_name'].values[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data['product_name'].unique()[0]) + '&' + str(data['product_name'].unique()[1])
        else:
            dic['产品名称'] = str(data['product_name'].unique()[0])
    elif rule_limit == '其他':
        data_prod = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]['product_name'].unique()
        len_prod = len(data_prod)
        if len_prod >= 3:
            dic['产品名称'] = str(data_prod[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data_prod[0]) + '&' + str(data_prod[1])
        else:
            dic['产品名称'] = str(data_prod[0])
    elif rule_limit == sample_type_col[rule_limit][1][0]:
        data_prod = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]['product_name'].unique()
        len_prod = len(data_prod)
        if len_prod >= 3:
            dic['产品名称'] = str(data_prod[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data_prod[0]) + '&' + str(data_prod[1])
        else:
            dic['产品名称'] = str(data_prod[0])
    dic['样本类型'] = rule_limit
    dic['风险类型'] = '短期风险' if target == 'fpd_30_act' else '中长期风险'
    dic['目标字段'] = target
    dic['规则名称'] = rule_name
    dic['规则类型'] = rule_type
    dic['申请周'] = week
    dic['申请量'] = len(data)
    dic['通过量'] = len(data[data[rule_all] == 0])
    dic['通过率'] = dic['通过量'] / dic['申请量'] if dic['申请量'] > 0 else None
    dic['用信量'] = len(data[data[use_credit_flag] == 1])
    dic['用信率'] = dic['用信量'] / dic['通过量'] if dic['通过量'] > 0 else None
    dic['置前触碰量'] = len(data[data[rule_all] == 1])
    if '>' in direction:
        if rule_limit == 'Total':
            dic['策略触碰量'] = len(data[data[var] > cut_point])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point)])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point)])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | (data[var] > cut_point)])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == '其他':
            dic['策略触碰量'] = len(data[(data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            dic['策略触碰量'] = len(data[(data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
    else:
        if rule_limit == 'Total':
            dic['策略触碰量'] = len(data[data[var] <= cut_point])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point)])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point)])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | (data[var] <= cut_point)])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == '其他':
            dic['策略触碰量'] = len(data[(data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            dic['策略触碰量'] = len(data[(data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
    return df_ls


# 1.9按日对规则进行泛化，分析规则触碰情况
def get_days_hit(data, rule_name, rule_type, cut_point, target, rule_limit, rule_all, use_credit_flag, circle_day, day, direction, var):
    """
    :param data:  数据框
    :param rule_name: 规则名
    :param rule_type: 规则类型
    :param cut_point: 切点，cut-off
    :param target: 目标字段
    :param rule_limit: 泛化样本类型
    :param rule_all: 线上已有规则触碰情况对应的字段列名
    :param use_credit_flag: 是否用信列
    :param circle_day: 申请日对应的列
    :param day: 要泛化的具体日
    :param direction: 变量取值方向
    :param var: 泛化变量名
    :return:  按日泛化分析结果
    """
    data = copy.deepcopy(data)
    data = data[data[circle_day] == day]
    dic = {}
    dic['报告日期'] = datetime.datetime.now().strftime('%Y-%m-%d')
    if rule_limit == 'Total':
        len_prod = len(data['product_name'].unique())
        if len_prod >= 3:
            dic['产品名称'] = str(data['product_name'].values[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data['product_name'].unique()[0]) + '&' + str(data['product_name'].unique()[1])
        else:
            dic['产品名称'] = str(data['product_name'].unique()[0])
    elif rule_limit == '其他':
        data_prod = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]['product_name'].unique()
        len_prod = len(data_prod)
        if len_prod >= 3:
            dic['产品名称'] = str(data_prod[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data_prod[0]) + '&' + str(data_prod[1])
        else:
            dic['产品名称'] = str(data_prod[0])
    elif rule_limit == sample_type_col[rule_limit][1][0]:
        data_prod = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]['product_name'].unique()
        len_prod = len(data_prod)
        if len_prod >= 3:
            dic['产品名称'] = str(data_prod[0]) + '等' + str(len_prod) + '个产品'
        elif len_prod >= 2:
            dic['产品名称'] = str(data_prod[0]) + '&' + str(data_prod[1])
        else:
            dic['产品名称'] = str(data_prod[0])
    dic['样本类型'] = rule_limit
    dic['风险类型'] = '短期风险' if target == 'fpd_30_act' else '中长期风险'
    dic['目标字段'] = target
    dic['规则名称'] = rule_name;
    dic['规则类型'] = rule_type;
    dic['申请日'] = day;
    dic['申请量'] = len(data);
    dic['通过量'] = len(data[data[rule_all] == 0])
    dic['通过率'] = dic['通过量'] / dic['申请量'] if dic['申请量'] > 0 else None
    dic['用信量'] = len(data[data[use_credit_flag] == 1])
    dic['用信率'] = dic['用信量'] / dic['通过量'] if dic['通过量'] > 0 else None
    dic['置前触碰量'] = len(data[data[rule_all] == 1])
    if '>' in direction:
        if rule_limit == 'Total':
            dic['策略触碰量'] = len(data[data[var] > cut_point])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point)])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point)])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | (data[var] > cut_point)])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == '其他':
            dic['策略触碰量'] = len(data[(data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            dic['策略触碰量'] = len(data[(data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] > cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
    else:
        if rule_limit == 'Total':
            dic['策略触碰量'] = len(data[data[var] <= cut_point])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point)])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point)])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | (data[var] <= cut_point)])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == '其他':
            dic['策略触碰量'] = len(data[(data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            dic['策略触碰量'] = len(data[(data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['重复触碰量'] = len(data[(data[rule_all] == 1) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['额外触碰量'] = len(data[(data[rule_all] == 0) & (data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1]))])
            dic['置后触碰量'] = len(data[(data[rule_all] == 1) | ((data[var] <= cut_point) & (
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])))])
            dic['置前触碰率'] = dic['置前触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['策略触碰率'] = dic['策略触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰率'] = dic['重复触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['额外触碰率'] = dic['额外触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['置后触碰率'] = dic['置后触碰量'] / dic['申请量'] if dic['申请量'] > 0 else None
            dic['重复触碰占策略触碰比例'] = dic['重复触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            dic['额外触碰占策略触碰比例'] = dic['额外触碰量'] / dic['策略触碰量'] if dic['策略触碰量'] > 0 else None
            df_ls = pd.DataFrame.from_dict(dic, orient='index').T
    return df_ls


# 1.10 按月泛化结果汇总
def get_mths_result(data, rule_name, rule_type, var, cut_point, target, rule_limit, rule_all, use_credit_flag, circle_mth, direction, base_lift, subset_total):
    """
    :param data:  数据框
    :param rule_name: 规则名
    :param rule_type: 规则类型
    :param var: 泛化变量名
    :param cut_point: 切点，cut-off
    :param target: 目标字段
    :param rule_limit: 泛化样本类型
    :param rule_all: 线上已有规则触碰情况对应的字段列名
    :param use_credit_flag: 是否用信列
    :param circle_mth: 申请月对应的列
    :param direction: 变量取值方向
    :param base_lift: 临界Lift
    :param subset_total: 是否获取每个子集作为计算整体
    :return:  所有泛化月泛化结果汇总
    """
    if subset_total:  ##获取每个计算整体
        if rule_limit == 'Total':
            data = data
        elif rule_limit == '其他':
            data = data[
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            data = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]
    mths = sorted(set(data[circle_mth]))
    df_contra_monthly = get_month_odds(data=data, rule_name=rule_name, rule_type=rule_type,
                                       cut_point=cut_point, target=target, rule_limit=rule_limit, rule_all=rule_all, use_credit_flag=use_credit_flag, direction=direction,
                                       circle_mth=circle_mth, mth=mths[0], var=var)

    for mth in mths[1:]:
        df2_contra_monthly = get_month_odds(data=data, rule_name=rule_name, rule_type=rule_type,
                                            cut_point=cut_point, target=target, rule_limit=rule_limit, rule_all=rule_all, use_credit_flag=use_credit_flag, direction=direction,
                                            circle_mth=circle_mth, mth=mth, var=var)

        df_contra_monthly = pd.concat([df_contra_monthly, df2_contra_monthly])
    df_contra_monthly['额外触碰Lift大于目标值的月份占比'] = sum(df_contra_monthly['额外触碰Lift'] > base_lift) / sum(
        df_contra_monthly['额外触碰Lift'].notnull()) if sum(df_contra_monthly['额外触碰Lift'].notnull()) > 0 else None
    df_contra_monthly['置前触碰率离散度'] = cv_type(cv(df_contra_monthly['置前触碰率']) if df_contra_monthly['置前触碰率'].sum() > 0 else 0)
    df_contra_monthly['策略触碰率离散度'] = cv_type(cv(df_contra_monthly['策略触碰率']) if df_contra_monthly['策略触碰率'].sum() > 0 else 0)
    df_contra_monthly['重复触碰率离散度'] = cv_type(cv(df_contra_monthly['重复触碰率']) if df_contra_monthly['重复触碰率'].sum() > 0 else 0)
    df_contra_monthly['额外触碰率离散度'] = cv_type(cv(df_contra_monthly['额外触碰率']) if df_contra_monthly['额外触碰率'].sum() > 0 else 0)
    df_contra_monthly['置后触碰率离散度'] = cv_type(cv(df_contra_monthly['置后触碰率']) if df_contra_monthly['置后触碰率'].sum() > 0 else 0)
    try:
        df_contra_monthly['评估结论'] = '建议上线' if (df_contra_monthly['额外触碰Lift大于目标值的月份占比'].values[0] >= 0.8) \
                                                      & ((df_contra_monthly['额外触碰率离散度'].values[0] == '非常稳定') | (df_contra_monthly['策略触碰率离散度'].values[0] == '相对稳定')) else '不建议上线'
    except:
        df_contra_monthly['评估结论'] = '不建议上线'
    return df_contra_monthly


# 1.11 按周泛化结果汇总
def get_weeks_result(data, rule_name, rule_type, var, cut_point, target, rule_limit, rule_all, use_credit_flag, circle_week, direction, subset_total):
    """
    :param data:  数据框
    :param rule_name: 规则名
    :param rule_type: 规则类型
    :param var: 泛化变量名
    :param cut_point: 切点，cut-off
    :param target: 目标字段
    :param rule_limit: 泛化样本类型
    :param rule_all: 线上已有规则触碰情况对应的字段列名
    :param use_credit_flag: 是否用信列
    :param circle_week: 申请周对应的列
    :param direction: 变量取值方向
    :param subset_total: 是否获取每个子集作为计算整体
    :return:  所有泛化周泛化结果汇总
    """
    if subset_total:  ##获取每个计算整体
        if rule_limit == 'Total':
            data = data
        elif rule_limit == '其他':
            data = data[
                data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            data = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]
    td = datetime.datetime.now().strftime('%Y-%m-%d')
    max_week = data[circle_week].max()
    if td <= max_week:
        weeks = sorted(set(data[circle_week]))[-11:-1]
        data = data[data[circle_week].map(lambda x: x in weeks)]
    else:
        weeks = sorted(set(data[circle_week]))[-10:]
        data = data[data[circle_week].map(lambda x: x in weeks)]
    df_monthly = get_weeks_hit(data=data, rule_name=rule_name, rule_type=rule_type, cut_point=cut_point, target=target, rule_limit=rule_limit, rule_all=rule_all,
                               use_credit_flag=use_credit_flag, direction=direction, circle_week=circle_week, week=weeks[0], var=var)

    for week in weeks[1:]:
        df2_monthly = get_weeks_hit(data=data, rule_name=rule_name, rule_type=rule_type, cut_point=cut_point, target=target, rule_limit=rule_limit, rule_all=rule_all,
                                    use_credit_flag=use_credit_flag, direction=direction, circle_week=circle_week, week=week, var=var)

        df_monthly = pd.concat([df_monthly, df2_monthly])
    df_monthly['置前触碰率离散度'] = cv_type(cv(df_monthly['置前触碰率']) if df_monthly['置前触碰率'].sum() > 0 else 0)
    df_monthly['策略触碰率离散度'] = cv_type(cv(df_monthly['策略触碰率']) if df_monthly['策略触碰率'].sum() > 0 else 0)
    df_monthly['重复触碰率离散度'] = cv_type(cv(df_monthly['重复触碰率']) if df_monthly['重复触碰率'].sum() > 0 else 0)
    df_monthly['额外触碰率离散度'] = cv_type(cv(df_monthly['额外触碰率']) if df_monthly['额外触碰率'].sum() > 0 else 0)
    df_monthly['置后触碰率离散度'] = cv_type(cv(df_monthly['置后触碰率']) if df_monthly['置后触碰率'].sum() > 0 else 0)
    return df_monthly


# 1.12 按日泛化结果汇总
def get_days_result(data, rule_name, rule_type, var, cut_point, target, rule_limit, rule_all, use_credit_flag, circle_day, direction, subset_total):
    """
    :param data:  数据框
    :param rule_name: 规则名
    :param rule_type: 规则类型
    :param var: 泛化变量名
    :param cut_point: 切点，cut-off
    :param target: 目标字段
    :param rule_limit: 泛化样本类型
    :param rule_all: 线上已有规则触碰情况对应的字段列名
    :param use_credit_flag: 是否用信列
    :param circle_day: 申请日对应的列
    :param direction: 变量取值方向
    :param subset_total: 是否获取每个子集作为计算整体
    :return:  所有泛化日泛化结果汇总
    """
    if subset_total:  ##获取每个计算整体
        if rule_limit == 'Total':
            data = data
        elif rule_limit == '其他':
            data = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) not in sample_type_col[rule_limit][1])]
        elif rule_limit == sample_type_col[rule_limit][1][0]:
            data = data[data[sample_type_col[rule_limit][0]].map(lambda x: str(x) in sample_type_col[rule_limit][1])]
    days = sorted(set(data[circle_day]))[-10:]
    data = data[data[circle_day].map(lambda x: x in days)]
    df_monthly = get_days_hit(data=data, rule_name=rule_name, rule_type=rule_type, cut_point=cut_point, target=target, rule_limit=rule_limit, rule_all=rule_all,
                              use_credit_flag=use_credit_flag, direction=direction, circle_day=circle_day, day=days[0], var=var)

    for day in days[1:]:
        df2_monthly = get_days_hit(data=data, rule_name=rule_name, rule_type=rule_type, cut_point=cut_point, target=target, rule_limit=rule_limit, rule_all=rule_all,
                                   use_credit_flag=use_credit_flag, direction=direction, circle_day=circle_day, day=day, var=var)

        df_monthly = pd.concat([df_monthly, df2_monthly])
    df_monthly['置前触碰率离散度'] = cv_type(cv(df_monthly['置前触碰率']) if df_monthly['置前触碰率'].sum() > 0 else 0)
    df_monthly['策略触碰率离散度'] = cv_type(cv(df_monthly['策略触碰率']) if df_monthly['策略触碰率'].sum() > 0 else 0)
    df_monthly['重复触碰率离散度'] = cv_type(cv(df_monthly['重复触碰率']) if df_monthly['重复触碰率'].sum() > 0 else 0)
    df_monthly['额外触碰率离散度'] = cv_type(cv(df_monthly['额外触碰率']) if df_monthly['额外触碰率'].sum() > 0 else 0)
    df_monthly['置后触碰率离散度'] = cv_type(cv(df_monthly['置后触碰率']) if df_monthly['置后触碰率'].sum() > 0 else 0)
    return df_monthly


def rule_combine_results(data, rules_dict, rule_all, use_credit_flag, circle_mth, circle_week, circle_day, base_lift, subset_total):
    """基于数据字典泛化结果汇总并输出

    :param data:  需要分析的数据
    :param rules_dict:  阈值测算输出的规则字典
    :param rule_all:  已有所有规则汇总后的字段名称
    :param use_credit_flag: 是否用信列
    :param circle_mth: 申请月对应的列
    :param circle_week: 申请周对应的列
    :param circle_day: 申请日对应的列
    :param base_lift: 临界Lift
    :param subset_total: 是否获取每个子集作为计算整体

    :return:   泛化结果汇总输出
    """
    for var in rules_dict.Var.unique():
        rules_dict01 = rules_dict[rules_dict.Var == var]
        min_index = rules_dict01.index.min()
        starttime_init = datetime.datetime.now()
        for row in rules_dict01.loc[min_index:min_index].iterrows():
            print('正在泛化的规则为：' + row[1]['Rule_Name'] + '\n' + '规则适用范围为：' + str(row[1]['Rule_Limit']) + '\n' + '目标字段为：' + row[1]['Target'])
            print('泛化的变量名称为:', var)
            rule_name = row[1]['Rule_Name']
            sample_type1 = str(row[1]['Rule_Limit'])
            rule_type = row[1]['Rule_Type']
            target = row[1]['Target']
            cut_point = row[1]['Threshold']
            var = var
            direction = row[1]['Direction']
            df_monthly = get_mths_result(data=data
                                         , rule_name=rule_name
                                         , rule_type=rule_type
                                         , var=var
                                         , cut_point=cut_point
                                         , target=target
                                         , rule_limit=sample_type1
                                         , rule_all=rule_all
                                         , use_credit_flag=use_credit_flag
                                         , circle_mth=circle_mth
                                         , direction=direction
                                         , base_lift=base_lift
                                         , subset_total=subset_total)
            df_weekly = get_weeks_result(data=data
                                         , rule_name=rule_name
                                         , rule_type=rule_type
                                         , var=var
                                         , cut_point=cut_point
                                         , target=target
                                         , rule_limit=sample_type1
                                         , rule_all=rule_all
                                         , use_credit_flag=use_credit_flag
                                         , circle_week=circle_week
                                         , direction=direction
                                         , subset_total=subset_total)
            df_daily = get_days_result(data=data
                                       , rule_name=rule_name
                                       , rule_type=rule_type
                                       , var=var
                                       , cut_point=cut_point
                                       , target=target
                                       , rule_limit=sample_type1
                                       , rule_all=rule_all
                                       , use_credit_flag=use_credit_flag
                                       , circle_day=circle_day
                                       , direction=direction
                                       , subset_total=subset_total)
        if len(rules_dict01) > 1:
            for row in rules_dict01.loc[(min_index + 1):].iterrows():
                print('正在泛化的规则为：' + row[1]['Rule_Name'] + '\n' + '规则适用范围为：' + str(
                    row[1]['Rule_Limit']) + '\n' + '目标字段为：' + row[1]['Target'])
                print('泛化的变量名称为:', var)
                rule_name = row[1]['Rule_Name']
                sample_type1 = row[1]['Rule_Limit']
                rule_type = row[1]['Rule_Type']
                target = row[1]['Target']
                cut_point = row[1]['Threshold']
                var = var
                direction = row[1]['Direction']
                df_monthly_01 = get_mths_result(data=data, rule_name=rule_name, rule_type=rule_type, var=var, cut_point=cut_point, target=target, rule_limit=sample_type1, rule_all=rule_all,
                                                use_credit_flag=use_credit_flag, circle_mth=circle_mth, direction=direction, base_lift=base_lift, subset_total=subset_total)
                df_monthly = pd.concat([df_monthly, df_monthly_01])
                df_weekly_01 = get_weeks_result(data=data, rule_name=rule_name, rule_type=rule_type, var=var, cut_point=cut_point, target=target, rule_limit=sample_type1, rule_all=rule_all,
                                                use_credit_flag=use_credit_flag, circle_week=circle_week, direction=direction, subset_total=subset_total)
                df_weekly = pd.concat([df_weekly, df_weekly_01])
                df_daily_01 = get_days_result(data=data, rule_name=rule_name, rule_type=rule_type, var=var, cut_point=cut_point, target=target, rule_limit=sample_type1, rule_all=rule_all,
                                              use_credit_flag=use_credit_flag, circle_day=circle_day, direction=direction, subset_total=subset_total)
                df_daily = pd.concat([df_daily, df_daily_01])

        df_monthly.insert(loc=0, column="序号", value=df_monthly["规则名称"].map({rule: index for index, rule in enumerate(df_monthly['规则名称'].unique(), start=1)}).values)
        df_weekly.insert(loc=0, column="序号", value=df_weekly["规则名称"].map({rule: index for index, rule in enumerate(df_weekly['规则名称'].unique(), start=1)}).values)
        df_daily.insert(loc=0, column="序号", value=df_daily["规则名称"].map({rule: index for index, rule in enumerate(df_daily['规则名称'].unique(), start=1)}).values)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 结果输出 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        writer = ExcelWriter()

        # 相关规则说明
        worksheet = writer.get_sheet_by_name("0.说明")

        end_row, _ = writer.insert_value2sheet(worksheet, "B2", value="1、规则说明", style="header_middle", end_space="E2", align={"horizontal": "left"})

        _, end_col = writer.insert_value2sheet(worksheet, (end_row + 1, 2), value="规则类型", style="header_middle")
        end_row, _ = writer.insert_value2sheet(worksheet, (end_row + 1, end_col), value="规则上线标准", style="header_middle")

        _, end_col = writer.insert_value2sheet(worksheet, (end_row, 2), value="HC", style="middle")
        end_row, _ = writer.insert_value2sheet(worksheet, (end_row, end_col), value="准入类规则需结合产品和适用场景进行考虑", style="middle", align={"horizontal": "left"})

        _, end_col = writer.insert_value2sheet(worksheet, (end_row, 2), value="FR/CR", style="last_middle")
        # end_row, _ = writer.insert_value2sheet(worksheet, (end_row, end_col), value=f"同时满足如下条件：\r\n1.近n个月额外触碰Lift>={base_lift}的月份占比>=80%\r\n2.按月统计的额外触碰率离散度稳定（非常稳定或相对稳定）\r\n注：追踪月份至少 >= 3", style="last_middle", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(worksheet, (end_row, end_col), value=f'="同时满足如下条件：" & CHAR(10) & "1.近n个月额外触碰Lift>={base_lift}的月份占比>=80%" & CHAR(10) & "2.按月统计的额外触碰率离散度稳定（非常稳定或相对稳定）" & CHAR(10) & "注：追踪月份至少 >= 3"', style="last_middle", align={"horizontal": "left", "wrap_text": True})

        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row + 2}", value="2、稳定性划分标准", style="header_middle", end_space=f"E{end_row + 2}", align={"horizontal": "left"})
        cv_describe = pd.DataFrame({"序号": [1, 2, 3, 4], "波动幅度": ["[0%, 15%)", "[15%, 40%)", "[40%, 75%)", "[75%, 正无穷)"], "类型": ["非常稳定", "相对稳定", "不稳定", "极不稳定"], "备注": ["波动幅度=离散度"] * 4})
        end_row, _ = dataframe2excel(cv_describe, writer, sheet_name=worksheet, start_row=end_row + 1, merge_column=["备注"], merge=True)

        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row + 2}", value="3、统计指标含义", style="header_middle", end_space=f"E{end_row + 2}", align={"horizontal": "left"})
        end_row, _ = dataframe2excel(word_desc, writer, sheet_name=worksheet, start_row=end_row + 1)

        # 规则增益效果评估
        worksheet = writer.get_sheet_by_name("1.规则泛化")

        # 评估规则明细
        end_row, _ = writer.insert_value2sheet(worksheet, f"B2", value="0、规则逻辑", style="header_middle", end_space=f"AY2")
        end_row, _ = dataframe2excel(rules_dict01, writer, sheet_name=worksheet, start_row=end_row + 1)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 每月规则泛化 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row + 2}", value="1、月度规则泛化", style="header_middle", end_space=f"AY{end_row + 2}", align={"horizontal": "left"})

        # 每月规则命中拒绝及效果评估
        text1, text2, text3 = std_result_output(df_monthly, "M")
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text1, style="merge_middle", end_space=f"AY{end_row}", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text2, style="merge_middle", end_space=f"AY{end_row}", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text3, style="middle", end_space=f"AY{end_row}", align={"horizontal": "left"})

        end_row, _ = dataframe2excel(df_monthly, writer, sheet_name=worksheet, start_row=end_row + 1, merge_column=["序号", "报告日期", "产品名称", "样本类型"], merge=True)

        # # 规则增益效果评估
        # end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row + 2}", value="2、规则效能", style="header_middle", end_space=f"AY{end_row + 2}")

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 每周规则泛化 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row + 2}", value="2、每周规则泛化", style="header_middle", end_space=f"AY{end_row + 2}", align={"horizontal": "left"})

        text1, text2, text3 = std_result_output(df_weekly, "W")
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text1, style="merge_middle", end_space=f"AY{end_row}", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text2, style="merge_middle", end_space=f"AY{end_row}", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text3, style="middle", end_space=f"AY{end_row}", align={"horizontal": "left"})

        end_row, _ = dataframe2excel(df_weekly, writer, sheet_name=worksheet, start_row=end_row + 1, merge_column=["序号", "报告日期", "产品名称", "样本类型"], merge=True)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 每日规则泛化 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row + 2}", value="3、每日规则泛化", style="header_middle", end_space=f"AY{end_row + 2}", align={"horizontal": "left"})

        text1, text2, text3 = std_result_output(df_daily, "D")
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text1, style="merge_middle", end_space=f"AY{end_row}", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text2, style="merge_middle", end_space=f"AY{end_row}", align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(worksheet, f"B{end_row}", value=text3, style="middle", end_space=f"AY{end_row}", align={"horizontal": "left"})

        end_row, _ = dataframe2excel(df_daily, writer, sheet_name=worksheet, start_row=end_row + 1, merge_column=["序号", "报告日期", "产品名称", "样本类型"], merge=True)

        writer.save(path_rule + var + '_single_rule_' + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + '.xlsx')


# 2.9 规则合并泛化时  按月、周、日输出分析结果函数
def std_result_output_01(wb, sheetname, data, offset):
    """
    泛化结果输出函数
    """
    nrows, ncols = data.shape
    first_title = wb.add_format(title_dic)
    body_text_title = wb.add_format(subtitle_format)
    body_text_center = wb.add_format(body_text_format_01)
    body_text_left = wb.add_format(body_text_left_format_01)
    body_text_percent = wb.add_format(body_text_per_format_01)
    body_text_center2 = wb.add_format(body_text_format_02)
    body_text_left2 = wb.add_format(body_text_left_format_02)
    body_text_percent2 = wb.add_format(body_text_per_format_02)
    ws = wb.add_worksheet(sheetname)
    ws.hide_gridlines({'option': 1})
    ws.autofilter(offset + 5, offset, offset + 5 + nrows, ncols)
    ws.set_column(0, 0, 2)
    ws.freeze_panes(7, 10)  ## 冻结单元格
    ws.merge_range(offset, offset, offset, ncols, sheetname, first_title)
    if '规则泛化' in sheetname:
        data_01 = data[-3:]
        data_02 = data[data['额外触碰成熟量'] >= 10][-3:]
        data_03 = data[data['整体逾期率'] > 0][-3:]
        hit_rate = sum(data_01['额外触碰量']) / sum(data_01['申请量'])
        risk_double = (data_02['额外触碰成熟坏样本量'].sum() / data_02['额外触碰成熟量'].sum()) / (sum(data_02['整体成熟坏样本量']) / sum(data_02['整体成熟量'])) if sum(data_02['整体成熟坏样本量']) > 0 else np.nan
        overdue_minus = sum(data_03['整体成熟坏样本量']) / sum(data_03['整体成熟量']) - sum(data_03['整体成熟坏样本量'] - data_03['额外触碰成熟坏样本量']) / sum(data_03['整体成熟量'] - data_03['额外触碰成熟量']) if sum(data_03['整体成熟量'] - data_03['额外触碰成熟量']) > 0 else np.nan
        overdue_range = overdue_minus / (sum(data_03['整体成熟坏样本量']) / sum(data_03['整体成熟量'])) if sum(data_03['整体成熟坏样本量']) > 0 else np.nan
        text1 = "1.策略上线后通过率下降预估值 = 近3个月额外触碰量之和/近3个月申请量之和；策略上线后逾期率变化预估值 = （近n个有表现月逾期样本之和/近n个有表现月成熟样本之和）-（近n个有表现月置后触碰坏样本之和/近n个有表现月置后触碰成熟样本之和） ；备注：取整体逾期率大于0的近n个月计算策略上线后逾期率变化值，n默认取3，若不足3个月，有几个月计算几个月;"
        text2 = "2.策略上线后额外触碰Lift预估值 = (近n个有表现月额外触碰成熟坏样本量之和/近n个有表现月额外触碰成熟量之和)/(近n个有表现月整体成熟坏样本量之和/近n个有表现月整体成熟量之和)；备注：取额外触碰成熟量大于等于10的近n个月计算策略上线后额外触碰Lift预估值，n默认取3，若不足3个月，有几个月计算几个月;"
        if overdue_minus > 0:
            text3 = "3.策略上线后通过率预计下降：{:.2f}%，逾期率下降：{:.2f}%，逾期率下降幅度：{:.2f}%，额外触碰Lift预估值为：{:.2f}。".format(hit_rate * 100, overdue_minus * 100, overdue_range * 100, risk_double)
        else:
            text3 = "3.策略上线后通过率预计下降：{:.2f}%，逾期率上升：{:.2f}%，逾期率上升幅度：{:.2f}%，额外触碰Lift预估值为：{:.2f}。".format(hit_rate * 100, abs(overdue_minus) * 100, abs(overdue_range) * 100, risk_double)
    if 'Weekly' in sheetname:
        week_num = str(len(data['申请周'].unique()))
        mean_obs = data['置前触碰率'].mean() * 100
        cv_rate = cv(data['置前触碰率'])
        if data['置前触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='置前触碰率')
        text1 = "1.近" + week_num + "个申请周平均置前触碰率%.2f" % mean_obs + '%, ' + '置前触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + ';置前触碰率整体趋势：' + trend + "；"
        mean_obs = data['策略触碰率'].mean() * 100
        cv_rate = cv(data['策略触碰率'])
        if data['策略触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='策略触碰率')
        text2 = "2.近" + week_num + "个申请周策略触碰率%.2f" % mean_obs + '%, ' + '策略触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 策略触碰率整体趋势：' + trend + "；"
        mean_obs = data['额外触碰率'].mean() * 100
        cv_rate = cv(data['额外触碰率'])
        if data['额外触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='额外触碰率')
        text3 = "3.近" + week_num + "个申请周额外触碰率%.2f" % mean_obs + '%, ' + '额外触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 额外触碰率整体趋势：' + trend + "；"
    if 'Dayly' in sheetname:
        day_num = str(len(data['申请日'].unique()))
        mean_obs = data['置前触碰率'].mean() * 100
        cv_rate = cv(data['置前触碰率'])
        if data['置前触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='置前触碰率')
        text1 = "1.近" + day_num + "个申请日平均置前触碰率%.2f" % mean_obs + '%, ' + '置前触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + ';置前触碰率整体趋势：' + trend + "；"
        mean_obs = data['策略触碰率'].mean() * 100
        cv_rate = cv(data['策略触碰率'])
        if data['策略触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='策略触碰率')
        text2 = "2.近" + day_num + "个申请日策略触碰率%.2f" % mean_obs + '%, ' + '策略触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 策略触碰率整体趋势：' + trend + "；"
        mean_obs = data['额外触碰率'].mean() * 100
        cv_rate = cv(data['额外触碰率'])
        if data['额外触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='额外触碰率')
        text3 = "3.近" + day_num + "个申请日额外触碰率%.2f" % mean_obs + '%, ' + '额外触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 额外触碰率整体趋势：' + trend + "；"
    ws.merge_range(offset + 1, offset, offset + 1, ncols, text1, body_text_left2)
    ws.merge_range(offset + 2, offset, offset + 2, ncols, text2, body_text_left2)
    ws.merge_range(offset + 3, offset, offset + 3, ncols, text3, body_text_left2)
    ws.set_row(offset + 4, 7)
    column = data.columns
    need_len = get_max_len(data=data, nrows=nrows, ncols=ncols, max_len=40)
    for i in range(len(need_len)):
        ws.set_column(i + 1, i + 1, need_len[i])
    data = data.replace(np.nan, '')
    data = data.replace(np.inf, 'Inf')
    for j in range(ncols):
        ws.write(offset + 5, j + 1, column[j], body_text_title)
    for i in range(nrows):
        for j in range(ncols):
            if '量' in column[j]:
                ws.conditional_format(offset + 6, j + 1, offset + 6 + nrows - 1, j + 1, condition_format_green_no)
            elif (('率' in column[j]) or ('占比' in column[j])):
                ws.conditional_format(offset + 6, j + 1, offset + 6 + nrows - 1, j + 1, condition_format_pink_no)
            elif 'Odds' in column[j] or 'Lift' in column[j] or '风险' in column[j]:
                ws.conditional_format(offset + 6, j + 1, offset + 6 + nrows - 1, j + 1, condition_format_red_no)
            elif '%' in column[j] or '比例' in column[j]:
                ws.conditional_format(offset + 6, j + 1, offset + 6 + nrows - 1, j + 1, condition_format_blue_no)
            value = data.iloc[i][j]
            if int(data.iloc[i, 0] % 2) == 1:
                if '%' in column[j] or 'rate' in column[j] or '占比' in column[j] or '率' in column[j] or '比例' in column[j]:
                    ws.write(i + offset + 6, j + 1, value, body_text_percent)
                else:
                    ws.write(i + offset + 6, j + 1, value, body_text_center)
            else:
                if '%' in column[j] or 'rate' in column[j] or '占比' in column[j] or '率' in column[j] or '比例' in column[j]:
                    ws.write(i + offset + 6, j + 1, value, body_text_percent2)
                else:
                    ws.write(i + offset + 6, j + 1, value, body_text_center2)
    ws.set_row(offset + 5 + nrows + 1, 7)
    ws.merge_range(offset + 5 + nrows + 2, offset, offset + 5 + nrows + 2, ncols, '2.规则效能', first_title)
    ws.set_row(offset + 5 + nrows + 3, 7)
    if '规则泛化' in sheetname:
        add_combine_plot(wb=wb, ws=ws, column=column, label_x_start=offset + 5, x_start=offset + 5 + 1, x_end=offset + 5 + 1 + nrows,
                         title='规则效能分析', x_label='申请月', y1_list=['额外触碰Lift', '额外触碰fpd15_Lift'],
                         y2_list=['置前触碰率', '策略触碰率', '重复触碰率', '额外触碰率', '置后触碰率'],
                         x_title='申请月', y1_title='额外触碰Lift', y2_title='规则触碰率', width=750, height=500, if_combine=True,
                         chart1_type='column', chart2_type='line', chart_row=offset + 5 + nrows + 4, chart_col=1)
    if 'Weekly' in sheetname:
        add_plot(wb=wb, ws=ws, column=column, label_x_start=offset + 5, x_start=offset + 5 + 1,
                 x_end=offset + 6 + nrows, title='规则效能分析', x_label='申请周',
                 y1_list=['置前触碰率', '策略触碰率', '重复触碰率', '额外触碰率', '置后触碰率'],
                 x_title='申请周', y1_title='规则触碰率', width=720, height=500, chart1_type='line', chart_row=offset + 5 + nrows + 4, chart_col=1)
    if 'Dayly' in sheetname:
        add_plot(wb=wb, ws=ws, column=column, label_x_start=offset + 5, x_start=offset + 5 + 1,
                 x_end=offset + 5 + nrows, title='规则效能分析', x_label='申请日',
                 y1_list=['置前触碰率', '策略触碰率', '重复触碰率', '额外触碰率', '置后触碰率'],
                 x_title='申请日', y1_title='规则触碰率', width=720, height=500, chart1_type='line',
                 chart_row=offset + 5 + nrows + 4, chart_col=1)


# 1.14 规则合并泛化结果汇总并输出
def rule_combine_results_01(data, rule_all, use_credit_flag, cut_point, rule_name, rule_name_chinese,
                            rule_type, rule_limit, circle_mth, circle_week, circle_day, target, direction, var, base_lift, subset_total):
    """
    :param data:  需要分析的数据
    :param rule_all:  已有所有规则汇总后的字段名称
    :param use_credit_flag: 是否用信列
    :param cut_point: 切点，cut-off
    :param rule_name: 泛化规则英文名
    :param rule_name_chinese: 泛化规则中文名
    :param rule_type: 泛化规则类型
    :param rule_limit: 泛化样本类型
    :param circle_mth: 申请月对应的列
    :param circle_week: 申请周对应的列
    :param circle_day: 申请日对应的列
    :param target: 目标字段
    :param direction: 变量取值方向
    :param var: 泛化变量名
    :param base_lift: 临界Lift
    :param subset_total: 是否获取每个子集作为计算整体
    :return:   泛化结果汇总输出
    """
    ana_date = datetime.datetime.now().strftime('%Y-%m-%d')
    rules_dict01 = pd.DataFrame(
        {'Ana_Date': [ana_date], 'Rule_Name': [rule_name], 'Description': [rule_name_chinese], 'Rule_Type': [rule_type],
         'Rule_Limit': [rule_limit], 'Target': [target], 'Threshold': [cut_point], 'Direction': [direction], 'var': [var]})
    starttime_init = datetime.datetime.now()
    print('程序开始执行时间为: ' + str(starttime_init))
    print('正在泛化的规则为：' + rule_name + '\n' + '规则适用范围为：' + rule_limit + '\n' + '目标字段为：' + target)
    df_monthly = get_mths_result(data=data, rule_name=rule_name, rule_type=rule_type, var=var,
                                 cut_point=cut_point, direction=direction, target=target, rule_limit=rule_limit, rule_all=rule_all, use_credit_flag=use_credit_flag,
                                 circle_mth=circle_mth, base_lift=base_lift, subset_total=subset_total)
    df_weekly = get_weeks_result(data=data, rule_name=rule_name, rule_type=rule_type, var=var,
                                 cut_point=cut_point, direction=direction, target=target, rule_limit=rule_limit, rule_all=rule_all, use_credit_flag=use_credit_flag,
                                 circle_week=circle_week, subset_total=subset_total)
    df_daily = get_days_result(data=data, rule_name=rule_name, rule_type=rule_type, var=var,
                               cut_point=cut_point, direction=direction, target=target, rule_limit=rule_limit, rule_all=rule_all, use_credit_flag=use_credit_flag,
                               circle_day=circle_day, subset_total=subset_total)
    cols = df_monthly.columns.tolist()
    df_monthly['序号'] = 1
    for i, j in enumerate(list(df_monthly['规则名称'].unique())):
        df_monthly['序号'][df_monthly['规则名称'] == j] = (i + 1)
    col1 = ['序号']
    col1.extend(cols)
    df_monthly = df_monthly[col1]

    cols = df_weekly.columns.tolist()
    df_weekly['序号'] = 1
    for i, j in enumerate(list(df_weekly['规则名称'].unique())):
        df_weekly['序号'][df_weekly['规则名称'] == j] = (i + 1)
    col1 = ['序号']
    col1.extend(cols)
    df_weekly = df_weekly[col1]

    cols = df_daily.columns.tolist()
    df_daily['序号'] = 1
    for i, j in enumerate(list(df_daily['规则名称'].unique())):
        df_daily['序号'][df_daily['规则名称'] == j] = (i + 1)
    col1 = ['序号']
    col1.extend(cols)
    df_daily = df_daily[col1]
    endtime_init = datetime.datetime.now()
    print('程序结束执行时间为: ' + str(endtime_init))
    print('程序执行时间为: ' + str(endtime_init - starttime_init))

    # 结果输出
    wb = xlsxwriter.Workbook(path_rule + rule_name + '_' + str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + '.xlsx')
    description_output(wb=wb, sheetname='0.说明', df1=rules_dict01, df2=word_desc, offset=1, base_lift=base_lift)
    std_result_output_01(wb=wb, sheetname='1.规则泛化', data=df_monthly, offset=1)
    std_result_output_01(wb=wb, sheetname='2A.Weekly触碰模拟', data=df_weekly, offset=1)
    std_result_output_01(wb=wb, sheetname='2B.Dayly触碰模拟', data=df_daily, offset=1)
    wb.close()


"""
2.规则泛化分析结果输出相关函数
"""

# 2.1 excel格式设置
biaotou = '#44546A'
title = '#44546A'
title_sub1 = '#00868B'
title_sub2 = '#00E5EE'
text = '#F2F2F2'
xunhuan1 = '#D1D1D1'
xunhuan2 = '#E3E3E3'
title_size = 12
text_title_size = 9
text_size = 8
split_color = 'white'

# 条件格式: 蓝色数据条，不指定最大、最小值，绿色
condition_format_green_no = {'type': 'data_bar', 'bar_solid': True, 'data_bar_2010': True,
                             'bar_color': '#65d97d'}
# 条件格式: 黄色数据条，不指定最大、最小值，橙色
condition_format_red_no = {'type': 'data_bar', 'bar_solid': True, 'data_bar_2010': True,
                           'bar_color': '#f2572d'}
# 条件格式: 黄色数据条，不指定最大、最小值，橙色
condition_format_pink_no = {'type': 'data_bar', 'bar_solid': True, 'data_bar_2010': True,
                            'bar_color': '#FF69B4'}
condition_format_blue_no = {'type': 'data_bar', 'bar_solid': True, 'data_bar_2010': True,
                            'bar_color': '#1E90FF'}
condition_format_3_color = {'type': '3_color_scale',
                            'max_color': '#F8696B', 'mid_color': '#FFEB84', 'min_color': '#63BE7B'}

#  总标题
title_dic = {'bold': True, 'font_name': 'Arial', 'font_size': title_size, 'font_color': 'white',
             'top_color': biaotou, 'bottom_color': biaotou, 'left_color': biaotou,
             'right_color': biaotou, 'bg_color': biaotou}

# 表格正文: 边框白色，字体12，背景深灰色、居中
subtitle_format = {'border': True, 'font_size': text_title_size, 'font_name': 'Arial', 'font_color': 'white',
                   'left_color': 'white', 'right_color': 'white', 'bg_color': biaotou,
                   'align': 'center', 'valign': 'vcenter'}

# 表格正文: 边框白色，字体12，背景灰色1、居中
body_text_format_01 = {'border': True, 'font_size': text_size, 'font_name': 'Arial',
                       'bg_color': text, 'top_color': split_color,
                       'bottom_color': split_color,
                       'left_color': split_color,
                       'right_color': split_color,
                       'align': 'center', 'valign': 'vcenter'}

# 正文左对齐
body_text_left_format_01 = copy.deepcopy(body_text_format_01)
body_text_left_format_01['align'] = 'left'
# 正文为比率
body_text_per_format_01 = copy.deepcopy(body_text_format_01)
body_text_per_format_01['num_format'] = '0.00%'

# 表格正文: 边框白色，字体12，背景灰色2、居中
body_text_format_02 = {'border': True, 'font_size': text_size, 'font_name': 'Arial',
                       'bg_color': xunhuan2, 'top_color': split_color,
                       'bottom_color': split_color,
                       'left_color': split_color,
                       'right_color': split_color,
                       'align': 'center', 'valign': 'vcenter'}


# 正文左对齐
body_text_left_format_02 = copy.deepcopy(body_text_format_02)
body_text_left_format_02['align'] = 'left'
# 正文为比率
body_text_per_format_02 = copy.deepcopy(body_text_format_02)
body_text_per_format_02['num_format'] = '0.00%'


# 2.2 规则稳定性说明
type_dic = pd.DataFrame([[1, '0%~15%', '非常稳定'], [2, '15%~40%', '相对稳定'], [3, '40%~75%', '不稳定'], [4, '>75%', '极不稳定']])


# 2.3自动获取单元格内容的length，如果包含汉字，则一个汉字计数为2，如果是英文，则计数为1
def get_same_len(x):
    """
    :param x: 输入内容
    :return: 字符串长度
    """
    import re
    if type(x) != str:
        x = str(x)
    num = 0
    for i in list(x):
        if re.match("[\u4e00-\u9fa5]+", i):
            num = num + 2
        else:
            num = num + 1
    return num


# 2.4 自动获取每一列的最大长度
def get_max_len(data, nrows, ncols, max_len=60):
    """
    自动计算每一列的长度
    （1）先获得每一列的最长的一个单元格内容的长度
    （2）如果超过最大长度限制，则只取最大程度，其余内容自动换行，否则无需限制
    字符串length对应到excel的单元格列宽需要缩小到0.73684，具体缩放比例可以修改
    """
    data = data.fillna('')
    res = []
    res1 = []
    column = data.columns
    for i in range(len(column)):
        x = column[[i]][0]
        res1.append(get_same_len(x))
    res2 = []
    for i in range(ncols):
        tmp = []
        for j in range(nrows):
            tmp.append(get_same_len(data.iloc[j, i]))
        tmp = max(tmp)
        if tmp > max_len:
            tmp = max_len
        res2.append(tmp * 0.73684)
    for i, j in zip(res1, res2):
        res.append(max(i, j))
    return res


# 2.5 规则泛化说明函数
def description_output(wb, sheetname, df1, df2, offset, base_lift):
    """
    :param wb:              excel 文件
    :param sheetname:       sheetname
    :param df1:            待输出数据框1
    :param df2:            待输出数据框2
    :param offset:          输出位置
    :param base_lift:       最小lift
    :return:        策略泛化结果说明
    """
    nrows, ncols = df1.shape
    first_title = wb.add_format(title_dic)
    body_text_title = wb.add_format(subtitle_format)
    body_text_center = wb.add_format(body_text_format_01)
    body_text_left = wb.add_format(body_text_left_format_01)
    body_text_percent = wb.add_format(body_text_per_format_01)
    body_text_format_01_01 = copy.deepcopy(body_text_format_01)
    body_text_format_01_01['text_wrap'] = 1
    body_text_center_01 = wb.add_format(body_text_format_01_01)
    column = df1.columns
    ws = wb.add_worksheet(sheetname)
    ws.hide_gridlines({'option': 1})
    ws.set_column(0, 0, 2)
    for i in range(len(column)):
        x = column[[i]][0]
        ll = get_same_len(x)
        lll = max(8, ll)
        ws.set_column(i + offset, i + offset, lll)
    ws.merge_range(offset, offset, offset, ncols, '1.规则说明', first_title)
    ws.set_row(2, 7)
    title_01 = ['规则类型', '规则上线标准']
    ws.write(3, 1, title_01[0], body_text_title)
    ws.merge_range(3, 2, 3, 5, title_01[1], body_text_title)
    rule_type01 = 'HC';
    rule_type02 = 'FR/CR';
    text_01 = '准入类规则需结合产品和适用场景进行考虑'
    text_021 = "同时满足如下条件："
    text_022 = "1.近n个月额外触碰Lift>=" + str(base_lift) + "的月份占比>=80%"
    text_023 = "2.按月统计的额外触碰率离散度稳定（非常稳定或相对稳定）"
    text_024 = "注：追踪月份至少 >= 3"
    ws.write(4, 1, rule_type01, body_text_center)
    ws.merge_range(4, 2, 4, 5, text_01, body_text_left)
    ws.merge_range(5, 1, 8, 1, rule_type02, body_text_center)
    ws.merge_range(5, 2, 5, 5, text_021, body_text_left)
    ws.merge_range(6, 2, 6, 5, text_022, body_text_left)
    ws.merge_range(7, 2, 7, 5, text_023, body_text_left)
    ws.merge_range(8, 2, 8, 5, text_024, body_text_left)
    ws.set_row(9, 7)
    # 增加备注
    title = ['序号', '波动幅度', '类型', '备注']
    for i in range(len(title) - 1):
        ws.write(10, i + 1, title[i], body_text_title)
    ws.merge_range(10, len(title), 10, len(title) + 1, '备注', body_text_title)
    rshape, cshape = type_dic.shape
    for i in range(rshape):
        for j in range(cshape):
            ws.write(10 + 1 + i, j + 1, type_dic.iloc[i][j], body_text_center)
    ws.merge_range(10 + 1, cshape + 1, 10 + rshape, cshape + 2, '波动幅度=离散度', body_text_center)
    ws.set_row(15, 7)
    ws.merge_range(16, 1, 16, ncols, '2.规则逻辑', first_title)
    ws.set_row(17, 7)
    df1 = df1.replace(np.nan, '')
    df1 = df1.replace(np.inf, 'Inf')
    for i in range(len(column)):
        x = column[[i]][0]
        ll = get_same_len(x)
        lll = max(8, ll)
        ws.set_column(i + 1, i + 1, lll)
    for j in range(ncols):
        ws.write(18, j + 1, column[j], body_text_title)
    for i in range(nrows):
        for j in range(ncols):
            if 'odds' in column[j] or 'Odds' in column[j]:
                ws.conditional_format(19, j + 1, 19 + nrows - 1, j + 1, condition_format_pink_no)
            value = df1.iloc[i][j]
            if '%' in column[j] or 'Rate' in column[j] or '占比' in column[j] or '率' in column[j]:
                ws.write(19 + i, j + 1, value, body_text_percent)
            else:
                ws.write(19 + i, j + 1, value, body_text_center)
    ws.set_row(19 + nrows, 7)
    ws.merge_range(19 + nrows + 1, 1, 19 + nrows + 1, ncols, '3.规则字段', first_title)
    ws.set_row(19 + nrows + 2, 7)
    nrows2, ncols2 = df2.shape
    df2 = df2.replace(np.nan, '')
    df2 = df2.replace(np.inf, 'Inf')
    column = df2.columns
    need_len = get_max_len(data=df2.iloc[:, 1:], nrows=nrows2, ncols=ncols2 - 1, max_len=45)
    for i in range(len(need_len)):
        ws.set_column(i + 2, i + 2, need_len[i])
    for j in range(ncols2):
        ws.write(19 + nrows + 3, j + 1, column[j], body_text_title)
    for i in range(nrows2):
        for j in range(ncols2):
            value = df2.iloc[i][j]
            ws.write(19 + nrows + 4 + i, j + 1, value, body_text_center_01)


# 2.6 规则泛化指标说明
word_desc = [{'评估内容': '样本类型', '内容详细含义': '测算和泛化样本细分类型', '备注': '取值可以是地域、渠道等'},
             {'评估内容': '风险类型', '内容详细含义': '测算和泛化变量为欺诈类变量则属于短期风险；测算和泛化变量为信用变量则属于中长期风险', '备注': None},
             {'评估内容': '目标字段', '内容详细含义': '测算和泛化的目标字段', '备注': None},
             {'评估内容': '规则类型', '内容详细含义': 'FR/CR/HC', '备注': 'FR是欺诈类规则；CR为信用类规则；HC为强规则'},
             {'评估内容': '申请量', '内容详细含义': '申请量', '备注': None},
             {'评估内容': '通过量', '内容详细含义': '审批通过量', '备注': None},
             {'评估内容': '通过率', '内容详细含义': '通过量/申请量', '备注': None},
             {'评估内容': '用信量', '内容详细含义': '审批通过且支用量', '备注': None},
             {'评估内容': '用信率', '内容详细含义': '用信量/通过量', '备注': None},
             {'评估内容': '置前触碰量', '内容详细含义': '已有所有规则触碰量', '备注': None},
             {'评估内容': '策略触碰量', '内容详细含义': '待上线规则触碰量（包含申请拒绝触碰和申请通过触碰）', '备注': None},
             {'评估内容': '重复触碰量', '内容详细含义': '待上线规则和已有规则重复触碰量', '备注': None},
             {'评估内容': '额外触碰量', '内容详细含义': '待上线规则在申请通过样本上触碰量', '备注': None},
             {'评估内容': '置后触碰量', '内容详细含义': '已有所有规则和待上线规则一共触碰量', '备注': None},
             {'评估内容': '置前触碰率', '内容详细含义': '置前触碰量/申请量', '备注': None},
             {'评估内容': '策略触碰率', '内容详细含义': '策略触碰量/申请量', '备注': None},
             {'评估内容': '重复触碰率', '内容详细含义': '重复触碰量/申请量', '备注': None},
             {'评估内容': '额外触碰率', '内容详细含义': '额外触碰量/申请量', '备注': None},
             {'评估内容': '置后触碰率', '内容详细含义': '置后触碰量/申请量', '备注': None},
             {'评估内容': '重复触碰占策略触碰比例', '内容详细含义': '重复触碰量）/策略触碰量', '备注': None},
             {'评估内容': '额外触碰占策略触碰比例', '内容详细含义': '额外触碰量）/策略触碰量', '备注': None},
             {'评估内容': '整体成熟量', '内容详细含义': '观测成熟量', '备注': None},
             {'评估内容': '额外触碰成熟量', '内容详细含义': '观测成熟且额外触碰量', '备注': None},
             {'评估内容': '整体成熟坏样本量', '内容详细含义': '观测成熟坏样本量', '备注': None},
             {'评估内容': '额外触碰成熟坏样本量', '内容详细含义': '观测成熟且额外触碰坏样本量', '备注': None},
             {'评估内容': 'fpd15整体成熟量', '内容详细含义': 'fpd15观测成熟量', '备注': None},
             {'评估内容': 'fpd15额外触碰成熟量', '内容详细含义': 'fpd15观测成熟且额外触碰量', '备注': None},
             {'评估内容': 'fpd15整体成熟坏样本量', '内容详细含义': 'fpd15观测成熟坏样本量', '备注': None},
             {'评估内容': 'fpd15额外触碰成熟坏样本量', '内容详细含义': 'fpd15观测成熟且额外触碰坏样本量', '备注': None},
             {'评估内容': '整体逾期率', '内容详细含义': '观测成熟样本中逾期样本占比', '备注': None},
             {'评估内容': '额外触碰逾期率', '内容详细含义': '观测成熟且额外触碰样本中逾期样本占比', '备注': None},
             {'评估内容': '额外触碰后整体逾期率', '内容详细含义': '观测成熟样本中剔除额外触碰样本后逾期样本占比', '备注': None},
             {'评估内容': '逾期率下降值', '内容详细含义': '策略上线后逾期率下降值', '备注': None},
             {'评估内容': '逾期率下降幅度', '内容详细含义': '策略上线后逾期率下降幅度', '备注': None},
             {'评估内容': '额外触碰Odds', '内容详细含义': '（额外触碰成熟申请单中坏好比）/（非额外触碰成熟申请单坏/好）', '备注': None},
             {'评估内容': '额外触碰Lift', '内容详细含义': '（额外触碰成熟申请单中坏样本占比）/（全量成熟申请单坏样本占比）', '备注': None},
             {'评估内容': '额外触碰fpd15_Odds', '内容详细含义': '（额外触碰fpd15成熟申请单中坏好比）/（非额外触碰fpd15成熟申请单中坏/好）', '备注': None},
             {'评估内容': '额外触碰fpd15_Lift', '内容详细含义': '（额外触碰fpd15成熟申请单中坏样本占比）/（全量fpd15成熟申请单坏样本占比）', '备注': None},
             {'评估内容': '额外触碰Lift大于目标值的月份占比', '内容详细含义': '额外触碰非空Lift大于阈值月份数/额外触碰非空Lift月份数', '备注': '阈值可根据实际情况自己设定，如取值为3'},
             {'评估内容': '置前触碰率离散度', '内容详细含义': '置前触碰率是否稳定', '备注': '0%~15%：非常稳定\n15%~40%：相对稳定\n40%~75%：不稳定\n>75%：极不稳定'},
             {'评估内容': '策略触碰率离散度', '内容详细含义': '策略触碰率是否稳定', '备注': '0%~15%：非常稳定\n15%~40%：相对稳定\n40%~75%：不稳定\n>75%：极不稳定'},
             {'评估内容': '重复触碰率离散度', '内容详细含义': '重复触碰率是否稳定', '备注': '0%~15%：非常稳定\n15%~40%：相对稳定\n40%~75%：不稳定\n>75%：极不稳定'},
             {'评估内容': '额外触碰率离散度', '内容详细含义': '额外触碰率是否稳定', '备注': '0%~15%：非常稳定\n15%~40%：相对稳定\n40%~75%：不稳定\n>75%：极不稳定'},
             {'评估内容': '置后触碰率离散度', '内容详细含义': '置后触碰率是否稳定', '备注': '0%~15%：非常稳定\n15%~40%：相对稳定\n40%~75%：不稳定\n>75%：极不稳定'},
             {'评估内容': '评估结论', '内容详细含义': '基于策略筛选规则，评估策略是否进行上线', '备注': None}]

word_desc = pd.DataFrame(word_desc)[['评估内容', '内容详细含义', '备注']]


# 2.8 按月泛化时规则风险表现情况画图函数
def add_combine_plot(wb, ws, column, label_x_start=1 + 6, x_start=1 + 6 + 1, x_end=int(1 + 6 + 1 + 12 * 1 / 2), title='客户级规则效能分析', x_label='申请月',
                     y1_list=['额外触碰Lift', '额外触碰fpd15_Lift'], y2_list=['置前触碰率', '策略触碰率', '重复触碰率', '额外触碰率', '置后触碰率'],
                     x_title='申请月', y1_title='额外触碰Lift', y2_title='规则触碰率', width=760, height=520, if_combine=True, chart1_type='column', chart2_type='line',
                     chart_row=1 + 6 + 12 + 4, chart_col=1):
    """
    :param column: 数据框列名
    :param label_x_start:  数据框列名开始行
    :param x_start:   数据框取值开始行
    :param x_end:    数据框取值结束行
    :param title:    图表名称
    :param x_label:  X轴数据列名
    :param y1_list:  Y1轴数据列名
    :param y2_list:   Y2轴数据列名
    :param x_title:  X轴标题
    :param y1_title:   Y1轴标题
    :param y2_title:    Y2轴标题
    :param width:   图表宽
    :param height:  图表高
    :param if_combine: 是否画组合图，默认是
    :param chart1_type: 组合图1类型，默认是柱状图
    :param chart2_type: 组合图2类型，默认是折线图
    :param chart_row:  图表插入行
    :param chart_col:  图表插入列
    :return:
    """
    x1 = list(column).index(x_label) + 1
    y1_index_list = [list(column).index(i) + 1 for i in y1_list]
    y2_index_list = [list(column).index(i) + 1 for i in y2_list]
    column_chart1 = wb.add_chart({'type': chart1_type})
    column_chart1.set_size({'width': width, 'height': height})
    for k in y1_index_list:
        column_chart1.add_series(
            {'name': [ws.name, label_x_start, k],
             'num_font': {'name': '微软雅黑', 'size': 9},
             'categories': [ws.name, x_start, x1, x_end, x1],
             'values': [ws.name, x_start, k, x_end, k],
             'data_labels': {'value': False}
             })
    column_chart1.set_title({'name': title, 'name_font': {'name': '微软雅黑', 'size': 10, 'bold': False}})
    column_chart1.set_x_axis({'name': x_title, 'name_font': {'name': '微软雅黑', 'size': 9, 'bold': False}})
    column_chart1.set_y_axis({'name': y1_title, 'name_font': {'name': '微软雅黑', 'size': 9, 'bold': False}})
    column_chart1.set_chartarea({'border': {'none': True}, 'fill': {'color': text}})
    column_chart1.set_plotarea({'border': {'none': True}, 'fill': {'color': text}})
    if if_combine:
        line_chart2 = wb.add_chart({'type': chart2_type})
        for k in y2_index_list:
            line_chart2.add_series(
                {'name': [ws.name, label_x_start, k],
                 'num_font': {'name': '微软雅黑', 'size': 9},
                 'categories': [ws.name, x_start, x1, x_end, x1],
                 'values': [ws.name, x_start, k, x_end, k],
                 'data_labels': {'value': False},
                 'y2_axis': True
                 })
        column_chart1.combine(line_chart2)
        line_chart2.set_y2_axis({'name': y2_title, 'name_font': {'name': '微软雅黑', 'size': 9, 'bold': False}})
    ws.insert_chart(chart_row, chart_col, column_chart1)


def add_plot(wb, ws, column, label_x_start=1 + 6, x_start=1 + 6 + 1, x_end=int(1 + 6 + 1 + 12 * 1 / 2), title='客户级规则效能分析', x_label='申请周',
             y1_list=['置前触碰率', '策略触碰率', '重复触碰率', '额外触碰率', '置后触碰率'], x_title='申请周', y1_title='规则触碰率', width=760, height=520, chart1_type='line',
             chart_row=1 + 6 + 12 + 4, chart_col=1):
    """
    插入图形函数
    """
    x1 = list(column).index(x_label) + 1
    y1_index_list = [list(column).index(i) + 1 for i in y1_list]
    column_chart1 = wb.add_chart({'type': chart1_type})
    column_chart1.set_size({'width': width, 'height': height})
    for k in y1_index_list:
        column_chart1.add_series(
            {'name': [ws.name, label_x_start, k],
             'num_font': {'name': '微软雅黑', 'size': 9},
             'categories': [ws.name, x_start, x1, x_end, x1],
             'values': [ws.name, x_start, k, x_end, k],
             'data_labels': {'value': False}
             })
    column_chart1.set_title({'name': title, 'name_font': {'name': '微软雅黑', 'size': 10, 'bold': False}})
    column_chart1.set_x_axis({'name': x_title, 'name_font': {'name': '微软雅黑', 'size': 9, 'bold': False}})
    column_chart1.set_y_axis({'name': y1_title, 'name_font': {'name': '微软雅黑', 'size': 9, 'bold': False}})
    column_chart1.set_chartarea({'border': {'none': True}, 'fill': {'color': text}})
    column_chart1.set_plotarea({'border': {'none': True}, 'fill': {'color': text}})
    ws.insert_chart(chart_row, chart_col, column_chart1)


def std_result_output(data, freq):
    """泛化结果输出函数,按月、周、日输出分析结果函数
    """
    if freq == "M":
        data_01 = data[-3:]
        data_02 = data[data['额外触碰成熟量'] >= 10][-3:]
        data_03 = data[data['整体逾期率'] > 0][-3:]
        hit_rate = sum(data_01['额外触碰量']) / sum(data_01['申请量'])
        risk_double = (data_02['额外触碰成熟坏样本量'].sum() / data_02['额外触碰成熟量'].sum()) / (sum(data_02['整体成熟坏样本量']) / sum(data_02['整体成熟量'])) if sum(data_02['整体成熟坏样本量']) > 0 else np.nan
        overdue_minus = sum(data_03['整体成熟坏样本量']) / sum(data_03['整体成熟量']) - sum(data_03['整体成熟坏样本量'] - data_03['额外触碰成熟坏样本量']) / sum(data_03['整体成熟量'] - data_03['额外触碰成熟量']) if sum(data_03['整体成熟量'] - data_03['额外触碰成熟量']) > 0 else np.nan
        overdue_range = overdue_minus / (sum(data_03['整体成熟坏样本量']) / sum(data_03['整体成熟量'])) if sum(data_03['整体成熟坏样本量']) > 0 else np.nan
        text1 = "1.策略上线后通过率下降预估值 = 近3个月额外触碰量之和/近3个月申请量之和；策略上线后逾期率变化预估值 = （近n个有表现月逾期样本之和/近n个有表现月成熟样本之和）-（近n个有表现月置后触碰坏样本之和/近n个有表现月置后触碰成熟样本之和） ；备注：取整体逾期率大于0的近n个月计算策略上线后逾期率变化值，n默认取3，若不足3个月，有几个月计算几个月;"
        text2 = "2.策略上线后额外触碰Lift预估值 = (近n个有表现月额外触碰成熟坏样本量之和/近n个有表现月额外触碰成熟量之和)/(近n个有表现月整体成熟坏样本量之和/近n个有表现月整体成熟量之和)；备注：取额外触碰成熟量大于等于10的近n个月计算策略上线后额外触碰Lift预估值，n默认取3，若不足3个月，有几个月计算几个月;"
        if overdue_minus > 0:
            text3 = "3.策略上线后通过率预计下降：{:.2f}%，逾期率下降：{:.2f}%，逾期率下降幅度：{:.2f}%，额外触碰Lift预估值为：{:.2f}。".format(hit_rate * 100, overdue_minus * 100, overdue_range * 100, risk_double)
        else:
            text3 = "3.策略上线后通过率预计下降：{:.2f}%，逾期率上升：{:.2f}%，逾期率上升幅度：{:.2f}%，额外触碰Lift预估值为：{:.2f}。".format(hit_rate * 100, abs(overdue_minus) * 100, abs(overdue_range) * 100, risk_double)
    elif freq == "W":
        week_num = str(len(data['申请周'].unique()))
        mean_obs = data['置前触碰率'].mean() * 100
        cv_rate = cv(data['置前触碰率'])
        if data['置前触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='置前触碰率')
        text1 = "1.近" + week_num + "个申请周平均置前触碰率%.2f" % mean_obs + '%, ' + '置前触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + ';置前触碰率整体趋势：' + trend + "；"
        mean_obs = data['策略触碰率'].mean() * 100
        cv_rate = cv(data['策略触碰率'])
        if data['策略触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='策略触碰率')
        text2 = "2.近" + week_num + "个申请周策略触碰率%.2f" % mean_obs + '%, ' + '策略触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 策略触碰率整体趋势：' + trend + "；"
        mean_obs = data['额外触碰率'].mean() * 100
        cv_rate = cv(data['额外触碰率'])
        if data['额外触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='额外触碰率')
        text3 = "3.近" + week_num + "个申请周额外触碰率%.2f" % mean_obs + '%, ' + '额外触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 额外触碰率整体趋势：' + trend + "；"
    elif freq == "D":
        day_num = str(len(data['申请日'].unique()))
        mean_obs = data['置前触碰率'].mean() * 100
        cv_rate = cv(data['置前触碰率'])
        if data['置前触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='置前触碰率')
        text1 = "1.近" + day_num + "个申请日平均置前触碰率%.2f" % mean_obs + '%, ' + '置前触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + ';置前触碰率整体趋势：' + trend + "；"
        mean_obs = data['策略触碰率'].mean() * 100
        cv_rate = cv(data['策略触碰率'])
        if data['策略触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='策略触碰率')
        text2 = "2.近" + day_num + "个申请日策略触碰率%.2f" % mean_obs + '%, ' + '策略触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 策略触碰率整体趋势：' + trend + "；"
        mean_obs = data['额外触碰率'].mean() * 100
        cv_rate = cv(data['额外触碰率'])
        if data['额外触碰率'].sum() == 0:
            cv_rate = 0
        info = cv_type(cv_rate)
        trend = get_trend(data, key='额外触碰率')
        text3 = "3.近" + day_num + "个申请日额外触碰率%.2f" % mean_obs + '%, ' + '额外触碰率波动幅度%.2f' % (100 * cv_rate) + '%, ' + info + '; 额外触碰率整体趋势：' + trend + "；"
    else:
        raise "frqp must in [M\W\D]"

    return text1, text2, text3


if __name__ == '__main__':
    import os
    import xlsxwriter
    import sys

    sys.path.append("../")
    from scorecardpipeline import *

    init_setting()

    # 参数值设置
    sample_range = '202108-202110'
    seq = 1
    # sample_type取值只有Total，表示测算全量样本，不需要配置参数sample_type_col；若还有其他样本类型，也支持同时测算
    sample_type = ['Total']

    # 若sample_type需测算类型1和类型2的样本，类型1和类型2所在的字段为ordertype，sample_type_col参数示例如下
    # sample_type_col = {
    #                    '类型1': ['ordertype', ['类型1']],
    #                    '类型2': ['ordertype', ['类型2']]
    #                   }

    sample_type_target = {'Total': ['fpd_30_act', 'mob3_dpd_30_act']}

    target_ripe = {'fpd_30_act': ['agr_fpd_30'], 'mob3_dpd_30_act': ['agr_mob3_dpd_30']}

    target_del_col = {'fpd_30_act': ['agr_fpd_30', 'agr_mob3_dpd_30', 'mob3_dpd_30_act'],
                      'mob3_dpd_30_act': ['agr_mob3_dpd_30', 'agr_fpd_30', 'fpd_30_act']}

    sub_div_bin = 0.1
    target_min_rate = {'fpd_30_act': [0.015], 'mob3_dpd_30_act': [0.02]}
    min_num = 40
    hit_num = 30
    sample_type_lift = {'Total': {'fpd_30_act': 1.8, 'mob3_dpd_30_act': 3}}

    # 泛化结果输出路径
    path_rule = "rule_data/rule_result/"

    if not os.path.exists(path_rule):
        os.makedirs(path_rule)

    # 4.读入数据并进行数据预处理
    #  读入测算环节筛选的待泛化规则
    rules_dict = pd.read_excel("rule_data/规则字典.xlsx")

    # 加载数据
    mydata = pd.read_csv("rule_data/rule_data.csv", encoding="gbk")
    mydata.columns = mydata.columns.map(lambda x: x.lower())

    for i in mydata.columns[mydata.dtypes != 'object']:
        mydata[i][mydata[i].map(lambda x: x in [-999, -9999, -999999])] = np.nan

    for i in mydata.columns[mydata.dtypes == 'object']:
        mydata[i] = mydata[i].map(lambda x: str(x).strip())
        mydata[i][mydata[i].map(lambda x: x in ['-999', '-9999', '-999999'])] = np.nan
        try:
            mydata[i] = mydata[i].astype('float64')
        except:
            pass

    # 处理灰样本
    mydata['mob3_dpd_30_act'] = mydata['mob3_dpd_30_act'].map(lambda x: 1 if x == 1 else 0)

    # 5.基于加载的函数进行策略自动泛化；（需要基于阈值测算的数据字典）

    """ # 泛化参数说明
    target_ripe      获取目标字段对应的是否成熟标签，与测算环节参数一样
    sample_type_col  若测算样本是整个样本的子集，需配置sample_type_col参数，指名样本类型从哪个字段获取，与测算环节参数一样
    rules_dict       测算环节筛选的待泛化规则集
    rule_all         当前已经在线上运行的所有规则对应的决策结果标签字段，1代表拒绝，0代表未拒绝
    use_credit_flag  授信申请通过后是否用信标签字段，1代表用信，0代表未用信
    circle_mth       申请月对应的字段，在进行策略泛化的时候的会按月进行泛化，分析策略触碰情况和风险表现情况
    circle_week      申请周对应的字段，在进行策略泛化的时候会按申请周进行泛化，分析近10周策略触碰情况
    circle_day       申请日对应的字段，在进行策略泛化的时候会按申请日进行泛化，分析近10日策略触碰情况
    base_lift        策略泛化时，取额外触碰样本的Lift值与base_lift进行比较，若额外触碰Lift值大于base_lift则说明策略效果较好
    subset_total     是否获取全量样本作为计算整体，若取值为False则分母为全样本，计算各种指标时是从全量样本维度考虑的，若取值为True则分母为样本类型对应的样本
    """

    target_ripe = {'fpd_30_act': ['agr_fpd_30'], 'mob3_dpd_30_act': ['agr_mob3_dpd_30']}

    # 在进行策略测算时，若除测算全量样本（Total）外还测算了类型1和类型2的样本，假设类型1和类型2所在的字段为ordertype，则需设置sample_type_col参数，否则不需要进行设置。
    # sample_type_col = {
    #                    '类型1': ['ordertype', ['类型1']],
    #                    '类型2': ['ordertype', ['类型2']]
    #                   }

    # 对测算环节筛选的样本进行自动化泛化
    starttime = datetime.datetime.now()
    print('程序开始执行时间为: ' + str(starttime))

    rule_combine_results(data=mydata, rules_dict=rules_dict, rule_all='apply_refuse_flag', use_credit_flag='if_loan_flag', circle_mth='apply_mth', circle_week='apply_week', circle_day='apply_day', base_lift=3, subset_total=True)

    endtime = datetime.datetime.now()
    print('程序开始执行时间为: ' + str(endtime - starttime))

    # 6.基于步骤3策略泛化结果筛选泛化效果好的规则集进行合并泛化（不需要测算结果数据字典）

    # 此次未筛选到足够多的泛化效果好的规则，只为跑通代码流程，代码执行结果供参考。基于泛化效果好的规则集，重构一条规则进行泛化，变量取值为1表示规则触碰，取值为0表示规则未触碰。

    mydata['all_need_online_rule'] = mydata.apply(lambda x: 1 if (x['var3'] > 3 or x['var6'] > 5) else 0, axis=1)

    starttime = datetime.datetime.now()
    print('程序开始执行时间为: ' + str(starttime))
    rule_combine_results_01(data=mydata, rule_all='apply_refuse_flag', use_credit_flag='if_loan_flag', cut_point=0, direction='>', rule_name='待上线规则合并泛化', rule_name_chinese='待上线规则合并泛化', var='all_need_online_rule', rule_type='CR', rule_limit='Total', circle_mth='apply_mth', circle_week='apply_week', circle_day='apply_day', target='mob3_dpd_30_act', base_lift=3, subset_total=True)
    endtime = datetime.datetime.now()
    print('程序执行时间为: ' + str(endtime - starttime))
