# -*- coding: utf-8 -*-
"""
@Time    : 2024/3/22 15:25
@Author  : itlubber
@Site    : itlubber.art
"""
import pandas as pd


principal = 5000.00  # 本金
annual_rate = 0.36  # 年化利率
term_rate = annual_rate / 12  # 月利率
day_rate = annual_rate / 12 / 30  # 日利率


repayPattern = {
    "MCAT": "随借随还,按日计息",
    "BFTO": "利随本清,按日计息",
    "MIRD": "按月付息到期还本",
    "EPEI": "等本等息,按月付息",  # 对借款人最不利
    "MCEP": "等额本金,按月付息",  # 借款总利息最低
    "MCEI": "等额等息,按月付息"
}


# "EPEI": "等本等息,按月付息"               每期利息和本金一样,最后一期本金为上一期的剩余本金，最后一期利息为总利息-已还利息
def get_EPEI(num):
    line = [["期数", "还款本息", "当期本金", "当期利息", "剩余本金", "还款方式"]]
    term_int = principal * term_rate  # 每期还利息（2位小数）
    total_int = principal * term_rate * num  # 总利息（2位小数）
    term_prin = principal / num  # 每期还本金（2位小数）
    prin = principal  # 初始化剩余本金为借款本金
    for i in range(1, num + 1):
        # 如果是最后一期，还款本金为上一期的剩余本金; 利息为总利息-已还利息
        if i == num:
            term_prin = prin
            term_int = total_int - (term_int * (num - 1))
        term_amt = term_prin + term_int
        prin = prin - term_prin  # 剩余本金=上期剩余本金-当期还本金
        line.append([i, term_amt, term_prin, term_int, prin, 'EPEI'])
    print('等本等息总利息为：', total_int)
    print(pd.DataFrame(line))


# "MIRD": "按月付息到期还本"      最后一期还本金，每期利息固定
def get_MIRD(num):
    line = [["期数", "还款本息", "当期本金", "当期利息", "剩余本金", "还款方式"]]
    term_int = principal * term_rate
    term_prin = 0.0
    total_int = principal * term_rate * num  # 总利息（2位小数）
    prin = principal
    for i in range(1, num + 1):
        # 如果是最后一期，还款本金为上一期的剩余本金; 利息为总利息-已还利息
        if i == num:
            term_prin = prin
            term_int = total_int - (term_int * (num - 1))
        term_amt = term_prin + term_int
        prin = prin - term_prin  # 剩余本金=上期剩余本金-当期还本金
        line.append([i, term_amt, term_prin, term_int, prin, 'MIRD'])
    print('按月付息到期还本总利息为：', total_int)
    print(pd.DataFrame(line))


# "等额本金,按月付息"
def get_MCEP(num):
    line = [["期数", "还款本息", "当期本金", "当期利息", "剩余本金", "还款方式"]]
    ## 每月本金相同，利息递减，相当于剩余本金的利息,每期利息固定：上一期本金*利率
    term_prin = principal / num  # 每期还本金（2位小数）
    prin = principal
    total_int = 0.0
    for i in range(1, num + 1):
        term_int = prin * term_rate  # 每期利息固定：上一期本金*利率
        # 如果是最后一期，还款本金为上一期的剩余本金;
        if i == num:
            term_prin = prin
        term_amt = term_prin + term_int
        prin = prin - term_prin  # 剩余本金=上期剩余本金-当期还本金
        line.append([i, term_amt, term_prin, term_int, prin, 'MCEP'])
        total_int = total_int + term_int
    print('等额本金总利息为：', total_int)
    print(pd.DataFrame(line))


# "MCEI": "等额本息,按月付息"
def get_MCEI(num):
    line = [["期数", "还款本息", "当期本金", "当期利息", "剩余本金", "还款方式"]]
    # 本金+利息保持相同，本金逐月递增，利息逐月递减，月还款数不变。
    term_amt = (principal * term_rate * (1 + term_rate) ** num) / ((1 + term_rate) ** num - 1)  # 每期还款总额       **是幂运算
    prin = principal
    total_int = 0.0
    for i in range(1, num + 1):
        term_int = prin * term_rate  # 每期利息计算固定：上一期本金*利率
        term_prin = term_amt - term_int
        # 如果是最后一期，还款本金为上一期的剩余本金;
        if i == num:
            term_prin = prin
        term_amt = term_prin + term_int
        prin = prin - term_prin  # 剩余本金=上期剩余本金-当期还本金
        line.append([i, term_amt, term_prin, term_int, prin, 'MCEI'])
        total_int = total_int + term_int
    print('等额本息总利息为：', total_int)
    print(pd.DataFrame(line))


# "MCAT": "随借随还,按日计息"
def get_MCAT(num, pay):
    line = [["期数", "还款本息", "当期本金", "当期利息", "剩余本金", "还款方式"]]
    day_int = pay * day_rate * num
    repay_amt = day_int + pay
    re_prin = principal - pay
    if pay > principal:
        print("还款金额:", pay, "不能大于本金:", principal)
    else:
        line.append([1, repay_amt, pay, day_int, re_prin, 'MCAT'])
        print('随借随还总利息为：', day_int)
        print(pd.DataFrame(line))


# "BFTO": "利随本清,按日计息"
def get_BFTO(num):
    line = [["期数", "还款本息", "当期本金", "当期利息", "剩余本金", "还款方式"]]
    day_int = principal * day_rate * num
    repay_amt = day_int + principal
    line.append([1, repay_amt, principal, day_int, 0.0, 'BFTO'])
    print('利随本清总利息为：', day_int)
    print(pd.DataFrame(line))


if __name__ == '__main__':
    get_EPEI(6)  # 等本等息,按月付息
    get_MCEP(6)  # 等额本金,按月付息
    get_MCEI(6)  # 等额本息,按月付息
    # get_MIRD(24)  # 到期还本,按月付息
    # get_MCAT(720, 195)  # 随借随还,按日计息
    # get_BFTO(720)  # 利随本清,按日计息
