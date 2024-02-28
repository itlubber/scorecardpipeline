WITH loan AS (
    # 筛选需要计算的订单数据
    SELECT id, user_id, flow_channel, bank_channel, apply_time, loan_time, periods, amount
    FROM database_name.loan
    WHERE l.loan_state = 'SUCCEED'
),
replan as (
	SELECT  l.flow_channel 流量渠道
			, l.bank_channel 放款资方
			, l.id 放款编号
			, l.user_id 用户编号
			, l.apply_time 申请时间
			, l.loan_time 放款时间
			, date_format(l.loan_time, '%Y-%m') 放款月份
			, l.periods 放款期数
    		, l.amount 放款金额
    		, p.period 还款期数
    		, p.plan_repay_date 应还日期
    		, date_add(p.plan_repay_date, INTERVAL 30 DAY) 观察日期
            , date(p.act_repay_time) 实还日期
            , p.principal_amt 应还本金
            , p.act_principal_amt 实还本金
	FROM loan l INNER JOIN database_name.repay_plan p ON l.id = p.loan_id
	ORDER BY 放款编号, 还款期数
),
amount as (
    SELECT date_format(loan_time, '%Y-%m') 放款月份, count(id) 放款件数, count(user_id) 放款人数, sum(amount) 放款金额
    FROM loan
    GROUP BY date_format(loan_time, '%Y-%m')
)
SELECT vintage.放款月份
		, SUM(IF(还款期数 = 1, 放款金额, 0)) 放款金额
		, SUM(IF(还款期数 = 1, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额  when '递延逾期率' then 逾期余额 / 出账金额  when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM1
		, SUM(IF(还款期数 = 2, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额  when '递延逾期率' then 逾期余额 / 出账金额 when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM2
		, SUM(IF(还款期数 = 3, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额  when '递延逾期率' then 逾期余额 / 出账金额  when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM3
		, SUM(IF(还款期数 = 4, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额 when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM4
		, SUM(IF(还款期数 = 5, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额 when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM5
		, SUM(IF(还款期数 = 6, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额  when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM6
		, SUM(IF(还款期数 = 7, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额 when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM7
		, SUM(IF(还款期数 = 8, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额 when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM8
		, SUM(IF(还款期数 = 9, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额  when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM9
		, SUM(IF(还款期数 = 10, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额  when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM10
		, SUM(IF(还款期数 = 11, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额 when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM11
		, SUM(IF(还款期数 = 12, case '递延逾期率' when '逾期率' then 逾期余额 / 放款金额 when '递延逾期率' then 逾期余额 / 出账金额 when '出账订单' then 出账订单  when '出账金额' then 出账金额  when '逾期余额' then 逾期余额 end, NULL)) TERM12
FROM (
	SELECT 放款月份, 还款期数
			, SUM(IF(观察点逾期 = 1, 观察点余额, 0)) 逾期余额
			, SUM(IF(观察日期 <= current_date(), 放款金额, 0)) 出账金额
			, COUNT(distinct IF(观察日期 <= current_date(), 放款编号, null)) 出账订单
			-- , SUM(IF(观察点逾期 = 1, 观察点余额, 0)) / SUM(放款金额) 逾期率
			-- , SUM(IF(观察点逾期 = 1, 观察点余额, 0)) / SUM(IF(观察日期 <= current_date(), 放款金额, 0)) 递延逾期率
	FROM (
		SELECT p1.放款编号, p1.放款月份, p1.放款金额, p1.还款期数, p1.观察日期
				, p1.放款金额 - sum(if(p2.实还日期 <= p1.观察日期, p2.实还本金, 0)) 观察点余额
				, if(p1.观察日期 < current_date() AND (p1.实还日期 IS NULL OR p1.实还日期 > p1.观察日期), 1, 0) 观察点逾期
		FROM replan p1
		LEFT JOIN replan p2 ON p1.放款编号 = p2.放款编号
		GROUP BY p1.放款编号, p1.放款月份, p1.放款金额, p1.还款期数, p1.观察日期, p1.实还日期
		ORDER BY 放款编号, 还款期数
	) balance
	GROUP BY 放款月份, 还款期数
) vintage
LEFT JOIN amount a ON vintage.放款月份 = a.放款月份
GROUP BY vintage.放款月份