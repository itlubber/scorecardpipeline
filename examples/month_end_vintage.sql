# MONTH END VINTAGE DPD
WITH RECURSIVE months_end(观察时点) AS (
    SELECT date_add(last_day(current_date()), INTERVAL 1 DAY)
    UNION
    SELECT date_sub(观察时点, INTERVAL 1 MONTH)
    FROM months_end
    WHERE 观察时点 > date '2023-12-01' # 限定开始统计时间使用
)
, loan AS (
    # 筛选需要计算的订单数据
    SELECT id, user_id, flow_channel, bank_channel, apply_time, loan_time, periods, amount
    FROM database_name.loan l
    WHERE l.loan_state = 'SUCCEED'
)
, replan as (
	SELECT  l.flow_channel 流量渠道
			, l.bank_channel 放款资方
			, l.id 放款编号
			, l.user_id 用户编号
			, l.apply_time 申请时间
			, l.loan_time 放款时间
			, date_format(l.loan_time, '%Y-%m') 放款月份
			, l.periods 放款期数
    		, p.period 还款期数
    		, l.amount 放款金额
    		, p.plan_repay_date 应还日期
            , date(p.act_repay_time) 实还日期
            , p.principal_amt 应还本金
            , p.act_principal_amt 实还本金
	FROM loan l INNER JOIN database_name.repay_plan p ON l.id = p.loan_id
    ORDER BY 放款编号, 还款期数
)
, amount AS (
    SELECT date_format(loan_time, '%Y-%m') 放款月份, count(id) 放款件数, count(user_id) 放款人数, sum(amount) 放款金额
    FROM loan
    GROUP BY date_format(loan_time, '%Y-%m')
)
, balance AS (
    SELECT 观察时点, 放款月份, 用户编号, 放款编号
            , 放款金额 - sum(IF(实还日期 IS NOT NULL
   AND 实还日期 < 观察时点, 实还本金, 0)) 余额
            , concat('MOB', TIMESTAMPDIFF(month, concat(substr(放款时间, 1, 7), '-01'), concat(substr(date_sub(观察时点, INTERVAL 1 DAY), 1, 7), '-01'))) MOB
            , ifnull(
                max(
                    CASE
                        WHEN 应还日期 >= IF(观察时点 > current_date(), current_date(), 观察时点) THEN 0  # 未到还款日，当天出账不计入
                        WHEN 应还日期 <= IF(观察时点 > current_date(), current_date(), 观察时点) AND 实还日期 <= 应还日期 THEN 0  # 按时还款
                        -- CURR 口径: 观察时点之前出账且截止观察时点的未还的订单逾期天数
                        WHEN 应还日期 < IF(观察时点 > current_date(), current_date(), 观察时点) AND 实还日期 < IF(观察时点 > current_date(), current_date(), 观察时点) THEN 0   # 观察时点已还清,也就是说逾期但已还的不算在内
                        WHEN 应还日期 < IF(观察时点 > current_date(), current_date(), 观察时点) AND (实还日期 IS NULL
    OR 实还日期 >= IF(观察时点 > current_date(), current_date(), 观察时点)) THEN DATEDIFF(IF(观察时点 > current_date(), current_date(), 观察时点), 应还日期)  # 观察时点时点未还款
                        ELSE 0
                    END
                ), 0
            ) 逾期天数
    FROM replan
    LEFT JOIN months_end ON replan.放款时间 < months_end.观察时点
    GROUP BY 观察时点, 放款月份, 用户编号, 放款编号, 放款金额
    ORDER BY 逾期天数 DESC
)
SELECT 放款月份, 放款金额
	, sum(CASE WHEN MOB = 'MOB1' THEN 逾期率 END) MOB1
	, sum(CASE WHEN MOB = 'MOB2' THEN 逾期率 END) MOB2
	, sum(CASE WHEN MOB = 'MOB3' THEN 逾期率 END) MOB3
	, sum(CASE WHEN MOB = 'MOB4' THEN 逾期率 END) MOB4
	, sum(CASE WHEN MOB = 'MOB5' THEN 逾期率 END) MOB5
	, sum(CASE WHEN MOB = 'MOB6' THEN 逾期率 END) MOB6
	, sum(CASE WHEN MOB = 'MOB7' THEN 逾期率 END) MOB7
	, sum(CASE WHEN MOB = 'MOB8' THEN 逾期率 END) MOB8
	, sum(CASE WHEN MOB = 'MOB9' THEN 逾期率 END) MOB9
	, sum(CASE WHEN MOB = 'MOB10' THEN 逾期率 END) MOB10
	, sum(CASE WHEN MOB = 'MOB11' THEN 逾期率 END) MOB11
	, sum(CASE WHEN MOB = 'MOB12' THEN 逾期率 END) MOB12
	, sum(CASE WHEN MOB = 'MOB13' THEN 逾期率 END) MOB13
	, sum(CASE WHEN MOB = 'MOB14' THEN 逾期率 END) MOB14
	, sum(CASE WHEN MOB = 'MOB15' THEN 逾期率 END) MOB15
-- 	, sum(CASE WHEN MOB = 'MOB16' THEN 逾期率 END) MOB16
-- 	, sum(CASE WHEN MOB = 'MOB17' THEN 逾期率 END) MOB17
-- 	, sum(CASE WHEN MOB = 'MOB18' THEN 逾期率 END) MOB18
-- 	, sum(CASE WHEN MOB = 'MOB19' THEN 逾期率 END) MOB19
-- 	, sum(CASE WHEN MOB = 'MOB20' THEN 逾期率 END) MOB20
-- 	, sum(CASE WHEN MOB = 'MOB21' THEN 逾期率 END) MOB21
-- 	, sum(CASE WHEN MOB = 'MOB22' THEN 逾期率 END) MOB22
-- 	, sum(CASE WHEN MOB = 'MOB23' THEN 逾期率 END) MOB23
-- 	, sum(CASE WHEN MOB = 'MOB24' THEN 逾期率 END) MOB24
-- 	, sum(CASE WHEN MOB = 'MOB25' THEN 逾期率 END) MOB25
FROM (
    SELECT a.放款月份, a.放款金额, t.MOB, t.逾期余额, ifnull(t.逾期余额,0) / a.放款金额 逾期率
    FROM (
        SELECT 放款月份, MOB, sum(CASE WHEN 逾期天数 > 30 THEN 余额 END) 逾期余额 # 逾期天数修改这里的30
        FROM balance
        GROUP BY 放款月份, MOB
    ) t LEFT JOIN amount a ON t.放款月份 = a.放款月份
) v
GROUP BY 放款月份, 放款金额
ORDER BY 放款月份