WITH months_end(观察时点) AS (
    select date(month_start)
    FROM (
        SELECT month_start
        FROM (
            SELECT CONCAT(SUBSTR(DATE_FORMAT(created_time, '%Y-%m-%d'), 1, 7), '-01') AS month_start 
            FROM qy_ods.s03_qh_loan
            GROUP BY 1
            ORDER BY 1 DESC
        ) AS subquery
        UNION ALL
        SELECT DATE_FORMAT(DATE_ADD(LAST_DAY(NOW()), INTERVAL 1 DAY), '%Y-%m-01') AS month_start
    ) g
    WHERE month_start >= '2023-12-01'
    ORDER BY month_start DESC
)
, loan AS (
    SELECT l.id, l.user_id, l.flow_channel, l.bank_channel, l.apply_time, l.loan_time, l.periods, l.amount
    FROM qy_ods.s03_qh_loan l
    LEFT JOIN qy_ods.s03_qh_order o on o.id = l.order_id
    LEFT JOIN qy_ods.s03_qh_user_risk_record urr on urr.id = o.risk_id
    LEFT JOIN qy_ods.s03_qh_user_periods_rights up on urr.id = up.risk_id
    WHERE l.loan_state = 'SUCCEED'
)
, cycle_replan as (
	SELECT  l.flow_channel 流量渠道
			, l.bank_channel 放款资方
			, l.id 放款编号
			, l.user_id 用户编号
			, l.apply_time 申请时间
			, l.amount 放款金额
			, l.loan_time 放款时间
			, date_format(l.loan_time, '%Y-%m') 放款月份
			, l.periods 放款期数
    		, p.period 还款期数
    		, p.plan_repay_date 应还日期
    		, date_add(p.plan_repay_date, INTERVAL 0 DAY) 入催观察日期
            , date(p.act_repay_time) 实还日期
            , p.principal_amt 应还本金
            , p.act_principal_amt 实还本金
	FROM loan l 
	INNER JOIN qy_ods.s03_qh_repay_plan p ON l.id = p.loan_id
)
, cycle_balance as (
    SELECT p1.放款编号
        , p1.还款期数
        , p1.放款金额 - case when p1.还款期数 = 1 then if(p1.实还本金 != p1.放款金额 or p1.实还本金 is null, 0, p1.实还本金) else sum(p2.实还本金) over(PARTITION by p1.放款编号 ORDER BY p1.还款期数) end 余额
	FROM cycle_replan p1 LEFT JOIN cycle_replan p2 ON p1.放款编号 = p2.放款编号 AND p1.还款期数 = p2.还款期数 + 1
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
	FROM loan l INNER JOIN qy_ods.s03_qh_repay_plan p ON l.id = p.loan_id
)
, amount AS (
    SELECT date_format(loan_time, '%Y-%m') 放款月份, count(l.id) 放款件数, count(l.user_id) 放款人数, sum(l.amount) 放款金额
    FROM loan l
    GROUP BY 1
)
, balance AS (
   select 观察时点, 放款月份, 用户编号, 放款编号 -- , 放款金额-sum(COALESCE (实还本金,0)) as 余额
        , 放款金额 - sum(IF(实还日期 IS NOT NULL AND 实还日期 < 观察时点, 实还本金, 0)) 余额
        , concat('MOB', TIMESTAMPDIFF(month, concat(substr(放款时间, 1, 7), '-01'), concat(substr(date_sub(观察时点, INTERVAL 1 DAY), 1, 7), '-01'))) MOB
        , IFNULL(
            MAX(
                CASE
                    WHEN 应还日期 >= IF(观察时点 > CURRENT_DATE(), CURRENT_DATE(), 观察时点) THEN 0  -- 未到还款日，当天出账不计入
                    WHEN 应还日期 <= IF(观察时点 > CURRENT_DATE(), CURRENT_DATE(), 观察时点) AND 实还日期 <= 应还日期 THEN 0  -- 按时还款
                    WHEN 应还日期 < IF(观察时点 > CURRENT_DATE(), CURRENT_DATE(), 观察时点) AND 实还日期 <= IF(观察时点 > CURRENT_DATE(), CURRENT_DATE(), 观察时点) THEN 0  -- 观察时点已还清, 逾期但已还的不算在内
                    WHEN 应还日期 < IF(观察时点 > CURRENT_DATE(), CURRENT_DATE(), 观察时点) AND (实还日期 IS NULL OR 实还日期 >= IF(观察时点 > CURRENT_DATE(), CURRENT_DATE(), 观察时点)) THEN DATEDIFF(IF(观察时点 > CURRENT_DATE(), CURRENT_DATE(), 观察时点), 应还日期)  -- 观察时点时点未还款
                    ELSE 0
                END
            ), 0
        ) AS 逾期天数
    FROM replan
    LEFT JOIN months_end ON replan.放款时间 < months_end.观察时点
    GROUP BY 观察时点, 放款月份, 用户编号, 放款编号, 放款金额, 放款时间
)
SELECT 统计口径, 放款月份, 放款金额, MOB1, MOB2, MOB3, MOB4, MOB5, MOB6, MOB7, MOB8, MOB9, MOB10, MOB11, MOB12, MOB13, `MOB1%`, `MOB2%`, `MOB3%`, `MOB4%`, `MOB5%`, `MOB6%`, `MOB7%`, `MOB8%`, `MOB9%`, `MOB10%`, `MOB11%`, `MOB12%`, (IF(坏账余额 = 0, NULL, 坏账余额) / 放款金额) as 'LOSS%'
FROM (
    SELECT 'CYCLE' 统计口径
            , c.放款月份
            , a.放款金额
    		, SUM(IF(还款期数 = 1, 净入催本金, NULL)) / SUM(IF(还款期数 = 1, 净出账本金, NULL)) MOB1
    		, SUM(IF(还款期数 = 2, 净入催本金, NULL)) / SUM(IF(还款期数 = 2, 净出账本金, NULL)) MOB2
    		, SUM(IF(还款期数 = 3, 净入催本金, NULL)) / SUM(IF(还款期数 = 3, 净出账本金, NULL)) MOB3
    		, SUM(IF(还款期数 = 4, 净入催本金, NULL)) / SUM(IF(还款期数 = 4, 净出账本金, NULL)) MOB4
    		, SUM(IF(还款期数 = 5, 净入催本金, NULL)) / SUM(IF(还款期数 = 5, 净出账本金, NULL)) MOB5
    		, SUM(IF(还款期数 = 6, 净入催本金, NULL)) / SUM(IF(还款期数 = 6, 净出账本金, NULL)) MOB6
    		, SUM(IF(还款期数 = 7, 净入催本金, NULL)) / SUM(IF(还款期数 = 7, 净出账本金, NULL)) MOB7
    		, SUM(IF(还款期数 = 8, 净入催本金, NULL)) / SUM(IF(还款期数 = 8, 净出账本金, NULL)) MOB8
    		, SUM(IF(还款期数 = 9, 净入催本金, NULL)) / SUM(IF(还款期数 = 9, 净出账本金, NULL)) MOB9
    		, SUM(IF(还款期数 = 10, 净入催本金, NULL)) / SUM(IF(还款期数 = 10, 净出账本金, NULL)) MOB10
    		, SUM(IF(还款期数 = 11, 净入催本金, NULL)) / SUM(IF(还款期数 = 11, 净出账本金, NULL)) MOB11
    		, SUM(IF(还款期数 = 12, 净入催本金, NULL)) / SUM(IF(还款期数 = 12, 净出账本金, NULL)) MOB12
    		, NULL MOB13
    
    		, SUM(IF(还款期数 = 1, 净催回本金, NULL)) / SUM(IF(还款期数 = 1, 净入催本金, NULL)) 'MOB1%'
    		, SUM(IF(还款期数 = 2, 净催回本金, NULL)) / SUM(IF(还款期数 = 2, 净入催本金, NULL)) 'MOB2%'
    		, SUM(IF(还款期数 = 3, 净催回本金, NULL)) / SUM(IF(还款期数 = 3, 净入催本金, NULL)) 'MOB3%'
    		, SUM(IF(还款期数 = 4, 净催回本金, NULL)) / SUM(IF(还款期数 = 4, 净入催本金, NULL)) 'MOB4%'
    		, SUM(IF(还款期数 = 5, 净催回本金, NULL)) / SUM(IF(还款期数 = 5, 净入催本金, NULL)) 'MOB5%'
    		, SUM(IF(还款期数 = 6, 净催回本金, NULL)) / SUM(IF(还款期数 = 6, 净入催本金, NULL)) 'MOB6%'
    		, SUM(IF(还款期数 = 7, 净催回本金, NULL)) / SUM(IF(还款期数 = 7, 净入催本金, NULL)) 'MOB7%'
    		, SUM(IF(还款期数 = 8, 净催回本金, NULL)) / SUM(IF(还款期数 = 8, 净入催本金, NULL)) 'MOB8%'
    		, SUM(IF(还款期数 = 9, 净催回本金, NULL)) / SUM(IF(还款期数 = 9, 净入催本金, NULL)) 'MOB9%'
    		, SUM(IF(还款期数 = 10, 净催回本金, NULL)) / SUM(IF(还款期数 = 10, 净入催本金, NULL)) 'MOB10%'
    		, SUM(IF(还款期数 = 11, 净催回本金, NULL)) / SUM(IF(还款期数 = 11, 净入催本金, NULL)) 'MOB11%'
    		, SUM(IF(还款期数 = 12, 净催回本金, NULL)) / SUM(IF(还款期数 = 12, 净入催本金, NULL)) 'MOB12%'
    		
    		, (SUM(IF(还款期数 = 1, 净入催本金, 0)) - SUM(IF(还款期数 = 1, 净催回本金, 0))) + (SUM(IF(还款期数 = 2, 净入催本金, 0)) - SUM(IF(还款期数 = 2, 净催回本金, 0))) + (SUM(IF(还款期数 = 3, 净入催本金, 0)) - SUM(IF(还款期数 = 3, 净催回本金, 0))) + (SUM(IF(还款期数 = 4, 净入催本金, 0)) - SUM(IF(还款期数 = 4, 净催回本金, 0)))
    		    + (SUM(IF(还款期数 = 5, 净入催本金, 0)) - SUM(IF(还款期数 = 5, 净催回本金, 0))) + (SUM(IF(还款期数 = 6, 净入催本金, 0)) - SUM(IF(还款期数 = 6, 净催回本金, 0))) + (SUM(IF(还款期数 = 7, 净入催本金, 0)) - SUM(IF(还款期数 = 7, 净催回本金, 0)))
    		    + (SUM(IF(还款期数 = 8, 净入催本金, 0)) - SUM(IF(还款期数 = 8, 净催回本金, 0))) + (SUM(IF(还款期数 = 9, 净入催本金, 0)) - SUM(IF(还款期数 = 9, 净催回本金, 0))) + (SUM(IF(还款期数 = 10, 净入催本金, 0)) - SUM(IF(还款期数 = 10, 净催回本金, 0)))
    		    + (SUM(IF(还款期数 = 11, 净入催本金, 0)) - SUM(IF(还款期数 = 11, 净催回本金, 0))) + (SUM(IF(还款期数 = 12, 净入催本金, 0)) - SUM(IF(还款期数 = 12, 净催回本金, 0))) 坏账余额,
    		sum(if(还款期数 = 1, 净出账本金, 0)) + sum(if(还款期数 = 2, 净出账本金, 0)) + sum(if(还款期数 = 3, 净出账本金, 0)) + sum(if(还款期数 = 4, 净出账本金, 0)) + sum(if(还款期数 = 5, 净出账本金, 0))+sum(if(还款期数 = 6, 净出账本金, 0))+sum(if(还款期数 = 7, 净出账本金, 0))+sum(if(还款期数 = 8, 净出账本金, 0))
			+sum(if(还款期数 = 9, 净出账本金, 0))+sum(if(还款期数 = 10, 净出账本金, 0))+sum(if(还款期数 = 11, 净出账本金, 0))+sum(if(还款期数 = 12, 净出账本金, 0)) 出账金额
    FROM (
    	SELECT p1.放款编号, p1.放款月份, p1.还款期数, p1.入催观察日期
    	       -- 出账订单，当期为首期或（上一期当前不逾期且上一期实还日期为当期应还日之前）
    			, if(p1.入催观察日期 < current_date() AND (p1.还款期数 = 1 OR (p2.实还日期 IS NOT NULL AND p2.实还日期 <= p1.入催观察日期) OR (p1.实还日期 IS NULL and  DATEDIFF(now(), p2.实还日期)>=0)), b.余额, 0) 净出账本金
    			, if(p1.入催观察日期 < current_date() AND (p1.还款期数 = 1 OR (p2.实还日期 IS NOT NULL AND p2.实还日期 <= p1.入催观察日期) OR (p1.实还日期 IS NULL and  DATEDIFF(now(), p2.实还日期)>=0)) AND (p1.实还日期 IS NULL OR DATEDIFF(p1.实还日期 , p1.入催观察日期) > {{day1}}), b.余额, 0) 净入催本金
    			, if(p1.入催观察日期 < current_date() AND (p1.还款期数 = 1 OR (p2.实还日期 IS NOT NULL AND p2.实还日期 <= p1.入催观察日期) OR (p1.实还日期 IS NULL and  DATEDIFF(now(), p2.实还日期)>=0)) AND DATEDIFF(p1.实还日期 , p1.入催观察日期) > {{day}}, b.余额, 0) 净催回本金
    	FROM cycle_replan p1 LEFT JOIN cycle_replan p2 ON p1.放款编号 = p2.放款编号 AND p1.还款期数 = p2.还款期数 + 1
    	LEFT JOIN cycle_balance b ON b.放款编号 = p1.放款编号 AND b.还款期数 = p1.还款期数
    	ORDER BY 放款编号, 还款期数
    ) c
    LEFT JOIN amount a ON c.放款月份 = a.放款月份
    GROUP BY 1, 2, 3
) t

UNION ALL

SELECT 'MONTH' 统计口径
    , 放款月份
    , 放款金额
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

    , NULL 'MOB1%'
	, NULL 'MOB2%'
	, NULL 'MOB3%'
	, NULL 'MOB4%'
	, NULL 'MOB5%'
	, NULL 'MOB6%'
	, NULL 'MOB7%'
	, NULL 'MOB8%'
	, NULL 'MOB9%'
	, NULL 'MOB10%'
	, NULL 'MOB11%'
	, NULL 'MOB12%'
	
	, NULL 'LOSS%'
FROM (
    SELECT a.放款月份, a.放款金额, t.MOB, t.逾期余额, ifnull(t.逾期余额,0) / a.放款金额 逾期率
    FROM (
        SELECT 放款月份, MOB, sum(CASE WHEN 逾期天数 > 0 THEN 余额 END) 逾期余额
        FROM balance
        GROUP BY 放款月份, MOB
    ) t LEFT JOIN amount a ON t.放款月份 = a.放款月份
) v
GROUP BY 1, 2, 3
ORDER BY 1, 2
