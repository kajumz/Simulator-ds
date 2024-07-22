WITH AllUsers AS (
    SELECT DISTINCT user_id
    FROM default.churn_submits
),
AllDays AS (
    SELECT DISTINCT DATE(timestamp) as day
    FROM default.churn_submits
),
UserStats as (
SELECT
    u.user_id as user_id,
    d.day as day,
    coalesce(n_submits, 0) as n_submits,
    coalesce(n_tasks, 0) as n_tasks,
    coalesce(n_solved, 0) as n_solved
FROM AllUsers u
CROSS JOIN AllDays d
LEFT JOIN (
    SELECT
        user_id,
        DATE(timestamp) as day,
        count(submit) as n_submits,
        count(distinct task_id) as n_tasks,
        countIf(is_solved) as n_solved
    FROM default.churn_submits
    GROUP BY user_id, day
) AS tmp
ON u.user_id = tmp.user_id AND d.day = tmp.day
ORDER BY u.user_id, d.day),
WithLag as 
(
    select
        user_id,
        day,
        n_submits,
        n_tasks,
        n_solved,
        max(case when n_submits > 0 then day else Null END) over (partition by user_id order by day) as last_activity,
        sum(n_submits) over (partition by user_id order by day rows between 13 preceding and current row) as submits,
        sum(n_solved) over (partition by user_id order by day rows between 13 preceding and current row) as solved,
        sum(n_solved) over (partition by user_id order by day) as solved_total,
        sum(n_submits) over (partition by user_id order by day rows between 1 following and 14 following) as target_day
        from
            UserStats
)
select
    day,
    user_id,
    Case
        when last_activity is Null then Null
        else day - last_activity
        end as days_offline,
    submits / 14 as avg_submits_14d,
    case 
        when submits > 0 then solved / submits
        else 0
        end as success_rate_14d,
    solved_total,
    case 
        when target_day > 0 then 0
        else 1
        end as target_14d
from
    WithLag
