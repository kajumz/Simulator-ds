WITH AllUsers AS (
    SELECT DISTINCT user_id
    FROM default.churn_submits
),
AllDays AS (
    SELECT DISTINCT DATE(timestamp) as day
    FROM default.churn_submits
)

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
ORDER BY u.user_id, d.day;
