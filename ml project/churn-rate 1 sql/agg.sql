select
    user_id,
    DATE(timestamp) as day,
    count(submit) as n_submits,
    count(distinct task_id) as n_tasks,
    count(is_solved) FILTER (where is_solved = True) as n_solved
from default.churn_submits
group by user_id, DATE(timestamp)
order by user_id, DATE(timestamp)