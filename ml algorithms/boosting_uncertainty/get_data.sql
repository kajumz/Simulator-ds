WITH sub AS (
    SELECT
        product_name,
        toMonday(dt) AS monday,
        max(price) AS max_price,
        count(*) AS y,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 1 PRECEDING AND 1 preceding) AS y_lag_1,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 2 PRECEDING AND 2 PRECEDING) AS y_lag_2,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 3 PRECEDING AND 3 PRECEDING) AS y_lag_3,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 4 PRECEDING AND 4 PRECEDING) AS y_lag_4,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 5 PRECEDING AND 5 PRECEDING) AS y_lag_5,
        any(y) OVER (PARTITION BY product_name ORDER BY toMonday(dt) ROWS BETWEEN 6 PRECEDING AND 6 PRECEDING) AS y_lag_6
    FROM
        default.data_sales_train
    GROUP BY
        product_name,
        monday
),
aggregated AS (
    SELECT
        monday,
        SUM(y_lag_1) AS y_all_lag_1,
        SUM(y_lag_2) AS y_all_lag_2,
        SUM(y_lag_3) AS y_all_lag_3,
        SUM(y_lag_4) AS y_all_lag_4,
        SUM(y_lag_5) AS y_all_lag_5,
        SUM(y_lag_6) AS y_all_lag_6
    FROM 
        sub
    GROUP BY
        monday
)
SELECT
    sub.product_name,
    sub.monday,
    sub.max_price,
    sub.y,
    sub.y_lag_1,
    sub.y_lag_2,
    sub.y_lag_3,
    sub.y_lag_4,
    sub.y_lag_5,
    sub.y_lag_6,
    (sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3) / 3 as y_avg_3,
    greatest(sub.y_lag_1, greatest(sub.y_lag_2, sub.y_lag_3))  AS y_max_3,
    least(sub.y_lag_1, least(sub.y_lag_2,sub.y_lag_3)) AS y_min_3,
    (sub.y_lag_1 + sub.y_lag_2 + sub.y_lag_3 + sub.y_lag_4 + sub.y_lag_5 + sub.y_lag_6) / 6  as y_avg_6,
    greatest(sub.y_lag_1, greatest(sub.y_lag_2, greatest(sub.y_lag_3, greatest(sub.y_lag_4, greatest(sub.y_lag_5, sub.y_lag_6))))) AS y_max_6,
    least(sub.y_lag_1, least(sub.y_lag_2, least(sub.y_lag_3, least(sub.y_lag_4, least(sub.y_lag_5, sub.y_lag_6))))) AS y_min_6,
    aggregated.y_all_lag_1,
    aggregated.y_all_lag_2,
    aggregated.y_all_lag_3,
    aggregated.y_all_lag_4,
    aggregated.y_all_lag_5,
    aggregated.y_all_lag_6,
    (aggregated.y_all_lag_1 + aggregated.y_all_lag_2 + aggregated.y_all_lag_3) / 3 as y_all_avg_3,
    greatest(aggregated.y_all_lag_1, greatest(aggregated.y_all_lag_2, aggregated.y_all_lag_3)) AS y_all_max_3,
    least(aggregated.y_all_lag_1, least(aggregated.y_all_lag_2,aggregated.y_all_lag_3)) AS y_all_min_3,
    (aggregated.y_all_lag_1 + aggregated.y_all_lag_2 + aggregated.y_all_lag_3 + aggregated.y_all_lag_4 + aggregated.y_all_lag_5 + aggregated.y_all_lag_6) / 6 as y_all_avg_6,
    greatest(aggregated.y_all_lag_1, greatest(aggregated.y_all_lag_2, greatest(aggregated.y_all_lag_3, greatest(aggregated.y_all_lag_4, greatest(aggregated.y_all_lag_5, aggregated.y_all_lag_6))))) AS y_all_max_6,
    least(aggregated.y_all_lag_1, least(aggregated.y_all_lag_2, least(aggregated.y_all_lag_3, least(aggregated.y_all_lag_4, least(aggregated.y_all_lag_5, aggregated.y_all_lag_6))))) AS y_all_min_6
FROM sub 
JOIN aggregated ON sub.monday = aggregated.monday