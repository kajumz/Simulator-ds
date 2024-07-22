WITH QuantityPerItem AS (
    SELECT
        user_id,
        item_id,
        SUM(units) AS qty
    FROM
        default.karpov_express_orders
    WHERE
        toDate(timestamp) BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY
        user_id, item_id
)

, AvgPricePerItem AS (
    SELECT
        item_id,
        ROUND(AVG(price), 2) AS price
    FROM
        default.karpov_express_orders
    WHERE
        toDate(timestamp) BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY
        item_id
)

SELECT
    QPI.user_id,
    QPI.item_id,
    QPI.qty AS qty,
    APPI.price AS price
FROM
    QuantityPerItem QPI
JOIN
    AvgPricePerItem APPI ON QPI.item_id = APPI.item_id
ORDER BY QPI.user_id, QPI.item_id



