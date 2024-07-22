SELECT
    Id as review_id,
    toDateTime(Time) AS dt,
    Score as rating,
CASE
WHEN Score = 1 THEN 'negative'
WHEN Score = 5 THEN 'positive'
ELSE 'neutral'
END AS sentiment,
Text as review
FROM
simulator.flyingfood_reviews
ORDER BY
review_id asc