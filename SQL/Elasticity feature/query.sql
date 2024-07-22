select
    sku,
    dates,
    avg(price) as price,
    count(*) as qty
from transactions
group by sku, dates