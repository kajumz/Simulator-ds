with tabl as
(
select *
from new_payments
where status = 'Confirmed'
)
select
    to_char(date_trunc('month', date), 'MM/DD/YY')::date as time,
    sum(amount) / count(distinct email_id) as arppu,
    sum(amount) / count(id) as aov
from tabl
group by to_char(date_trunc('month', date), 'MM/DD/YY')::date
order by 1