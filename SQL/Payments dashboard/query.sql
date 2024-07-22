select 
	to_char(DATE_TRUNC('month', date), 'YYYY-MM-DD')::date as time,
	mode,
	count(status) FILTER (where status = 'Confirmed') / count(status)::numeric * 100.0 as percents
from
	new_payments
group by 
	DATE_TRUNC('month', date), 2
order by 1,2 asc