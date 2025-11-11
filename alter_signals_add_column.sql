alter table signals
  add column if not exists prev_close double precision;

-- prices から前日終値を埋める
update signals s
set prev_close = sub.prev_close
from (
  select
    p.date,
    p.code,
    lag(p.close) over (partition by p.code order by p.date) as prev_close
  from prices p
) sub
where s.date = sub.date
  and s.code = sub.code
  and s.prev_close is null;
