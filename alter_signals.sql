alter table signals add column if not exists diff_pct double precision;

-- 価格テーブル prices(date, code, close) から LAG で前日比% を埋める
update signals s
set diff_pct = sub.diff_pct
from (
  select p.date, p.code,
         case when lag(p.close) over (partition by p.code order by p.date) > 0
              then (p.close - lag(p.close) over (partition by p.code order by p.date))
                   / lag(p.close) over (partition by p.code order by p.date) * 100
              else null end as diff_pct
  from prices p
) sub
where s.date = sub.date and s.code = sub.code and s.diff_pct is null;
