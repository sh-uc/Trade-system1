-- まずRLSが有効になっているか（なっていないなら有効化は任意）
alter table prices     enable row level security;
alter table indicators enable row level security;
alter table signals    enable row level security;

-- prices: anon に読み書き・更新を許可
drop policy if exists anon_select on prices;
drop policy if exists anon_insert on prices;
drop policy if exists anon_update on prices;

create policy anon_select on prices
for select using (true);

create policy anon_insert on prices
for insert with check (true);

create policy anon_update on prices
for update using (true) with check (true);

-- indicators
drop policy if exists anon_select on indicators;
drop policy if exists anon_insert on indicators;
drop policy if exists anon_update on indicators;

create policy anon_select on indicators
for select using (true);

create policy anon_insert on indicators
for insert with check (true);

create policy anon_update on indicators
for update using (true) with check (true);

-- signals
drop policy if exists anon_select on signals;
drop policy if exists anon_insert on signals;
drop policy if exists anon_update on signals;

create policy anon_select on signals
for select using (true);

create policy anon_insert on signals
for insert with check (true);

create policy anon_update on signals
for update using (true) with check (true);

-- #