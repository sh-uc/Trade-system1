## README.md（Supabase設定 追記）
```
### Supabase 環境変数（推奨: Streamlit Cloud の Secrets）
# .streamlit/secrets.toml に以下を設定
[supabase]
url = "https://xxxxxxxxxxxx.supabase.co"
key = "サービスロール or anon キー"


### 初期テーブル（SQL）
-- prices: 価格系列
create table if not exists prices (
date date not null,
code text not null,
open numeric, high numeric, low numeric, close numeric, volume bigint,
primary key (date, code)
);


-- indicators: 指標
create table if not exists indicators (
date date not null,
code text not null,
rsi14 numeric, macd numeric, macd_signal numeric, atr14 numeric,
stdev20 numeric, ma25 numeric, ma75 numeric, vol_ma20 numeric,
vol_spike boolean, swing_low20 numeric, swing_high20 numeric,
primary key (date, code)
);


-- signals: 判定結果
create table if not exists signals (
date date not null,
code text not null,
action text not null,
summary text,
reasons jsonb,
close numeric,
qty integer,
sl numeric,
tp numeric,
risk_jpy numeric,
created_at timestamp with time zone default now(),
primary key (date, code)
);
```


---


# -*- coding: utf-8 -*-