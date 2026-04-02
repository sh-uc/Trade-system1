import os
from typing import Tuple

import yfinance as yf

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None


def load_supabase_settings() -> Tuple[str, str, str]:
    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE")
        or os.environ.get("SUPABASE_KEY")
        or ""
    ).strip()
    source = "env"

    if (not url or not key) and tomllib is not None:
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            with open(secrets_path, "rb") as fh:
                data = tomllib.load(fh)
            sb = data.get("supabase", {})
            url = url or str(sb.get("url", "")).strip()
            key = key or str(sb.get("key", "")).strip()
            source = ".streamlit/secrets.toml"

    return url, key, source


def check_supabase() -> bool:
    url, key, source = load_supabase_settings()
    if not url or not key:
        print("[SKIP] Supabase: credentials not found in env or .streamlit/secrets.toml")
        return False
    if create_client is None:
        print("[NG] Supabase: supabase client library is not available")
        return False

    try:
        sb = create_client(url, key)
        res = sb.table("tickers").select("code").limit(1).execute()
        count = len(res.data or [])
        print(f"[OK] Supabase: connected via {source}, tickers sample rows={count}")
        return True
    except Exception as exc:
        print(f"[NG] Supabase: {exc}")
        return False


def check_yfinance() -> bool:
    ticker = (os.environ.get("CHECK_TICKER") or "3778.T").strip()
    try:
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=False, progress=False)
        if df.empty:
            print(f"[NG] yfinance: no rows returned for {ticker}")
            return False
        print(f"[OK] yfinance: {ticker} rows={len(df)} last_date={df.index[-1].date()}")
        return True
    except Exception as exc:
        print(f"[NG] yfinance: {exc}")
        return False


def main() -> int:
    supabase_ok = check_supabase()
    yfinance_ok = check_yfinance()

    if supabase_ok and yfinance_ok:
        print("[OK] connectivity_check: both checks passed")
        return 0

    print("[WARN] connectivity_check: one or more checks did not pass")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

