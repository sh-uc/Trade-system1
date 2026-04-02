import os

import requests

from daily_paper_monitor import build_line_summary, build_monitor_report, parse_label_set
from db_utils import get_supabase


def send_line_message_broadcast(text: str) -> None:
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if not token:
        print("[WARN] LINE_CHANNEL_TOKEN not set; skip LINE broadcast")
        return
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"messages": [{"type": "text", "text": text[:4900]}]}
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[LINE broadcast]", response.status_code, response.text)


def send_line_message_push(user_id: str, text: str) -> None:
    token = os.environ.get("LINE_CHANNEL_TOKEN")
    if not token:
        print("[WARN] LINE_CHANNEL_TOKEN not set; skip LINE push")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"to": user_id, "messages": [{"type": "text", "text": text[:4900]}]}
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print("[LINE push]", response.status_code, response.text)


def save_report(report: dict) -> None:
    sb = get_supabase()
    row = {
        "report_date": report["as_of"].isoformat(),
        "strategy_name": report["strategy_name"],
        "simulation_name": report["position_simulation_name"] or "",
        "latest_market_date": report["latest_market_date"].isoformat(),
        "open_positions_count": len(report["open_positions"]),
        "entry_candidates_count": len(report["candidates"]),
        "blocked_signals_count": len(report["blocked"]),
        "report_markdown": report["report_text"],
        "line_summary": build_line_summary(report),
        "meta": {
            "report_path": str(report["report_path"]),
            "tickers": report["tickers"],
        },
    }
    sb.table("daily_monitor_reports").upsert(
        row, on_conflict="report_date,strategy_name,simulation_name"
    ).execute()


def main() -> None:
    from datetime import date

    as_of = os.getenv("MONITOR_DATE")
    strategy_name = os.getenv("STRATEGY_NAME", "swing_v1_candidate9")
    position_simulation_name = os.getenv("POSITION_SIMULATION_NAME", "")
    report_dir = os.getenv("REPORT_DIR", "docs/daily_reports")

    report = build_monitor_report(
        as_of=date.fromisoformat(as_of) if as_of else date.today(),
        strategy_name=strategy_name,
        position_simulation_name=position_simulation_name,
        report_dir=report_dir,
        earnings_block_days=int(os.getenv("EARNINGS_BLOCK_DAYS", "2")),
        earnings_block_labels=parse_label_set(os.getenv("EARNINGS_BLOCK_LABELS", "FY")),
        ex_dividend_block_days=int(os.getenv("EX_DIVIDEND_BLOCK_DAYS", "1")),
        upcoming_event_days=int(os.getenv("UPCOMING_EVENT_DAYS", "7")),
    )

    save_report(report)
    summary = build_line_summary(report)
    user_id = os.getenv("LINE_USER_ID", "").strip()
    if user_id:
        send_line_message_push(user_id, summary)
    else:
        send_line_message_broadcast(summary)

    print(f"[MONITOR_JOB] stored report for {report['as_of'].isoformat()} strategy={strategy_name}")


if __name__ == "__main__":
    main()
