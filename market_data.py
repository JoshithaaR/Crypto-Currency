"""
Live market-data acquisition with graceful fallbacks.

Priority:
  1. Binance Vision public API  (live daily klines, no auth)   -> "binance-vision"
  2. Local cached CSV           (BTCUSDT-1d.csv)               -> "local-cache"

Returns raw OHLCV frames shaped exactly like BTCUSDT-1d.csv
(Date, Open, High, Low, Close, Volume) so feature engineering is identical
to training.
"""
import io
import json
import os
import urllib.request
from datetime import datetime, timezone

import pandas as pd

LOCAL_CSV = "BTCUSDT-1d.csv"
VISION_URL = ("https://data-api.binance.vision/api/v3/klines"
              "?symbol={sym}&interval={itv}&limit={lim}")
TIMEOUT = 10


def _utc_today() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def fetch_live(symbol="BTCUSDT", interval="1d", limit=400) -> pd.DataFrame:
    url = VISION_URL.format(sym=symbol, itv=interval, lim=limit)
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        raw = json.loads(r.read().decode())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"unexpected payload from Binance Vision: {str(raw)[:120]}")
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    rows = []
    for k in raw:
        rows.append({
            "Date": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "Open": float(k[1]), "High": float(k[2]), "Low": float(k[3]),
            "Close": float(k[4]), "Volume": float(k[5]),
            "in_progress": float(k[6]) > now_ms,   # candle not yet closed
        })
    df = pd.DataFrame(rows)
    return df


def fetch_local() -> pd.DataFrame:
    df = pd.read_csv(LOCAL_CSV, parse_dates=["Date"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["in_progress"] = False
    return df


def get_market() -> dict:
    """Returns {all, completed, source, live_price, fetched_at, last_completed_date}."""
    try:
        all_df = fetch_live()
        source = "binance-vision (live)"
    except Exception as e:
        all_df = fetch_local()
        source = f"local-cache (stale: {type(e).__name__})"

    # split completed vs in-progress (today's forming candle)
    today = _utc_today().strftime("%Y-%m-%d")
    if "in_progress" in all_df.columns and bool(all_df["in_progress"].iloc[-1]) \
            and all_df["Date"].iloc[-1] >= today:
        completed = all_df.iloc[:-1].copy()
        in_progress = True
    else:
        completed = all_df.copy()
        in_progress = False
        all_df = all_df.copy()
        all_df["in_progress"] = False

    completed = completed.sort_values("Date").reset_index(drop=True)
    all_df = all_df.sort_values("Date").reset_index(drop=True)

    return {
        "all": all_df,
        "completed": completed,
        "source": source,
        "live_price": float(all_df["Close"].iloc[-1]),
        "in_progress": in_progress,
        "target_date": (pd.Timestamp(completed["Date"].iloc[-1]) + pd.Timedelta(days=1))
                       .strftime("%Y-%m-%d"),
        "last_completed_date": completed["Date"].iloc[-1],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    m = get_market()
    print(m["source"], "| live:", m["live_price"], "| completed thru:",
          m["last_completed_date"], "-> predicting", m["target_date"])
    print(m["completed"].tail(3).to_string(index=False))
