"""
Feature engineering for the Binance OHLCV crypto dataset.

Input : BTCUSDT-1d.csv   (Date, Open, High, Low, Close, Volume)
Output: BTCUSDT-1d-features.csv  (+ the same columns plus technical indicators)

Usage:
    python prepare_features.py                       # BTCUSDT 1d (defaults)
    python prepare_features.py ETHUSDT 1d            # another symbol/interval
"""
import sys
import glob
import os
import urllib.request
import concurrent.futures

import pandas as pd
import numpy as np

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
INTERVAL = sys.argv[2] if len(sys.argv) > 2 else "1d"

BASE_RAW = "https://media.githubusercontent.com/media/Speirsy11/crypto-dataset/main/"
REPO = "crypto-dataset"
PARTITION = f"data/interval_id={INTERVAL}/symbol_id={SYMBOL}"


# ---------------------------------------------------------------- data loading
def ensure_data() -> pd.DataFrame:
    """Sparse-fetch the parquet partition for SYMBOL/INTERVAL if missing."""
    pattern = os.path.join(REPO, PARTITION, "**", "*.parquet")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        raise SystemExit(
            f"No local parquet files for {SYMBOL}/{INTERVAL}.\n"
            f"Clone with: git clone --filter=blob:none --sparse "
            f"https://github.com/Speirsy11/crypto-dataset.git {REPO}\n"
            f"git -C {REPO} sparse-checkout set {PARTITION}"
        )

    def fetch(path):
        if open(path, "rb").read(4) == b"PAR1":
            return
        url = BASE_RAW + os.path.relpath(path, REPO).replace(os.sep, "/")
        urllib.request.urlretrieve(url, path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(fetch, files))

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df.rename(columns={"timestamp": "Date"})
    df["Date"] = df["Date"].dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    return df[["Date", "open", "high", "low", "close", "volume"]].rename(
        columns={"open": "Open", "high": "High", "low": "Low",
                 "close": "Close", "volume": "Volume"}
    )


# ------------------------------------------------------------- helper indicators
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# ------------------------------------------------------------------- build features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["Close"]

    # Moving averages
    for w in (7, 14, 30):
        out[f"SMA_{w}"] = c.rolling(w).mean()
        out[f"EMA_{w}"] = c.ewm(span=w, adjust=False).mean()

    # Momentum
    out["RSI"] = rsi(c, 14)
    out["MACD"], out["MACD_Signal"] = macd(c)

    # Bollinger Bands (20, 2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    out["Bollinger_Upper"] = sma20 + 2 * std20
    out["Bollinger_Lower"] = sma20 - 2 * std20

    # Returns / risk / volume
    out["Daily_Return"] = c.pct_change()
    out["Volatility"] = out["Daily_Return"].rolling(14).std() * np.sqrt(365)  # annualised
    out["Price_Change"] = c - out["Open"]              # intraday move
    out["Volume_Change"] = out["Volume"].pct_change()

    # Target column for supervised learning (next-day close)
    out["Target_Close_t1"] = c.shift(-1)

    return out


if __name__ == "__main__":
    raw_path = f"{SYMBOL}-{INTERVAL}.csv"
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path, parse_dates=["Date"])
        raw["Date"] = raw["Date"].dt.strftime("%Y-%m-%d")
    else:
        raw = ensure_data()
        raw.to_csv(raw_path, index=False)

    feat = build_features(raw)
    out_path = f"{SYMBOL}-{INTERVAL}-features.csv"
    feat.to_csv(out_path, index=False)

    print(f"raw      : {raw_path}  ({len(raw)} rows)")
    print(f"features : {out_path}  ({len(feat)} rows, {len(feat.columns)} cols)")
    print("columns  :", ", ".join(feat.columns))
    print("\nfirst fully-popular row (after 30d warm-up):")
    print(feat.dropna().head(2).to_string(index=False))
