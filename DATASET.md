# Dataset: Binance BTCUSDT Daily OHLCV + Technical Indicators

Prepared for an **AI-based cryptocurrency price prediction system**
(historical trend analysis → next-day price forecasting → investor decision support / risk reduction).

## Files

| File | Rows | Description |
|---|---|---|
| `BTCUSDT-1d.csv` | 3,325 | Raw daily OHLCV candles (recommended starting subset) |
| `BTCUSDT-1d-features.csv` | 3,325 | OHLCV + 16 technical/derived features + next-day target |
| `prepare_features.py` | — | Reproducible loader + feature engineering script |
| `crypto-dataset/` | — | Sparse Git checkout of the source repo (BTCUSDT/1d partition only, ~0.5 MB) |

## Provenance

- **Source:** [Speirsy11/crypto-dataset](https://github.com/Speirsy11/crypto-dataset) — public Binance USDT OHLCV candles in partitioned Parquet (built from Binance public 1‑minute market data).
- **Coverage:** 2017‑08‑17 → 2026‑09‑23 (last complete UTC day at download time; refreshed daily upstream).
- **Exchange / pair:** Binance spot, `BTCUSDT`. Candles use **UTC** boundaries.
- **License:** data **CC0 / public domain** (`DATA_LICENSE`); scripts MIT. Not affiliated with Binance.
- **Full repo scope:** 10 symbols (BTC, ETH, SOL, BNB, XRP, TRX, DOGE, ZEC, ADA, BCH) × 8 intervals (1m, 5m, 15m, 1h, 4h, 1d, 1w, 1mo), 41.8 M+ 1‑minute candles.

## Quality checks performed

- ✅ 3,325 daily rows, **no missing days** (max gap = 1), **0 nulls**, no duplicate timestamps
- ✅ OHLC integrity: `high ≥ max(open, close)`, `low ≤ min(open, close)` on every row
- ✅ Price range 3,189.02 → 124,658.54 USDT; first row matches the known reference values (2017‑08‑17 open 4261.48, volume 795.15)

## Schema

**Raw (`BTCUSDT-1d.csv`)**
`Date, Open, High, Low, Close, Volume` (Volume in base asset, BTC)

**Features (`BTCUSDT-1d-features.csv`)** — raw plus:

| Group | Columns |
|---|---|
| Trend | `SMA_7/14/30`, `EMA_7/14/30` |
| Momentum | `RSI` (Wilder, 14), `MACD` (12/26), `MACD_Signal` (9) |
| Volatility bands | `Bollinger_Upper/Lower` (20, 2σ) |
| Returns & risk | `Daily_Return`, `Volatility` (14d annualised σ), `Price_Change` (Close−Open), `Volume_Change` |
| Supervised target | `Target_Close_t1` = Close(t+1) — null on the final row only |

Rows before the 30‑day warm‑up have partial indicator values (first fully populated row: 2017‑09‑15 → **3,295 training‑ready rows**).

## Refresh / extend

```bash
python prepare_features.py                 # BTCUSDT 1d (default)
python prepare_features.py ETHUSDT 1d      # any symbol/interval in the repo
git -C crypto-dataset pull                 # upstream updates daily
```

Note: parquet files are **Git LFS** objects; `prepare_features.py` auto-fetches
missing LFS content from `media.githubusercontent.com`.

## Modelling notes (anti‑leakage)

- Split **chronologically** (e.g. train 2017‑2023, validation 2024‑2025, test 2026) — never random‑shuffle time series.
- Scale features *after* splitting (fit scaler on train only).
- Strong baselines to beat: last‑close persistence, linear regression, ARIMA → then RandomForest/XGBoost → LSTM/GRU/Transformer.
- Predict `Target_Close_t1` or `Daily_Return(t+1)`; report MAE, RMSE, MAPE, directional accuracy.

## Companion / alternative datasets

- **Sentiment (2nd source for your architecture):** [Speirsy11/crypto-news-dataset](https://github.com/Speirsy11/crypto-news-dataset) — FinBERT‑scored crypto news & macro sentiment, article‑level + **daily** parquet, same 10 coins, updated daily through 2026‑09‑23. Pairs naturally with the OHLCV data for the OHLCV + sentiment → model pipeline.
- **Ready‑made ML competition data:** [G‑Research Crypto Forecasting (Kaggle)](https://www.kaggle.com/c/g-research-crypto-forecasting/data) — multi‑asset crypto returns with auxiliary market data, designed explicitly for return prediction.
- **Continuous self‑generated data:** [CoinGecko API](https://www.coingecko.com/en/api) — historical price, market cap, volume, OHLC (10+ years on paid plans) for keeping the dataset current after the project ends.
- **Official raw source:** [binance/binance-public-data](https://github.com/binance/binance-public-data) — kline CSVs direct from Binance if you ever need more pairs.
