# CryptoForecast AI — Realtime BTCUSDT Prediction System

End-to-end AI cryptocurrency price-prediction system: **Binance OHLCV → technical
indicators → 5-model ensemble → next-day close forecast → live dashboard with a
graded track record.**

## Quick start

```bash
python3 train_models.py     # (already run) trains & saves all models -> artifacts/
python3 app.py              # serves dashboard on 0.0.0.0:5000
```

### One-command implementation (new machine)

```bash
./run.sh          # venv + deps + data + train + serve   (first run ~5 min)
./run.sh train    # retrain with latest data, then exit
./run.sh serve    # just start the dashboard
```

`run.sh` is idempotent: it only installs/builds what's missing.

## Deployment

**Your laptop:** `./run.sh` → open `http://localhost:5000`.

**24/7 cloud (any free VPS, $0):**
```bash
git clone <your-repo> && cd <repo> && ./run.sh serve     # under tmux/systemd
# daily retrain at 00:10 UTC:
crontab -e
10 0 * * * cd /path/to/repo && ./run.sh train >> retrain.log 2>&1
```
Live market data auto-updates every day at 00:00 UTC — the dashboard needs
**no action** to keep forecasting fresh; only `./run.sh train` (cron, weekly is
fine) refreshes the models themselves.

**Programmatic integration:** use `client_example.py` —
`GET /api/predict` returns the full forecast payload as JSON (tested end-to-end).

## Architecture

```
Binance Vision API (live daily klines)          GitHub: Speirsy11/crypto-dataset
        │  fallback: BTCUSDT-1d.csv cache        (2017-08-17 → present, CC0)
        └──────────────┬─────────────────────────────┘
                       ▼
        Feature engineering (prepare_features.py)
        SMA/EMA 7·14·30 · RSI(14) · MACD(12/26/9) · Bollinger(20,2)
        Daily_Return · Volatility · Volume_Change · price lags · wicks
                       ▼
   ┌────────────────────────────────────────────────────────┐
   │ Linear Regression │ ARIMA(2,1,2) │ Random Forest       │
   │ XGBoost            │ LSTM (PyTorch, 30-day window)      │
   └───────────────┬────────────────────────────────────────┘
                   ▼            ensemble = mean of 4 ML models
        Next-day close forecast + 90% band + direction agreement
                   ▼
        Flask dashboard (app.py) — live price, forecast,
        risk panel (RSI/vol/ATR/stops), 2026 backtest chart,
        model comparison, auto-graded prediction log
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Web dashboard + JSON API (the realtime system) |
| `engine.py` | Loads trained artifacts, produces one forecast from latest completed candle |
| `market_data.py` | Live klines fetch (Binance Vision) with local-cache fallback |
| `train_models.py` | Trains/evaluates all models, saves `artifacts/` |
| `prepare_features.py` | OHLCV → indicators feature builder (reused everywhere) |
| `BTCUSDT-1d.csv` | Historical daily dataset (3,325 rows, 2017→2026) |
| `BTCUSDT-1d-features.csv` | Dataset + 16 indicators + `Target_Close_t1` label |
| `DATASET.md` | Dataset data card (provenance, schema, checks) |
| `artifacts/` | `models.joblib`, `arima.pkl`, `lstm.pt`, `metrics.json`, `test_predictions.csv`, `prediction_log.json` |

## Evaluation protocol (no leakage)

- **TRAIN** ≤ 2023‑12‑31 · **VAL** 2024‑2025 · **TEST** 2026 (held out)
- Production models are fit on train+val only; the dashboard's backtest is a
  true out-of-sample run over 265 days of 2026.
- Every live forecast is logged and later graded against the realized close
  (`artifacts/prediction_log.json`) → the **track record** table is the system's
  honest scorecard.

### Held-out 2026 results (n = 265)

| Model | MAE $ | RMSE $ | MAPE % | Dir acc % |
|---|---|---|---|---|
| Persistence (naive) | 1,201 | 1,758 | **1.66** | n/a |
| Linear Regression | **1,210** | **1,776** | **1.68** | **50.6** |
| ARIMA(2,1,2) | 1,819 | 2,472 | 2.51 | 47.9 |
| XGBoost | 1,834 | 2,459 | 2.51 | 49.4 |
| Random Forest | 2,359 | 3,075 | 3.21 | 46.0 |
| LSTM | 3,357 | 4,340 | 4.55 | 51.7 |
| **Ensemble (avg of 4)** | 1,857 | 2,492 | 2.54 | 47.9 |

Interpretation (also shown on the dashboard): next-day BTC is extremely hard to
beat — persistence ("tomorrow ≈ today") achieves ~1.7% MAPE. Linear regression
matches it while adding ~51% directional signal. Beating persistence over long
horizons is where the real edge (and risk reduction) would come from; the system
reports this honestly rather than overpromising.

## API

| Endpoint | Returns |
|---|---|
| `GET /` | Dashboard UI |
| `GET /api/predict` | Live price, forecast (per model + ensemble), 90% band, risk panel, graded log |
| `GET /api/metrics` | Val/test metrics for every model |
| `GET /api/chart` | 2026 actual closes vs ensemble/persistence backtest |
| `GET /api/log` | Raw track record |
| `GET /health` | Liveness |

## Realtime behaviour

- Forecasts refresh every **45 s**; data comes from Binance Vision's live daily
  klines (no API key). The in-progress (today) candle is used only for the
  *live price* card — forecasts are always computed from **completed** candles.
- When a target day closes, the log grades it automatically (error %, direction
  hit/miss) — predictions are never silently dropped.
- Re-run `python3 train_models.py` to retrain with fresh history; the dashboard
  picks up new artifacts on next load (restart `app.py` to clear caches).

---
*Educational project — not financial advice. Models will be wrong sometimes;
the track record table exists so you can see exactly how wrong.*
