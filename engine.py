"""
Realtime prediction engine.

Loads the artifacts saved by train_models.py and produces a one-step-ahead
(next completed daily close) forecast from the latest completed candle,
plus indicator/risk snapshots for the dashboard.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import torch

from prepare_features import build_features
from train_models import LSTMNet, make_lag_features

ART = "artifacts"
_cache = {}


def load_models():
    """Load artifacts; prefer portable JSON/NPZ (any OS / package version),
    fall back to the legacy joblib bundle for older artifact sets."""
    if _cache:
        return _cache
    meta = json.load(open(os.path.join(ART, "metrics.json")))
    cfg = json.load(open(os.path.join(ART, "lstm_cfg.json")))
    lstm = LSTMNet(cfg["n_feat"], cfg["hidden"], cfg["layers"])
    lstm.load_state_dict(torch.load(os.path.join(ART, "lstm.pt"), weights_only=True))
    lstm.eval()
    _cache.update(meta=meta, lstm=lstm, lstm_cfg=cfg)

    ls_path = os.path.join(ART, "linear_scalers.json")
    rf_path = os.path.join(ART, "rf_trees.npz")
    xgb_path = os.path.join(ART, "xgboost.json")

    if os.path.exists(ls_path) and os.path.exists(rf_path):
        # ---- portable path (version/OS independent) --------------------
        from portable_io import PortableLinear, PortableRF, PortableScaler
        p = json.load(open(ls_path))
        _cache["linear_scaler"] = PortableScaler(p["linear_mean"], p["linear_scale"])
        _cache["linear"] = PortableLinear(p["coef"], p["intercept"])
        _cache["random_forest"] = PortableRF(np.load(rf_path))
        _cache["lstm_scaler_X"] = PortableScaler(p["lstm_x_mean"], p["lstm_x_scale"])
        _cache["lstm_scaler_y"] = PortableScaler(p["lstm_y_mean"], p["lstm_y_scale"])
        _cache["portable"] = True
    else:
        # ---- legacy path: joblib bundle (trained pre-portable) ---------
        bundle = joblib.load(os.path.join(ART, "models.joblib"))
        _cache.update(bundle)

    # XGBoost: always via native JSON — pickled boosters break across platforms
    if os.path.exists(xgb_path):
        from xgboost import XGBRegressor
        xgm = XGBRegressor()
        xgm.load_model(xgb_path)
        _cache["xgb_model"] = xgm
    return _cache


def _features_frame(df_completed: pd.DataFrame) -> pd.DataFrame:
    """Identical pipeline to training: build_features -> lag features -> drop warm-up NaNs."""
    feat = make_lag_features(build_features(df_completed))
    meta = load_models()["meta"]
    # inference: only feature columns must be complete — the newest row has no
    # realized target yet, but it IS the row we want to predict from.
    feat = feat.dropna(subset=meta["feature_cols"]).reset_index(drop=True)
    return feat


def predict(df_completed: pd.DataFrame) -> dict:
    """df_completed: raw OHLCV (Date..Volume), ascending, latest COMPLETE candle last."""
    art = load_models()
    meta = art["meta"]
    cols = meta["feature_cols"]

    feat = _features_frame(df_completed)
    if len(feat) < art["lstm_cfg"]["seq_len"] + 1:
        raise ValueError("not enough history for feature warm-up")

    row = feat.iloc[-1]
    X = row[cols].values.reshape(1, -1)
    last_close = float(row["Close"])
    target_date = (pd.Timestamp(row["Date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # --- tree / linear models
    p_lin = float(art["linear"].predict(art["linear_scaler"].transform(X))[0])
    p_rf = float(art["random_forest"].predict(X)[0])
    xgm = art.get("xgb_model") or art.get("xgboost")
    p_xgb = float(xgm.predict(X)[0])

    # --- LSTM (sequence of the last seq_len rows, scaled like training)
    seq = feat[cols].values[-art["lstm_cfg"]["seq_len"]:]
    seq_s = art["lstm_scaler_X"].transform(seq)[None, :, :]
    with torch.no_grad():
        p_lstm = float(art["lstm_scaler_y"].inverse_transform(
            art["lstm"](torch.tensor(seq_s, dtype=torch.float32)).numpy()
            .reshape(-1, 1)).ravel()[0])

    # --- ARIMA: one-step forecast on log close (refit only when data changed)
    order = tuple(meta["arima_order"])
    closes = df_completed["Close"].values.astype(float)
    key = (order, len(closes), str(df_completed["Date"].iloc[-1]), float(closes[-1]))
    if _cache.get("arima_key") != key:
        ar = ARIMA(np.log(closes), order=order).fit()
        _cache["arima_key"], _cache["arima_value"] = key, float(np.exp(ar.forecast(1)[0]))
    p_arima = _cache["arima_value"]

    # --- ensemble + band
    core = {"linear": p_lin, "random_forest": p_rf, "xgboost": p_xgb, "lstm": p_lstm}
    ensemble = float(np.mean(list(core.values())))
    q90 = float(meta["residual_q90"])
    lo, hi = ensemble - q90, ensemble + q90

    preds_all = dict(core, arima=p_arima)
    ups = sum(1 for v in preds_all.values() if v > last_close)

    # --- indicator / risk snapshot (from the last completed candle's row)
    h = df_completed["High"].values.astype(float)
    lo_ = df_completed["Low"].values.astype(float)
    c = df_completed["Close"].values.astype(float)
    tr = np.maximum(h - lo_, np.maximum(np.abs(h - np.roll(c, 1)),
                                        np.abs(lo_ - np.roll(c, 1))))
    atr = float(tr[-14:].mean())  # ATR(14)

    risk = {
        "rsi": round(float(row["RSI"]), 1),
        "macd_hist": round(float(row["MACD"] - row["MACD_Signal"]), 1),
        "vol_ann_pct": round(float(row["Volatility"]) * 100, 1),
        "sma30": round(float(row["SMA_30"]), 2),
        "ema14": round(float(row["EMA_14"]), 2),
        "price_vs_sma30_pct": round((last_close / float(row["SMA_30"]) - 1) * 100, 2),
        "atr14": round(atr, 2),
        "bollinger_pos": round(float((last_close - row["Bollinger_Lower"]) /
                                     (row["Bollinger_Upper"] - row["Bollinger_Lower"]) * 100), 1),
        "stop_long": round(last_close - 2 * atr, 2),
        "stop_short": round(last_close + 2 * atr, 2),
    }

    exp_move_pct = (ensemble / last_close - 1) * 100
    return {
        "target_date": target_date,
        "basis_date": str(row["Date"]),
        "last_close": round(last_close, 2),
        "predictions": {k: round(v, 2) for k, v in preds_all.items()},
        "ensemble": round(ensemble, 2),
        "band90": {"low": round(lo, 2), "high": round(hi, 2), "half": round(q90, 2)},
        "expected_move_pct": round(exp_move_pct, 2),
        "direction": "UP" if exp_move_pct > 0 else ("DOWN" if exp_move_pct < 0 else "FLAT"),
        "agreement": {"up": int(ups), "down": int(len(preds_all) - ups), "total": len(preds_all)},
        "risk": risk,
    }


if __name__ == "__main__":
    from market_data import get_market
    m = get_market()
    r = predict(m["completed"])
    print(json.dumps(r, indent=2))
