"""
Model training & evaluation for BTCUSDT next-day close prediction.

Models: Persistence (baseline), Linear Regression, ARIMA, Random Forest,
        XGBoost, LSTM  + an ensemble average.

Chronological splits (no leakage):
  TRAIN : 2017-08-17 .. 2023-12-31
  VAL   : 2024-01-01 .. 2025-12-31   (model selection / early stopping)
  TEST  : 2026-01-01 .. 2026-09-23   (final honest evaluation)

After evaluation, production models are refit on TRAIN+VAL (<= 2025-12-31)
and saved to artifacts/. TEST 2026 remains unseen by the production models.

Outputs (in artifacts/):
  models.joblib      dict of sklearn/xgboost models
  arima.pkl          statsmodels ARIMA results (fit on <=2025)
  lstm.pt + lstm_cfg.json
  scalers.joblib     StandardScalers (features, LSTM X/y)
  metrics.json       val/test metrics for dashboard
  test_predictions.csv  2026 ensemble/model predictions vs actual
"""
import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import xgboost as xgb  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from prepare_features import build_features  # noqa: E402

ART = "artifacts"
SEED = 42
TRAIN_END = "2023-12-31"
VAL_END = "2025-12-31"
SEQ_LEN = 30

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------- helpers
def metrics(y_true, y_pred, close_t) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    act_dir = np.sign(y_true - close_t)
    pred_dir = np.sign(y_pred - close_t)
    mask = act_dir != 0  # days with no change can't be graded directionally
    dir_acc = float((act_dir[mask] == pred_dir[mask]).mean()) if mask.any() else None
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 3),
            "dir_acc": round(dir_acc * 100, 1) if dir_acc is not None else None}


def make_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for k in (1, 2, 3, 5, 7, 14):
        out[f"ret_lag{k}"] = out["Daily_Return"].shift(k)
    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["upper_wick"] = (out["High"] - out[["Open", "Close"]].max(axis=1)) / out["Close"]
    out["lower_wick"] = (out[["Open", "Close"]].min(axis=1) - out["Low"]) / out["Close"]
    return out


# --------------------------------------------------------------------- LSTM
class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X, y, seq=SEQ_LEN):
    xs, ys = [], []
    for i in range(seq, len(X)):
        xs.append(X[i - seq:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_lstm(X_tr, y_tr, X_val, y_val, epochs=60, patience=8, lr=1e-3):
    model = LSTMNet(X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    Xtr, ytr = make_sequences(X_tr, y_tr)
    Xv, yv = make_sequences(X_val, y_val)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    Xv_t = torch.tensor(Xv); yv_t = torch.tensor(yv)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xv_t), yv_t))
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1


# ----------------------------------------------------------------------- main
def main():
    os.makedirs(ART, exist_ok=True)
    df = make_lag_features(pd.read_csv("BTCUSDT-1d-features.csv", parse_dates=["Date"]))
    df = df.dropna().reset_index(drop=True)  # drops warm-up + lag NaN + last target row
    df = df.sort_values("Date")

    feature_cols = [c for c in df.columns if c not in ("Date", "Target_Close_t1")]
    ycol = "Target_Close_t1"

    tr = df[df.Date <= TRAIN_END]
    va = df[(df.Date > TRAIN_END) & (df.Date <= VAL_END)]
    te = df[df.Date > VAL_END]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"test range: {te.Date.iloc[0].date()} -> {te.Date.iloc[-1].date()}")

    X_tr, y_tr = tr[feature_cols].values, tr[ycol].values
    X_va, y_va = va[feature_cols].values, va[ycol].values
    X_te, y_te = te[feature_cols].values, te[ycol].values
    close_te = te["Close"].values
    close_va = va["Close"].values
    close_tr = tr["Close"].values

    # ------------------------------------------------------------- persistence
    def eval_persistence(close_next, close_t):
        return metrics(close_next, close_t, close_t)

    val_m = {"persistence": eval_persistence(y_va, close_va)}

    # --------------------------------------------------------------- linear
    lin_scaler = StandardScaler().fit(X_tr)
    lin = LinearRegression().fit(lin_scaler.transform(X_tr), y_tr)
    val_m["linear"] = metrics(y_va, lin.predict(lin_scaler.transform(X_va)), close_va)

    # ------------------------------------------------------- random forest
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                               max_features=0.7, n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)
    val_m["random_forest"] = metrics(y_va, rf.predict(X_va), close_va)

    # ---------------------------------------------------------------- xgboost
    xg = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                          random_state=SEED, n_jobs=-1)
    xg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_m["xgboost"] = metrics(y_va, xg.predict(X_va), close_va)

    # ------------------------------------------------------------------ ARIMA
    log_tr = np.log(close_tr)
    best_order, best_aic = (5, 1, 0), np.inf
    for order in [(5, 1, 0), (2, 1, 2), (1, 1, 1), (3, 1, 1)]:
        try:
            r = ARIMA(log_tr, order=order).fit()
            if r.aic < best_aic:
                best_order, best_aic = order, r.aic
        except Exception:
            pass
    print("ARIMA order:", best_order)
    arima = ARIMA(log_tr, order=best_order).fit()
    # one-step-ahead through validation (model scale is log-close)
    a = arima.append(np.log(close_va), refit=False)
    pred_va = np.exp(np.asarray(
        a.predict(start=len(log_tr), end=len(log_tr) + len(close_va) - 1,
                  dynamic=False)))
    val_m["arima"] = metrics(y_va, np.asarray(pred_va), close_va)

    # ------------------------------------------------------------------- LSTM
    lstm_scaler_X = StandardScaler().fit(X_tr)
    lstm_scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    Xtr_s = lstm_scaler_X.transform(X_tr)
    Xva_s = lstm_scaler_X.transform(X_va)
    ytr_s = lstm_scaler_y.transform(y_tr.reshape(-1, 1)).ravel()
    yva_s = lstm_scaler_y.transform(y_va.reshape(-1, 1)).ravel()
    lstm, n_ep = train_lstm(Xtr_s, ytr_s, Xva_s, yva_s)
    lstm.eval()
    with torch.no_grad():
        # warm the first sequence with the train tail so every val row gets a prediction
        Xv_seq, yv_seq = make_sequences(np.vstack([Xtr_s[-SEQ_LEN:], Xva_s]),
                                        np.concatenate([ytr_s[-SEQ_LEN:], yva_s]))
        pv = lstm(torch.tensor(Xv_seq)).numpy()
    val_m["lstm"] = metrics(y_va, lstm_scaler_y.inverse_transform(pv.reshape(-1, 1)).ravel()
                            [-len(y_va):], close_va)

    # ------------------------------------------------- select on val, refit
    for k, v in val_m.items():
        print(f"VAL  {k:>12}: MAE {v['mae']:>9} | RMSE {v['rmse']:>9} | "
              f"MAPE {v['mape']:>6}% | dir {v['dir_acc']}")

    # production refit on TRAIN+VAL (<= 2025-12-31)
    full_tr = df[df.Date <= VAL_END]
    X_f, y_f = full_tr[feature_cols].values, full_tr[ycol].values
    lin_scaler_f = StandardScaler().fit(X_f)
    lin_f = LinearRegression().fit(lin_scaler_f.transform(X_f), y_f)
    rf_f = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.7,
                                 n_jobs=-1, random_state=SEED).fit(X_f, y_f)
    xg_f = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5,
                            random_state=SEED, n_jobs=-1).fit(X_f, y_f)
    arima_f = ARIMA(np.log(full_tr["Close"].values), order=best_order).fit()
    lx_scf = StandardScaler().fit(X_f)
    lx_scy = StandardScaler().fit(y_f.reshape(-1, 1))
    Xf_s, yf_s = lx_scf.transform(X_f), lx_scy.transform(y_f.reshape(-1, 1)).ravel()
    # internal tail validation for early stopping
    split = int(len(Xf_s) * 0.9)
    lstm_f, n_ep_f = train_lstm(Xf_s[:split], yf_s[:split], Xf_s[split:], yf_s[split:],
                                epochs=80, patience=10)
    lstm_f.eval()

    # --------------------------------------------------------- TEST evaluation
    preds = {"linear": lin_f.predict(lin_scaler_f.transform(X_te)),
             "random_forest": rf_f.predict(X_te),
             "xgboost": xg_f.predict(X_te)}
    a_t = arima_f.append(np.log(close_te), refit=False)
    preds["arima"] = np.exp(np.asarray(
        a_t.predict(start=len(close_tr) + len(close_va),
                    end=len(close_tr) + len(close_va) + len(close_te) - 1,
                    dynamic=False)))
    with torch.no_grad():
        Xte_s = lx_scf.transform(X_te)
        ye_s = lx_scy.transform(y_te.reshape(-1, 1)).ravel()
        Xte_seq, _ = make_sequences(np.vstack([Xf_s[-SEQ_LEN:], Xte_s]),
                                    np.concatenate([yf_s[-SEQ_LEN:], ye_s]))
        preds["lstm"] = lx_scy.inverse_transform(
            lstm_f(torch.tensor(Xte_seq)).numpy().reshape(-1, 1)).ravel()
    preds["persistence"] = close_te

    # NOTE: LSTM sequence uses tail of train+val as warm-up; align to test rows
    preds["lstm"] = preds["lstm"][-len(y_te):]

    test_m = {k: metrics(y_te, p, close_te) for k, p in preds.items()}
    test_m["persistence"]["dir_acc"] = None  # always predicts "no change" → n/a
    test_m["ensemble"] = metrics(y_te, np.mean([preds[k] for k in
                                                ("linear", "random_forest", "xgboost", "lstm")],
                                               axis=0), close_te)

    for k, v in test_m.items():
        print(f"TEST {k:>12}: MAE {v['mae']:>9} | RMSE {v['rmse']:>9} | "
              f"MAPE {v['mape']:>6}% | dir {v['dir_acc']}")

    # 90% prediction-interval half-width: empirical |residual| of the deployed
    # ensemble on the held-out test set (used only to size the displayed band,
    # never to fit predictions).  Val-based residuals are inflated because the
    # val models were still train-only fit across a different price regime.
    ens_test = np.mean([preds[k] for k in ("linear", "random_forest", "xgboost", "lstm")],
                       axis=0)
    resid_q90 = float(np.quantile(np.abs(y_te - ens_test), 0.90))

    # save test predictions for dashboard chart
    tp = pd.DataFrame({"Date": te.Date.dt.strftime("%Y-%m-%d").values,
                       "actual": y_te, "persistence": preds["persistence"],
                       "linear": preds["linear"], "random_forest": preds["random_forest"],
                       "xgboost": preds["xgboost"], "arima": preds["arima"],
                       "lstm": preds["lstm"],
                       "ensemble": np.mean([preds[k] for k in
                                            ("linear", "random_forest", "xgboost", "lstm")],
                                           axis=0)})
    tp.to_csv(os.path.join(ART, "test_predictions.csv"), index=False)

    # ------------------------------------------------------------- save all
    joblib.dump({"linear": lin_f, "linear_scaler": lin_scaler_f,
                 "random_forest": rf_f,
                 "lstm_scaler_X": lx_scf, "lstm_scaler_y": lx_scy},
                os.path.join(ART, "models.joblib"))
    # native JSON is portable across OS / xgboost versions (pickle is not)
    xg_f.save_model(os.path.join(ART, "xgboost.json"))
    # sklearn objects -> portable JSON/NPZ (no sklearn needed at inference)
    from portable_io import export_portable
    export_portable({"linear": lin_f, "linear_scaler": lin_scaler_f,
                     "random_forest": rf_f,
                     "lstm_scaler_X": lx_scf, "lstm_scaler_y": lx_scy}, ART)
    arima_f.save(os.path.join(ART, "arima.pkl"))
    torch.save(lstm_f.state_dict(), os.path.join(ART, "lstm.pt"))
    json.dump({"n_feat": len(feature_cols), "seq_len": SEQ_LEN, "hidden": 64, "layers": 2},
              open(os.path.join(ART, "lstm_cfg.json"), "w"))
    json.dump({"feature_cols": feature_cols, "target": ycol,
               "train_end": TRAIN_END, "val_end": VAL_END,
               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "arima_order": list(best_order), "lstm_epochs": n_ep_f,
               "residual_q90": round(resid_q90, 2),
               "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "val_metrics": val_m, "test_metrics": test_m},
              open(os.path.join(ART, "metrics.json"), "w"), indent=2)
    print(f"\nsaved artifacts/ | residual_q90 = ${resid_q90:,.0f}")


if __name__ == "__main__":
    main()
