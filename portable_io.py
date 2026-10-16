"""
Version/OS-portable model serialization + inference.

sklearn pickles break across versions and platforms ("incompatible dtype",
"input stream corrupted", ...).  This module saves the fitted LinearRegression,
RandomForest and StandardScaler state as plain JSON/NPZ arrays and provides
drop-in objects with the same .predict/.transform/.inverse_transform interface,
so the realtime engine never depends on a specific sklearn build.
"""
import json
import os

import numpy as np


def export_portable(bundle: dict, out_dir: str = "artifacts"):
    """Convert the joblib bundle's sklearn objects -> portable files."""
    lin, sc = bundle["linear"], bundle["linear_scaler"]
    lx, ly = bundle["lstm_scaler_X"], bundle["lstm_scaler_y"]
    json.dump({
        "coef": np.asarray(lin.coef_, dtype=float).tolist(),
        "intercept": float(lin.intercept_),
        "linear_mean": np.asarray(sc.mean_, dtype=float).tolist(),
        "linear_scale": np.asarray(sc.scale_, dtype=float).tolist(),
        "lstm_x_mean": np.asarray(lx.mean_, dtype=float).tolist(),
        "lstm_x_scale": np.asarray(lx.scale_, dtype=float).tolist(),
        "lstm_y_mean": np.asarray(ly.mean_, dtype=float).tolist(),
        "lstm_y_scale": np.asarray(ly.scale_, dtype=float).tolist(),
    }, open(os.path.join(out_dir, "linear_scalers.json"), "w"))

    rf = bundle["random_forest"]
    arr = {"n_trees": np.array(len(rf.estimators_), np.int32)}
    for i, est in enumerate(rf.estimators_):
        tr = est.tree_
        arr[f"{i}_left"] = tr.children_left.astype(np.int32)
        arr[f"{i}_right"] = tr.children_right.astype(np.int32)
        arr[f"{i}_feat"] = tr.feature.astype(np.int32)
        arr[f"{i}_thr"] = np.asarray(tr.threshold, dtype=np.float64)
        arr[f"{i}_val"] = np.asarray(tr.value, dtype=np.float64).reshape(
            tr.value.shape[0], -1)[:, 0]
    np.savez_compressed(os.path.join(out_dir, "rf_trees.npz"), **arr)


class PortableScaler:
    """Same interface as sklearn StandardScaler for .transform/.inverse_transform."""
    def __init__(self, mean, scale):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, y):
        y = np.asarray(y, dtype=np.float64)
        return y * self.scale_ + self.mean_


class PortableLinear:
    """LinearRegression.predict on already-scaled features (matches sklearn)."""
    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)

    def predict(self, X):
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


class PortableRF:
    """RandomForestRegressor.predict — pure-numpy traversal of exported trees."""
    def __init__(self, npz):
        self._n = int(npz["n_trees"])
        self._t = npz

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(len(X))
        for i in range(self._n):
            left, right = self._t[f"{i}_left"], self._t[f"{i}_right"]
            feat, thr, val = self._t[f"{i}_feat"], self._t[f"{i}_thr"], self._t[f"{i}_val"]
            pred = np.empty(len(X))
            for s in range(len(X)):
                node = 0
                while left[node] != -1:
                    node = left[node] if X[s, feat[node]] <= thr[node] else right[node]
                pred[s] = val[node]
            out += pred
        return out / self._n
