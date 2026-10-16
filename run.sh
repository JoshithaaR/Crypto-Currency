#!/usr/bin/env bash
# CryptoForecast AI — one-command implementation
#   ./run.sh          -> setup deps (if needed) + train (if needed) + serve dashboard
#   ./run.sh train    -> force retrain with latest data, then exit
#   ./run.sh serve    -> just start the dashboard
set -euo pipefail
cd "$(dirname "$0")"
CMD="${1:-all}"

# ---------- 1. virtualenv + dependencies (only if missing) ----------
if [ ! -d .venv ]; then
  echo "[1/4] creating virtualenv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python3 -c "import flask, xgboost, statsmodels, torch, sklearn, pyarrow" 2>/dev/null; then
  echo "[2/4] installing dependencies (first run only, ~2 min)..."
  pip install -q -r requirements.txt
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
else
  echo "[2/4] dependencies OK"
fi

# ---------- 2. dataset (uses shipped CSVs, or rebuilds from GitHub) ----------
if [ ! -f BTCUSDT-1d.csv ] || [ ! -f BTCUSDT-1d-features.csv ]; then
  echo "[3/4] building dataset..."
  if [ ! -d crypto-dataset ]; then
    git clone --filter=blob:none --sparse https://github.com/Speirsy11/crypto-dataset.git
    git -C crypto-dataset sparse-checkout set data/interval_id=1d/symbol_id=BTCUSDT metadata
  fi
  python3 prepare_features.py
else
  echo "[3/4] dataset OK ($(wc -l < BTCUSDT-1d.csv) rows)"
fi

# ---------- 3. train (skip if artifacts exist, unless forced) ----------
if [ "$CMD" = "train" ] || [ ! -f artifacts/metrics.json ]; then
  echo "[4/4] training models (~2 min)..."
  python3 train_models.py
  [ "$CMD" = "train" ] && exit 0
else
  echo "[4/4] artifacts OK (use './run.sh train' to retrain)"
fi

# ---------- 4. serve ----------
echo "starting dashboard on http://0.0.0.0:5000 ..."
python3 app.py
