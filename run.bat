@echo off
rem ============================================================
rem  CryptoForecast AI - Windows one-command implementation
rem    run.bat          -> setup + train if needed + serve
rem    run.bat train    -> retrain models, then exit
rem    run.bat serve    -> just start the dashboard
rem ============================================================
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found.
  echo Install it from https://www.python.org/downloads/ and tick
  echo "Add python.exe to PATH" in the installer, then re-run this.
  pause
  exit /b 1
)

rem ---------- 1. virtualenv + dependencies ----------
if not exist .venv (
  echo [1/4] creating virtualenv...
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -c "import flask, xgboost, statsmodels, torch, sklearn, pyarrow" >nul 2>&1
if errorlevel 1 (
  echo [2/4] installing dependencies first run only, about 3-5 min...
  pip install -q -r requirements.txt
  if errorlevel 1 pause ^& exit /b 1
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 pause ^& exit /b 1
) else (
  echo [2/4] dependencies OK
)

rem ---------- 2. dataset (CSVs ship with the project) ----------
if not exist BTCUSDT-1d-features.csv (
  echo [3/4] building dataset...
  if not exist crypto-dataset (
    git clone --filter=blob:none --sparse https://github.com/Speirsy11/crypto-dataset.git
    git -C crypto-dataset sparse-checkout set data/interval_id=1d/symbol_id=BTCUSDT metadata
  )
  python prepare_features.py
) else (
  echo [3/4] dataset OK
)

rem ---------- 3. train or skip ----------
if "%1"=="train" goto :train
if not exist artifacts\metrics.json goto :train
echo [4/4] trained models OK use "run.bat train" to retrain
goto :serve

:train
echo [4/4] training models about 2 min...
python train_models.py
if errorlevel 1 pause ^& exit /b 1
if "%1"=="train" exit /b 0

:serve
echo.
echo  Dashboard: http://localhost:5000   press Ctrl+C to stop
echo.
python app.py
