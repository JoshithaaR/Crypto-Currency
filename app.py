"""
Realtime AI cryptocurrency prediction system — web dashboard + JSON API.

Endpoints
  GET /                dashboard (inline CSS/JS, no external resources)
  GET /api/predict     live data + next-day forecast + risk panel + track record
  GET /api/metrics     held-out test/val metrics for all models
  GET /api/chart       2026 actual closes vs ensemble backtest predictions
  GET /api/log         prediction track record (graded past calls)
  GET /health          liveness

Run:  python3 app.py     (binds 0.0.0.0:5000)
"""
import json
import os
import threading
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, Response, jsonify, request

import engine
from market_data import get_market

app = Flask(__name__)

ART = "artifacts"
LOG_PATH = os.path.join(ART, "prediction_log.json")
_lock = threading.Lock()
_refresh_hint = 45  # seconds


# --------------------------------------------------------------- prediction log
def _load_log() -> list:
    if os.path.exists(LOG_PATH):
        return json.load(open(LOG_PATH))
    return []


def _save_log(entries: list):
    with open(LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def _grade(entries: list, closes: dict):
    """Attach actuals to past forecasts whose target date has completed."""
    for e in entries:
        if e.get("graded"):
            continue
        actual = closes.get(e["target_date"])
        if actual is None:
            continue
        basis = e["basis_close"]
        e["graded"] = True
        e["actual"] = round(actual, 2)
        e["abs_err"] = round(abs(actual - e["ensemble"]), 2)
        e["pct_err"] = round(abs(actual - e["ensemble"]) / actual * 100, 3)
        e["dir_correct"] = bool((e["ensemble"] - basis > 0) == (actual - basis > 0))


def _update_log(pred: dict, market: dict):
    entries = _load_log()
    closes = dict(zip(market["completed"]["Date"], market["completed"]["Close"]))
    _grade(entries, closes)

    td = pred["target_date"]
    rec = next((e for e in entries if e["target_date"] == td), None)
    if rec is None:
        rec = {"target_date": td, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        entries.append(rec)
    rec.update({
        "basis_date": pred["basis_date"],
        "basis_close": pred["last_close"],
        "ensemble": pred["ensemble"],
        "band90": pred["band90"],
        "expected_move_pct": pred["expected_move_pct"],
        "direction": pred["direction"],
        "agreement": pred["agreement"],
        "predictions": pred["predictions"],
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": market["source"],
        "live_at_call": round(market["live_price"], 2),
        "graded": rec.get("graded", False),
    })
    entries.sort(key=lambda e: e["target_date"], reverse=True)
    _save_log(entries[:60])
    return entries[:12]


# --------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/predict")
def api_predict():
    try:
        with _lock:
            m = get_market()
            pred = engine.predict(m["completed"])
            log = _update_log(pred, m)
        return jsonify({
            "ok": True,
            "source": m["source"],
            "fetched_at": m["fetched_at"],
            "in_progress": m["in_progress"],
            "live_price": round(m["live_price"], 2),
            "last_completed_date": m["last_completed_date"],
            "prediction": pred,
            "log": log,
            "model_trained_at": engine.load_models()["meta"]["trained_at"],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500


@app.get("/api/metrics")
def api_metrics():
    meta = engine.load_models()["meta"]
    return jsonify({"val": meta["val_metrics"], "test": meta["test_metrics"],
                    "n_train": meta["n_train"], "n_val": meta["n_val"],
                    "n_test": meta["n_test"], "trained_at": meta["trained_at"],
                    "arima_order": meta["arima_order"], "lstm_epochs": meta["lstm_epochs"]})


@app.get("/api/chart")
def api_chart():
    start = request.args.get("start", "2026-01-01")
    actual = pd.read_csv("BTCUSDT-1d.csv", parse_dates=["Date"])
    try:
        live = get_market()
        lv = live["all"][["Date", "Close"]].copy()
        lv["Date"] = pd.to_datetime(lv["Date"])
        actual = pd.concat([actual, lv]).drop_duplicates("Date", keep="last")
    except Exception:
        pass
    actual["Date"] = pd.to_datetime(actual["Date"])
    actual = actual[actual["Date"] >= start].sort_values("Date")

    pred = pd.read_csv(os.path.join(ART, "test_predictions.csv"), parse_dates=["Date"])
    pred = pred[pred["Date"] >= start]

    return jsonify({
        "actual": [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 2)}
                   for d, v in zip(actual.Date, actual.Close)],
        "pred": [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 2)}
                 for d, v in zip(pred.Date, pred.ensemble)],
        "persistence": [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 2)}
                        for d, v in zip(pred.Date, pred.persistence)],
    })


@app.get("/api/log")
def api_log():
    return jsonify(_load_log()[:20])


@app.get("/")
def index():
    return Response(PAGE, mimetype="text/html")


# ------------------------------------------------------------------ dashboard
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CryptoForecast AI — BTCUSDT Realtime Predictor</title>
<style>
:root{--bg:#0b0f1a;--card:#121a2b;--card2:#0f1524;--line:#1e2a44;--txt:#e8eefc;--dim:#8fa3c8;
--up:#22d39a;--down:#ff5c7a;--acc:#5b8cff;--acc2:#a78bfa;--warn:#ffb020;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:20px}
a{color:var(--acc)}
.wrap{max-width:1180px;margin:0 auto}
header{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:18px}
.logo{font-size:20px;font-weight:800;letter-spacing:.3px}
.logo span{background:linear-gradient(90deg,var(--acc),var(--acc2));-webkit-background-clip:text;background-clip:text;color:transparent}
.sub{color:var(--dim);font-size:12px}
.badge{font-size:11px;padding:3px 10px;border-radius:99px;border:1px solid var(--line);color:var(--dim);background:var(--card2)}
.badge.live{color:var(--up);border-color:#14532d}
.badge.stale{color:var(--warn);border-color:#5b4413}
.spacer{flex:1}
.clock{color:var(--dim);font-variant-numeric:tabular-nums;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.card .lab{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.8px}
.card .val{font-size:26px;font-weight:800;margin-top:4px;font-variant-numeric:tabular-nums}
.card .sub{margin-top:2px}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700}
.chip.up{background:#0d3b2e;color:var(--up)} .chip.down{background:#461722;color:var(--down)}
.chip.flat{background:#333;color:#ddd}
.up{color:var(--up)} .down{color:var(--down)}
.row{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
.pill{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:8px 12px;min-width:120px}
.pill .k{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.6px}
.pill .v{font-weight:700;font-size:15px;margin-top:1px}
h2{font-size:14px;margin:22px 0 10px;color:#c6d4f2;letter-spacing:.4px}
h2 small{color:var(--dim);font-weight:400;font-size:11.5px;margin-left:8px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
th,td{padding:9px 12px;text-align:right;font-variant-numeric:tabular-nums;font-size:13px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.6px;background:var(--card2)}
th:first-child,td:first-child{text-align:left}
tr:last-child td{border-bottom:none}
tr.hl td{background:#10203a;font-weight:700}
td.best{color:var(--up)}
.canvas-card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
#chart{width:100%;height:320px;display:block}
.legend{display:flex;gap:16px;color:var(--dim);font-size:12px;margin-top:6px}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.err{background:#451a1a;border:1px solid #7f1d1d;color:#ffb4b4;padding:10px 14px;border-radius:10px;margin:10px 0;display:none}
.foot{color:var(--dim);font-size:11.5px;margin-top:26px;padding-top:14px;border-top:1px solid var(--line);line-height:1.7}
.hist{color:var(--dim)}
.verdict{font-weight:700;font-size:12px}
.verdict.hit{color:var(--up)} .verdict.miss{color:var(--down)} .verdict.open{color:var(--warn)}
#countdown{color:var(--acc)}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
</style></head><body><div class="wrap">

<header>
  <div class="logo">📈 CryptoForecast <span>AI</span></div>
  <div class="sub">BTCUSDT · next-day close · Binance OHLCV + XGBoost/LSTM ensemble</div>
  <span class="badge" id="src">connecting…</span>
  <span class="badge" id="trained">model: –</span>
  <div class="spacer"></div>
  <div class="clock">UTC <span id="utc">–</span> · refresh in <span id="countdown">–</span>s</div>
</header>

<div class="err" id="err"></div>

<div class="grid">
  <div class="card"><div class="lab">Live price</div>
    <div class="val" id="live">–</div>
    <div class="sub hist" id="liveSub">–</div></div>
  <div class="card"><div class="lab">AI forecast close · <span id="tdate">–</span></div>
    <div class="val" id="ens">–</div>
    <div class="sub hist" id="band">90% band: –</div></div>
  <div class="card"><div class="lab">Expected move</div>
    <div class="val" id="move">–</div>
    <div class="sub"><span class="chip flat" id="dir">–</span>
      <span class="hist" id="agree"></span></div></div>
  <div class="card"><div class="lab">Backtest 2026 (held-out)</div>
    <div class="val" id="btMape">–</div>
    <div class="sub hist" id="btSub">ensemble MAPE · directional accuracy –</div></div>
</div>

<div class="row" id="risk">
  <div class="pill"><div class="k">RSI(14)</div><div class="v" id="rsi">–</div></div>
  <div class="pill"><div class="k">Volatility (ann.)</div><div class="v" id="vol">–</div></div>
  <div class="pill"><div class="k">Trend vs SMA30</div><div class="v" id="trend">–</div></div>
  <div class="pill"><div class="k">MACD hist</div><div class="v" id="macd">–</div></div>
  <div class="pill"><div class="k">ATR(14)</div><div class="v" id="atr">–</div></div>
  <div class="pill"><div class="k">Suggested long stop</div><div class="v" id="stop">–</div></div>
  <div class="pill"><div class="k">Bollinger position</div><div class="v" id="bp">–</div></div>
</div>

<h2>2026 out-of-sample backtest <small>models trained only on data ≤ 2025-12-31</small></h2>
<div class="canvas-card">
  <canvas id="chart"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#5b8cff"></span>Actual close</span>
    <span><span class="dot" style="background:#22d39a"></span>Ensemble prediction</span>
    <span><span class="dot" style="background:#8fa3c8"></span>Persistence baseline (t → t+1)</span>
  </div>
</div>

<h2>Model comparison — held-out 2026 test set <small>lower error better · dir = directional accuracy</small></h2>
<table id="mt"><thead><tr><th>Model</th><th>MAE ($)</th><th>RMSE ($)</th><th>MAPE (%)</th><th>Dir acc (%)</th></tr></thead>
<tbody></tbody></table>

<h2>Prediction track record <small>every forecast this system has issued, graded against the realized close</small></h2>
<table id="lt"><thead><tr><th>Target day</th><th>Forecast</th><th>Actual</th><th>Error %</th><th>Direction</th><th>Verdict</th></tr></thead>
<tbody></tbody></table>

<div class="foot">
  <b>How it works:</b> Binance daily OHLCV → 20+ technical indicators (SMA/EMA/RSI/MACD/Bollinger/volatility/lags)
  → Linear Regression · ARIMA · Random Forest · XGBoost · LSTM ensemble → next-day close with a 90% band from validation residuals.
  Data refreshed from Binance Vision; forecasts logged and graded automatically.
  <br><b>Disclaimer:</b> educational project — not financial advice. Crypto is volatile; models can and will be wrong.
</div>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=n=>n==null?'–':'$'+Number(n).toLocaleString('en-US',{maximumFractionDigits:0});
const fmt2=n=>n==null?'–':'$'+Number(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
let secs=45, chartData=null;

function clock(){$('utc').textContent=new Date().toISOString().slice(11,19);}

async function loadPredict(){
  const r=await fetch('/api/predict'); const j=await r.json();
  if(!j.ok){showErr(j.error);return;}
  showErr(null);
  $('src').textContent=j.source;
  $('src').className='badge '+(j.source.includes('live')?'live':'stale');
  $('trained').textContent='model: '+j.model_trained_at.slice(0,16).replace('T',' ')+'Z';
  $('live').textContent=fmt2(j.live_price);
  $('liveSub').textContent=(j.in_progress?'live (day forming)':'last trade')+' · completed thru '+j.last_completed_date;
  const p=j.prediction;
  $('tdate').textContent=p.target_date;
  $('ens').textContent=fmt2(p.ensemble);
  $('band').textContent='90% band: '+fmt(p.band90.low)+' – '+fmt(p.band90.high)+' (±'+fmt(p.band90.half)+')';
  const m=p.expected_move_pct;
  $('move').textContent=(m>0?'+':'')+m+'%';
  $('move').className='val '+(m>0?'up':m<0?'down':'');
  const d=$('dir'); d.textContent=p.direction; d.className='chip '+(p.direction==='UP'?'up':p.direction==='DOWN'?'down':'flat');
  $('agree').textContent=p.agreement.up+'/'+p.agreement.total+' models agree';
  const r2=p.risk;
  $('rsi').textContent=r2.rsi+(r2.rsi>70?' overbought':r2.rsi<30?' oversold':'');
  $('vol').textContent=r2.vol_ann_pct+'%';
  $('trend').textContent=(r2.price_vs_sma30_pct>0?'+':'')+r2.price_vs_sma30_pct+'%';
  $('trend').className='v '+(r2.price_vs_sma30_pct>0?'up':'down');
  $('macd').textContent=r2.macd_hist>0?'+'+r2.macd_hist:r2.macd_hist;
  $('macd').className='v '+(r2.macd_hist>0?'up':'down');
  $('atr').textContent=fmt2(r2.atr14);
  $('stop').textContent=fmt2(r2.stop_long);
  $('bp').textContent=r2.bollinger_pos+'%';
  renderLog(j.log);
  return j;
}

async function loadMetrics(){
  const j=await (await fetch('/api/metrics')).json();
  const t=j.test;
  const names={persistence:'Persistence (naive)',linear:'Linear Regression',arima:'ARIMA '+j.arima_order.join(','),
    random_forest:'Random Forest',xgboost:'XGBoost',lstm:'LSTM (PyTorch)',ensemble:'★ Ensemble (avg of 4)'};
  const order=['persistence','linear','arima','random_forest','xgboost','lstm','ensemble'];
  const bestMape=Math.min(...order.filter(k=>t[k]).map(k=>t[k].mape));
  let rows='';
  for(const k of order){
    const m=t[k]; if(!m)continue;
    rows+=`<tr class="${k==='ensemble'?'hl':''}"><td>${names[k]}</td>
      <td>${m.mae.toLocaleString()}</td><td>${m.rmse.toLocaleString()}</td>
      <td class="${m.mape===bestMape?'best':''}">${m.mape}</td>
      <td>${m.dir_acc==null?'n/a':m.dir_acc}</td></tr>`;
  }
  document.querySelector('#mt tbody').innerHTML=rows;
  const e=t.ensemble;
  $('btMape').textContent=e.mape+'%';
  $('btSub').textContent='ensemble MAPE · dir acc '+(e.dir_acc==null?'–':e.dir_acc+'%')+' · MAE $'+e.mae.toLocaleString();
}

async function loadChart(){
  chartData=await (await fetch('/api/chart')).json();
  drawChart();
}

function drawChart(){
  if(!chartData)return;
  const c=$('chart'), dpr=window.devicePixelRatio||1;
  const W=c.clientWidth, H=320;
  c.width=W*dpr; c.height=H*dpr; c.style.height=H+'px';
  const g=c.getContext('2d'); g.scale(dpr,dpr); g.clearRect(0,0,W,H);
  const A=chartData.actual, P=chartData.pred, B=chartData.persistence||[];
  if(!A.length)return;
  const all=[...A,...P].map(x=>x.v);
  let min=Math.min(...all), max=Math.max(...all);
  const pad=(max-min)*0.06||1; min-=pad; max+=pad;
  const L=64,R=14,T=14,Bt=26;
  const X=i=>L+(W-L-R)*(i/(A.length-1||1));
  const mapD=new Map(A.map((x,i)=>[x.i,i]));
  const Y=v=>T+(H-T-Bt)*(1-(v-min)/(max-min));
  // grid + y labels
  g.strokeStyle='#1e2a44'; g.fillStyle='#8fa3c8'; g.font='11px sans-serif'; g.textAlign='right';
  for(let k=0;k<=4;k++){
    const v=min+(max-min)*k/4, y=Y(v);
    g.beginPath(); g.moveTo(L,y); g.lineTo(W-R,y); g.stroke();
    g.fillText('$'+Math.round(v).toLocaleString(), L-8, y+4);
  }
  // x labels
  g.textAlign='center';
  const step=Math.ceil(A.length/7);
  for(let i=0;i<A.length;i+=step){
    g.fillText(A[i].d.slice(5), X(i), H-8);
  }
  const idx=new Map(A.map((x,i)=>[x.d,i]));
  const drawSeries=(arr,color,width,dash)=>{
    g.strokeStyle=color; g.lineWidth=width; g.setLineDash(dash||[]);
    g.beginPath(); let started=false;
    for(const p of arr){
      const i=idx.get(p.d); if(i==null)continue;
      const x=X(i), y=Y(p.v);
      started?g.lineTo(x,y):g.moveTo(x,y); started=true;
    }
    g.stroke(); g.setLineDash([]);
  };
  drawSeries(B,'#8fa3c8',1.4,[5,4]);
  drawSeries(P,'#22d39a',2);
  drawSeries(A,'#5b8cff',2);
  // live dot
  const last=A[A.length-1];
  g.fillStyle='#5b8cff'; g.beginPath(); g.arc(X(A.length-1),Y(last.v),4,0,7); g.fill();
}

function renderLog(log){
  let rows='';
  for(const e of log){
    let verdict,cls;
    if(e.graded){const hit=e.dir_correct;verdict=hit?'HIT':'MISS';cls=hit?'hit':'miss';}
    else {verdict='OPEN';cls='open';}
    rows+=`<tr><td>${e.target_date}</td><td>${fmt2(e.ensemble)}</td>
      <td>${e.graded?fmt2(e.actual):'<span class="hist">pending</span>'}</td>
      <td>${e.graded?e.pct_err+'%':'–'}</td>
      <td>${e.graded?(e.dir_correct?'<span class="up">✓ correct</span>':'<span class="down">✗ wrong</span>'):e.direction}</td>
      <td><span class="verdict ${cls}">${verdict}</span></td></tr>`;
  }
  document.querySelector('#lt tbody').innerHTML=rows||'<tr><td colspan="6" class="hist">No forecasts logged yet.</td></tr>';
}

function showErr(msg){
  const e=$('err');
  if(!msg){e.style.display='none';return;}
  e.style.display='block'; e.textContent='⚠ '+msg;
}

async function tick(){
  clock();
  secs--;
  $('countdown').textContent=secs;
  if(secs<=0){secs=45; loadPredict().catch(()=>{});}
}
async function boot(){
  clock(); setInterval(tick,1000);
  await Promise.all([loadPredict().catch(e=>showErr(String(e))),
                     loadMetrics().catch(()=>{}),
                     loadChart().catch(()=>{})]);
  setInterval(()=>{loadPredict().catch(()=>{});},45000);
  setInterval(()=>{loadChart().catch(()=>{});},300000);
}
window.addEventListener('resize',drawChart);
boot();
</script>
</body></html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
