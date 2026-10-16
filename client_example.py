"""
Minimal client for the CryptoForecast realtime API.

    python3 client_example.py

Shows the current live price, the AI next-day forecast, risk panel, and the
graded track record — straight from a running instance of app.py.
"""
import json
import urllib.request

BASE = "http://localhost:5000"  # change to your hosted URL


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    p = get("/api/predict")
    if not p["ok"]:
        raise SystemExit("API error: " + p.get("error", "?"))

    pred = p["prediction"]
    print(f"data source     : {p['source']}")
    print(f"live price      : ${p['live_price']:,.2f}")
    print(f"basis candle    : {pred['basis_date']} (close ${pred['last_close']:,.2f})")
    print(f"Ai forecast for : {pred['target_date']}")
    print(f"  ensemble      : ${pred['ensemble']:,.2f}  ({pred['expected_move_pct']:+.2f}% {pred['direction']})")
    print(f"  90% band      : ${pred['band90']['low']:,.0f} – ${pred['band90']['high']:,.0f}")
    print(f"  agreement     : {pred['agreement']['up']}/{pred['agreement']['total']} models up")
    for name, val in pred["predictions"].items():
        print(f"    {name:<14}: ${val:,.2f}")
    r = pred["risk"]
    print(f"risk            : RSI {r['rsi']} | vol {r['vol_ann_pct']}%/yr | "
          f"ATR ${r['atr14']:,.0f} | suggested long stop ${r['stop_long']:,.0f}")

    m = get("/api/metrics")
    e = m["test"]["ensemble"]
    print(f"\nheld-out 2026   : ensemble MAPE {e['mape']}% | MAE ${e['mae']:,.0f} | "
          f"dir acc {e['dir_acc']}% (n={m['n_test']} days)")

    open_entries = [x for x in p["log"] if x.get("graded")]
    print(f"track record    : {len(open_entries)} graded, "
          f"latest: {open_entries[0]['target_date']} err {open_entries[0]['pct_err']}%"
          if open_entries else "track record    : awaiting first grading")


if __name__ == "__main__":
    main()
