"""
Kronos SOL/USDT 30m prediction server with TradingView Lightweight Charts.
Run:  python server.py
Then open http://localhost:8080
"""
import json
import urllib.request
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor
import http.server
import socketserver


def fetch_binance_klines(symbol="SOLUSDT", interval="30m", limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    rows = []
    for k in data:
        rows.append({
            "time": int(k[0]) // 1000,
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
        })
    return rows


def run_prediction():
    print("Loading models...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print("Fetching SOL/USDT 30m data from Binance...")
    raw = fetch_binance_klines()
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    lookback, pred_len = 400, 24
    hist = df.iloc[-lookback:].reset_index(drop=True)
    x_df = hist[["open", "high", "low", "close", "volume"]].copy()
    x_df["amount"] = 0.0
    x_ts = hist["time"]
    y_ts = pd.Series(pd.date_range(
        start=hist["time"].iloc[-1] + pd.Timedelta(minutes=30),
        periods=pred_len, freq="30min", tz="UTC",
    ))

    print(f"Latest: {hist['time'].iloc[-1]} close={hist['close'].iloc[-1]:.2f}")
    pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                pred_len=pred_len, T=0.8, top_p=0.9, sample_count=1, verbose=True)

    show_last = 60
    hist_show = hist.iloc[-show_last:]

    hist_candles, hist_vol = [], []
    for _, r in hist_show.iterrows():
        t = int(r["time"].timestamp())
        hist_candles.append({"time": t, "open": float(r["open"]), "high": float(r["high"]),
                             "low": float(r["low"]), "close": float(r["close"])})
        hist_vol.append({"time": t, "value": float(r["volume"]),
                         "color": "#26a69a" if r["close"] >= r["open"] else "#ef5350"})

    pred_candles, pred_vol = [], []
    for t, (_, r) in zip(y_ts, pred_df.iterrows()):
        ts = int(t.timestamp())
        pred_candles.append({"time": ts, "open": float(r["open"]), "high": float(r["high"]),
                             "low": float(r["low"]), "close": float(r["close"])})
        pred_vol.append({"time": ts, "value": float(r["volume"]),
                         "color": "#42a5f5" if r["close"] >= r["open"] else "#ff7043"})

    sep_ts = int(hist_show["time"].iloc[-1].timestamp())
    return json.dumps({"hist": hist_candles, "pred": pred_candles,
                       "histVol": hist_vol, "predVol": pred_vol, "sep": sep_ts})


print("Running Kronos prediction...")
data_json = run_prediction()

HTML = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>SOL/USDT 30m — Kronos Forecast</title>
<style>
  body {{ margin: 0; background: #131722; color: #d1d4dc; font-family: -apple-system, sans-serif; }}
  #chart {{ width: 100vw; height: 100vh; }}
  .header {{ padding: 10px 20px; background: #1e222d; border-bottom: 1px solid #2a2e39; display: flex; align-items: center; gap: 12px; }}
  .header h1 {{ margin: 0; font-size: 16px; font-weight: 600; color: #fff; }}
  .badge {{ padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 500; }}
  .badge.hist {{ background: #26a69a33; color: #26a69a; border: 1px solid #26a69a; }}
  .badge.pred {{ background: #42a5f533; color: #42a5f5; border: 1px solid #42a5f5; }}
  .badge.sep {{ background: #ff704333; color: #ff7043; border: 1px solid #ff7043; }}
</style></head>
<body>
<div class="header">
  <h1>SOL/USDT — 30m</h1>
  <span class="badge hist">historical</span>
  <span class="badge pred">kronos forecast</span>
  <span class="badge sep">| prediction start</span>
</div>
<div id="chart"></div>
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
<script>
const DATA = {data_json};
const chart = LightweightCharts.createChart(document.getElementById("chart"), {{
  layout: {{ background: {{ color: "#131722" }}, textColor: "#d1d4dc" }},
  grid: {{ vertLines: {{ color: "#2a2e39" }}, horzLines: {{ color: "#2a2e39" }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  rightPriceScale: {{ borderColor: "#2a2e39", scaleMargins: {{ top: 0.05, bottom: 0.25 }} }},
  timeScale: {{ borderColor: "#2a2e39", timeVisible: true, secondsVisible: false }},
}});
const histSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {{
  upColor: "#26a69a", downColor: "#ef5350", borderDownColor: "#ef5350", borderUpColor: "#26a69a",
  wickDownColor: "#ef5350", wickUpColor: "#26a69a", priceFormat: {{ type: "price", precision: 2, minMove: 0.01 }},
}});
histSeries.setData(DATA.hist);
const predSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {{
  upColor: "#42a5f5", downColor: "#ff7043", borderDownColor: "#ff7043", borderUpColor: "#42a5f5",
  wickDownColor: "#ff7043", wickUpColor: "#42a5f5", priceFormat: {{ type: "price", precision: 2, minMove: 0.01 }},
}});
predSeries.setData(DATA.pred);
const volSeries = chart.addSeries(LightweightCharts.HistogramSeries, {{ priceFormat: {{ type: "volume" }}, priceScaleId: "volume" }});
volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.80, bottom: 0 }} }});
volSeries.setData([...DATA.histVol, ...DATA.predVol]);
const allCandles = [...DATA.hist, ...DATA.pred];
const allHighs = allCandles.map(c => c.high), allLows = allCandles.map(c => c.low);
const rng = Math.max(...allHighs) - Math.min(...allLows);
const sepSeries = chart.addSeries(LightweightCharts.LineSeries, {{
  color: "#ff7043", lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
  lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
}});
sepSeries.setData([
  {{ time: DATA.sep, value: Math.min(...allLows) - rng * 0.05 }},
  {{ time: DATA.sep, value: Math.max(...allHighs) + rng * 0.05 }},
]);
</script></body></html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        else:
            super().do_GET()


PORT = 8080
print(f"\nOpen http://localhost:{PORT}")
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    httpd.serve_forever()
