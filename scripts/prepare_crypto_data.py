"""
Fetch years of 30m klines from Binance for crypto finetuning.
Saves CSV files consumable by finetune_csv/train_sequential.py.
"""
import json
import urllib.request
import pandas as pd
import time


def fetch_klines(symbol, interval="30m", start_time=None, end_time=None, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    if start_time:
        url += f"&startTime={int(start_time.timestamp() * 1000)}"
    if end_time:
        url += f"&endTime={int(end_time.timestamp() * 1000)}"
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode())


def fetch_all_klines(symbol, interval="30m", years=3):
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.DateOffset(years=years)
    all_rows = []
    current_start = start
    while current_start < end:
        try:
            raw = fetch_klines(symbol, interval, start_time=current_start, end_time=end)
            if not raw:
                break
            rows = [{
                "timestamps": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "amount": float(k[7]),
            } for k in raw]
            all_rows.extend(rows)
            last_ts = pd.Timestamp(raw[-1][0], unit="ms", tz="UTC")
            print(f"  {symbol}: fetched up to {last_ts} ({len(rows)} candles)")
            current_start = last_ts + pd.Timedelta(minutes=1)
            time.sleep(0.1)
        except Exception as e:
            print(f"  error: {e}, retrying in 5s...")
            time.sleep(5)
    df = pd.DataFrame(all_rows).drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)
    return df


SYMBOLS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
YEARS = 3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "finetune_csv", "data")

for sym in SYMBOLS:
    print(f"\nFetching {sym} {YEARS}yr 30m data...")
    df = fetch_all_klines(sym, "30m", YEARS)
    path = f"{OUTPUT_DIR}/{sym.lower()}_30m.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows → {path}")
    print(f"  Range: {df['timestamps'].min()} → {df['timestamps'].max()}")
