import json, os, datetime as dt
import numpy as np, pandas as pd, requests, yfinance as yf

PAIRS = {
    "EUR/USD": ("EURUSD=X", 0.0001, 4),
    "GBP/USD": ("GBPUSD=X", 0.0001, 4),
    "USD/JPY": ("USDJPY=X", 0.01, 2),
    "AUD/USD": ("AUDUSD=X", 0.0001, 4),
    "XAU/USD": ("GC=F", 0.1, 2),
}
TFS = {
    "M1": ("1m", "5d", None), "M5": ("5m", "30d", None), "M15": ("15m", "30d", None),
    "M30": ("30m", "30d", None), "H1": ("60m", "60d", None), "H4": ("60m", "60d", "4h"),
    "D1": ("1d", "1y", None), "W1": ("1wk", "5y", None), "MN": ("1mo", "10y", None),
}
BT_DAYS = {"M1": 5, "M5": 30, "M15": 30, "M30": 30, "H1": 30, "H4": 30, "D1": 90, "W1": 365, "MN": 1095}
cache = {}

def fetch(t, iv, per, rs):
    k = (t, iv, per, rs)
    if k not in cache:
        df = yf.Ticker(t).history(interval=iv, period=per)[["Open", "High", "Low", "Close"]].dropna()
        if rs:
            df = df.resample(rs).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
        cache[k] = df
    return cache[k]

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean(); rd = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))

def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def prep(df):
    df = df.copy()
    df["e9"] = ema(df["Close"], 9); df["e21"] = ema(df["Close"], 21)
    df["rsi"] = rsi(df["Close"]); df["atr"] = atr(df)
    m = ema(df["Close"], 12) - ema(df["Close"], 26)
    df["mh"] = m - ema(m, 9)
    return df

def backtest(df, days, pip):
    idx = df.index; cut = idx[-1] - pd.Timedelta(days=days)
    e9 = df["e9"].values; e21 = df["e21"].values; rs = df["rsi"].values; at = df["atr"].values
    hi = df["High"].values; lo = df["Low"].values; cl = df["Close"].values
    n = len(df); i = 22; c = w = 0; tot = slsum = eq = peak = dd = 0.0
    while i < n - 1:
        if idx[i] < cut or np.isnan(at[i]) or np.isnan(rs[i]):
            i += 1; continue
        cu = e9[i-1] <= e21[i-1] and e9[i] > e21[i] and rs[i] > 50
        cd = e9[i-1] >= e21[i-1] and e9[i] < e21[i] and rs[i] < 50
        if not (cu or cd):
            i += 1; continue
        d = 1 if cu else -1; entry = cl[i]; risk = 1.5 * at[i]
        sl = entry - d * risk; tp = entry + d * 1.5 * risk
        res = None; ex = min(n - 1, i + 59)
        for j in range(i + 1, ex + 1):
            if d == 1:
                if lo[j] <= sl: res = -1.0; ex = j; break
                if hi[j] >= tp: res = 1.5; ex = j; break
            else:
                if hi[j] >= sl: res = -1.0; ex = j; break
                if lo[j] <= tp: res = 1.5; ex = j; break
        if res is None: res = d * (cl[ex] - entry) / risk
        c += 1; w += res > 0; tot += res; slsum += risk
        eq += res; peak = max(peak, eq); dd = max(dd, peak - eq)
        i = ex + 1
    if c == 0: return {"c": 0, "w": 0, "r": 0, "slp": 1, "dd": 0}
    return {"c": c, "w": round(100 * w / c, 1), "r": round(tot, 2), "slp": round(slsum / c / pip, 1), "dd": round(dd, 2)}

def build(df, pip, dec, days):
    r = df.iloc[-1]; px = float(r["Close"]); risk = 1.5 * float(r["atr"])
    up = r["e9"] > r["e21"] and r["mh"] > 0 and 50 <= r["rsi"] <= 70
    dn = r["e9"] < r["e21"] and r["mh"] < 0 and 30 <= r["rsi"] <= 50
    d = "buy" if up else "sell" if dn else "wait"
    s = -1 if d == "sell" else 1
    rd = lambda v: round(float(v), dec + 1)
    h = df.tail(40); H = df["High"]; L = df["Low"]
    a = H.tail(10).max() > H.iloc[-20:-10].max() and L.tail(10).min() > L.iloc[-20:-10].min()
    b = H.tail(10).max() < H.iloc[-20:-10].max() and L.tail(10).min() < L.iloc[-20:-10].min()
    return {
        "dir": d, "price": rd(px), "sl": rd(px - s * risk), "tp1": rd(px + s * 1.5 * risk), "tp2": rd(px + s * 3 * risk),
        "slp": round(risk / pip, 1), "rsi": int(round(r["rsi"])),
        "ema": "Bullish" if r["e9"] > r["e21"] else "Bearish", "macd": "Bullish" if r["mh"] > 0 else "Bearish",
        "st": "HH / HL" if a else "LH / LL" if b else "Sideways",
        "candles": [[rd(x) for x in row] for row in h[["Open", "High", "Low", "Close"]].values],
        "e9": [rd(x) for x in h["e9"]], "e21": [rd(x) for x in h["e21"]],
        "bt": backtest(df, days, pip), "btdays": days,
    }

def news():
    try:
        data = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"}).json()
        return [{"t": e["title"], "c": e["country"], "i": e["impact"], "d": e["date"]} for e in data
                if e.get("impact") in ("High", "Medium") and e.get("country") in ("USD", "EUR", "GBP", "JPY", "AUD")][:80]
    except Exception as e:
        print("news gagal", e); return []

def telegram(old, new):
    tk = os.getenv("TG_TOKEN"); ch = os.getenv("TG_CHAT")
    if not tk or not ch: return
    msgs = []
    for p, tfs in new["pairs"].items():
        for tf, s in tfs.items():
            if tf not in ("M15", "H1", "H4", "D1") or s["dir"] == "wait": continue
            o = old.get("pairs", {}).get(p, {}).get(tf)
            if not o or o["dir"] == s["dir"]: continue
            msgs.append(f'{p} {tf} {s["dir"].upper()} | Entry {s["price"]} SL {s["sl"]} TP1 {s["tp1"]} TP2 {s["tp2"]}')
    if msgs:
        requests.post(f"https://api.telegram.org/bot{tk}/sendMessage", json={"chat_id": ch, "text": "\n".join(msgs[:15])}, timeout=20)

def main():
    try: old = json.load(open("data/signals.json"))
    except Exception: old = {}
    out = {"updated": dt.datetime.now(dt.timezone.utc).isoformat(), "pairs": {}, "news": news()}
    for name, (tk, pip, dec) in PAIRS.items():
        out["pairs"][name] = {}
        for tf, (iv, per, rs) in TFS.items():
            try:
                df = fetch(tk, iv, per, rs)
                if len(df) < 60: continue
                out["pairs"][name][tf] = build(prep(df), pip, dec, BT_DAYS[tf])
            except Exception as e:
                print("skip", name, tf, e)
    os.makedirs("data", exist_ok=True)
    json.dump(out, open("data/signals.json", "w"), separators=(",", ":"))
    telegram(old, out)

main()
