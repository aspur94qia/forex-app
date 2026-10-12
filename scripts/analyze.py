import json, os, time, datetime as dt
import numpy as np, pandas as pd, requests, yfinance as yf

F, X, M, C = "Forex", "Eksotis", "Emas dan logam", "Crypto"
INS = {}
for n in ["EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD",
          "EUR/GBP","EUR/JPY","EUR/CHF","EUR/AUD","EUR/NZD","EUR/CAD","GBP/JPY","GBP/CHF","GBP/AUD",
          "GBP/NZD","GBP/CAD","AUD/JPY","AUD/CHF","AUD/NZD","AUD/CAD","NZD/JPY","NZD/CHF","NZD/CAD",
          "CAD/JPY","CAD/CHF","CHF/JPY"]:
    jpy = n.endswith("JPY")
    INS[n] = dict(t=n.replace("/", "") + "=X", pip=0.01 if jpy else 0.0001, dec=3 if jpy else 5, ct=100000, g=F)
for n in ["USD/SGD","USD/ZAR","USD/TRY","USD/MXN","USD/HKD","USD/NOK","USD/SEK","USD/DKK","USD/PLN"]:
    INS[n] = dict(t=n.replace("/", "") + "=X", pip=0.0001, dec=5, ct=100000, g=X)
INS["XAU/USD"] = dict(t="GC=F", pip=0.1, dec=2, ct=100, g=M)
INS["XAG/USD"] = dict(t="SI=F", pip=0.01, dec=3, ct=5000, g=M)
INS["XPT/USD"] = dict(t="PL=F", pip=0.1, dec=2, ct=100, g=M)
INS["XPD/USD"] = dict(t="PA=F", pip=0.1, dec=2, ct=100, g=M)
INS["BTC/USD"] = dict(t="BTC-USD", pip=1.0, dec=2, ct=1, g=C)
INS["ETH/USD"] = dict(t="ETH-USD", pip=0.1, dec=2, ct=1, g=C)
INS["LTC/USD"] = dict(t="LTC-USD", pip=0.01, dec=2, ct=1, g=C)
INS["XRP/USD"] = dict(t="XRP-USD", pip=0.0001, dec=4, ct=1, g=C)
INS["BCH/USD"] = dict(t="BCH-USD", pip=0.1, dec=2, ct=1, g=C)

PLAN = [("scalp","M1"),("scalp","M5"),("scalp","M15"),("intra","M30"),("intra","H1"),("intra","H4"),
        ("swing","H4"),("swing","D1"),("long","W1"),("long","MN")]
SRC = {"M1":("1m","5d",None),"M5":("5m","60d",None),"M15":("15m","60d",None),"M30":("30m","60d",None),
       "H1":("60m","200d",None),"H4":("60m","200d","4h"),"D1":("1d","3y",None),"W1":("1wk","10y",None),"MN":("1mo","max",None)}
BT = {("scalp","M1"):5,("scalp","M5"):30,("scalp","M15"):30,("intra","M30"):30,("intra","H1"):30,("intra","H4"):60,
      ("swing","H4"):120,("swing","D1"):365,("long","W1"):1095,("long","MN"):3650}
RULE = {
    "scalp": dict(txt="EMA 9/21 cross + filter EMA 50 + RSI", f=9, s=21, slm=1.0, tp1=1.5, tp2=3, K=3),
    "intra": dict(txt="Pullback ke EMA 20 searah tren EMA 20/50 + RSI", f=20, s=50, slm=1.5, tp1=1.5, tp2=3, K=2),
    "swing": dict(txt="Breakout 20 candle searah EMA 50/200 + RSI", f=50, s=200, slm=2.0, tp1=2, tp2=4, K=2),
    "long": dict(txt="EMA 10/30 + MACD, re-entry di EMA 10", f=10, s=30, slm=2.0, tp1=2, tp2=4, K=1),
}

def dl(tickers, iv, per):
    out = {}; raw = None
    for _ in range(2):
        try:
            raw = yf.download(tickers, interval=iv, period=per, group_by="ticker",
                              auto_adjust=False, progress=False, threads=True)
            break
        except Exception as e:
            print("download gagal", iv, e); time.sleep(3)
    if raw is None: return out
    for t in tickers:
        try:
            d = raw[t][["Open", "High", "Low", "Close"]].dropna()
            if len(d) > 50: out[t] = d
        except Exception:
            pass
    return out

def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean(); rd = dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + ru / rd.replace(0, np.nan))

def atr(d, n=14):
    pc = d["Close"].shift()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(), (d["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def prep(df):
    d = df.copy(); c = d["Close"]
    for n in (9, 10, 20, 21, 30, 50, 200):
        d[f"e{n}"] = c.ewm(span=n, adjust=False).mean()
    d["rsi"] = rsi(c); d["atr"] = atr(d)
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["m"] = m; d["mh"] = m - m.ewm(span=9, adjust=False).mean()
    return d

def events(d, mode):
    c = d["Close"]; o = d["Open"]; h = d["High"]; l = d["Low"]; r = d["rsi"]
    if mode == "scalp":
        cu = (d["e9"] > d["e21"]) & (d["e9"].shift() <= d["e21"].shift())
        cd = (d["e9"] < d["e21"]) & (d["e9"].shift() >= d["e21"].shift())
        b = cu & (c > d["e50"]) & (r > 50) & (r < 72)
        s = cd & (c < d["e50"]) & (r < 50) & (r > 28)
    elif mode == "intra":
        up = (d["e20"] > d["e50"]) & (c > d["e50"]); dn = (d["e20"] < d["e50"]) & (c < d["e50"])
        b = up & (l <= d["e20"]) & (c > d["e20"]) & (c > o) & (r > 45) & (r < 70)
        s = dn & (h >= d["e20"]) & (c < d["e20"]) & (c < o) & (r < 55) & (r > 30)
    elif mode == "swing":
        hh = h.rolling(20).max().shift(); ll = l.rolling(20).min().shift()
        b = (c > hh) & (c.shift() <= hh.shift()) & (d["e50"] > d["e200"]) & (r > 55)
        s = (c < ll) & (c.shift() >= ll.shift()) & (d["e50"] < d["e200"]) & (r < 45)
    else:
        b = (d["e10"] > d["e30"]) & (d["m"] > 0) & (c > d["e10"]) & (c.shift() <= d["e10"].shift())
        s = (d["e10"] < d["e30"]) & (d["m"] < 0) & (c < d["e10"]) & (c.shift() >= d["e10"].shift())
    return (b.astype(int) - s.astype(int)).values

def backtest(d, ev, R, days, pip):
    idx = d.index; cut = idx[-1] - pd.Timedelta(days=days)
    at = d["atr"].values; hi = d["High"].values; lo = d["Low"].values; cl = d["Close"].values
    n = len(d); c = 0; w = 0; tot = slsum = eq = peak = dd = 0.0; nxt = 0
    for i in np.nonzero(ev)[0]:
        if i < nxt or i >= n - 1 or idx[i] < cut or np.isnan(at[i]): continue
        dr = int(ev[i]); entry = cl[i]; risk = R["slm"] * at[i]
        sl = entry - dr * risk; tp = entry + dr * R["tp1"] * risk
        res = None; ex = min(n - 1, i + 60)
        for j in range(i + 1, ex + 1):
            if dr == 1:
                if lo[j] <= sl: res = -1.0; ex = j; break
                if hi[j] >= tp: res = float(R["tp1"]); ex = j; break
            else:
                if hi[j] >= sl: res = -1.0; ex = j; break
                if lo[j] <= tp: res = float(R["tp1"]); ex = j; break
        if res is None: res = float(dr * (cl[ex] - entry) / risk)
        c += 1; w += 1 if res > 0 else 0; tot += res; slsum += float(risk)
        eq += res; peak = max(peak, eq); dd = max(dd, peak - eq); nxt = ex + 1
    if c == 0: return {"c": 0, "w": 0, "r": 0, "slp": 1, "dd": 0}
    return {"c": c, "w": round(100.0 * w / c, 1), "r": round(tot, 2), "slp": round(slsum / c / pip, 1), "dd": round(dd, 2)}

def build(d, mode, tf, ins):
    R = RULE[mode]; ev = events(d, mode); n = len(d)
    r = d.iloc[-1]; px = float(r["Close"]); risk = R["slm"] * float(r["atr"])
    dr = 0
    for k in range(1, R["K"] + 1):
        if ev[n - k] != 0:
            dr = int(ev[n - k]); ec = float(d["Close"].iloc[n - k]); break
    if dr != 0:
        mv = (px - ec) * dr / risk
        if mv > 1.0 or mv < -1.0: dr = 0
    dec = ins["dec"]; pip = ins["pip"]; rd = lambda v: round(float(v), dec)
    s = -1 if dr == -1 else 1
    f, sl_ = R["f"], R["s"]
    H = d["High"]; L = d["Low"]
    a = H.tail(10).max() > H.iloc[-20:-10].max() and L.tail(10).min() > L.iloc[-20:-10].min()
    b = H.tail(10).max() < H.iloc[-20:-10].max() and L.tail(10).min() < L.iloc[-20:-10].min()
    if mode == "scalp": watch = [["EMA 21", r["e21"]], ["EMA 50", r["e50"]]]
    elif mode == "intra": watch = [["Zona pullback EMA 20", r["e20"]], ["Tren EMA 50", r["e50"]]]
    elif mode == "swing": watch = [["Buy di atas", H.iloc[-20:].max()], ["Sell di bawah", L.iloc[-20:].min()]]
    else: watch = [["EMA 10", r["e10"]], ["EMA 30", r["e30"]]]
    t = d.tail(36); rv = r["rsi"]
    res = {
        "dir": "buy" if dr == 1 else "sell" if dr == -1 else "wait",
        "price": rd(px), "sl": rd(px - s * risk), "tp1": rd(px + s * R["tp1"] * risk), "tp2": rd(px + s * R["tp2"] * risk),
        "slp": round(risk / pip, 1), "rsi": int(round(rv)) if rv == rv else 50,
        "tr": "Bullish" if r[f"e{f}"] > r[f"e{sl_}"] else "Bearish",
        "macd": "Bullish" if r["mh"] > 0 else "Bearish",
        "st": "HH / HL" if a else "LH / LL" if b else "Sideways",
        "lines": [f"EMA {f}", f"EMA {sl_}"],
        "e1": [rd(x) for x in t[f"e{f}"]], "e2": [rd(x) for x in t[f"e{sl_}"]],
        "candles": [[rd(x) for x in row] for row in t[["Open", "High", "Low", "Close"]].values],
        "watch": [[w[0], rd(w[1])] for w in watch],
        "rr": f"RR 1:{R['tp1']:g} / 1:{R['tp2']:g}", "method": R["txt"],
        "bt": backtest(d, ev, R, BT[(mode, tf)], pip), "btdays": BT[(mode, tf)],
    }
    json.dumps(res, allow_nan=False)
    return res

def pip_value(name, ins, price):
    pq = ins["pip"] * ins["ct"]
    if ins["g"] in (M, C): return pq
    base, quote = name.split("/")
    if quote == "USD": return pq
    if base == "USD" and name in price: return pq / price[name]
    if f"USD/{quote}" in price: return pq / price[f"USD/{quote}"]
    if f"{quote}/USD" in price: return pq * price[f"{quote}/USD"]
    return 10.0

def news():
    try:
        data = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=20,
                            headers={"User-Agent": "Mozilla/5.0"}).json()
        keep = ("USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "CNY")
        return [{"t": e["title"], "c": e["country"], "i": e["impact"], "d": e["date"]} for e in data
                if e.get("impact") in ("High", "Medium") and e.get("country") in keep][:120]
    except Exception as e:
        print("news gagal", e); return []

def telegram(old, new):
    tk = os.getenv("TG_TOKEN"); ch = os.getenv("TG_CHAT")
    if not tk or not ch: return
    msgs = []
    for p, modes in new["pairs"].items():
        for mode, tfs in modes.items():
            for tf, s in tfs.items():
                if tf not in ("M15", "H1", "H4", "D1") or s["dir"] == "wait": continue
                o = old.get("pairs", {}).get(p, {}).get(mode, {}).get(tf)
                if not o or o["dir"] == s["dir"]: continue
                msgs.append(f'{p} {mode} {tf} {s["dir"].upper()} | Entry {s["price"]} SL {s["sl"]} TP1 {s["tp1"]} TP2 {s["tp2"]}')
    if msgs:
        requests.post(f"https://api.telegram.org/bot{tk}/sendMessage", json={"chat_id": ch, "text": "\n".join(msgs[:15])}, timeout=20)

def main():
    repo = os.getenv("REPO", "aspur94qia/forex-app"); old = {}
    try: old = requests.get(f"https://raw.githubusercontent.com/{repo}/data/signals.json", timeout=20).json()
    except Exception: pass
    names = list(INS); tk = [INS[n]["t"] for n in names]; raw = {}
    for key in {(v[0], v[1]) for v in SRC.values()}:
        raw[key] = dl(tk, key[0], key[1]); print("unduh", key, len(raw[key]), "dari", len(tk))
    prices = {}
    for n in names:
        d = raw[("1d", "3y")].get(INS[n]["t"])
        if d is not None: prices[n] = float(d["Close"].iloc[-1])
    out = {"updated": dt.datetime.now(dt.timezone.utc).isoformat(), "meta": {}, "pairs": {}, "news": news()}
    for n in names:
        ins = INS[n]; pv = round(pip_value(n, ins, prices), 6)
        out["meta"][n] = {"g": ins["g"], "pip": ins["pip"], "dec": ins["dec"], "pv": pv}
        out["pairs"][n] = {}
        for mode, tf in PLAN:
            iv, per, rs = SRC[tf]
            try:
                d = raw[(iv, per)].get(ins["t"])
                if d is None: continue
                if rs: d = d.resample(rs).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
                if len(d) < 60: continue
                out["pairs"][n].setdefault(mode, {})[tf] = build(prep(d), mode, tf, ins)
            except Exception as e:
                print("skip", n, mode, tf, e)
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/signals.json", "w"), separators=(",", ":"), allow_nan=False)
    telegram(old, out)

main()
