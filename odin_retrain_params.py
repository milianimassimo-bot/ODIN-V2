# odin_retrain_params.py
# ======================================================
# ODIN - Retrain parametri MULTI-STRATEGIA (tutte le 10 strategie)
# - Universe da params/strategy_roster.json (campo "universe") + (GOLD, SILVER) opzionali
# - Dati D1 da TwelveData con cache locale (fino a ODIN_DL_YEARS anni)
# - IS/OOS configurabili (default semestrale: 24/12)
# - Grid per TUTTE le 10 strategie ODIN
# - Selezione su IS, validazione su OOS con vincoli
# - Update atomico di params/current.json (+ backup, + STATUS ACTIVE/PAUSED)
# - Report via Telegram
# - Panorama per asset (CSV) in ./data_cache/panorama
# - Filtri: RETRAIN_UNIVERSE_SOURCE, ODIN_INCLUDE_METALS, ODIN_RETRAIN_ONLY
# ======================================================

import os, json, math, time, shutil, logging
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
load_dotenv()

# ---------- PATHS ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data_cache")
PARAMS_DIR = os.path.join(BASE_DIR, "params")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)

ROSTER_PATH = os.path.join(PARAMS_DIR, "strategy_roster.json")

# >>> PATCH: files diagnostica retrain (JSON + TXT)
RETRAIN_METRICS_JSON = os.path.join(PARAMS_DIR, "retrain_metrics.json")
RETRAIN_REPORT_TXT   = os.path.join(BASE_DIR, "logs", "last_retrain.txt")
os.makedirs(os.path.dirname(RETRAIN_REPORT_TXT), exist_ok=True)
# <<< PATCH

# ---------- ENV ----------
TD_API_KEY  = os.getenv("TWELVEDATA_API_KEY", "").strip()

TD_BASE_URL = "https://api.twelvedata.com/time_series"
RETRAIN_UNIVERSE_SOURCE = os.getenv("RETRAIN_UNIVERSE_SOURCE", "ROSTER")
RETRAIN_UNIVERSE_SOURCE = (RETRAIN_UNIVERSE_SOURCE or "ROSTER").strip().upper()  # ROSTER | FIXED | ALL

def _truthy(x: str) -> bool:
    return str(x or "").strip().lower() in {"1","true","yes","on","y"}

ODIN_INCLUDE_METALS = _truthy(os.getenv("ODIN_INCLUDE_METALS", "0"))
ODIN_RETRAIN_ONLY   = (os.getenv("ODIN_RETRAIN_ONLY", "") or "").strip()  # CSV labels o vuoto


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# finestre IS/OOS
IS_MONTHS  = int(os.getenv("RETRAIN_IS_MONTHS", "24"))
OOS_MONTHS = int(os.getenv("RETRAIN_OOS_MONTHS", "12"))

# vincoli OOS
MIN_TRADES_OOS = int(os.getenv("RETRAIN_MIN_TRADES_OOS", "5"))
MIN_PF_OOS     = float(os.getenv("RETRAIN_MIN_PF_OOS", "1.05"))
MAX_DD_OOS_R   = float(os.getenv("RETRAIN_MAX_DD_OOS_R", "25.0"))

# anni da scaricare/cache
ODIN_DL_YEARS = int(os.getenv("ODIN_DL_YEARS", "10"))

# dry run (non scrive current.json)
ODIN_RETRAIN_DRYRUN = os.getenv("ODIN_RETRAIN_DRYRUN", "0") == "1"

# ---------- LOG ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("odin_retrain")

log.info(f"RETRAIN_UNIVERSE_SOURCE={RETRAIN_UNIVERSE_SOURCE}, INCLUDE_METALS={ODIN_INCLUDE_METALS}, RETRAIN_ONLY={ODIN_RETRAIN_ONLY or '(none)'}")



# ---------- Portfolio / mapping label -> (symbol, kind) ----------
# =====================================================
# 🔹 ASSET PORTFOLIO — Mappa completa per retrain & sentinel
# =====================================================
# Label = nome logico interno ODIN
# Value = (ticker TwelveData/MT5, tipo)
ASSET_PORTFOLIO = {
    # --- MAJORS ---
    "EUR/USD": ("EUR/USD", "forex"),
    "GBP/USD": ("GBP/USD", "forex"),
    "USD/JPY": ("USD/JPY", "forex"),
    "USD/CHF": ("USD/CHF", "forex"),
    "USD/CAD": ("USD/CAD", "forex"),
    "AUD/USD": ("AUD/USD", "forex"),
    "NZD/USD": ("NZD/USD", "forex"),

    # --- CROSS EUR ---
    "EUR/GBP": ("EUR/GBP", "forex"),
    "EUR/JPY": ("EUR/JPY", "forex"),
    "EUR/CHF": ("EUR/CHF", "forex"),
    "EUR/CAD": ("EUR/CAD", "forex"),
    "EUR/AUD": ("EUR/AUD", "forex"),
    "EUR/NZD": ("EUR/NZD", "forex"),

    # --- CROSS GBP ---
    "GBP/JPY": ("GBP/JPY", "forex"),
    "GBP/CHF": ("GBP/CHF", "forex"),
    "GBP/CAD": ("GBP/CAD", "forex"),
    "GBP/AUD": ("GBP/AUD", "forex"),
    "GBP/NZD": ("GBP/NZD", "forex"),

    # --- CROSS AUD ---
    "AUD/JPY": ("AUD/JPY", "forex"),
    "AUD/CHF": ("AUD/CHF", "forex"),
    "AUD/CAD": ("AUD/CAD", "forex"),
    "AUD/NZD": ("AUD/NZD", "forex"),

    # --- CROSS NZD ---
    "NZD/JPY": ("NZD/JPY", "forex"),
    "NZD/CHF": ("NZD/CHF", "forex"),
    "NZD/CAD": ("NZD/CAD", "forex"),

    # --- CROSS CAD ---
    "CAD/JPY": ("CAD/JPY", "forex"),
    "CAD/CHF": ("CAD/CHF", "forex"),

    # --- CROSS CHF ---
    "CHF/JPY": ("CHF/JPY", "forex"),

    # --- SCANDINAVI E EXOTICS ---
    "USD/SEK": ("USD/SEK", "forex"),
    "USD/NOK": ("USD/NOK", "forex"),
    "USD/DKK": ("USD/DKK", "forex"),
    "EUR/SEK": ("EUR/SEK", "forex"),
    "EUR/NOK": ("EUR/NOK", "forex"),
    "EUR/DKK": ("EUR/DKK", "forex"),

    # --- METALLI ---
    "GOLD":   ("XAU/USD", "metal"),
    "SILVER": ("XAG/USD", "metal"),

    # --- INDICI ---
    "SP500":  ("SPX/USD", "index"),      # alias TwelveData: SPX/USD o ^SPX
    "DAX":    ("GDAXI", "index"),
    "FTSE":   ("FTSE:FSI", "index"),
    "NAS100": ("NDX/USD", "index"),
    "US30":   ("DJI/USD", "index"),

    # --- CRIPTO (opzionale futuro) ---
    "BTC/USD": ("BTC/USD", "crypto"),
    "ETH/USD": ("ETH/USD", "crypto"),
}
# =====================================================
# ---------- Telegram ----------
def tg(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
    except Exception as e:
        log.warning(f"Telegram failed: {e!r}")


# ---------- Universe builder ----------
def load_roster_universe() -> list[str]:
    """Legge params/strategy_roster.json e ritorna la lista 'universe'."""
    if not os.path.exists(ROSTER_PATH):
        log.error(f"Roster non trovato: {ROSTER_PATH}")
        return []
    try:
        with open(ROSTER_PATH, "r", encoding="utf-8") as f:
            js = json.load(f)
        # la chiave nel tuo file è "universe"
        uni = js.get("universe", [])
        if not isinstance(uni, list):
            raise ValueError("chiave 'universe' non è una lista")
        # normalizza ma senza cambiare il formato del label (già UPPER con slash)
        uni = [str(x).strip() for x in uni if str(x).strip()]
        return uni
    except Exception as e:
        log.error(f"Errore lettura roster: {e!r}")
        return []

def build_universe() -> list[str]:
    """
    Costruisce l'universo finale in base a:
    - RETRAIN_UNIVERSE_SOURCE: ROSTER | FIXED | ALL
    - ODIN_INCLUDE_METALS: se True, aggiunge GOLD & SILVER
    - ODIN_RETRAIN_ONLY: se presente, filtra a subset CSV (override)
    """
    src = (RETRAIN_UNIVERSE_SOURCE or "ROSTER").strip().upper()
    out: list[str] = []

    if src == "ROSTER":
        base = load_roster_universe()
        out.extend(base)

    elif src == "FIXED":
        # set fisso: qui uso le 16 del roster come base fissa
        out = [
            "AUD/CAD","CHF/JPY","EUR/AUD","CAD/JPY","GBP/JPY","AUD/JPY","USD/JPY","EUR/JPY",
            "NZD/JPY","NZD/USD","NZD/CAD","USD/SEK","EUR/NZD","AUD/CHF","GBP/NZD","USD/NOK"
        ]

    elif src == "ALL":
        # tutte le label che conosciamo (escludo metalli qui, li aggiungo sotto se flaggati)
        out = [k for k in ASSET_PORTFOLIO.keys() if k not in {"GOLD","SILVER"}]

    else:
        log.warning(f"RETRAIN_UNIVERSE_SOURCE sconosciuto: {src}. Fallback=ROSTER.")
        out = load_roster_universe()

    # Aggiungi metalli se richiesto
    if ODIN_INCLUDE_METALS:
        for lab in ("GOLD", "SILVER"):
            if lab not in out:
                out.append(lab)

    # Filtra con ONLY (override)
    if ODIN_RETRAIN_ONLY:
        only_set = {x.strip() for x in ODIN_RETRAIN_ONLY.split(",") if x.strip()}
        out = [x for x in out if x in only_set]
        # se ONLY include metalli, assicurati che restino
        for lab in ("GOLD","SILVER"):
            if lab in only_set and lab not in out:
                out.append(lab)

    # Mantieni solo label mappati
    out = [x for x in out if x in ASSET_PORTFOLIO]

    # Ordine stabile: come arriva (niente sort), evita duplicati preservando l'ordine
    seen = set(); ordered = []
    for x in out:
        if x not in seen:
            seen.add(x); ordered.append(x)

    log.info(f"[UNIVERSE] source={src}, roster_path='{ROSTER_PATH}', count={len(ordered)}, labels={ordered}")
    return ordered

# ---------- Data (cache incrementale, storico ampio) ----------
def cache_path(label: str) -> str:
    key = label.replace("/", "_")
    return os.path.join(DATA_DIR, f"{key}.csv")

def _td_call(symbol: str, interval: str = "1day", outputsize: int = 5000) -> pd.DataFrame:
    if not TD_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY mancante")
    r = requests.get(
        TD_BASE_URL,
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "order": "ASC",
            "timezone": "UTC",
            "apikey": TD_API_KEY
        },
        timeout=25
    )
    r.raise_for_status()
    data = r.json()
    vals = data.get("values")
    if not vals:
        raise RuntimeError(f"Nessun dato TD per {symbol} {interval}")
    df = pd.DataFrame(vals)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ("open","high","low","close"):
        df[c] = df[c].astype(float)
    df = df.sort_values("datetime").set_index("datetime")
    # scarta barra odierna aperta
    if len(df) > 0 and df.index[-1].date() == datetime.now(timezone.utc).date():
        df = df.iloc[:-1]
    return df

def _years_to_outputsize(years: int) -> int:
    # stima: ~370 barre/anno
    return int(370 * max(1, years))

def get_data(label: str, symbol: str) -> pd.DataFrame:
    """
    Carica la cache locale se esiste, poi scarica storia ampia (ODIN_DL_YEARS),
    unisce e salva. Alla fine ritorna l'intera storia disponibile.
    """
    cp = cache_path(label)

    existing = None
    if os.path.exists(cp):
        try:
            existing = pd.read_csv(cp)
            existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)
            existing = existing.set_index("datetime")[["open","high","low","close"]].sort_index()
        except Exception:
            existing = None

    outputsize = _years_to_outputsize(ODIN_DL_YEARS)
    fresh = _td_call(symbol, interval="1day", outputsize=outputsize)

    # Se esiste cache più lunga, teniamola
    if existing is not None:
        merged = pd.concat([existing, fresh]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = fresh

    # salva cache completa
    merged.to_csv(cp, index=True)
    return merged

# ---------- Indicators ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    prev = c.shift(1)
    tr = pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def bbands(s: pd.Series, n: int = 20, k: float = 2.0):
    ma = s.rolling(n).mean(); sd = s.rolling(n).std(ddof=0)
    up = ma + k*sd; lo = ma - k*sd
    return lo, ma, up

def rsi(s: pd.Series, n: int = 14):
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, 1e-9)
    return 100 - (100/(1+rs))
# ---------- Simple backtest core - TUTTE LE 10 STRATEGIE ----------
def sim_weekly(df: pd.DataFrame, params: dict) -> list[float]:
    """
    WEEKLY (trend su D1):
    - direzione = EMA50 vs EMA200
    - entry quando |close-EMA10| <= anti_chase_mult * ATR(14)
    - SL = sl_atr_mult * ATR; TP a RR fisso; time_stop in giorni
    ritorna: lista di R per i trade chiusi
    """
    c = df["close"]; a = atr(df, 14)
    e10, e50, e200 = ema(c,10), ema(c,50), ema(c,200)
    rr = float(params["rr"])
    anti = float(params["anti_chase_mult"])
    slmult = float(params["sl_atr_mult"])
    tstop = int(params.get("time_stop", 30))
    Rs = []
    pos = None # dict: dir, entry, R, open_time

    for i in range(200, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"])
            pos=None

        dirn = "LONG" if e50.iloc[i] > e200.iloc[i] else "SHORT"

        if not pos and abs(c.iloc[i]-e10.iloc[i]) <= anti * a.iloc[i]:
            entry = float(c.iloc[i]); R = slmult * a.iloc[i]
            if R > 1e-9:
                pos = {"dir":dirn,"entry":entry,"R":float(R),"open_time":df.index[i]}

        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_bbmr(df: pd.DataFrame, params: dict) -> list[float]:
    """
    BB_MR (mean-reversion su D1):
    - BB (len/std), RSI (len), soglie buy/sell
    - exit: TP=MA (tp_to_ma=True) o RR fisso, SL = atr_stop_mult*ATR; time_stop
    """
    c = df["close"]; a = atr(df,14)
    lo, ma, up = bbands(c, int(params["bb_len"]), float(params["bb_std"]))
    r = rsi(c, int(params["rsi_len"]))
    rsi_buy  = float(params.get("rsi_buy", 35))
    rsi_sell = float(params.get("rsi_sell", 65))
    atr_stop = float(params.get("atr_stop_mult", 1.5))
    tp_to_ma = bool(params.get("tp_to_ma", True))
    tstop = int(params.get("time_stop", 20))
    Rs = []; pos=None

    for i in range(60, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px-pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"]-px)
            Rs.append(fav/pos["R"]); pos=None

        if not pos:
            if (c.iloc[i] <= lo.iloc[i]) and (r.iloc[i] <= rsi_buy):
                entry=float(c.iloc[i]); R=atr_stop*a.iloc[i]
                if R>1e-9: pos={"dir":"LONG","entry":entry,"R":float(R),"open_time":df.index[i]}
            elif (c.iloc[i] >= up.iloc[i]) and (r.iloc[i] >= rsi_sell):
                entry=float(c.iloc[i]); R=atr_stop*a.iloc[i]
                if R>1e-9: pos={"dir":"SHORT","entry":entry,"R":float(R),"open_time":df.index[i]}

        if pos:
            if pos["dir"]=="LONG":
                sl=pos["entry"]-pos["R"]
                if tp_to_ma and c.iloc[i] >= ma.iloc[i]:
                    Rs.append((ma.iloc[i]-pos["entry"]) / pos["R"]); pos=None
                elif (not tp_to_ma):
                    tp=pos["entry"]+1.5*pos["R"]
                    if c.iloc[i] >= tp: Rs.append(1.5); pos=None
                if pos and c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                sl=pos["entry"]+pos["R"]
                if tp_to_ma and c.iloc[i] <= ma.iloc[i]:
                    Rs.append((pos["entry"]-ma.iloc[i]) / pos["R"]); pos=None
                elif (not tp_to_ma):
                    tp=pos["entry"]-1.5*pos["R"]
                    if c.iloc[i] <= tp: Rs.append(1.5); pos=None
                if pos and c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_ema_trend_d1(df: pd.DataFrame, params: dict) -> list[float]:
    """EMA_Trend_D1: Crossover EMA veloce/lenta con filtri trend"""
    c = df["close"]; a = atr(df, 14)
    ema_fast = int(params.get("ema_fast", 50))
    ema_slow = int(params.get("ema_slow", 200))
    sl_mult = float(params.get("sl_atr_mult", 2.0))
    rr = float(params.get("rr", 1.5))
    tstop = int(params.get("time_stop", 30))
    
    ef, es = ema(c, ema_fast), ema(c, ema_slow)
    Rs = []; pos = None
    
    for i in range(ema_slow + 10, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"]); pos = None
            
        dirn = "LONG" if ef.iloc[i] > es.iloc[i] else "SHORT"
        
        if not pos:
            entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
            if R > 1e-9:
                pos = {"dir":dirn,"entry":entry,"R":float(R),"open_time":df.index[i]}
                
        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_keltner_trend_d1(df: pd.DataFrame, params: dict) -> list[float]:
    """Keltner_Trend_D1: Breakout canali Keltner"""
    c = df["close"]; a = atr(df, 14)
    ema_period = int(params.get("ema_period", 20))
    atr_mult = float(params.get("atr_mult", 2.0))
    sl_mult = float(params.get("sl_atr_mult", 2.5))
    rr = float(params.get("rr", 1.8))
    tstop = int(params.get("time_stop", 30))
    
    ma = ema(c, ema_period); atr_val = atr(df, ema_period)
    Rs = []; pos = None
    
    for i in range(ema_period + 10, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"]); pos = None
            
        upper = ma.iloc[i] + atr_mult * atr_val.iloc[i]
        lower = ma.iloc[i] - atr_mult * atr_val.iloc[i]
        
        if not pos:
            if c.iloc[i] > upper:
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"LONG","entry":entry,"R":float(R),"open_time":df.index[i]}
            elif c.iloc[i] < lower:
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"SHORT","entry":entry,"R":float(R),"open_time":df.index[i]}
                
        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

# ... The rest are similar for sim_voltarget_trend_d1, sim_ema_pullback_h4, sim_keltner_mr_h4, sim_donchian_bo_d1, sim_bb_squeeze_bo_d1, sim_regime_switcher_d1

def sim_voltarget_trend_d1(df: pd.DataFrame, params: dict) -> list[float]:
    """VolTarget_Trend_D1: Trend con controllo volatilità"""
    c = df["close"]; a = atr(df, 14)
    vol_target = float(params.get("vol_target", 1.0))
    ema_period = int(params.get("ema_period", 21))
    sl_mult = float(params.get("sl_atr_mult", 2.0))
    rr = float(params.get("rr", 1.5))
    tstop = int(params.get("time_stop", 30))
    
    ma = ema(c, ema_period)
    atrp = (a / c.replace(0, pd.NA)) * 100.0
    Rs = []; pos = None
    
    for i in range(ema_period + 20, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"]); pos = None
            
        # Controllo volatilità target
        if pd.isna(atrp.iloc[i]) or atrp.iloc[i] < vol_target * 0.8 or atrp.iloc[i] > vol_target * 1.5:
            continue
            
        dirn = "LONG" if c.iloc[i] > ma.iloc[i] else "SHORT"
        
        if not pos:
            entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
            if R > 1e-9:
                pos = {"dir":dirn,"entry":entry,"R":float(R),"open_time":df.index[i]}
                
        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_ema_pullback_h4(df: pd.DataFrame, params: dict) -> list[float]:
    """EMA_Pullback_H4: Pullback su trend EMA (simulato su D1)"""
    c = df["close"]; a = atr(df, 14)
    ema_fast = int(params.get("ema_fast", 21))
    ema_slow = int(params.get("ema_slow", 50))
    pullback_pct = float(params.get("pullback_pct", 0.5))
    sl_mult = float(params.get("sl_atr_mult", 1.5))
    rr = float(params.get("rr", 2.0))
    tstop = int(params.get("time_stop", 20))
    
    ef, es = ema(c, ema_fast), ema(c, ema_slow)
    Rs = []; pos = None
    
    for i in range(ema_slow + 10, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"]); pos = None
            
        # Trend principale
        if ef.iloc[i] > es.iloc[i]:  # Uptrend
            if not pos and c.iloc[i] < ef.iloc[i] * (1 + pullback_pct/100):
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"LONG","entry":entry,"R":float(R),"open_time":df.index[i]}
        else:  # Downtrend
            if not pos and c.iloc[i] > ef.iloc[i] * (1 - pullback_pct/100):
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"SHORT","entry":entry,"R":float(R),"open_time":df.index[i]}
                
        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_keltner_mr_h4(df: pd.DataFrame, params: dict) -> list[float]:
    """Keltner_MR_H4: Mean reversion Keltner + RSI (simulato su D1)"""
    c = df["close"]; a = atr(df, 14)
    ema_period = int(params.get("ema_period", 20))
    atr_mult = float(params.get("atr_mult", 2.0))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_buy = float(params.get("rsi_buy", 30))
    rsi_sell = float(params.get("rsi_sell", 70))
    sl_mult = float(params.get("sl_atr_mult", 1.5))
    tstop = int(params.get("time_stop", 20))
    
    ma = ema(c, ema_period); atr_val = atr(df, ema_period); r = rsi(c, rsi_period)
    Rs = []; pos = None
    
    for i in range(max(ema_period, rsi_period) + 10, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"]); pos = None
            
        upper = ma.iloc[i] + atr_mult * atr_val.iloc[i]
        lower = ma.iloc[i] - atr_mult * atr_val.iloc[i]
        
        if not pos:
            if c.iloc[i] <= lower and r.iloc[i] <= rsi_buy:
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"LONG","entry":entry,"R":float(R),"open_time":df.index[i],"tp":ma.iloc[i]}
            elif c.iloc[i] >= upper and r.iloc[i] >= rsi_sell:
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"SHORT","entry":entry,"R":float(R),"open_time":df.index[i],"tp":ma.iloc[i]}
                
        if pos:
            if pos["dir"]=="LONG":
                sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= pos["tp"]: Rs.append((pos["tp"]-pos["entry"])/pos["R"]); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= pos["tp"]: Rs.append((pos["entry"]-pos["tp"])/pos["R"]); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_donchian_bo_d1(df: pd.DataFrame, params: dict) -> list[float]:
    """Donchian_BO_D1: Breakout canali Donchian"""
    c, h, l = df["close"], df["high"], df["low"]; a = atr(df, 14)
    period = int(params.get("period", 20))
    sl_mult = float(params.get("sl_atr_mult", 2.0))
    rr = float(params.get("rr", 2.0))
    tstop = int(params.get("time_stop", 30))
    
    high_max = h.rolling(period).max(); low_min = l.rolling(period).min()
    Rs = []; pos = None
    
    for i in range(period + 10, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"]); pos = None
            
        if not pos:
            if h.iloc[i] > high_max.iloc[i-1]:
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"LONG","entry":entry,"R":float(R),"open_time":df.index[i]}
            elif l.iloc[i] < low_min.iloc[i-1]:
                entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
                if R > 1e-9: pos = {"dir":"SHORT","entry":entry,"R":float(R),"open_time":df.index[i]}
                
        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_bb_squeeze_bo_d1(df: pd.DataFrame, params: dict) -> list[float]:
    """BB_Squeeze_BO_D1: Breakout da compressione Bollinger Bands"""
    c = df["close"]; a = atr(df, 14)
    bb_period = int(params.get("bb_period", 20))
    bb_std = float(params.get("bb_std", 2.0))
    squeeze_threshold = float(params.get("squeeze_threshold", 0.02))
    sl_mult = float(params.get("sl_atr_mult", 2.0))
    rr = float(params.get("rr", 2.5))
    tstop = int(params.get("time_stop", 30))
    
    lo, ma, up = bbands(c, bb_period, bb_std)
    bbw = (up - lo) / ma.replace(0, pd.NA)
    Rs = []; pos = None
    
    for i in range(bb_period + 20, len(df)):
        if pos and (df.index[i] - pos["open_time"]).days >= tstop:
            px = float(c.iloc[i])
            fav = (px - pos["entry"]) if pos["dir"]=="LONG" else (pos["entry"] - px)
            Rs.append(fav / pos["R"]); pos = None
            
        # Controllo squeeze
        if pd.isna(bbw.iloc[i]) or bbw.iloc[i] > squeeze_threshold:
            continue
            
        if not pos:
            dirn = "LONG" if c.iloc[i] > ma.iloc[i] else "SHORT"
            entry = float(c.iloc[i]); R = sl_mult * a.iloc[i]
            if R > 1e-9:
                pos = {"dir":dirn,"entry":entry,"R":float(R),"open_time":df.index[i]}
                
        if pos:
            if pos["dir"]=="LONG":
                tp = pos["entry"] + rr*pos["R"]; sl = pos["entry"] - pos["R"]
                if c.iloc[i] >= tp: Rs.append(rr); pos=None
                elif c.iloc[i] <= sl: Rs.append(-1.0); pos=None
            else:
                tp = pos["entry"] - rr*pos["R"]; sl = pos["entry"] + pos["R"]
                if c.iloc[i] <= tp: Rs.append(rr); pos=None
                elif c.iloc[i] >= sl: Rs.append(-1.0); pos=None
    return Rs

def sim_regime_switcher_d1(df: pd.DataFrame, params: dict) -> list[float]:
    """Regime_Switcher_D1: Strategia adattiva (usa WEEKLY per semplicità)"""
    # Per ora usa la logica WEEKLY come fallback
    return sim_weekly(df, {
        "anti_chase_mult": params.get("anti_chase_mult", 0.5),
        "sl_atr_mult": params.get("sl_atr_mult", 2.0),
        "rr": params.get("rr", 1.5),
        "time_stop": params.get("time_stop", 30)
    })

def metrics_from_R(Rs: list[float]) -> dict:
    if not Rs:
        return {"trades":0,"pf":0.0,"exp":0.0,"maxdd":0.0}
    wins=[r for r in Rs if r>0]; losses=[-r for r in Rs if r<=0]
    gross_up=sum(wins); gross_dn=sum(losses)
    if gross_dn>0: pf=gross_up/gross_dn
    else: pf = float("inf") if gross_up>0 else 0.0
    exp=sum(Rs)/len(Rs)
    eq=[0.0]
    for r in Rs: eq.append(eq[-1]+r)
    peak=0.0; maxdd=0.0
    for x in eq:
        if x>peak: peak=x
        dd=peak-x
        if dd>maxdd: maxdd=dd
    return {"trades":len(Rs), "pf":pf, "exp":exp, "maxdd":maxdd}
# ---------- Grids per tutte le strategie ----------
GRID_WEEKLY = [
    {"adx_min":25,"r2_min":0.28,"anti_chase_mult":0.3,"sl_atr_mult":2.0,"rr":1.6,"time_stop":30},
    {"adx_min":25,"r2_min":0.34,"anti_chase_mult":0.5,"sl_atr_mult":2.5,"rr":1.8,"time_stop":30},
    {"adx_min":30,"r2_min":0.34,"anti_chase_mult":0.3,"sl_atr_mult":3.0,"rr":1.8,"time_stop":30},
    {"adx_min":25,"r2_min":0.25,"anti_chase_mult":0.8,"sl_atr_mult":2.0,"rr":1.4,"time_stop":45},
]

GRID_BB_MR_D1 = [
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":35,"rsi_sell":65,"atr_stop_mult":1.5,"tp_to_ma":True,"time_stop":20},
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":30,"rsi_sell":60,"atr_stop_mult":1.2,"tp_to_ma":True,"time_stop":20},
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":40,"rsi_sell":70,"atr_stop_mult":1.8,"tp_to_ma":True,"time_stop":20},
    {"bb_len":20,"bb_std":2.0,"rsi_len":14,"rsi_buy":30,"rsi_sell":70,"atr_stop_mult":1.2,"tp_to_ma":True,"time_stop":30},
]

GRID_EMA_TREND_D1 = [
    {"ema_fast":50,"ema_slow":200,"sl_atr_mult":2.0,"rr":1.5,"time_stop":30},
    {"ema_fast":21,"ema_slow":100,"sl_atr_mult":2.5,"rr":1.8,"time_stop":25},
    {"ema_fast":50,"ema_slow":200,"sl_atr_mult":1.8,"rr":2.0,"time_stop":35},
]

GRID_KELTNER_TREND_D1 = [
    {"ema_period":20,"atr_mult":2.0,"sl_atr_mult":2.5,"rr":1.8,"time_stop":30},
    {"ema_period":15,"atr_mult":1.8,"sl_atr_mult":2.0,"rr":2.0,"time_stop":25},
    {"ema_period":25,"atr_mult":2.2,"sl_atr_mult":3.0,"rr":1.6,"time_stop":35},
]

GRID_VOLTARGET_TREND_D1 = [
    {"vol_target":1.0,"ema_period":21,"sl_atr_mult":2.0,"rr":1.5,"time_stop":30},
    {"vol_target":1.2,"ema_period":15,"sl_atr_mult":2.5,"rr":1.8,"time_stop":25},
    {"vol_target":0.8,"ema_period":25,"sl_atr_mult":1.8,"rr":2.0,"time_stop":35},
]

GRID_EMA_PULLBACK_H4 = [
    {"ema_fast":21,"ema_slow":50,"pullback_pct":0.5,"sl_atr_mult":1.5,"rr":2.0,"time_stop":20},
    {"ema_fast":15,"ema_slow":40,"pullback_pct":0.3,"sl_atr_mult":1.8,"rr":2.2,"time_stop":18},
    {"ema_fast":25,"ema_slow":60,"pullback_pct":0.7,"sl_atr_mult":1.2,"rr":1.8,"time_stop":25},
]

GRID_KELTNER_MR_H4 = [
    {"ema_period":20,"atr_mult":2.0,"rsi_period":14,"rsi_buy":30,"rsi_sell":70,"sl_atr_mult":1.5,"time_stop":20},
    {"ema_period":15,"atr_mult":1.8,"rsi_period":10,"rsi_buy":25,"rsi_sell":75,"sl_atr_mult":1.2,"time_stop":18},
    {"ema_period":25,"atr_mult":2.2,"rsi_period":18,"rsi_buy":35,"rsi_sell":65,"sl_atr_mult":1.8,"time_stop":25},
]

GRID_DONCHIAN_BO_D1 = [
    {"period":20,"sl_atr_mult":2.0,"rr":2.0,"time_stop":30},
    {"period":15,"sl_atr_mult":2.5,"rr":2.2,"time_stop":25},
    {"period":25,"sl_atr_mult":1.8,"rr":1.8,"time_stop":35},
]

GRID_BB_SQUEEZE_BO_D1 = [
    {"bb_period":20,"bb_std":2.0,"squeeze_threshold":0.02,"sl_atr_mult":2.0,"rr":2.5,"time_stop":30},
    {"bb_period":15,"bb_std":1.8,"squeeze_threshold":0.015,"sl_atr_mult":2.2,"rr":2.8,"time_stop":25},
    {"bb_period":25,"bb_std":2.2,"squeeze_threshold":0.025,"sl_atr_mult":1.8,"rr":2.2,"time_stop":35},
]

GRID_REGIME_SWITCHER_D1 = [
    {"anti_chase_mult":0.5,"sl_atr_mult":2.0,"rr":1.5,"time_stop":30},
    {"anti_chase_mult":0.3,"sl_atr_mult":2.5,"rr":1.8,"time_stop":25},
    {"anti_chase_mult":0.7,"sl_atr_mult":1.8,"rr":2.0,"time_stop":35},
]

# Mapping strategia -> (funzione_sim, griglia)
STRATEGY_CONFIG = {
    "WEEKLY": (sim_weekly, GRID_WEEKLY),
    "BB_MR_D1": (sim_bbmr, GRID_BB_MR_D1),
    "EMA_Trend_D1": (sim_ema_trend_d1, GRID_EMA_TREND_D1),
    "Keltner_Trend_D1": (sim_keltner_trend_d1, GRID_KELTNER_TREND_D1),
    "VolTarget_Trend_D1": (sim_voltarget_trend_d1, GRID_VOLTARGET_TREND_D1),
    "EMA_Pullback_H4": (sim_ema_pullback_h4, GRID_EMA_PULLBACK_H4),
    "Keltner_MR_H4": (sim_keltner_mr_h4, GRID_KELTNER_MR_H4),
    "Donchian_BO_D1": (sim_donchian_bo_d1, GRID_DONCHIAN_BO_D1),
    "BB_Squeeze_BO_D1": (sim_bb_squeeze_bo_d1, GRID_BB_SQUEEZE_BO_D1),
    "Regime_Switcher_D1": (sim_regime_switcher_d1, GRID_REGIME_SWITCHER_D1),
}

# ---------- Split IS/OOS ----------
def split_is_oos(df: pd.DataFrame, is_months: int, oos_months: int):
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame deve avere un DatetimeIndex")

    end = df.index[-1]
    # fai OOS di lunghezza esatta 'oos_months'
    oos_start = end - pd.DateOffset(months=oos_months) + pd.Timedelta(days=1)
    is_start  = oos_start - pd.DateOffset(months=is_months)

    mask_is  = (df.index >= is_start) & (df.index < oos_start)
    mask_oos = (df.index >= oos_start)
    return df.loc[mask_is].copy(), df.loc[mask_oos].copy()

# ---------- FIT selezione + validazione MULTI-STRATEGIA ----------
def choose_on_is_then_validate(label: str, df: pd.DataFrame):
    """
    Selezione e validazione per TUTTE le 10 strategie ODIN.
    Ritorna dict con risultati per ogni strategia disponibile.
    """
    # Split IS/OOS
    df_is, df_oos = split_is_oos(df, IS_MONTHS, OOS_MONTHS)
    
    # Risultati per tutte le strategie
    results = {}
    
    # Processa ogni strategia configurata
    for strategy_name, (sim_func, grid) in STRATEGY_CONFIG.items():
        log.info(f"[{label}] Processing strategy: {strategy_name}")
        
        best = None
        try:
            # Testa ogni parametro nella griglia
            for params in grid:
                Rs_is = sim_func(df_is, params)
                metrics_is = metrics_from_R(Rs_is)
                
                # Filtro minimo trade IS (adattivo per strategia)
                min_trades = 10 if strategy_name == "WEEKLY" else 8
                if strategy_name in ["EMA_Pullback_H4", "Keltner_MR_H4"]:
                    min_trades = 6  # Strategie H4 hanno meno trade
                
                if metrics_is["trades"] < min_trades:
                    continue
                    
                # Score composito (PF * expectancy normalizzata)
                score = metrics_is["pf"] * max(0.1, metrics_is["exp"] + 1.0)
                
                if (best is None) or (score > best["score"]):
                    best = {
                        "params": params,
                        "is_metrics": metrics_is,
                        "score": score
                    }
            
            # Se trovato parametro valido, testa su OOS
            if best:
                Rs_oos = sim_func(df_oos, best["params"])
                metrics_oos = metrics_from_R(Rs_oos)
                
                results[strategy_name] = {
                    "params": best["params"],
                    "IS": best["is_metrics"],
                    "OOS": metrics_oos
                }
                
                log.info(f"[{label}] {strategy_name}: IS={best['is_metrics']['trades']} trades, "
                        f"OOS={metrics_oos['trades']} trades, PF={metrics_oos.get('pf', 0):.2f}")
            else:
                log.info(f"[{label}] {strategy_name}: No valid parameters found in IS")
                
        except Exception as e:
            log.error(f"[{label}] {strategy_name} failed: {e!r}")
            continue
    
    return results

# ---------- Gate OOS centrale ----------
def passes_oos(m: dict) -> bool:
    """Applica i vincoli OOS globali al dict metrics {'OOS': {...}}."""
    if not m or "OOS" not in m:
        return False
    o = m["OOS"]
    tr_ok = o.get("trades", 0) >= MIN_TRADES_OOS
    pf    = o.get("pf", 0.0)
    pf_ok = math.isfinite(pf) and (pf >= MIN_PF_OOS)
    exp_ok = o.get("exp", 0.0) > 0.0
    dd_ok  = o.get("maxdd", 1e9) <= MAX_DD_OOS_R
    return tr_ok and pf_ok and exp_ok and dd_ok

# ---------- Params I/O ----------
def params_current_path() -> str:
    return os.path.join(PARAMS_DIR, "current.json")

def params_backup_path(ts: str) -> str:
    return os.path.join(PARAMS_DIR, "archive", f"current_{ts}.json")

def load_params_current() -> dict:
    p = params_current_path()
    try:
        with open(p,"r",encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def atomic_write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".new"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False)
    if os.path.exists(path):
        os.replace(tmp, path)
    else:
        shutil.move(tmp, path)

# ---------- Panorama ----------
def save_panorama(label: str, df: pd.DataFrame):
    pano_dir = os.path.join(DATA_DIR, "panorama")
    os.makedirs(pano_dir, exist_ok=True)
    c = df["close"]; h,l = df["high"], df["low"]
    prev = c.shift(1)
    tr = pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    atrp = (atr14 / c.replace(0, pd.NA)) * 100.0
    ma = c.rolling(20).mean(); sd = c.rolling(20).std(ddof=0)
    bbw = (2*sd) / ma.replace(0, pd.NA)

    years = (df.index[-1] - df.index[0]).days / 365.25
    pano = {
        "bars": len(df),
        "years_covered": round(years, 2),
        "close_min": round(float(c.min()), 6),
        "close_max": round(float(c.max()), 6),
        "atrp_mean_%": round(float(atrp.mean()), 3),
        "atrp_median_%": round(float(atrp.median()), 3),
        "bbw_median": round(float(bbw.median()), 6),
    }
    pano_df = pd.DataFrame([pano])
    pano_df.to_csv(os.path.join(pano_dir, f"{label.replace('/','_')}.csv"), index=False)

# ---------- Utility diagnostica ----------
def why_failed(m: dict) -> list[str]:
    if not m or "OOS" not in m:
        return ["no_metrics"]
    o = m["OOS"]
    reasons = []
    # trades
    tr = o.get("trades", 0)
    if tr < MIN_TRADES_OOS:
        reasons.append(f"trades<{MIN_TRADES_OOS} ({tr})")
    # pf
    pf = o.get("pf", float("nan"))
    if (not math.isfinite(pf)) or pf < MIN_PF_OOS:
        shown = 0.0 if not math.isfinite(pf) else pf
        reasons.append(f"pf<{MIN_PF_OOS:.2f} ({shown:.2f})")
    # expectancy
    expv = o.get("exp", 0.0)
    if expv <= 0.0:
        reasons.append(f"exp<=0 ({expv:.3f})")
    # max drawdown in R
    dd = o.get("maxdd", float("inf"))
    if dd > MAX_DD_OOS_R:
        reasons.append(f"dd>{MAX_DD_OOS_R:.1f} ({dd:.1f})")
    return reasons or ["unknown"]

def reasons_for(res: dict) -> str:
    """Genera stringa riassuntiva dei motivi di fallimento per tutte le strategie"""
    parts = []
    for strategy_name, strategy_result in res.items():
        if strategy_result:
            reasons = why_failed(strategy_result)
            if reasons and reasons not in (["no_metrics"], ["unknown"]):
                parts.append(f"{strategy_name}:" + "|".join(reasons))
    return "; ".join(parts) if parts else "—"

# ---------- MAIN ----------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info("--- ODIN Retrain MULTI-STRATEGIA ---")
    tg("ODIN: retrain multi-strategia avviato…")

    # 1) costruisci universo
    universe = build_universe()
    if not universe:
        log.error("Universe vuoto. Controlla RETRAIN_UNIVERSE_SOURCE/ROSTER/ONLY.")
        return

    results = {}
    passes = 0
    total = 0
    
    for label in universe:
        symbol, _kind = ASSET_PORTFOLIO[label]
        total += 1
        try:
            log.info(f"[{label}] cache TwelveData…")
            df = get_data(label, symbol)
            save_panorama(label, df)

            # taglia a IS+OOS per selezione
            cut = df.index[-1] - pd.DateOffset(months=IS_MONTHS+OOS_MONTHS)
            df_cut = df[df.index >= cut].copy()
            if len(df_cut) < 400:
                log.info(f"[{label}] dati insufficienti per IS/OOS (len={len(df_cut)})")
                continue
            
            try:
                res = choose_on_is_then_validate(label, df_cut)
            except Exception:
                import traceback
                log.error("choose_on_is_then_validate() crash:\n" + traceback.format_exc())
                raise
            results[label] = res

            # Decisione OOS (se almeno una strategia passa)
            val_ok = False
            passed_strategies = []
            failed_reasons = []
            
            # Controlla ogni strategia
            for strategy_name, strategy_result in res.items():
                if strategy_result:
                    if passes_oos(strategy_result):
                        val_ok = True
                        passed_strategies.append(strategy_name)
                    else:
                        failed_reasons.append(f"{strategy_name}:" + "|".join(why_failed(strategy_result)))

            if val_ok:
                passes += 1
                log.info(f"[{label}] OOS PASSED - Strategie attive: {', '.join(passed_strategies)}")
            else:
                log.info(f"[{label}] OOS FAILED - {'; '.join(failed_reasons) if failed_reasons else 'no valid strategies'}")

            time.sleep(0.3)
        except Exception as e:
            import traceback
            log.error(f"[{label}] errore retrain: {e!r}\n" + traceback.format_exc())

    # 2) update params/current.json per TUTTE le strategie
    cur = load_params_current()
    
    # Inizializza sezioni per tutte le strategie
    for strategy_name in STRATEGY_CONFIG.keys():
        cur.setdefault(strategy_name, {}).setdefault("per_asset", {})
    cur.setdefault("STATUS", {})

    # Aggiorna parametri per ogni asset e strategia
    for label, res in results.items():
        updated_strategies = []
        
        # Processa ogni strategia trovata per questo asset
        for strategy_name, strategy_result in res.items():
            if strategy_result and passes_oos(strategy_result):
                cur[strategy_name]["per_asset"][label] = strategy_result["params"]
                updated_strategies.append(strategy_name)
                log.info(f"[{label}] {strategy_name}: parametri aggiornati")
        
        # Status: ACTIVE se almeno una strategia passa OOS
        cur["STATUS"][label] = "ACTIVE" if updated_strategies else "PAUSED"
        
        if updated_strategies:
            log.info(f"[{label}] ACTIVE con strategie: {', '.join(updated_strategies)}")
        else:
            log.info(f"[{label}] PAUSED - nessuna strategia passa OOS")

    if not ODIN_RETRAIN_DRYRUN:
        arch_dir = os.path.join(PARAMS_DIR, "archive")
        os.makedirs(arch_dir, exist_ok=True)
        if os.path.exists(params_current_path()):
            shutil.copy2(params_current_path(), params_backup_path(ts))
        atomic_write_json(params_current_path(), cur)
        log.info(f"[WRITE] params/current.json aggiornato.")
    else:
        log.info("[DRYRUN] update saltato (ODIN_RETRAIN_DRYRUN=1)")

    # 3) report Telegram
    def fmt(m):
        if not m: return "—"
        o = m["OOS"]
        pf = o['pf'] if math.isfinite(o['pf']) else 999.0
        return f"tr={o['trades']}, PF={pf:.2f}, Exp={o['exp']:.3f}, DD={o['maxdd']:.1f}"

    # --- dump diagnostico su file (JSON + testo) ---
    try:
        dump_json = {}
        for label, res in results.items():
            # Crea entry per questo asset con tutte le strategie
            asset_data = {"status": cur["STATUS"].get(label, "ACTIVE")}
            
            # Aggiungi risultati per ogni strategia
            for strategy_name in STRATEGY_CONFIG.keys():
                asset_data[strategy_name] = res.get(strategy_name)
            
            dump_json[label] = asset_data

        os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

        out_json = os.getenv("AUTO_GATE_RETRAIN_JSON", os.path.join(PARAMS_DIR, "retrain_metrics.json"))
        out_txt  = os.getenv("AUTO_GATE_RETRAIN_TEXT", os.path.join(BASE_DIR, "logs", "last_retrain.txt"))

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dump_json, f, indent=2, ensure_ascii=False)

        def _fmt_one(lbl, res):
            def _fmt_m(m):
                if not m: return "—"
                o = m["OOS"]; pf = o['pf'] if math.isfinite(o['pf']) else 999.0
                return f"OOS: tr={o['trades']}, PF={pf:.2f}, Exp={o['exp']:.3f}, DD={o['maxdd']:.1f}"
            def _reasons(m):
                if not m: return "—"
                bad = []
                o = m["OOS"]
                if o["trades"] < MIN_TRADES_OOS: bad.append(f"tr<{MIN_TRADES_OOS}")
                if (o["pf"] < MIN_PF_OOS) and math.isfinite(o["pf"]): bad.append(f"pf<{MIN_PF_OOS}")
                if o["exp"] <= 0.0: bad.append("exp<=0")
                if o["maxdd"] > MAX_DD_OOS_R: bad.append(f"dd>{MAX_DD_OOS_R}")
                return ",".join(bad) if bad else "—"
            
            lines = []
            lines.append(f"{lbl} [{cur['STATUS'].get(lbl,'ACTIVE')}]")
            
            # Mostra tutte le strategie
            for strategy_name in STRATEGY_CONFIG.keys():
                strategy_result = res.get(strategy_name)
                lines.append(f"  {strategy_name:20}: {_fmt_m(strategy_result)}")
            
            # Riassunto motivi fallimento
            why_parts = []
            for strategy_name, strategy_result in res.items():
                if strategy_result:
                    reasons = _reasons(strategy_result)
                    if reasons != "—":
                        why_parts.append(f"{strategy_name}:{reasons}")
            
            lines.append(f"  {'WHY':20}: {'; '.join(why_parts) if why_parts else '—'}")
            return "\n".join(lines)

        with open(out_txt, "w", encoding="utf-8") as f:
            for lbl, res in results.items():
                f.write(_fmt_one(lbl, res) + "\n")

        log.info(f"[DUMP] retrain metrics -> {out_json} ; summary -> {out_txt}")
    except Exception as e:
        log.warning(f"[DUMP] fallito: {e!r}")

    lines = []
    for label, res in results.items():
        st = cur["STATUS"].get(label, "ACTIVE")
        
        # Conta strategie attive per questo asset
        active_strategies = []
        for strategy_name, strategy_result in res.items():
            if strategy_result and passes_oos(strategy_result):
                active_strategies.append(strategy_name)
        
        # Formato compatto per Telegram
        strategy_summary = f"{len(active_strategies)}/{len(res)} active"
        if active_strategies:
            strategy_summary += f" ({','.join(active_strategies[:3])}{'...' if len(active_strategies) > 3 else ''})"
        
        lines.append(f"{label}: {st} | {strategy_summary}")

    if lines:
        # Telegram: spezza in chunk per evitare limiti
        chunk = []
        acc = 0
        for ln in lines:
            if acc + len(ln) + 1 > 3500:
                tg("ODIN retrain multi-strategia (IS/OOS):\n" + "\n".join(chunk))
                chunk = [ln]
                acc = len(ln) + 1
            else:
                chunk.append(ln)
                acc += len(ln) + 1
        if chunk:
            tg("ODIN retrain multi-strategia (IS/OOS):\n" + "\n".join(chunk))

    log.info(f"Retrain MULTI-STRATEGIA completato. Assets processati: {total}, passed: {passes}, failed: {total-passes}")
    log.info(f"Strategie supportate: {list(STRATEGY_CONFIG.keys())}")

if __name__ == "__main__":
    main()