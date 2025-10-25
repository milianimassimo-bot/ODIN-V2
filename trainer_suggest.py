# ============================================================
# ODIN - Trainer Suggestor v4.0 MULTI-STRATEGIA (2025)
# ------------------------------------------------------------
# NUOVO: Supporta tutte le 10 strategie ODIN con logica avanzata
# Integra con retrain_metrics.json per decisioni intelligenti
# Algoritmo adattivo per ottimizzazione parametri dinamica
# ============================================================

import os, json, sqlite3, statistics, requests, time, math
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from collections import defaultdict

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

load_dotenv()

# --- Telegram setup ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def tg_send(msg: str):
    """Invia messaggio Telegram (3 retry, non blocca)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "disable_web_page_preview": True}
    for i in range(3):
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            return
        except Exception:
            time.sleep(0.8 * (2 ** i))

# --- Percorsi principali ---
ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
PARAMS = os.path.join(ROOT, "params")
DB_PATH = os.path.join(LOGS, "odin_ml.db")
CURRENT_PARAMS = os.getenv("ODIN_PARAMS_FILE", os.path.join(PARAMS, "current.json"))
RETRAIN_METRICS = os.getenv("AUTO_GATE_RETRAIN_JSON", os.path.join(PARAMS, "retrain_metrics.json"))
SOFTENERS_OUT = os.path.join(PARAMS, "softeners_plan.json")
REPORT_OUT = os.path.join(LOGS, f"trainer_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt")

# --- Parametri di training ---
N_MIN = int(os.getenv("TRAINER_MIN_TRADES", "4"))
TTL_MIN = int(os.getenv("SOFTENER_TTL_MIN", "720"))  # 12h
DECAY_DAYS = int(os.getenv("TRAINER_DECAY_DAYS", "30"))
STEP_STRICT = float(os.getenv("TRAINER_STEP_STRICT", "0.10"))
STEP_RELAX = float(os.getenv("TRAINER_STEP_RELAX", "0.10"))
CONFIDENCE_THRESHOLD = float(os.getenv("TRAINER_CONFIDENCE_THRESHOLD", "0.7"))

# ============================================================
# CONFIGURAZIONE MULTI-STRATEGIA
# ============================================================

# Definizione parametri ottimizzabili per ogni strategia
STRATEGY_PARAMS = {
    "WEEKLY": {
        "pullback_atr_max": {"min": 0.8, "max": 3.5, "default": 1.4, "step": 0.1},
        "atr_pct_max": {"min": 0.6, "max": 4.0, "default": 2.0, "step": 0.1}
    },
    "BB_MR_D1": {
        "adx_max": {"min": 12.0, "max": 35.0, "default": 18.0, "step": 1.0},
        "r2_max": {"min": 0.10, "max": 0.70, "default": 0.20, "step": 0.02}
    },
    "EMA_Trend_D1": {
        "ema_fast": {"min": 8, "max": 25, "default": 12, "step": 1},
        "ema_slow": {"min": 20, "max": 50, "default": 26, "step": 2},
        "atr_mult": {"min": 1.0, "max": 3.0, "default": 2.0, "step": 0.1}
    },
    "Keltner_Trend_D1": {
        "keltner_period": {"min": 15, "max": 35, "default": 20, "step": 1},
        "keltner_mult": {"min": 1.5, "max": 3.0, "default": 2.0, "step": 0.1},
        "trend_strength": {"min": 0.3, "max": 0.8, "default": 0.5, "step": 0.05}
    },
    "VolTarget_Trend_D1": {
        "vol_lookback": {"min": 10, "max": 30, "default": 20, "step": 2},
        "vol_target": {"min": 0.8, "max": 2.0, "default": 1.2, "step": 0.1},
        "trend_filter": {"min": 0.4, "max": 0.9, "default": 0.6, "step": 0.05}
    },
    "EMA_Pullback_H4": {
        "ema_period": {"min": 15, "max": 40, "default": 21, "step": 1},
        "pullback_pct": {"min": 0.3, "max": 1.2, "default": 0.618, "step": 0.05},
        "rsi_threshold": {"min": 25, "max": 45, "default": 35, "step": 2}
    },
    "Keltner_MR_H4": {
        "keltner_period": {"min": 15, "max": 35, "default": 20, "step": 1},
        "mean_revert_mult": {"min": 1.8, "max": 3.2, "default": 2.5, "step": 0.1},
        "rsi_oversold": {"min": 20, "max": 35, "default": 30, "step": 1}
    },
    "Donchian_BO_D1": {
        "donchian_period": {"min": 15, "max": 35, "default": 20, "step": 1},
        "breakout_mult": {"min": 1.2, "max": 2.5, "default": 1.5, "step": 0.1},
        "volume_filter": {"min": 0.8, "max": 1.5, "default": 1.0, "step": 0.05}
    },
    "BB_Squeeze_BO_D1": {
        "bb_period": {"min": 15, "max": 30, "default": 20, "step": 1},
        "bb_std": {"min": 1.8, "max": 2.5, "default": 2.0, "step": 0.1},
        "squeeze_threshold": {"min": 0.1, "max": 0.4, "default": 0.2, "step": 0.02}
    },
    "Regime_Switcher_D1": {
        "regime_lookback": {"min": 15, "max": 40, "default": 25, "step": 2},
        "volatility_threshold": {"min": 0.5, "max": 1.5, "default": 0.8, "step": 0.05},
        "trend_momentum": {"min": 0.3, "max": 0.8, "default": 0.5, "step": 0.05}
    }
}

# Mapping nomi alternativi
STRATEGY_ALIASES = {
    "BB_MR": "BB_MR_D1",
    "BBMR": "BB_MR_D1"
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _read_json(p, default=None):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def _write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _clamp(x, lo, hi):
    try:
        return max(lo, min(hi, float(x)))
    except:
        return lo

def _now_utc():
    return datetime.now(timezone.utc)

def normalize_strategy_name(strat_name):
    """Normalizza i nomi delle strategie per consistenza."""
    return STRATEGY_ALIASES.get(strat_name, strat_name)

# ============================================================
# 1) CARICA OUTCOMES DAL DB + RETRAIN METRICS
# ============================================================

def load_outcomes():
    """Carica outcomes dal database ML."""
    if not os.path.exists(DB_PATH):
        print(f"⚠️ DB non trovato: {DB_PATH}")
        return []
    
    con = sqlite3.connect(DB_PATH)
    c = con.cursor()
    rows = c.execute("""
        SELECT ts_utc, asset, strategy, pnl_abs, exit_reason
        FROM outcomes
        WHERE pnl_abs IS NOT NULL
        ORDER BY ts_utc DESC
    """).fetchall()
    con.close()
    return rows

def load_retrain_metrics():
    """Carica metriche dal sistema retrain multi-strategia."""
    return _read_json(RETRAIN_METRICS, {})

# ============================================================
# 2) ANALISI AVANZATA CON MULTIPLE DATA SOURCES
# ============================================================

def compute_advanced_metrics(outcomes, retrain_data):
    """Calcola metriche avanzate combinando outcomes ML e retrain data."""
    stats = {}
    now = _now_utc()
    
    # 1) Processa outcomes dal database ML
    for ts, asset, strat, pnl, reason in outcomes:
        try:
            strat = normalize_strategy_name(strat)
            pnl = float(pnl)
            t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            age_days = (now - t).days
            w = max(0.1, 1.0 - (age_days / DECAY_DAYS))
        except Exception:
            continue
            
        k = (asset, strat)
        bucket = stats.setdefault(k, {
            "ml_pnl": [], "ml_weights": [], "ml_reasons": [],
            "retrain_metrics": None, "confidence": 0.0
        })
        bucket["ml_pnl"].append(pnl)
        bucket["ml_weights"].append(w)
        bucket["ml_reasons"].append(reason)
    
    # 2) Integra dati retrain
    for asset, asset_data in retrain_data.items():
        if not isinstance(asset_data, dict):
            continue
            
        for strat_name in STRATEGY_PARAMS.keys():
            strat_data = asset_data.get(strat_name)
            if strat_data and isinstance(strat_data, dict):
                k = (asset, strat_name)
                bucket = stats.setdefault(k, {
                    "ml_pnl": [], "ml_weights": [], "ml_reasons": [],
                    "retrain_metrics": None, "confidence": 0.0
                })
                bucket["retrain_metrics"] = strat_data
    
    # 3) Calcola metriche finali
    for k, bucket in stats.items():
        # ML metrics
        if bucket["ml_pnl"]:
            try:
                wavg = sum(p * w for p, w in zip(bucket["ml_pnl"], bucket["ml_weights"])) / sum(bucket["ml_weights"])
            except ZeroDivisionError:
                wavg = statistics.mean(bucket["ml_pnl"])
            bucket["ml_avg"] = wavg
            bucket["ml_n"] = len(bucket["ml_pnl"])
            bucket["ml_win"] = len([x for x in bucket["ml_pnl"] if x > 0])
            bucket["ml_winrate"] = bucket["ml_win"] / bucket["ml_n"] if bucket["ml_n"] else 0
        else:
            bucket["ml_avg"] = 0.0
            bucket["ml_n"] = 0
            bucket["ml_winrate"] = 0.0
        
        # Retrain metrics
        retrain = bucket["retrain_metrics"]
        if retrain and "OOS" in retrain:
            oos = retrain["OOS"]
            bucket["retrain_pf"] = oos.get("pf", 0.0)
            bucket["retrain_trades"] = oos.get("trades", 0)
            bucket["retrain_exp"] = oos.get("exp", 0.0)
            bucket["retrain_dd"] = oos.get("maxdd", 0.0)
        else:
            bucket["retrain_pf"] = 0.0
            bucket["retrain_trades"] = 0
            bucket["retrain_exp"] = 0.0
            bucket["retrain_dd"] = 0.0
        
        # Confidence score (combina ML e retrain)
        ml_confidence = min(1.0, bucket["ml_n"] / 10.0) if bucket["ml_n"] > 0 else 0.0
        retrain_confidence = min(1.0, bucket["retrain_trades"] / 20.0) if bucket["retrain_trades"] > 0 else 0.0
        bucket["confidence"] = (ml_confidence + retrain_confidence) / 2.0
    
    return stats

# ============================================================
# 3) ALGORITMO INTELLIGENTE DI PARAMETER TUNING
# ============================================================

def get_current_params(cur, asset, strategy):
    """Estrae parametri correnti per asset/strategia."""
    strat_data = cur.get(strategy, {})
    asset_params = strat_data.get("per_asset", {}).get(asset, {})
    
    # Fallback ai default se non trovati
    defaults = {}
    if strategy in STRATEGY_PARAMS:
        for param, config in STRATEGY_PARAMS[strategy].items():
            defaults[param] = asset_params.get(param, config["default"])
    
    return defaults

def calculate_adjustment_factor(bucket):
    """Calcola fattore di aggiustamento basato su performance e confidence."""
    ml_performance = bucket["ml_avg"] if bucket["ml_n"] >= N_MIN else 0.0
    retrain_performance = bucket["retrain_pf"] - 1.0 if bucket["retrain_trades"] >= N_MIN else 0.0
    
    # Combina performance ML e retrain con pesi
    ml_weight = min(1.0, bucket["ml_n"] / 10.0)
    retrain_weight = min(1.0, bucket["retrain_trades"] / 15.0)
    
    if ml_weight + retrain_weight == 0:
        return 0.0  # Dati insufficienti
    
    combined_performance = (ml_performance * ml_weight + retrain_performance * retrain_weight) / (ml_weight + retrain_weight)
    
    # Scala per confidence
    adjustment = combined_performance * bucket["confidence"]
    
    # Limita aggiustamenti estremi
    return _clamp(adjustment, -0.5, 0.5)

def build_intelligent_softeners(stats, cur):
    """Costruisce piano softeners intelligente per tutte le strategie."""
    plan = {}
    lines = []
    
    for (asset, strategy), bucket in stats.items():
        if strategy not in STRATEGY_PARAMS:
            continue
            
        # Verifica confidence minima
        if bucket["confidence"] < CONFIDENCE_THRESHOLD:
            lines.append(f"⚪️ {asset}/{strategy}: confidence {bucket['confidence']:.2f} < {CONFIDENCE_THRESHOLD}, skip")
            continue
        
        # Calcola aggiustamento
        adj_factor = calculate_adjustment_factor(bucket)
        if abs(adj_factor) < 0.05:  # Soglia minima per cambiamenti
            continue
        
        # Applica aggiustamenti ai parametri
        current_params = get_current_params(cur, asset, strategy)
        new_params = {}
        param_changes = []
        
        for param_name, param_config in STRATEGY_PARAMS[strategy].items():
            current_val = current_params.get(param_name, param_config["default"])
            
            # Calcola nuovo valore
            if adj_factor > 0:  # Performance buona -> relax
                new_val = current_val * (1 + STEP_RELAX * adj_factor)
            else:  # Performance cattiva -> tighten
                new_val = current_val * (1 + STEP_STRICT * adj_factor)
            
            # Clamp ai limiti
            new_val = _clamp(new_val, param_config["min"], param_config["max"])
            
            # Arrotonda al step
            step = param_config["step"]
            new_val = round(new_val / step) * step
            
            if abs(new_val - current_val) > step * 0.1:  # Cambio significativo
                new_params[param_name] = {"set": new_val, "ttl_min": TTL_MIN}
                param_changes.append(f"{param_name}: {current_val:.3f}→{new_val:.3f}")
        
        if new_params:
            # Aggiungi al piano
            if strategy not in plan:
                plan[strategy] = {"per_asset": {}}
            plan[strategy]["per_asset"][asset] = new_params
            
            # Log del cambiamento
            direction = "🟢 RELAX" if adj_factor > 0 else "🔻 TIGHTEN"
            ml_info = f"ML: n={bucket['ml_n']}, avg={bucket['ml_avg']:.3f}" if bucket['ml_n'] > 0 else "ML: none"
            retrain_info = f"OOS: tr={bucket['retrain_trades']}, pf={bucket['retrain_pf']:.2f}" if bucket['retrain_trades'] > 0 else "OOS: none"
            
            lines.append(f"{direction} {asset}/{strategy} (conf={bucket['confidence']:.2f})")
            lines.append(f"  {ml_info} | {retrain_info}")
            lines.append(f"  Changes: {', '.join(param_changes)}")
    
    return plan, lines

# ============================================================
# 4) SUMMARY AVANZATO MULTI-STRATEGIA
# ============================================================

def summarize_advanced_portfolio(stats):
    """Summary avanzato con dati ML + retrain."""
    if not stats:
        return ["(nessun dato)"]
    
    # Statistiche globali
    total_ml_trades = sum(v["ml_n"] for v in stats.values())
    total_retrain_combinations = sum(1 for v in stats.values() if v["retrain_trades"] > 0)
    avg_confidence = statistics.mean([v["confidence"] for v in stats.values()]) if stats else 0.0
    
    # Per strategia
    per_strategy = defaultdict(lambda: {"ml_trades": 0, "ml_pnl_sum": 0.0, "retrain_active": 0, "retrain_pf_sum": 0.0})
    
    for (asset, strategy), bucket in stats.items():
        s = per_strategy[strategy]
        s["ml_trades"] += bucket["ml_n"]
        s["ml_pnl_sum"] += bucket["ml_avg"] * bucket["ml_n"]
        if bucket["retrain_trades"] > 0:
            s["retrain_active"] += 1
            s["retrain_pf_sum"] += bucket["retrain_pf"]
    
    # Top performers
    strategy_performance = []
    for strategy, data in per_strategy.items():
        ml_avg = data["ml_pnl_sum"] / data["ml_trades"] if data["ml_trades"] > 0 else 0.0
        retrain_avg = data["retrain_pf_sum"] / data["retrain_active"] if data["retrain_active"] > 0 else 1.0
        combined_score = (ml_avg + (retrain_avg - 1.0)) / 2.0
        strategy_performance.append((strategy, combined_score, data))
    
    strategy_performance.sort(key=lambda x: x[1], reverse=True)
    
    lines = [
        f"📊 MULTI-STRATEGY ANALYSIS",
        f"ML trades: {total_ml_trades} | Retrain combinations: {total_retrain_combinations}",
        f"Avg confidence: {avg_confidence:.2f} | Strategies: {len(per_strategy)}",
        "",
        "🏆 STRATEGY RANKING:"
    ]
    
    for i, (strategy, score, data) in enumerate(strategy_performance[:5]):
        ml_info = f"ML:{data['ml_trades']}tr" if data['ml_trades'] > 0 else "ML:none"
        retrain_info = f"OOS:{data['retrain_active']}active" if data['retrain_active'] > 0 else "OOS:none"
        lines.append(f"{i+1}. {strategy}: score={score:.3f} ({ml_info}, {retrain_info})")
    
    return lines

# ============================================================
# 5) MAIN FUNCTION
# ============================================================

def main():
    tg_send("🧠 ODIN Trainer v4.0 MULTI-STRATEGIA — avvio analisi avanzata…")
    
    try:
        # Carica dati
        outcomes = load_outcomes()
        retrain_data = load_retrain_metrics()
        
        if not outcomes and not retrain_data:
            tg_send("ℹ️ Trainer: nessun dato ML o retrain trovato.")
            return
        
        # Analisi avanzata
        stats = compute_advanced_metrics(outcomes, retrain_data)
        cur = load_baseline()
        plan, lines = build_intelligent_softeners(stats, cur)
        
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Salva piano
        out = {
            "generated_utc": ts, 
            "version": "4.0_multi_strategy",
            "notes": "Intelligent multi-strategy parameter optimization",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "strategies_supported": list(STRATEGY_PARAMS.keys()),
            **plan
        }
        _write_json(SOFTENERS_OUT, out)
        
        # Summary avanzato
        summary_lines = summarize_advanced_portfolio(stats)
        
        # Report completo
        with open(REPORT_OUT, "w", encoding="utf-8") as f:
            f.write(f"=== ODIN Trainer v4.0 MULTI-STRATEGY REPORT {ts} ===\n")
            f.write(f"DB: {DB_PATH}\nRetrain: {RETRAIN_METRICS}\n\n")
            
            f.write("[ADVANCED SUMMARY]\n")
            for s in summary_lines:
                f.write(s + "\n")
            
            f.write("\n[DETAILED ANALYSIS]\n")
            for (asset, strategy), bucket in sorted(stats.items()):
                f.write(f"\n{asset}/{strategy}:\n")
                f.write(f"  ML: n={bucket['ml_n']}, avg={bucket['ml_avg']:.3f}, wr={bucket['ml_winrate']:.2f}\n")
                f.write(f"  Retrain: trades={bucket['retrain_trades']}, pf={bucket['retrain_pf']:.2f}, exp={bucket['retrain_exp']:.3f}\n")
                f.write(f"  Confidence: {bucket['confidence']:.2f}\n")
            
            f.write("\n[OPTIMIZATION DECISIONS]\n")
            if lines:
                for ln in lines: 
                    f.write(ln + "\n")
            else:
                f.write("Nessuna ottimizzazione proposta (confidence o performance insufficienti).\n")
        
        # Telegram report
        header = f"🧠 ODIN Trainer v4.0 MULTI ({ts})"
        summary_text = "\n".join(summary_lines[:6])
        
        if lines:
            changes_preview = "\n".join([l for l in lines if l.startswith(("🟢", "🔻"))][:8])
            tg_send(f"{header}\n{summary_text}\n\n📝 OPTIMIZATIONS ({len([l for l in lines if l.startswith(('🟢', '🔻'))])} changes):\n{changes_preview}")
        else:
            tg_send(f"{header}\n{summary_text}\n\n✅ No optimizations needed (all strategies performing well)")
        
        print(f"✅ Advanced report: {REPORT_OUT}")
        print(f"🧠 Multi-strategy softeners: {SOFTENERS_OUT}")
        print(f"📊 Analyzed {len(stats)} asset/strategy combinations")
        print(f"🎯 Generated {len([l for l in lines if l.startswith(('🟢', '🔻'))])} parameter optimizations")
        
    except Exception as e:
        error_msg = f"❌ Trainer v4.0 ERROR: {type(e).__name__}: {e}"
        tg_send(error_msg)
        print(error_msg)
        raise

def load_baseline():
    """Carica parametri baseline da current.json."""
    return _read_json(CURRENT_PARAMS, {})

if __name__ == "__main__":
    main()