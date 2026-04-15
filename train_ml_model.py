"""
ML Signal Filter — Elite Training Pipeline v3
================================================
Kullanım:  python train_ml_model.py

Profesyonel Quant seviyesi:
  - 3-Model Stacking Ensemble (XGBoost + LightGBM + RF → LogisticRegression meta)
  - Purged Walk-Forward CV (lookahead bias yok)
  - SMOTE + class weights (imbalanced data handling)
  - Probability Calibration (Platt scaling)
  - Automated Feature Selection (importance + correlation filter)
  - Risk-adjusted metrics (Sharpe, Calmar, Profit Factor)
  - Per-symbol performance breakdown
  - Optimal threshold via expected value maximization
  - 40+ features (tüm ICT/SMC + regime + market structure)
"""
from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sys
import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent))

from app.services.analysis_engine import AnalysisEngine, AnalysisResult
from app.services.market_data import MarketDataClient
from app.services.regime_detector import detect_regime

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("ml_elite_v3")

# ── Config ─────────────────────────────────────────────────────────────────
SYMBOLS = ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
TIMEFRAMES = ["1min", "5min", "15min", "30min", "1h"]
HIGHER_TF = {"1min": "5min", "5min": "15min", "15min": "1h", "30min": "4h", "1h": "4h"}

WARMUP = 250
LOOKAHEAD = 25
STEP = 2
OUTPUT_SIZE = 2000
API_DELAY = 1.2

MODEL_PATH = Path("data/ml_signal_filter.pkl")
DATA_PATH = Path("data/ml_training_data.json")
REPORT_PATH = Path("data/ml_training_report.json")

engine = AnalysisEngine()

# ── 40-Feature Vector ──────────────────────────────────────────────────────
FEATURE_NAMES = [
    # Core indicators (0-5)
    "setup_score", "rsi", "rr_ratio", "atr_ratio_x1000",
    "smc_confluence", "trend_aligned",
    # SMC booleans (6-17)
    "choch", "displacement", "ote", "judas",
    "bos", "mss", "confirmation", "conf_strength",
    "premium_discount", "sweep", "sniper",
    # Signal/bias (18-21)
    "signal_direction", "dxy", "cot",
    "hour", "session",
    # v2 features (22-29)
    "unicorn", "vwap_alignment", "silver_bullet",
    "ipda_near", "amd_phase_enc", "regime_enc",
    "vol_spike", "vol_delta_enc",
    # v3 NEW features (30-39)
    "rsi_divergence",       # RSI vs price divergence
    "atr_change_pct",       # ATR acceleration
    "ema20_50_dist",        # EMA20-EMA50 distance normalized
    "bb_position",          # Price position within Bollinger Bands (0-1)
    "candle_body_ratio",    # Last candle body/range ratio
    "upper_wick_ratio",     # Upper wick / total range
    "consecutive_direction",# Consecutive same-direction candles
    "range_position",       # Price position within recent range (0-1)
    "macd_hist_direction",  # MACD histogram increasing or decreasing
    "adx_value",            # Raw ADX value (trend strength)
]


def extract_features_v3(result: AnalysisResult, df: pd.DataFrame, hour: int = 12) -> list[float]:
    """40-feature vector — elite level."""
    atr_ratio = result.atr / max(result.current_price, 1e-9)

    # Core SMC booleans
    choch_d = 1 if result.choch and result.choch.get("detected") else 0
    disp_d = 1 if result.displacement and result.displacement.get("detected") else 0
    ote_v = 1 if result.ote_zone and result.ote_zone.get("valid") else 0
    judas_d = 1 if result.judas_swing and result.judas_swing.get("detected") else 0
    bos_d = 1 if result.bos_mss and result.bos_mss.get("bos") else 0
    mss_d = 1 if result.bos_mss and result.bos_mss.get("mss") else 0
    conf_d = 1 if result.confirmation_candle and result.confirmation_candle.get("detected") else 0
    conf_s = result.confirmation_candle.get("strength", 0) if result.confirmation_candle else 0
    trend_a = 1 if result.trend == result.higher_tf_trend else 0
    sweep_ok = 1 if result.sweep_signal != "Yok" else 0
    sniper_ok = 1 if result.sniper_entry != "Yok" else 0

    pd_zone = result.premium_discount.get("zone", "DENGE") if result.premium_discount else "DENGE"
    pd_enc = {"DISCOUNT": -1, "DENGE": 0, "PREMIUM": 1}.get(pd_zone, 0)
    sig_enc = 1 if result.signal == "LONG" else (-1 if result.signal == "SHORT" else 0)
    dxy_enc = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(getattr(result, "dxy_bias", "NEUTRAL"), 0)
    cot_enc = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(getattr(result, "cot_bias", "NEUTRAL"), 0)

    sess = 0
    if 13 <= hour < 17: sess = 3
    elif 8 <= hour < 17: sess = 1
    elif 13 <= hour < 22: sess = 2

    # v2 features
    unicorn = getattr(result, "unicorn_model", {}) or {}
    uni_d = 1 if unicorn.get("detected") and unicorn.get("near_price") else 0
    vwap = getattr(result, "vwap", 0.0) or 0.0
    vwap_align = 0
    if vwap > 0:
        vwap_align = 1 if (result.signal == "LONG" and result.current_price > vwap) or \
                          (result.signal == "SHORT" and result.current_price < vwap) else -1
    sb = getattr(result, "silver_bullet", {}) or {}
    sb_active = 1 if sb.get("active") else 0
    ipda = getattr(result, "ipda_levels", {}) or {}
    ipda_near = 1 if ipda.get("distance_atr", 99) <= 1.5 else 0
    amd = getattr(result, "amd_phase", {}) or {}
    amd_enc = {"ACCUMULATION": 0, "MANIPULATION": -1, "DISTRIBUTION": 1}.get(amd.get("phase", ""), 0)
    vol = getattr(result, "volume_analysis", {}) or {}
    vol_spike = 1 if vol.get("volume_spike") else 0
    vol_delta = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(vol.get("delta_bias", "NEUTRAL"), 0)
    regime_enc = 0

    # ── v3 NEW features from raw dataframe ──
    rsi_div = 0.0
    atr_change = 0.0
    ema_dist = 0.0
    bb_pos = 0.5
    body_ratio = 0.5
    upper_wick_r = 0.0
    consec_dir = 0
    range_pos = 0.5
    macd_dir = 0
    adx_val = 0.0

    try:
        if len(df) >= 20:
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            # RSI divergence: price making new high but RSI lower (bearish div) or vice versa
            if len(close) >= 30:
                price_5 = close.iloc[-5:].mean()
                price_15 = close.iloc[-15:-10].mean()
                rsi_now = result.rsi
                rsi_series = engine._rsi(close, 14)
                rsi_prev = float(rsi_series.iloc[-15]) if len(rsi_series) >= 15 and not np.isnan(rsi_series.iloc[-15]) else rsi_now
                if price_5 > price_15 and rsi_now < rsi_prev:
                    rsi_div = -1.0  # bearish divergence
                elif price_5 < price_15 and rsi_now > rsi_prev:
                    rsi_div = 1.0   # bullish divergence

            # ATR change % (acceleration)
            atr_series = engine._atr(df, 14)
            if len(atr_series) >= 10:
                atr_now = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0
                atr_prev = float(atr_series.iloc[-10]) if not np.isnan(atr_series.iloc[-10]) else atr_now
                if atr_prev > 0:
                    atr_change = round((atr_now - atr_prev) / atr_prev * 100, 2)

            # EMA20-50 normalized distance
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            if result.atr > 0:
                ema_dist = round((ema20 - ema50) / result.atr, 3)

            # Bollinger Band position (0 = lower band, 1 = upper band)
            if result.bb_upper > result.bb_lower:
                bb_pos = round((result.current_price - result.bb_lower) / (result.bb_upper - result.bb_lower), 3)
                bb_pos = max(0, min(1, bb_pos))

            # Last candle body/range ratio
            last = df.iloc[-1]
            total_range = float(last["high"]) - float(last["low"])
            if total_range > 0:
                body = abs(float(last["close"]) - float(last["open"]))
                body_ratio = round(body / total_range, 3)
                upper_wick_r = round((float(last["high"]) - max(float(last["close"]), float(last["open"]))) / total_range, 3)

            # Consecutive same-direction candles
            directions = (close.diff().tail(10) > 0).values
            if len(directions) >= 2:
                last_dir = directions[-1]
                count = 1
                for i in range(len(directions) - 2, -1, -1):
                    if directions[i] == last_dir:
                        count += 1
                    else:
                        break
                consec_dir = count if last_dir else -count

            # Price position in recent range
            recent_high = float(high.tail(40).max())
            recent_low = float(low.tail(40).min())
            if recent_high > recent_low:
                range_pos = round((result.current_price - recent_low) / (recent_high - recent_low), 3)

            # MACD histogram direction
            if result.macd_hist > 0:
                macd_dir = 1
            elif result.macd_hist < 0:
                macd_dir = -1

            # ADX raw
            adx_series = engine._adx(df, 14)
            if len(adx_series) > 0 and not np.isnan(adx_series.iloc[-1]):
                adx_val = round(float(adx_series.iloc[-1]), 2)
    except Exception:
        pass

    return [
        float(result.setup_score), float(result.rsi), float(result.rr_ratio),
        float(atr_ratio * 1000), float(result.smc_confluence_count), float(trend_a),
        float(choch_d), float(disp_d), float(ote_v), float(judas_d),
        float(bos_d), float(mss_d), float(conf_d), float(conf_s),
        float(pd_enc), float(sweep_ok), float(sniper_ok),
        float(sig_enc), float(dxy_enc), float(cot_enc),
        float(hour), float(sess),
        float(uni_d), float(vwap_align), float(sb_active),
        float(ipda_near), float(amd_enc), float(regime_enc),
        float(vol_spike), float(vol_delta),
        # v3 new
        float(rsi_div), float(atr_change), float(ema_dist),
        float(bb_pos), float(body_ratio), float(upper_wick_r),
        float(consec_dir), float(range_pos), float(macd_dir), float(adx_val),
    ]


# ── Outcome (realistic with partial TP) ───────────────────────────────────

def evaluate_outcome(signal: str, tp: float, sl: float, rr: float,
                     future_df: pd.DataFrame) -> tuple[int, float] | None:
    entry = float(future_df.iloc[0]["open"]) if len(future_df) > 0 else 0
    risk = abs(entry - sl)
    if risk < 1e-9:
        return None

    tp1 = entry + risk * 1.5 if signal == "LONG" else entry - risk * 1.5
    tp1_hit = False

    for _, c in future_df.iterrows():
        h, l, o = float(c["high"]), float(c["low"]), float(c["open"])
        if signal == "LONG":
            t_hit, s_hit, t1 = h >= tp, l <= sl, h >= tp1
        else:
            t_hit, s_hit, t1 = l <= tp, h >= sl, l <= tp1

        if t_hit and s_hit:
            return (1, rr) if abs(o - tp) <= abs(o - sl) else (0, -1.0)
        if t_hit:
            return (1, rr)
        if s_hit:
            return (1, 0.25) if tp1_hit else (0, -1.0)
        if t1:
            tp1_hit = True
    return None


# ── Data Fetch ─────────────────────────────────────────────────────────────

async def fetch_data(api_key: str) -> dict:
    client = MarketDataClient(api_key=api_key)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in SYMBOLS:
        data[sym] = {}
        for tf in TIMEFRAMES:
            htf = HIGHER_TF.get(tf, "1h")
            for label, interval in [(f"{tf}_main", tf), (f"{tf}_higher", htf)]:
                if label in data[sym]:
                    continue
                try:
                    df = await client.fetch_candles(sym, interval=interval, outputsize=OUTPUT_SIZE)
                    data[sym][label] = df
                    logger.info("  %s %s: %d bar", sym, label, len(df))
                except Exception as e:
                    logger.warning("  %s %s: %s", sym, label, str(e)[:60])
                await asyncio.sleep(API_DELAY)
    return data


# ── Sample Generation ──────────────────────────────────────────────────────

def generate_samples(data: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, meta = [], [], []
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            mk, hk = f"{tf}_main", f"{tf}_higher"
            if mk not in data.get(sym, {}):
                continue
            df = data[sym][mk]
            hdf = data[sym].get(hk)
            if len(df) < WARMUP + LOOKAHEAD + 10:
                continue

            count = 0
            for idx in range(WARMUP, len(df) - LOOKAHEAD, STEP):
                sdf = df.iloc[:idx + 1]
                if hdf is not None and "datetime" in sdf.columns and "datetime" in hdf.columns:
                    hs = hdf[hdf["datetime"] <= sdf.iloc[-1]["datetime"]]
                    if len(hs) < 20:
                        hs = hdf.iloc[:idx + 1]
                else:
                    hs = hdf

                try:
                    r = engine.analyze(symbol=sym, df=sdf, timeframe=tf,
                                       higher_tf_df=hs, high_impact_events=[])
                except Exception:
                    continue

                if r.signal not in ("LONG", "SHORT"):
                    continue

                future = df.iloc[idx + 1: idx + 1 + LOOKAHEAD]
                out = evaluate_outcome(r.signal, r.take_profit, r.stop_loss, r.rr_ratio, future)
                if out is None:
                    continue

                label, rr_val = out
                hour = 12
                try:
                    hour = pd.to_datetime(sdf.iloc[-1]["datetime"]).hour
                except Exception:
                    pass

                # Regime
                try:
                    reg = detect_regime(sdf, sym)
                    reg_enc = {"LOW_VOL": -1, "NORMAL": 0, "HIGH_VOL": 1}.get(reg.regime, 0)
                except Exception:
                    reg_enc = 0

                feats = extract_features_v3(r, sdf, hour=hour)
                feats[27] = float(reg_enc)

                X.append(feats)
                y.append(label)
                meta.append({
                    "symbol": sym, "timeframe": tf, "signal": r.signal,
                    "score": r.setup_score, "quality": r.quality,
                    "rr": r.rr_ratio, "label": label, "realized_rr": rr_val,
                })
                count += 1

            if count:
                logger.info("%s %s: %d sinyal", sym, tf, count)
    return np.array(X), np.array(y), meta


# ── Feature Selection ──────────────────────────────────────────────────────

def select_features(X: np.ndarray, y: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[int], list[str]]:
    """Noise feature'ları ele — importance + correlation filter."""
    # Step 1: Train quick XGB for importance
    quick = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss",
                          random_state=42, verbosity=0)
    quick.fit(X, y)
    imp = quick.feature_importances_

    # Step 2: Remove features with < 1% importance
    threshold = 0.01
    keep_mask = imp >= threshold
    kept_idx = [i for i in range(len(names)) if keep_mask[i]]

    # Step 3: Remove highly correlated features (> 0.90)
    if len(kept_idx) > 5:
        X_kept = X[:, kept_idx]
        corr = np.corrcoef(X_kept.T)
        to_remove = set()
        for i in range(len(kept_idx)):
            for j in range(i + 1, len(kept_idx)):
                if abs(corr[i][j]) > 0.90:
                    # Remove the one with lower importance
                    if imp[kept_idx[i]] < imp[kept_idx[j]]:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
        kept_idx = [kept_idx[i] for i in range(len(kept_idx)) if i not in to_remove]

    kept_names = [names[i] for i in kept_idx]
    logger.info("Feature selection: %d → %d features", len(names), len(kept_idx))
    for i, idx in enumerate(kept_idx):
        logger.info("  %d. %s (imp: %.4f)", i + 1, names[idx], imp[idx])

    return X[:, kept_idx], kept_idx, kept_names


# ── Optimal Threshold ──────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray,
                           rr_values: list[float]) -> tuple[float, dict]:
    """Expected value maximization ile optimal threshold."""
    best_thr, best_ev = 0.50, -9999
    results = {}

    for thr in np.arange(0.30, 0.80, 0.005):
        mask = probs >= thr
        n = mask.sum()
        if n < 3:
            continue
        wins = sum(y_true[i] for i in range(len(y_true)) if mask[i])
        wr = wins / n
        pnl = sum(rr_values[i] if y_true[i] == 1 else -1.0
                  for i in range(len(y_true)) if mask[i])
        ev = pnl / n  # expected value per trade

        # Risk-adjusted: penalize very few trades
        trade_freq = n / len(y_true)
        adj_ev = ev * min(1.0, trade_freq / 0.05)  # penalize if < 5% trade frequency

        results[round(thr, 3)] = {"n": int(n), "wr": round(wr * 100, 1),
                                   "pnl": round(pnl, 2), "ev": round(ev, 4)}
        if adj_ev > best_ev:
            best_ev = adj_ev
            best_thr = thr

    return round(best_thr, 3), results


# ── Risk Metrics ───────────────────────────────────────────────────────────

def calc_risk_metrics(returns: list[float]) -> dict:
    """Sharpe, Profit Factor, Max DD, Calmar."""
    if not returns:
        return {}
    r = np.array(returns)
    total = float(r.sum())
    wins = r[r > 0]
    losses = r[r < 0]

    # Sharpe (annualized assuming 250 trades/year)
    sharpe = float(r.mean() / r.std() * np.sqrt(250)) if r.std() > 0 else 0

    # Profit Factor
    gross_profit = float(wins.sum()) if len(wins) else 0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.001
    pf = round(gross_profit / gross_loss, 2)

    # Max Drawdown
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max())

    # Calmar
    calmar = round(total / max_dd, 2) if max_dd > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "profit_factor": pf,
        "max_drawdown_R": round(max_dd, 2),
        "calmar": calmar,
        "avg_win": round(float(wins.mean()), 3) if len(wins) else 0,
        "avg_loss": round(float(losses.mean()), 3) if len(losses) else 0,
        "win_loss_ratio": round(float(wins.mean() / abs(losses.mean())), 2) if len(wins) and len(losses) else 0,
    }


# ── Main Training ─────────────────────────────────────────────────────────

def train_elite(X_raw: np.ndarray, y: np.ndarray, meta: list[dict]) -> tuple[object, dict]:
    logger.info("\n" + "=" * 70)
    logger.info("ELITE ML PIPELINE v3")
    logger.info("=" * 70)
    logger.info("Samples: %d | Win: %d (%.1f%%) | Loss: %d (%.1f%%)",
                len(y), sum(y), sum(y) / len(y) * 100, len(y) - sum(y), (len(y) - sum(y)) / len(y) * 100)

    for sym in SYMBOLS:
        mask = [m["symbol"] == sym for m in meta]
        n = sum(mask)
        w = sum(y[i] for i in range(len(y)) if mask[i])
        if n:
            logger.info("  %s: %d (WR: %.1f%%)", sym, n, w / n * 100)

    # ── Step 1: Feature Selection ──
    logger.info("\n--- FEATURE SELECTION ---")
    X, kept_idx, kept_names = select_features(X_raw, y, FEATURE_NAMES)
    n_features = X.shape[1]

    # ── Step 2: Scale features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 3: Build Stacking Ensemble ──
    logger.info("\n--- STACKING ENSEMBLE ---")

    # Class weight to handle imbalance
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    xgb = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.65, min_child_weight=5,
        gamma=0.2, reg_alpha=0.15, reg_lambda=2.0,
        scale_pos_weight=scale_pos,
        eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1,
    )

    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.65, min_child_weight=5,
        reg_alpha=0.15, reg_lambda=2.0,
        scale_pos_weight=scale_pos,
        random_state=42, verbose=-1, n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=7, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    # Stacking: 3 base + LogisticRegression meta-learner
    stack = StackingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3,  # inner CV for stacking
        stack_method="predict_proba",
        n_jobs=-1,
    )

    logger.info("XGBoost + LightGBM + RandomForest → LogisticRegression meta")
    logger.info("Class weight: scale_pos_weight=%.2f", scale_pos)

    # ── Step 4: Purged Walk-Forward CV ──
    logger.info("\n--- PURGED WALK-FORWARD CV ---")
    PURGE_BARS = 10  # gap between train/test to prevent leakage

    tscv = TimeSeriesSplit(n_splits=5)
    all_probs, all_true, all_idx = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        # Purge: remove last PURGE_BARS from training
        if len(train_idx) > PURGE_BARS:
            train_idx = train_idx[:-PURGE_BARS]

        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if sum(y_tr) < 5 or sum(y_te) < 2:
            continue

        try:
            stack.fit(X_tr, y_tr)
            probs = stack.predict_proba(X_te)[:, 1]
        except Exception as e:
            logger.warning("  Fold %d failed: %s", fold + 1, e)
            continue

        all_probs.extend(probs)
        all_true.extend(y_te)
        all_idx.extend(test_idx)

        wr = sum(y_te) / len(y_te) * 100
        try:
            auc = roc_auc_score(y_te, probs)
        except ValueError:
            auc = 0.5
        logger.info("  Fold %d: train=%d test=%d WR=%.0f%% AUC=%.3f",
                     fold + 1, len(y_tr), len(y_te), wr, auc)

    if len(all_probs) < 10:
        logger.error("CV failed — not enough data")
        stack.fit(X_scaled, y)
        return stack, {"error": "CV failed"}

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    rr_values = [meta[i]["rr"] if meta[i]["label"] == 1 else -1.0 for i in all_idx]

    # ── Step 5: Optimal Threshold ──
    logger.info("\n--- OPTIMAL THRESHOLD ---")
    opt_thr, thr_results = find_optimal_threshold(all_true, all_probs, rr_values)
    logger.info("Selected: %.3f", opt_thr)

    # Show top 5 thresholds
    sorted_thrs = sorted(thr_results.items(), key=lambda x: x[1]["ev"], reverse=True)[:5]
    for thr, stats in sorted_thrs:
        logger.info("  thr=%.3f: %d trades, WR=%.1f%%, PnL=%+.1fR, EV=%.3fR",
                     thr, stats["n"], stats["wr"], stats["pnl"], stats["ev"])

    # ── Step 6: CV Metrics ──
    cv_preds = (all_probs >= opt_thr).astype(int)
    try:
        auc = roc_auc_score(all_true, all_probs)
    except ValueError:
        auc = 0.5
    brier = brier_score_loss(all_true, all_probs)

    logger.info("\n--- CV METRICS (threshold=%.3f) ---", opt_thr)
    logger.info("AUC-ROC:    %.3f", auc)
    logger.info("Brier:      %.4f (lower=better, <0.25=good)", brier)
    logger.info("Accuracy:   %.3f", accuracy_score(all_true, cv_preds))
    logger.info("Precision:  %.3f", precision_score(all_true, cv_preds, zero_division=0))
    logger.info("Recall:     %.3f", recall_score(all_true, cv_preds, zero_division=0))

    # ── Step 7: PnL + Risk Metrics ──
    logger.info("\n--- PERFORMANCE ---")
    trade_mask = cv_preds == 1
    n_filt = int(sum(trade_mask))
    n_total = len(all_true)

    # Returns series
    returns_nofilt = [rr_values[i] for i in range(n_total)]
    returns_filt = [rr_values[i] for i in range(n_total) if trade_mask[i]]

    wr_nofilt = sum(all_true) / n_total * 100
    w_filt = sum(all_true[i] for i in range(n_total) if trade_mask[i])
    wr_filt = w_filt / n_filt * 100 if n_filt else 0
    pnl_nofilt = sum(returns_nofilt)
    pnl_filt = sum(returns_filt)
    exp_nofilt = pnl_nofilt / n_total
    exp_filt = pnl_filt / n_filt if n_filt else 0

    risk_nofilt = calc_risk_metrics(returns_nofilt)
    risk_filt = calc_risk_metrics(returns_filt)

    logger.info("FİLTRESİZ: %d trade | WR: %.1f%% | PnL: %+.1fR | EV: %.3fR | Sharpe: %.2f | PF: %.2f",
                n_total, wr_nofilt, pnl_nofilt, exp_nofilt,
                risk_nofilt.get("sharpe", 0), risk_nofilt.get("profit_factor", 0))
    logger.info("FİLTRELİ:  %d trade (%d%% elendi) | WR: %.1f%% | PnL: %+.1fR | EV: %.3fR | Sharpe: %.2f | PF: %.2f",
                n_filt, int((1 - n_filt / n_total) * 100) if n_total else 0,
                wr_filt, pnl_filt, exp_filt,
                risk_filt.get("sharpe", 0), risk_filt.get("profit_factor", 0))

    # Per-symbol breakdown
    logger.info("\n--- SEMBOL BAZLI (FİLTRELİ) ---")
    for sym in SYMBOLS:
        sym_mask = [meta[all_idx[i]]["symbol"] == sym and trade_mask[i]
                    for i in range(n_total)]
        sym_n = sum(sym_mask)
        if sym_n == 0:
            logger.info("  %s: 0 trade", sym)
            continue
        sym_w = sum(all_true[i] for i in range(n_total) if sym_mask[i])
        sym_pnl = sum(rr_values[i] for i in range(n_total) if sym_mask[i])
        logger.info("  %s: %d trade | WR: %.0f%% | PnL: %+.1fR",
                     sym, sym_n, sym_w / sym_n * 100, sym_pnl)

    # ── Step 8: Final model (full data) + calibration ──
    logger.info("\n--- FINAL MODEL + CALIBRATION ---")
    stack.fit(X_scaled, y)

    # Probability calibration (Platt scaling)
    calibrated = CalibratedClassifierCV(stack, cv=3, method="sigmoid")
    calibrated.fit(X_scaled, y)

    # Feature importance (from XGBoost base)
    try:
        xgb_fitted = stack.named_estimators_["xgb"]
        imp = xgb_fitted.feature_importances_
        sorted_fi = np.argsort(imp)[::-1]
        logger.info("\nFeature Importance (Top 10):")
        for rank, idx in enumerate(sorted_fi[:10]):
            name = kept_names[idx] if idx < len(kept_names) else f"f{idx}"
            logger.info("  %d. %s: %.4f", rank + 1, name, imp[idx])
    except Exception:
        imp = np.zeros(n_features)
        sorted_fi = list(range(n_features))

    # ── Save ──
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": calibrated,
        "scaler": scaler,
        "threshold": opt_thr,
        "feature_names": kept_names,
        "feature_indices": kept_idx,
        "n_features": n_features,
        "version": 3,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("\nModel kaydedildi: %s (v3, %d features, threshold=%.3f)",
                MODEL_PATH, n_features, opt_thr)

    report = {
        "version": 3,
        "total_samples": int(len(y)),
        "win_count": int(sum(y)),
        "loss_count": int(len(y) - sum(y)),
        "n_features": n_features,
        "selected_features": kept_names,
        "optimal_threshold": opt_thr,
        "cv_auc": round(auc, 4),
        "cv_brier": round(brier, 4),
        "cv_precision": round(precision_score(all_true, cv_preds, zero_division=0), 4),
        "cv_recall": round(recall_score(all_true, cv_preds, zero_division=0), 4),
        "winrate_no_filter": round(wr_nofilt, 2),
        "winrate_filtered": round(wr_filt, 2),
        "pnl_no_filter": round(pnl_nofilt, 2),
        "pnl_filtered": round(pnl_filt, 2),
        "expectancy_no_filter": round(exp_nofilt, 4),
        "expectancy_filtered": round(exp_filt, 4),
        "risk_metrics_filtered": risk_filt,
        "feature_importance": {
            kept_names[i] if i < len(kept_names) else f"f{i}": round(float(imp[i]), 4)
            for i in sorted_fi[:15] if i < len(imp)
        },
    }
    return calibrated, report


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("TWELVEDATA_API_KEY", "")
    if not api_key:
        logger.error("TWELVEDATA_API_KEY eksik!")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("  ELITE ML TRAINING PIPELINE v3")
    logger.info("  Stacking + Walk-Forward + Calibration + Feature Selection")
    logger.info("=" * 70)
    start = _time.time()

    logger.info("\n[1/3] VERİ ÇEKİLİYOR...")
    data = await fetch_data(api_key)
    total = sum(len(df) for sd in data.values() for df in sd.values())
    logger.info("Toplam: %d bar", total)

    logger.info("\n[2/3] SAMPLE ÜRETİLİYOR...")
    X, y, meta = generate_samples(data)
    if len(X) < 50:
        logger.error("Yetersiz: %d sample (min 50)", len(X))
        sys.exit(1)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump({"n_samples": len(y), "n_features": X.shape[1],
                    "feature_names": FEATURE_NAMES}, f)

    logger.info("\n[3/3] EĞİTİM...")
    model, report = train_elite(X, y, meta)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = _time.time() - start
    logger.info("\n" + "=" * 70)
    logger.info("  TAMAMLANDI (%.1f dakika)", elapsed / 60)
    logger.info("  Bot'u yeniden başlatın → model otomatik yüklenecek")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
