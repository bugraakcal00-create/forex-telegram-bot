from __future__ import annotations

"""
ML Signal Filter — XGBoost tabanlı sinyal kalite filtresi.

Çalışma mantığı:
  1. Bot her sinyali sqlite'a loglar (outcome: tp_hit / sl_hit / pending)
  2. Yeterli veri birikince (min_samples=80) model eğitilir
  3. Eğitimli model her yeni sinyali filtreler: P(success) < 0.50 → NO TRADE

Özellikler (Features):
  - setup_score, rsi, rr_ratio, atr_ratio
  - smc_confluence_count
  - trend_aligned (bool)
  - choch, displacement, ote, judas, bos, confirmation_candle
  - premium_discount zone
  - sweep_signal, sniper_entry
  - hour_of_day, session_encoded
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.analysis_engine import AnalysisResult
    from app.storage.sqlite_store import BotRepository

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent.parent / "data" / "ml_signal_filter.pkl"


@dataclass
class MLFilterResult:
    probability: float      # P(win) 0.0–1.0
    should_trade: bool      # probability >= threshold
    is_trained: bool        # Model eğitildi mi?
    feature_count: int      # Kaç özellik kullanıldı


class MLSignalFilter:
    """
    XGBoost sinyal filtresi.
    Eğitilmemişse her zaman should_trade=True döner (geçirici mod).
    """

    def __init__(self, model_path: Path = _MODEL_PATH, threshold: float = 0.52):
        self.model_path = model_path
        self.threshold  = threshold
        self._model     = None
        self._scaler    = None
        self._feature_indices = None
        self._version   = 1
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    bundle = pickle.load(f)
                if isinstance(bundle, dict) and "model" in bundle:
                    self._model = bundle["model"]
                    self.threshold = bundle.get("threshold", self.threshold)
                    self._version = bundle.get("version", 2)
                    self._scaler = bundle.get("scaler")
                    self._feature_indices = bundle.get("feature_indices")
                    logger.info("ML filter v%d loaded (threshold=%.3f, %d features): %s",
                                self._version, self.threshold,
                                bundle.get("n_features", "?"), self.model_path)
                else:
                    self._model = bundle
                    self._version = 1
                    logger.info("ML filter v1 loaded: %s", self.model_path)
            except Exception as exc:
                logger.warning("ML model yuklenemedi: %s", exc)
                self._model = None

    def is_trained(self) -> bool:
        return self._model is not None

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def extract_features(result: "AnalysisResult") -> list[float]:
        """AnalysisResult'tan ML özellik vektörü çıkar."""
        from datetime import datetime

        hour = datetime.now().hour

        # Session encode (UTC): London=1, NY=2, overlap=3, other=0
        session_enc = 0
        if 13 <= hour < 17:   session_enc = 3  # London+NY overlap (13:00-17:00 UTC)
        elif 8 <= hour < 17:  session_enc = 1  # London (08:00-17:00 UTC)
        elif 13 <= hour < 22: session_enc = 2  # NY (13:00-22:00 UTC)

        atr_ratio = result.atr / max(result.current_price, 1e-9)

        choch_detected    = 1 if result.choch and result.choch.get("detected") else 0
        disp_detected     = 1 if result.displacement and result.displacement.get("detected") else 0
        ote_valid         = 1 if result.ote_zone and result.ote_zone.get("valid") else 0
        judas_detected    = 1 if result.judas_swing and result.judas_swing.get("detected") else 0
        bos_detected      = 1 if result.bos_mss and result.bos_mss.get("bos") else 0
        mss_detected      = 1 if result.bos_mss and result.bos_mss.get("mss") else 0
        conf_detected     = 1 if result.confirmation_candle and result.confirmation_candle.get("detected") else 0
        conf_strength     = result.confirmation_candle.get("strength", 0) if result.confirmation_candle else 0
        trend_aligned     = 1 if result.trend == result.higher_tf_trend else 0
        sweep_ok          = 1 if result.sweep_signal != "Yok" else 0
        sniper_ok         = 1 if result.sniper_entry != "Yok" else 0

        pd_zone = result.premium_discount.get("zone", "DENGE") if result.premium_discount else "DENGE"
        pd_enc = {"DISCOUNT": -1, "DENGE": 0, "PREMIUM": 1}.get(pd_zone, 0)

        # Signal direction
        signal_enc = 1 if result.signal == "LONG" else (-1 if result.signal == "SHORT" else 0)

        # DXY / COT / sentiment
        dxy_enc = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(
            getattr(result, "dxy_bias", "NEUTRAL"), 0
        )
        cot_enc = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(
            getattr(result, "cot_bias", "NEUTRAL"), 0
        )

        return [
            float(result.setup_score),
            float(result.rsi),
            float(result.rr_ratio),
            float(atr_ratio * 1000),          # scale up
            float(result.smc_confluence_count),
            float(trend_aligned),
            float(choch_detected),
            float(disp_detected),
            float(ote_valid),
            float(judas_detected),
            float(bos_detected),
            float(mss_detected),
            float(conf_detected),
            float(conf_strength),
            float(pd_enc),
            float(sweep_ok),
            float(sniper_ok),
            float(signal_enc),
            float(dxy_enc),
            float(cot_enc),
            float(hour),
            float(session_enc),
        ]

    # ── Tahmin ────────────────────────────────────────────────────────────────

    def _extract_v3_features(self, result: "AnalysisResult", df) -> list[float]:
        """v3: 40 boyutlu feature (train_ml_model.py ile senkron)."""
        import numpy as np
        base = self._extract_v2_features(result)
        try:
            from app.services.analysis_engine import AnalysisEngine as _AE
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            rsi_div = 0.0
            if len(close) >= 30:
                p5 = close.iloc[-5:].mean()
                p15 = close.iloc[-15:-10].mean()
                rsi_s = _AE._rsi(close, 14)
                rsi_prev = float(rsi_s.iloc[-15]) if len(rsi_s) >= 15 and not np.isnan(rsi_s.iloc[-15]) else result.rsi
                if p5 > p15 and result.rsi < rsi_prev: rsi_div = -1.0
                elif p5 < p15 and result.rsi > rsi_prev: rsi_div = 1.0

            atr_s = _AE._atr(df, 14)
            atr_change = 0.0
            if len(atr_s) >= 10:
                a_now = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0
                a_prev = float(atr_s.iloc[-10]) if not np.isnan(atr_s.iloc[-10]) else a_now
                if a_prev > 0: atr_change = round((a_now - a_prev) / a_prev * 100, 2)

            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            ema_dist = round((ema20 - ema50) / max(result.atr, 1e-9), 3)

            bb_pos = 0.5
            if result.bb_upper > result.bb_lower:
                bb_pos = max(0, min(1, (result.current_price - result.bb_lower) / (result.bb_upper - result.bb_lower)))

            last = df.iloc[-1]
            tr = float(last["high"]) - float(last["low"])
            body_ratio = abs(float(last["close"]) - float(last["open"])) / tr if tr > 0 else 0.5
            uw_ratio = (float(last["high"]) - max(float(last["close"]), float(last["open"]))) / tr if tr > 0 else 0

            dirs = (close.diff().tail(10) > 0).values
            consec = 1
            if len(dirs) >= 2:
                for i in range(len(dirs) - 2, -1, -1):
                    if dirs[i] == dirs[-1]: consec += 1
                    else: break
                consec = consec if dirs[-1] else -consec

            rh = float(high.tail(40).max())
            rl = float(low.tail(40).min())
            range_pos = (result.current_price - rl) / (rh - rl) if rh > rl else 0.5

            macd_dir = 1 if result.macd_hist > 0 else (-1 if result.macd_hist < 0 else 0)

            adx_s = _AE._adx(df, 14)
            adx_val = float(adx_s.iloc[-1]) if len(adx_s) > 0 and not np.isnan(adx_s.iloc[-1]) else 0

            return base + [rsi_div, atr_change, ema_dist, bb_pos, body_ratio,
                           uw_ratio, float(consec), range_pos, float(macd_dir), adx_val]
        except Exception:
            return base + [0.0] * 10

    def _extract_v2_features(self, result: "AnalysisResult") -> list[float]:
        """v2: 30 boyutlu feature (yeni özellikler dahil)."""
        base = self.extract_features(result)
        # v2 ek feature'lar
        unicorn = getattr(result, "unicorn_model", {}) or {}
        uni_d = 1.0 if unicorn.get("detected") and unicorn.get("near_price") else 0.0

        vwap = getattr(result, "vwap", 0.0) or 0.0
        if vwap > 0:
            vwap_a = 1.0 if (result.signal == "LONG" and result.current_price > vwap) or \
                            (result.signal == "SHORT" and result.current_price < vwap) else -1.0
        else:
            vwap_a = 0.0

        sb = getattr(result, "silver_bullet", {}) or {}
        sb_a = 1.0 if sb.get("active") else 0.0

        ipda = getattr(result, "ipda_levels", {}) or {}
        ipda_n = 1.0 if ipda.get("distance_atr", 99) <= 1.5 else 0.0

        amd = getattr(result, "amd_phase", {}) or {}
        amd_e = {"ACCUMULATION": 0.0, "MANIPULATION": -1.0, "DISTRIBUTION": 1.0}.get(amd.get("phase", ""), 0.0)

        vol = getattr(result, "volume_analysis", {}) or {}
        vs = 1.0 if vol.get("volume_spike") else 0.0
        vd = {"BULLISH": 1.0, "NEUTRAL": 0.0, "BEARISH": -1.0}.get(vol.get("delta_bias", "NEUTRAL"), 0.0)

        return base + [uni_d, vwap_a, sb_a, ipda_n, amd_e, 0.0, vs, vd]

    def predict(self, result: "AnalysisResult", df=None) -> MLFilterResult:
        """Sinyalin başarılı olma olasılığını tahmin et."""
        if self._version >= 3 and df is not None:
            features = self._extract_v3_features(result, df)
        elif self._version >= 2:
            features = self._extract_v2_features(result)
        else:
            features = self.extract_features(result)

        if not self.is_trained():
            return MLFilterResult(
                probability=0.55,
                should_trade=True,
                is_trained=False,
                feature_count=len(features),
            )

        try:
            import numpy as np
            X = np.array([features])

            # v3: feature selection + scaling
            if self._version >= 3 and self._feature_indices is not None:
                X = X[:, self._feature_indices]
            if self._scaler is not None:
                X = self._scaler.transform(X)

            prob = float(self._model.predict_proba(X)[0][1])
            return MLFilterResult(
                probability=round(prob, 3),
                should_trade=(prob >= self.threshold),
                is_trained=True,
                feature_count=len(features),
            )
        except Exception as exc:
            logger.warning("ML predict error: %s", exc)
            return MLFilterResult(
                probability=0.55,
                should_trade=True,
                is_trained=False,
                feature_count=len(features),
            )

    # ── Eğitim ────────────────────────────────────────────────────────────────

    def train(self, repo: "BotRepository", min_samples: int = 80) -> bool:
        """
        SQLite signal_logs tablosundan labeled verileri çek, XGBoost eğit.
        outcome = 'tp_hit' → 1 (kazanç), 'sl_hit' → 0 (kayıp)
        """
        try:
            import numpy as np
            from xgboost import XGBClassifier  # type: ignore
        except ImportError:
            logger.warning("xgboost veya numpy yuklu degil. ML filter atlandi.")
            return False

        rows = repo.get_labeled_signal_logs(limit=2000)
        if len(rows) < min_samples:
            logger.info(
                "ML egitimi icin yeterli veri yok: %d/%d", len(rows), min_samples
            )
            return False

        X_list: list[list[float]] = []
        y_list: list[int] = []

        for row in rows:
            try:
                features = _row_to_features(row)
                label = 1 if str(row.get("outcome", "")) == "tp_hit" else 0
                X_list.append(features)
                y_list.append(label)
            except Exception:
                continue

        if len(X_list) < min_samples:
            return False

        X = np.array(X_list)
        y = np.array(y_list)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X, y)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)
        self._model = model

        win_rate = sum(y_list) / len(y_list)
        logger.info(
            "ML model egitildi: %d sample, win_rate=%.2f, saved to %s",
            len(y_list), win_rate, self.model_path
        )
        return True


def _row_to_features(row: dict) -> list[float]:
    """DB satırından feature vektörü çıkar (eğitim için)."""
    import json
    from datetime import datetime

    created_at = str(row.get("created_at", ""))
    try:
        hour = int(created_at[11:13]) if len(created_at) >= 13 else 12
    except Exception:
        hour = 12

    session_enc = 0
    if 13 <= hour < 17:   session_enc = 3  # overlap
    elif 8 <= hour < 17:  session_enc = 1  # London
    elif 13 <= hour < 22: session_enc = 2  # NY

    signal = str(row.get("signal", ""))
    signal_enc = 1 if signal == "LONG" else (-1 if signal == "SHORT" else 0)

    try:
        no_trade_reasons = json.loads(str(row.get("no_trade_reasons", "[]")))
        nr_count = len(no_trade_reasons)
    except Exception:
        nr_count = 0

    # Parse reason field for SMC indicators
    reason_str = str(row.get("reason", "")).lower()
    choch_detected = 1.0 if "choch" in reason_str else 0.0
    disp_detected = 1.0 if "displacement" in reason_str else 0.0
    ote_valid = 1.0 if "ote" in reason_str else 0.0
    judas_detected = 1.0 if "judas" in reason_str else 0.0
    bos_detected = 1.0 if "bos" in reason_str else 0.0
    mss_detected = 1.0 if "mss" in reason_str else 0.0
    conf_detected = 1.0 if "onay" in reason_str or "engulfing" in reason_str or "hammer" in reason_str else 0.0
    conf_strength = 80.0 if conf_detected else 0.0
    pd_enc = -1.0 if "discount" in reason_str else (1.0 if "premium" in reason_str else 0.0)

    return [
        float(row.get("setup_score", 50)),
        float(row.get("rsi", 50)),
        float(row.get("rr_ratio", 0)),
        0.0,   # atr_ratio (not stored)
        float(row.get("smc_confluence_count", 0)),
        1.0 if str(row.get("trend", "")) == str(row.get("higher_tf_trend", "X")) else 0.0,
        choch_detected, disp_detected, ote_valid, judas_detected,
        bos_detected, mss_detected, conf_detected, conf_strength,
        pd_enc,
        1.0 if str(row.get("sweep_signal", "Yok")) != "Yok" else 0.0,
        1.0 if str(row.get("sniper_entry", "Yok")) != "Yok" else 0.0,
        float(signal_enc),
        0.0, 0.0,  # dxy/cot (not stored)
        float(hour),
        float(session_enc),
    ]


# Global singleton
ml_filter = MLSignalFilter()
