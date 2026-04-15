from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from app.services.analysis_engine import AnalysisResult

# Dark theme colors
_BG = "#0b1120"
_CARD = "#111d35"
_GRID = "#1a2840"
_TEXT = "#c8d8f0"
_MUTED = "#4a6080"
_GREEN = "#10b981"
_RED = "#ef4444"
_BLUE = "#3b82f6"
_GOLD = "#f59e0b"
_PURPLE = "#a78bfa"


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def generate_signal_chart(df: pd.DataFrame, result: AnalysisResult, bars: int = 80) -> bytes:
    df = df.tail(bars).copy().reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    rsi_s = _rsi(close)
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = macd_fast - macd_slow
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_sig

    has_volume = "volume" in df.columns and df["volume"].sum() > 0

    fig = plt.figure(figsize=(14, 12), facecolor=_BG)
    if has_volume:
        gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[4, 1.0, 1.0, 1.0], hspace=0.06)
    else:
        gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[4, 1.3, 1.3], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1) if has_volume else None

    all_axes = [ax for ax in (ax1, ax2, ax3, ax4) if ax is not None]
    for ax in all_axes:
        ax.set_facecolor(_CARD)
        ax.tick_params(colors=_MUTED, labelsize=7)
        ax.grid(color=_GRID, linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)

    x = np.arange(len(df))

    # --- Candlesticks ---
    for i in range(len(df)):
        c = float(close.iloc[i])
        o = float(open_.iloc[i])
        h = float(high.iloc[i])
        lw = float(low.iloc[i])
        color = _GREEN if c >= o else _RED
        ax1.plot([i, i], [lw, h], color=color, linewidth=0.8, zorder=2)
        ax1.bar(i, abs(c - o), bottom=min(c, o), color=color, width=0.7, alpha=0.9, zorder=2)

    # --- EMAs & Bollinger ---
    ax1.plot(x, ema20.values, color=_GOLD, linewidth=1.2, label="EMA 20", zorder=3)
    ax1.plot(x, ema50.values, color=_BLUE, linewidth=1.2, label="EMA 50", zorder=3)
    ax1.fill_between(x, bb_lower.values, bb_upper.values, alpha=0.07, color=_BLUE, zorder=1)
    ax1.plot(x, bb_upper.values, color=_BLUE, linewidth=0.5, alpha=0.4, zorder=1)
    ax1.plot(x, bb_lower.values, color=_BLUE, linewidth=0.5, alpha=0.4, zorder=1)

    # --- Support / Resistance ---
    for lvl in result.support:
        ax1.axhline(lvl, color=_GREEN, linestyle="--", linewidth=0.9, alpha=0.75, zorder=4)
    for lvl in result.resistance:
        ax1.axhline(lvl, color=_RED, linestyle="--", linewidth=0.9, alpha=0.75, zorder=4)

    # --- Premium / Discount background ---
    if result.premium_discount and result.premium_discount.get("zone"):
        pd_zone = result.premium_discount["zone"]
        pd_low = result.premium_discount.get("range_low", 0)
        pd_high = result.premium_discount.get("range_high", 1)
        eq_mid = (pd_low + pd_high) / 2
        if pd_zone == "PREMIUM":
            ax1.axhspan(eq_mid, pd_high, alpha=0.04, color=_RED, zorder=0)
        elif pd_zone == "DISCOUNT":
            ax1.axhspan(pd_low, eq_mid, alpha=0.04, color=_GREEN, zorder=0)

    # --- OTE zone (Fibonacci 61.8–78.6%) ---
    if result.ote_zone and result.ote_zone.get("valid"):
        ote_lo = result.ote_zone.get("ote_low", 0)
        ote_hi = result.ote_zone.get("ote_high", 0)
        if ote_hi > ote_lo:
            ax1.axhspan(ote_lo, ote_hi, alpha=0.14, color=_GOLD, zorder=1)
            ax1.annotate("OTE", xy=(len(df) - 2, (ote_lo + ote_hi) / 2),
                         color=_GOLD, fontsize=7, fontweight="bold", va="center")

    # --- Previous Day High/Low (PDH/PDL) ---
    pdh_pdl = getattr(result, "pdh_pdl", {}) or {}
    pdh = pdh_pdl.get("prev_day_high", 0)
    pdl = pdh_pdl.get("prev_day_low", 0)
    pwh = pdh_pdl.get("prev_week_high", 0)
    pwl = pdh_pdl.get("prev_week_low", 0)
    if pdh > 0:
        ax1.axhline(pdh, color="#facc15", linestyle=(0, (6, 2)), linewidth=1.0, alpha=0.75, zorder=4)
        ax1.annotate("PDH", xy=(2, pdh), color="#facc15", fontsize=6, fontweight="bold", va="bottom")
    if pdl > 0:
        ax1.axhline(pdl, color="#c084fc", linestyle=(0, (6, 2)), linewidth=1.0, alpha=0.75, zorder=4)
        ax1.annotate("PDL", xy=(2, pdl), color="#c084fc", fontsize=6, fontweight="bold", va="top")
    if pwh > 0:
        ax1.axhline(pwh, color="#facc15", linestyle=(0, (2, 4)), linewidth=0.7, alpha=0.40, zorder=3)
    if pwl > 0:
        ax1.axhline(pwl, color="#c084fc", linestyle=(0, (2, 4)), linewidth=0.7, alpha=0.40, zorder=3)

    # --- Round Numbers ---
    for rn in (getattr(result, "round_numbers", []) or [])[:4]:
        ax1.axhline(rn, color=_MUTED, linestyle=":", linewidth=0.6, alpha=0.35, zorder=2)

    # --- Order blocks ---
    for ob in result.order_blocks[:4]:
        ob_color = _GREEN if ob["type"] == "bullish_ob" else _RED
        alpha = 0.13 if not ob.get("broken") else 0.05
        ax1.axhspan(ob["bottom"], ob["top"], alpha=alpha, color=ob_color, zorder=1)

    # --- Breaker blocks (broken OBs that flipped) ---
    for bb in (result.breaker_blocks or [])[:3]:
        bb_color = _GREEN if bb["type"] == "bullish_breaker" else _RED
        ax1.axhspan(bb["bottom"], bb["top"], alpha=0.09, color=bb_color,
                    hatch="//", zorder=1)

    # --- FVG zones ---
    for fvg in result.fvg_zones[:4]:
        if fvg.get("filled"):
            continue  # filled FVGs are shown as iFVG below
        fvg_color = _GREEN if fvg["type"] == "bullish_fvg" else _RED
        ax1.axhspan(fvg["bottom"], fvg["top"], alpha=0.07, color=fvg_color,
                    linestyle="dotted", zorder=1)

    # --- iFVG zones (Inversion FVG) ---
    for ifvg in (result.ifvg_zones or [])[:3]:
        ifvg_color = _PURPLE
        ax1.axhspan(ifvg["bottom"], ifvg["top"], alpha=0.12, color=ifvg_color, zorder=1)
        ax1.annotate("iFVG", xy=(len(df) - 4, (ifvg["bottom"] + ifvg["top"]) / 2),
                     color=_PURPLE, fontsize=6, va="center")

    # --- Equal Highs / Equal Lows ---
    for lvl in (result.equal_highs or [])[:3]:
        ax1.axhline(lvl, color=_GOLD, linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.8, zorder=4)
    for lvl in (result.equal_lows or [])[:3]:
        ax1.axhline(lvl, color=_PURPLE, linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.8, zorder=4)

    # --- CHoCH marker ---
    if result.choch and result.choch.get("detected"):
        choch_price = result.choch.get("price", 0)
        if choch_price:
            ax1.axhline(choch_price, color="#f97316", linestyle="--", linewidth=1.1,
                        alpha=0.85, zorder=4)
            ax1.annotate("CHoCH", xy=(5, choch_price), color="#f97316",
                         fontsize=7, fontweight="bold", va="bottom")

    # --- BOS / MSS marker ---
    bos_mss = getattr(result, "bos_mss", {}) or {}
    if bos_mss.get("bos") or bos_mss.get("mss"):
        bos_level = bos_mss.get("level", 0)
        bos_type  = bos_mss.get("type", "")
        bos_color = _GREEN if "bullish" in bos_type else _RED
        label_txt = "BOS" if bos_mss.get("bos") else "MSS"
        if bos_level:
            ax1.axhline(bos_level, color=bos_color, linestyle=(0, (5, 2)),
                        linewidth=0.9, alpha=0.75, zorder=4)
            ax1.annotate(label_txt, xy=(len(df) // 4, bos_level),
                         color=bos_color, fontsize=6, fontweight="bold", va="top")

    # --- Judas Swing marker ---
    judas = getattr(result, "judas_swing", {}) or {}
    if judas.get("detected"):
        js_lvl = judas.get("sweep_level", 0)
        js_dir = judas.get("direction", "")
        js_color = _GREEN if js_dir == "bullish" else _RED
        if js_lvl:
            ax1.axhline(js_lvl, color=js_color, linestyle=(0, (2, 2)),
                        linewidth=1.0, alpha=0.70, zorder=4)
            ax1.annotate("JS", xy=(len(df) * 3 // 4, js_lvl),
                         color=js_color, fontsize=6, fontweight="bold", va="bottom")

    # --- Confirmation candle marker ---
    conf = getattr(result, "confirmation_candle", {}) or {}
    if conf.get("detected"):
        ax1.axvline(len(df) - 1, color=_GOLD, linewidth=1.2, alpha=0.6, zorder=4, linestyle=":")
        ax1.annotate(conf.get("type", "Onay"), xy=(len(df) - 1, float(result.current_price)),
                     color=_GOLD, fontsize=6, fontweight="bold", ha="center", va="top")

    # --- Entry / SL / TP lines ---
    if result.signal != "NO TRADE":
        entry_mid = (result.entry_zone[0] + result.entry_zone[1]) / 2
        ax1.axhspan(result.entry_zone[0], result.entry_zone[1],
                    alpha=0.22, color=_GOLD, zorder=5)
        # SL line + label
        ax1.axhline(result.stop_loss, color=_RED, linewidth=1.5, linestyle="-.", zorder=5)
        ax1.annotate(f"SL  {result.stop_loss:.2f}", xy=(1, result.stop_loss),
                     xycoords=("axes fraction", "data"), color=_RED,
                     fontsize=7, fontweight="bold", ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", fc=_CARD, ec=_RED, alpha=0.85))
        # TP1 line + label
        ax1.axhline(result.take_profit, color=_GREEN, linewidth=1.5, linestyle="-.", zorder=5)
        ax1.annotate(f"TP1 {result.take_profit:.2f}", xy=(1, result.take_profit),
                     xycoords=("axes fraction", "data"), color=_GREEN,
                     fontsize=7, fontweight="bold", ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", fc=_CARD, ec=_GREEN, alpha=0.85))
        # TP2 line + label
        ax1.axhline(result.take_profit_2, color=_GREEN, linewidth=1.0, linestyle=":", alpha=0.75, zorder=5)
        ax1.annotate(f"TP2 {result.take_profit_2:.2f}", xy=(1, result.take_profit_2),
                     xycoords=("axes fraction", "data"), color=_GREEN,
                     fontsize=7, fontweight="bold", ha="right", va="top",
                     bbox=dict(boxstyle="round,pad=0.2", fc=_CARD, ec=_GREEN, alpha=0.75))
        # Entry label
        ax1.annotate(f"GİRİŞ {entry_mid:.2f}", xy=(1, entry_mid),
                     xycoords=("axes fraction", "data"), color=_GOLD,
                     fontsize=7, fontweight="bold", ha="right", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc=_CARD, ec=_GOLD, alpha=0.85))

    # ══════════════════════════════════════════════════════
    #  TAHMİNİ YÖN PROJEKSİYONU
    # ══════════════════════════════════════════════════════
    proj_bars = max(15, int(len(df) * 0.40))   # veri genişliğinin %40'ı
    last_x    = len(df) - 1
    proj_end  = last_x + proj_bars

    # Sinyal yoksa trende göre "potansiyel" göster
    _effective_signal = result.signal
    _is_potential     = (_effective_signal == "NO TRADE")
    if _is_potential:
        _effective_signal = "LONG" if result.trend == "Yukari" else "SHORT"

    is_long     = (_effective_signal == "LONG")
    arrow_color = _GREEN if is_long else _RED

    # Alpha / kalınlık: potansiyel = biraz soluk ama yine de NET görünür
    if _is_potential:
        _a   = 0.75
        _alw = 0.70
        _ahw = 0.80
        lw_mul = 0.85
    else:
        _a   = 0.95
        _alw = 0.80
        _ahw = 0.95
        lw_mul = 1.0

    # Hedef seviyeleri
    atr_val   = result.atr
    entry_mid = (result.entry_zone[0] + result.entry_zone[1]) / 2

    # Her durumda AnalysisResult'tan gelen seviyeleri kullan (tutarlılık)
    tp1 = result.take_profit
    tp2 = result.take_profit_2
    sl  = result.stop_loss
    if _is_potential:
        entry_mid = float(result.current_price)

    # ── Y eksenini projeksiyon değerlerini kapsayacak şekilde genişlet ──
    # Bu kritik: olmadan projeksiyon grafik dışına çıkıp görünmez
    data_ymin = float(low.min())
    data_ymax = float(high.max())
    proj_ymin = min(sl, tp1, tp2, entry_mid) - atr_val * 0.5
    proj_ymax = max(sl, tp1, tp2, entry_mid) + atr_val * 0.5
    y_margin  = (data_ymax - data_ymin) * 0.08
    ax1.set_ylim(min(data_ymin, proj_ymin) - y_margin,
                 max(data_ymax, proj_ymax) + y_margin)

    # ── X ekseni genişlet ──
    ax1.set_xlim(-1, proj_end + 2)

    # ── "SIMDI" ayirici ──
    ax1.axvline(last_x, color=_MUTED, linewidth=1.2, linestyle="--", alpha=0.6, zorder=4)
    ybot = ax1.get_ylim()[0]
    ax1.text(last_x + 0.4, ybot + (ax1.get_ylim()[1] - ybot) * 0.02,
             "SIMDI", color=_MUTED, fontsize=7, ha="left", va="bottom",
             fontweight="bold", alpha=0.8)

    # ── Projeksiyon arka plan ──
    ax1.axvspan(last_x, proj_end + 1,
                alpha=0.08 if not _is_potential else 0.05,
                color=arrow_color, zorder=0)

    # ── Projeksiyon yol noktaları ──
    px = [last_x,
          last_x + proj_bars * 0.20,
          last_x + proj_bars * 0.50,
          last_x + proj_bars * 0.80,
          proj_end]

    if is_long:
        py = [entry_mid,
              entry_mid + (tp1 - entry_mid) * 0.25,
              entry_mid + (tp1 - entry_mid) * 0.65,
              tp1,
              tp2]
    else:
        py = [entry_mid,
              entry_mid + (tp1 - entry_mid) * 0.25,
              entry_mid + (tp1 - entry_mid) * 0.65,
              tp1,
              tp2]

    line_style = (0, (5, 3)) if _is_potential else "-"

    # ── Ana projeksiyon çizgisi (KALİN) ──
    ax1.plot(px, py,
             color=arrow_color,
             linewidth=3.5 * lw_mul,
             linestyle=line_style,
             alpha=_a,
             zorder=7,
             solid_capstyle="round")

    # ── Projeksiyon noktaları ──
    ax1.scatter(px[1:], py[1:],
                color=arrow_color, s=60 * lw_mul,
                zorder=8, alpha=_a, edgecolors="white", linewidths=0.5)

    # ── Risk/Ödül bantı (dolu alan) ──
    ax1.fill_between(px,
                     py,
                     [sl] * len(px),
                     alpha=0.12 * lw_mul,
                     color=arrow_color,
                     zorder=1)

    # ── TP/SL yatay çizgiler (tüm projeksiyon boyunca) ──
    ax1.hlines(tp1, last_x, proj_end, colors=_GREEN,
               linewidth=1.8 * lw_mul, linestyle="--", alpha=_alw, zorder=5)
    ax1.hlines(tp2, last_x, proj_end, colors=_GREEN,
               linewidth=1.2 * lw_mul, linestyle=":", alpha=_alw * 0.8, zorder=5)
    ax1.hlines(sl, last_x, proj_end, colors=_RED,
               linewidth=1.8 * lw_mul, linestyle="--", alpha=_alw, zorder=5)

    # ── Yön oku (büyük, göze çarpan) ──
    arrow_dy = atr_val * (3.0 if is_long else -3.0)
    ax1.annotate(
        "",
        xy    =(last_x + 1.5, entry_mid + arrow_dy),
        xytext=(last_x + 0.1, entry_mid),
        arrowprops=dict(
            arrowstyle="-|>",
            color=arrow_color,
            lw=4.0 * lw_mul,
            mutation_scale=28,
        ),
        zorder=9,
    )

    # ── Sinyal etiketi (badge) ──
    if _is_potential:
        badge_txt = "? POTANSIYEL AL (LONG)" if is_long else "? POTANSIYEL SAT (SHORT)"
        badge_fs  = 9
    else:
        badge_txt = ">>> AL (LONG) <<<" if is_long else ">>> SAT (SHORT) <<<"
        badge_fs  = 11

    badge_y = entry_mid + arrow_dy * 1.5
    ax1.annotate(
        badge_txt,
        xy=(last_x + 2.0, badge_y),
        color=arrow_color,
        fontsize=badge_fs,
        fontweight="bold",
        ha="left", va="center",
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.6",
            fc=_CARD,
            ec=arrow_color,
            linewidth=2.5 * lw_mul,
            alpha=_ahw,
        ),
    )

    # ── Seviye etiketleri (sağ taraf) ──
    label_x = proj_end + 0.3
    ax1.annotate(
        f"TP1  {tp1:.2f}",
        xy=(label_x, tp1),
        color=_GREEN, fontsize=8, fontweight="bold",
        ha="left", va="center", zorder=9,
        bbox=dict(boxstyle="round,pad=0.25", fc=_CARD, ec=_GREEN, alpha=_ahw),
    )
    ax1.annotate(
        f"TP2  {tp2:.2f}",
        xy=(label_x, tp2),
        color=_GREEN, fontsize=7, fontweight="bold",
        ha="left", va="center", zorder=9,
        bbox=dict(boxstyle="round,pad=0.25", fc=_CARD, ec=_GREEN, alpha=_alw),
    )
    ax1.annotate(
        f"SL   {sl:.2f}",
        xy=(label_x, sl),
        color=_RED, fontsize=8, fontweight="bold",
        ha="left", va="center", zorder=9,
        bbox=dict(boxstyle="round,pad=0.25", fc=_CARD, ec=_RED, alpha=_ahw),
    )
    # Giriş etiketi
    ax1.annotate(
        f"GIRIS  {entry_mid:.2f}",
        xy=(last_x - 1, entry_mid),
        color=_GOLD, fontsize=7, fontweight="bold",
        ha="right", va="center", zorder=9,
        bbox=dict(boxstyle="round,pad=0.25", fc=_CARD, ec=_GOLD, alpha=_ahw),
    )

    signal_label_map = {"LONG": "AL ▲", "SHORT": "SAT ▼", "NO TRADE": "İŞLEM YOK"}
    sig_color = {"LONG": _GREEN, "SHORT": _RED, "NO TRADE": _MUTED}[result.signal]
    conf_cnt  = getattr(result, "smc_confluence_count", 0)
    dxy_bias  = getattr(result, "dxy_bias", "")
    ax1.set_title(
        f"{result.symbol}  {result.timeframe}   ►   {signal_label_map[result.signal]}   "
        f"|  {result.quality} / {result.setup_score}pt  |  R/R {result.rr_ratio}  "
        f"|  SMC×{conf_cnt}  |  DXY:{dxy_bias}",
        color=sig_color, fontsize=10, fontweight="bold", pad=8, loc="left",
    )
    ax1.legend(loc="upper left", fontsize=7, facecolor=_CARD, edgecolor=_GRID,
               labelcolor=_TEXT, framealpha=0.9, ncol=2)
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Fiyat", color=_MUTED, fontsize=8)

    # --- RSI ---
    ax2.plot(x, rsi_s.values, color=_PURPLE, linewidth=1.3)
    ax2.axhline(70, color=_RED, linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.axhline(30, color=_GREEN, linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.axhline(50, color=_MUTED, linestyle=":", linewidth=0.5, alpha=0.5)
    ax2.fill_between(x, rsi_s.values, 50,
                     where=(rsi_s.values >= 50), alpha=0.12, color=_GREEN, interpolate=True)
    ax2.fill_between(x, rsi_s.values, 50,
                     where=(rsi_s.values < 50), alpha=0.12, color=_RED, interpolate=True)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color=_MUTED, fontsize=8)
    ax2.tick_params(labelbottom=False)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))

    # --- MACD ---
    hist_colors = [_GREEN if v >= 0 else _RED for v in macd_hist.values]
    ax3.bar(x, macd_hist.values, color=hist_colors, alpha=0.75, width=0.8)
    ax3.plot(x, macd_line.values, color=_BLUE, linewidth=1.1, label="MACD")
    ax3.plot(x, macd_sig.values, color=_GOLD, linewidth=1.1, label="Sinyal")
    ax3.axhline(0, color=_MUTED, linewidth=0.5)
    ax3.set_ylabel("MACD", color=_MUTED, fontsize=8)
    ax3.legend(loc="upper left", fontsize=6, facecolor=_CARD, edgecolor=_GRID,
               labelcolor=_TEXT, framealpha=0.9)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(3))

    # --- Volume Panel (ax4) ---
    if ax4 is not None and has_volume:
        vol_series = df["volume"].astype(float)
        vol_avg    = vol_series.rolling(20).mean()
        # Renk: yükselen mum → yeşil, düşen → kırmızı
        vol_colors = [_GREEN if float(close.iloc[i]) >= float(open_.iloc[i]) else _RED
                      for i in range(len(df))]
        ax4.bar(x, vol_series.values, color=vol_colors, alpha=0.60, width=0.8, zorder=2)
        ax4.plot(x, vol_avg.values, color=_GOLD, linewidth=1.0, alpha=0.80, zorder=3, label="Vol MA20")
        # Spike işareti
        vol_analysis = getattr(result, "volume_analysis", {}) or {}
        if vol_analysis.get("volume_spike"):
            ax4.axvline(len(df) - 1, color=_GOLD, linewidth=1.5, alpha=0.7, linestyle=":", zorder=4)
        ax4.set_ylabel("Hacim", color=_MUTED, fontsize=7)
        ax4.tick_params(labelbottom=False)
        ax4.yaxis.set_major_locator(plt.MaxNLocator(2))
        # Delta bias etiketi
        delta_bias = vol_analysis.get("delta_bias", "NEUTRAL")
        bias_color = _GREEN if delta_bias == "BULLISH" else (_RED if delta_bias == "BEARISH" else _MUTED)
        ax4.annotate(f"Delta: {delta_bias}",
                     xy=(0.01, 0.85), xycoords="axes fraction",
                     color=bias_color, fontsize=6, fontweight="bold")

    # --- X axis ticks (en alt panel) ---
    _bottom_ax = ax4 if ax4 is not None else ax3
    # Üst paneller için x tick etiketlerini gizle
    ax2.tick_params(labelbottom=False)
    ax3.tick_params(labelbottom=(ax4 is None))
    if "datetime" in df.columns:
        step   = max(1, len(df) // 8)
        ticks  = list(range(0, len(df), step))
        labels = [str(df.iloc[i]["datetime"])[:16] for i in ticks]
        ticks.append(proj_end)
        labels.append("-> Tahmin")
        _bottom_ax.set_xticks(ticks)
        _bottom_ax.set_xticklabels(labels, rotation=20, fontsize=6, color=_MUTED)
    else:
        _bottom_ax.axvline(last_x, color=_MUTED, linewidth=0.6, linestyle=":", alpha=0.4)

    # RSI, MACD ve Volume panellerinde projeksiyon alanı
    for _pax in (ax2, ax3, ax4):
        if _pax is not None:
            _pax.axvspan(last_x, proj_end + 1, alpha=0.05, color=_MUTED, zorder=0)
            _pax.axvline(last_x, color=_MUTED, linewidth=0.6, linestyle=":", alpha=0.4)
    # ax2/ax3 genişliği güncelle
    ax2.set_xlim(-1, proj_end + 2)
    ax3.set_xlim(-1, proj_end + 2)
    if ax4 is not None:
        ax4.set_xlim(-1, proj_end + 2)

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=_BG)
        buf.seek(0)
        return buf.read()
    finally:
        plt.close(fig)


def generate_backtest_chart(result) -> bytes:
    """Backtest sonuclari icin equity curve + drawdown + monthly returns chart."""
    from app.services.backtest_service import BacktestResult

    equity = result.equity_curve or [0.0]
    monthly = result.monthly_returns or {}

    fig = plt.figure(figsize=(14, 10), facecolor=_BG)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.3)

    # ── 1. Equity Curve ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(_CARD)
    x = list(range(len(equity)))
    ax1.plot(x, equity, color=_GREEN, linewidth=1.5, label="Equity (R)")
    ax1.fill_between(x, 0, equity, where=[e >= 0 for e in equity], alpha=0.15, color=_GREEN)
    ax1.fill_between(x, 0, equity, where=[e < 0 for e in equity], alpha=0.15, color=_RED)
    ax1.axhline(y=0, color=_MUTED, linewidth=0.5, linestyle="--")
    ax1.set_title(
        f"Backtest — WR: {result.winrate}% | Sharpe: {result.sharpe_ratio} | PF: {result.profit_factor} | Max DD: {result.max_drawdown_pct}%",
        color=_TEXT, fontsize=11, fontweight="bold", pad=10,
    )
    ax1.set_ylabel("Kumulatif R", color=_TEXT, fontsize=9)
    ax1.tick_params(colors=_MUTED, labelsize=8)
    ax1.grid(True, color=_GRID, linewidth=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_color(_GRID)
    ax1.spines["left"].set_color(_GRID)

    # ── 2. Drawdown ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(_CARD)
    # Drawdown hesapla
    dd_series = []
    peak = equity[0]
    for val in equity:
        if val > peak:
            peak = val
        dd_series.append(val - peak)
    ax2.fill_between(x, dd_series, 0, color=_RED, alpha=0.35)
    ax2.plot(x, dd_series, color=_RED, linewidth=0.8)
    ax2.set_ylabel("Drawdown (R)", color=_TEXT, fontsize=9)
    ax2.set_title("Drawdown", color=_TEXT, fontsize=10, pad=5)
    ax2.tick_params(colors=_MUTED, labelsize=8)
    ax2.grid(True, color=_GRID, linewidth=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color(_GRID)
    ax2.spines["left"].set_color(_GRID)

    # ── 3. Monthly Returns ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(_CARD)
    if monthly:
        months = sorted(monthly.keys())
        values = [monthly[m] for m in months]
        colors = [_GREEN if v >= 0 else _RED for v in values]
        bars = ax3.bar(range(len(months)), values, color=colors, alpha=0.8, width=0.6)
        ax3.set_xticks(range(len(months)))
        ax3.set_xticklabels(months, rotation=45, ha="right", fontsize=7, color=_MUTED)
        ax3.axhline(y=0, color=_MUTED, linewidth=0.5, linestyle="--")
    else:
        ax3.text(0.5, 0.5, "Aylik veri yok", transform=ax3.transAxes,
                 ha="center", va="center", color=_MUTED, fontsize=10)
    ax3.set_title("Aylik Getiri (R)", color=_TEXT, fontsize=10, pad=5)
    ax3.set_ylabel("R", color=_TEXT, fontsize=9)
    ax3.tick_params(colors=_MUTED, labelsize=8)
    ax3.grid(True, color=_GRID, linewidth=0.3, axis="y")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_color(_GRID)
    ax3.spines["left"].set_color(_GRID)

    # ── Kaydet ───────────────────────────────────────────────────────────────
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.tight_layout(pad=1.0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=_BG)
        buf.seek(0)
        return buf.read()
    finally:
        plt.close(fig)
