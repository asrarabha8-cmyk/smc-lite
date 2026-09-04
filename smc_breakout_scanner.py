#!/usr/bin/env python3
"""
ماسح الاختراق — فلتر فينفيز + تأكيد SMC + أهداف ABC
====================================================
المنطق ثلاث طبقات، كل طبقة تجاوب على سؤال مختلف:

  1) فينفيز  → مَن المرشحون؟   (Float < 50M · RelVol > 2 · قمة 20 يوم · ...)
  2) SMC     → متى أدخل؟       (إغلاق شمعة 4 ساعات فوق آخر قمة هيكلية = BOS)
  3) ABC     → أين أخرج؟       (إسقاط فيبو 1.618 / 1.809 / 2.0 من قاع B)

قواعد مُلزَمة داخل الكود:
  • لا يُفحص إلا على شموع **مغلقة** — الشمعة الجارية تُستبعد دائماً.
  • الوقف = آخر قاع هيكلي قبل الاختراق (BOS)، لا رقم عشوائي.
  • يُستبعد أي سهم قفزت فيه شمعة واحدة أكثر من 200%.
  • كل تنبيه يُسجَّل في CSV — بعد شهر يصير عندك رقمك الخاص لا رقم غيرك.

التشغيل:
    pip install yfinance pandas requests lxml
    export TG_TOKEN="..."   # توكن البوت
    export TG_CHAT="..."    # chat id
    python smc_breakout_scanner.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------
# الإعدادات
# ------------------------------------------------------------------
FINVIZ_FILTERS = ",".join([
    "geo_usa",           # أمريكا
    "sh_curvol_o500",    # فوليوم اليوم > 500K
    "sh_float_u50",      # الفلوت < 50M   ← قلب الفلتر
    "sh_price_o1",       # السعر > $1
    "sh_relvol_o2",      # الفوليوم النسبي > 2
    "ta_change_u5",      # التغير اليوم > 5%
    "ta_highlow20d_nh",  # قمة جديدة لـ 20 يوم
])

PIVOT_LOOKBACK = 3       # عرض الفراكتال لتحديد القمم والقيعان
MAX_CANDLE_GAIN = 200.0  # تحذير صديقك: تجاهل من ارتفع 200% في شمعة
MIN_RR = 1.5             # لا ترسل صفقة عائدها أقل من 1.5 ضعف مخاطرتها
COOLDOWN_DAYS = 5        # لا تكرر تنبيه نفس السهم خلال 5 أيام
FIBS = [1.618, 1.809, 2.0]

STATE_FILE = Path("scanner_state.json")
LOG_FILE = Path("alerts_log.csv")

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"}


# ------------------------------------------------------------------
# الطبقة 1 — فينفيز
# ------------------------------------------------------------------
def finviz_screen() -> list[str]:
    """يسحب قائمة المرشحين من فلتر فينفيز نفسه."""
    url = f"https://finviz.com/screener.ashx?v=111&f={FINVIZ_FILTERS}"
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        tables = pd.read_html(r.text)
    except Exception as e:
        print(f"فينفيز فشل: {e}", file=sys.stderr)
        return []

    for tb in tables:
        if "Ticker" in tb.columns:
            return [str(t).strip().upper() for t in tb["Ticker"].dropna()]
    print("لم أجد جدول النتائج في صفحة فينفيز.", file=sys.stderr)
    return []


# ------------------------------------------------------------------
# البيانات — شموع 4 ساعات
# ------------------------------------------------------------------
def fetch_4h(ticker: str) -> pd.DataFrame | None:
    """ينزّل شموع الساعة ثم يجمّعها إلى 4 ساعات، ويحذف الشمعة الجارية."""
    try:
        df = yf.download(ticker, period="60d", interval="1h",
                         auto_adjust=False, progress=False, threads=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()

    # الشمعة الأخيرة قد تكون غير مكتملة — القاعدة تقول: إغلاق فقط
    return df.iloc[:-1] if len(df) > 1 else None


# ------------------------------------------------------------------
# الطبقة 2 — SMC: القمم والقيعان الهيكلية + BOS
# ------------------------------------------------------------------
def pivots(df: pd.DataFrame, L: int = PIVOT_LOOKBACK) -> tuple[list[int], list[int]]:
    h, l = df["High"].values, df["Low"].values
    ph, pl = [], []
    for i in range(L, len(df) - L):
        win_h, win_l = h[i - L:i + L + 1], l[i - L:i + L + 1]
        if h[i] >= win_h.max():
            ph.append(i)
        if l[i] <= win_l.min():
            pl.append(i)
    return ph, pl


def find_bos(df: pd.DataFrame) -> dict | None:
    """اختراق هيكلي طازج: إغلاق آخر شمعة مغلقة فوق آخر قمة هيكلية."""
    if len(df) < 40:
        return None

    ph, pl = pivots(df)
    if not ph or not pl:
        return None

    last = len(df) - 1                       # آخر شمعة مغلقة
    prior_highs = [i for i in ph if i < last - PIVOT_LOOKBACK]
    if not prior_highs:
        return None

    a_idx = prior_highs[-1]                  # القمة المخترقة = النقطة A
    level = float(df["High"].iloc[a_idx])

    close_now = float(df["Close"].iloc[last])
    close_prev = float(df["Close"].iloc[last - 1])

    # شرط الطزاجة: أُغلقت فوقه الآن ولم تكن مُغلقة فوقه قبل شمعة
    if not (close_now > level and close_prev <= level):
        return None

    # الوقف = آخر قاع هيكلي قبل الاختراق  ← النقطة B
    prior_lows = [i for i in pl if i > a_idx]
    b_idx = prior_lows[-1] if prior_lows else max(pl)
    stop = float(df["Low"].iloc[b_idx])
    if stop >= close_now:
        return None

    return {"entry": close_now, "stop": stop, "bos_level": level,
            "a_idx": a_idx, "b_idx": b_idx, "a_high": level,
            "bar_time": df.index[last]}


# ------------------------------------------------------------------
# الطبقة 3 — أهداف ABC
# ------------------------------------------------------------------
def abc_targets(df: pd.DataFrame, sig: dict) -> list[float]:
    """يقيس الساق الدافعة (قاع → A) ويسقطها من قاع B بنسب فيبو."""
    a_idx, b_idx = sig["a_idx"], sig["b_idx"]
    start = max(0, a_idx - 40)
    leg_low = float(df["Low"].iloc[start:a_idx + 1].min())
    leg = sig["a_high"] - leg_low
    if leg <= 0:
        return []
    b_low = float(df["Low"].iloc[b_idx])
    return [round(b_low + leg * f, 2) for f in FIBS]


def violent_candle(df: pd.DataFrame, bars: int = 6) -> float:
    """أكبر ارتفاع لشمعة واحدة خلال آخر عدة شموع — حارس الـ200%."""
    tail = df.tail(bars)
    gains = (tail["Close"] / tail["Open"] - 1) * 100
    return float(gains.max())


# ------------------------------------------------------------------
# التنبيه
# ------------------------------------------------------------------
def notify(text: str):
    token, chat = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if not token or not chat:
        print(text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=30,
    )
    if not r.ok:
        print("Telegram error:", r.text, file=sys.stderr)


def fmt(x: float) -> str:
    return f"{x:,.4f}" if abs(x) < 10 else f"{x:,.2f}"


# ------------------------------------------------------------------
# الحالة والسجل
# ------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=1))


def log_alert(row: dict):
    """كل تنبيه يُسجَّل — هذا ما سيعطيك أرقامك الحقيقية بعد شهر."""
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)


# ------------------------------------------------------------------
def main():
    now = dt.datetime.now(dt.timezone.utc)
    state = load_state()

    tickers = finviz_screen()
    print(f"scan {now:%Y-%m-%d %H:%M} UTC — فينفيز أعطى {len(tickers)} مرشحاً")
    if not tickers:
        return

    signals, watch, skipped = [], [], []

    for t in tickers:
        # منع تكرار نفس التنبيه
        seen = state.get(t)
        if seen:
            age = (now - dt.datetime.fromisoformat(seen)).days
            if age < COOLDOWN_DAYS:
                continue

        df = fetch_4h(t)
        if df is None or len(df) < 40:
            skipped.append(t)
            continue

        spike = violent_candle(df)
        if spike > MAX_CANDLE_GAIN:
            skipped.append(f"{t} (شمعة +{spike:.0f}%)")
            continue

        try:
            sig = find_bos(df)
        except Exception as e:
            print(f"{t}: {e}", file=sys.stderr)
            continue

        if not sig:
            watch.append(t)
            continue

        targets = abc_targets(df, sig)
        if not targets:
            continue

        risk = sig["entry"] - sig["stop"]
        rr = (targets[0] - sig["entry"]) / risk if risk > 0 else 0
        risk_pct = risk / sig["entry"] * 100

        if rr < MIN_RR:
            skipped.append(f"{t} (R:R {rr:.1f})")
            continue

        sig.update(ticker=t, targets=targets, rr=rr, risk_pct=risk_pct)
        signals.append(sig)
        state[t] = now.isoformat()

        log_alert({
            "date": now.strftime("%Y-%m-%d %H:%M"),
            "ticker": t,
            "entry": round(sig["entry"], 4),
            "stop": round(sig["stop"], 4),
            "risk_pct": round(risk_pct, 1),
            "t1": targets[0], "t2": targets[1], "t3": targets[2],
            "rr": round(rr, 2),
            # تُملأ يدوياً أو بسكربت متابعة لاحق
            "max_gain_pct": "", "outcome": "",
        })

    print(f"إشارات: {len(signals)} · مراقبة: {len(watch)} · مُستبعد: {len(skipped)}")

    if not signals:
        if watch:
            notify("🔍 <b>لا اختراق مؤكد اليوم</b>\n"
                   f"مرشحون تحت المراقبة: {', '.join(watch[:15])}")
        return

    lines = ["🚨 <b>اختراق هيكلي مؤكد — إغلاق 4 ساعات</b>", ""]
    for s in signals:
        t1, t2, t3 = s["targets"]
        lines += [
            f"<b>${s['ticker']}</b>",
            f"دخول: <code>{fmt(s['entry'])}</code>",
            f"وقف (BOS): <code>{fmt(s['stop'])}</code>  ({s['risk_pct']:.0f}%-)",
            f"أهداف ABC: <code>{fmt(t1)}</code> · {fmt(t2)} · {fmt(t3)}",
            f"العائد/المخاطرة: <b>{s['rr']:.1f}</b>",
            "",
        ]
    lines.append("<i>جني جزئي عند أول منطقة عرض ونقل الوقف للتعادل.</i>")

    notify("\n".join(lines))
    save_state(state)


if __name__ == "__main__":
    main()
