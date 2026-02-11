# -*- coding: utf-8 -*-
"""
업비트 멀티 에이전트 시스템 트레이딩 봇

┌─────────────────────────────────────────────────┐
│  Agent 1. 시장 분석가    │  Claude Sonnet 4      │
│  Agent 2. 트레이더       │  Claude Opus 4.5      │
│  Agent 3. 리스크 매니저  │  Claude Sonnet 4      │
└─────────────────────────────────────────────────┘

- Anthropic 프롬프트 캐싱으로 비용 최적화
- 영어 시스템 프롬프트로 토큰 효율 극대화
- 15분 간격 자동 매매 / 손절·익절 자동 관리
- 매수 매도 후 각 에이전트 실적 텔레그램 메시지 기능
"""

import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import anthropic
import schedule
import requests as req_lib

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────
TICKER = "KRW-BTC"
INTERVAL = "minute60"
CANDLE_COUNT = 200
TRADE_INTERVAL_MIN = 15
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-20250514"

# ─── Telegram ────────────────────────────────────────────

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg: str):
    if not TG_TOKEN or not TG_CHAT or "your_" in TG_TOKEN:
        logger.debug("Telegram not configured, skip")
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        req_lib.post(url, json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram send fail: {e}")

# Singleton Claude client
_client = None

def get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    return _client


# ─── Technical Indicators ────────────────────────────────

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0).rolling(p).mean()
    lo = (-d.where(d < 0, 0.0)).rolling(p).mean()
    return 100 - (100 / (1 + g / lo))

def calc_macd(s, f=12, sl=26, sg=9):
    ml = s.ewm(span=f).mean() - s.ewm(span=sl).mean()
    sig = ml.ewm(span=sg).mean()
    return ml, sig, ml - sig

def calc_bollinger(s, p=20, sd=2):
    sma = s.rolling(p).mean()
    std = s.rolling(p).std()
    return sma + sd * std, sma, sma - sd * std

def calc_stochastic(df, kp=14, dp=3):
    lo = df["low"].rolling(kp).min()
    hi = df["high"].rolling(kp).max()
    k = 100 * (df["close"] - lo) / (hi - lo)
    return k, k.rolling(dp).mean()

def calc_atr(df, p=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(p).mean()

def get_market_data(ticker, interval, count):
    df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")
    df["rsi"] = calc_rsi(df["close"])
    df["macd"], df["macd_sig"], df["macd_hist"] = calc_macd(df["close"])
    df["bb_up"], df["bb_mid"], df["bb_lo"] = calc_bollinger(df["close"])
    df["stoch_k"], df["stoch_d"] = calc_stochastic(df)
    df["atr"] = calc_atr(df)
    for p in [5, 20, 60, 120]:
        df[f"ma{p}"] = df["close"].rolling(p).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df


# ─── Cached System Prompts (English, token-efficient) ────

ANALYST_SYSTEM = [{"type": "text", "cache_control": {"type": "ephemeral"}, "text": (
    "You are a crypto technical analysis expert. Analyze market data objectively.\n"
    "Respond ONLY in JSON:\n"
    '{"trend":"bullish|bearish|sideways","trend_strength":1-10,'
    '"key_signals":["s1","s2","s3"],"support":number,"resistance":number,'
    '"volume":"increasing|decreasing|neutral","summary":"2-3 sentences"}\n'
    "Rules: RSI<30=oversold,RSI>70=overbought. MACD golden cross=bullish,death cross=bearish. "
    "Bollinger lower=buy zone,upper=sell zone. MA ascending=uptrend,descending=downtrend. "
    "Stochastic K>D in oversold=buy,K<D in overbought=sell. Volume confirms trend. "
    "ATR high=volatile,low=calm."
)}]

TRADER_SYSTEM = [{"type": "text", "cache_control": {"type": "ephemeral"}, "text": (
    "You are a crypto trader. Decide buy/sell/hold based on analyst report and portfolio.\n"
    "Respond ONLY in JSON:\n"
    '{"decision":"buy|sell|hold","confidence":0-100,"reason":"2-3 sentences","urgency":"immediate|wait|none"}\n'
    "Rules: 1.Enter only in trend direction 2.Sideways=hold 3.If holding,focus on hold/sell "
    "4.No coins=cannot sell 5.confidence<70=hold 6.Consider unrealized PnL"
)}]

RISK_SYSTEM = [{"type": "text", "cache_control": {"type": "ephemeral"}, "text": (
    "You are a risk management expert. Review trader decision, optimize position sizing.\n"
    "Respond ONLY in JSON:\n"
    '{"final":"buy|sell|hold","size_ratio":0.0-0.5,"sell_ratio":0.0-1.0,'
    '"stop_loss_pct":number,"take_profit_pct":number,"risk":1-10,"override":null|"reason"}\n'
    "Rules: 1.Max loss/trade=2% of total 2.Max invest=50% of total "
    "3.SL/TP based on ATR 4.Low confidence=smaller position 5.Trader hold=keep hold "
    "6.Can override if risk too high"
)}]


# ─── Agent Calls ─────────────────────────────────────────

def call_cached(system_blocks, user_msg, max_tok=300, model=None):
    client = get_client()
    resp = client.messages.create(
        model=model or MODEL_SONNET, max_tokens=max_tok,
        system=system_blocks,
        messages=[{"role": "user", "content": user_msg}],
    )
    u = resp.usage
    ci = getattr(u, "cache_read_input_tokens", 0)
    cc = getattr(u, "cache_creation_input_tokens", 0)
    logger.debug(f"Tokens in={u.input_tokens} cached={ci} cache_new={cc} out={u.output_tokens}")
    return resp.content[0].text

def parse_json(text):
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning(f"JSON parse fail: {text[:200]}")
        return {}

def agent_analyst(df, ticker):
    la = df.iloc[-1]
    pr = df.iloc[-2]
    chg = (la["close"] - pr["close"]) / pr["close"] * 100
    data = (
        f"{ticker} P={la['close']:.0f} chg={chg:+.2f}% vol={la['volume']:.2f} vMA20={la['vol_ma20']:.2f}\n"
        f"RSI={la['rsi']:.1f} MACD={la['macd']:.1f}/{la['macd_sig']:.1f}/{la['macd_hist']:.1f}\n"
        f"BB={la['bb_up']:.0f}/{la['bb_mid']:.0f}/{la['bb_lo']:.0f} "
        f"Stoch=K{la['stoch_k']:.1f}/D{la['stoch_d']:.1f} ATR={la['atr']:.0f}\n"
        f"MA=5:{la['ma5']:.0f}/20:{la['ma20']:.0f}/60:{la['ma60']:.0f}/120:{la['ma120']:.0f}\n"
        f"Close10={','.join(f'{v:.0f}' for v in df['close'].tail(10).values)}"
    )
    text = call_cached(ANALYST_SYSTEM, data, 300)
    r = parse_json(text)
    if not r:
        r = {"trend": "sideways", "trend_strength": 5, "key_signals": [], "summary": "fail"}
    logger.info(f"[Analyst] {r.get('trend')} str={r.get('trend_strength')} | {r.get('summary','')[:80]}")
    return r

def agent_trader(analysis, balance, price):
    pnl = ((price / balance["avg_price"] - 1) * 100) if balance["avg_price"] > 0 else 0
    msg = (
        f"Trend={analysis.get('trend')} str={analysis.get('trend_strength')}/10 "
        f"Sig={','.join(analysis.get('key_signals',[]))}\n"
        f"S={analysis.get('support','?')} R={analysis.get('resistance','?')} Vol={analysis.get('volume')}\n"
        f"{analysis.get('summary','')}\n"
        f"P={price:.0f} KRW={balance['krw']:.0f} Coins={balance['coin_qty']} "
        f"Avg={balance['avg_price']:.0f} PnL={pnl:.2f}%"
    )
    text = call_cached(TRADER_SYSTEM, msg, 250, model=MODEL_OPUS)
    r = parse_json(text)
    if not r:
        r = {"decision": "hold", "confidence": 0, "reason": "fail", "urgency": "none"}
    logger.info(f"[Trader] {r['decision'].upper()} conf={r.get('confidence')}% | {r.get('reason','')[:80]}")
    return r

def agent_risk(trader, analysis, balance, atr, price):
    total = balance["krw"] + balance["coin_value"]
    pnl = ((price / balance["avg_price"] - 1) * 100) if balance["avg_price"] > 0 else 0
    cash_pct = (balance["krw"] / total * 100) if total > 0 else 100
    coin_pct = (balance["coin_value"] / total * 100) if total > 0 else 0
    msg = (
        f"Dec={trader['decision']} conf={trader.get('confidence')}% urg={trader.get('urgency')}\n"
        f"Reason: {trader.get('reason')}\n"
        f"Trend={analysis.get('trend')} str={analysis.get('trend_strength')}/10\n"
        f"ATR={atr:.0f} vol%={atr/price*100:.2f}%\n"
        f"Total={total:.0f} Cash={cash_pct:.1f}% Coin={coin_pct:.1f}% PnL={pnl:.2f}%"
    )
    text = call_cached(RISK_SYSTEM, msg, 250)
    r = parse_json(text)
    if not r:
        r = {"final": "hold", "size_ratio": 0, "sell_ratio": 0,
             "stop_loss_pct": 3, "take_profit_pct": 5, "risk": 10, "override": "fail"}
    ov = r.get("override")
    if ov:
        logger.info(f"[Risk] OVERRIDE: {ov}")
    logger.info(
        f"[Risk] {r['final'].upper()} size={r.get('size_ratio',0)*100:.0f}% "
        f"SL=-{r.get('stop_loss_pct')}% TP=+{r.get('take_profit_pct')}% risk={r.get('risk')}/10"
    )
    return r


# ─── Upbit Trader ────────────────────────────────────────

class UpbitTrader:
    def __init__(self):
        ak = os.getenv("UPBIT_ACCESS_KEY")
        sk = os.getenv("UPBIT_SECRET_KEY")
        if not ak or not sk or "your_" in ak:
            logger.warning("Upbit keys not set -> simulation mode")
            self.upbit = None
        else:
            self.upbit = pyupbit.Upbit(ak, sk)

    @property
    def is_live(self):
        return self.upbit is not None

    def get_balance_info(self, ticker):
        coin = ticker.split("-")[1]
        if not self.is_live:
            return {"krw": 1_000_000, "coin_qty": 0, "avg_price": 0, "coin_value": 0}
        krw = self.upbit.get_balance("KRW") or 0
        qty = self.upbit.get_balance(coin) or 0
        avg = self.upbit.get_avg_buy_price(coin) or 0
        cur = pyupbit.get_current_price(ticker) or 0
        return {"krw": krw, "coin_qty": qty, "avg_price": avg, "coin_value": qty * cur}

    def buy(self, ticker, amount):
        if not self.is_live:
            logger.info(f"[SIM] BUY {ticker} {amount:,.0f} KRW")
            return {"simulated": True}
        try:
            r = self.upbit.buy_market_order(ticker, amount)
            logger.info(f"BUY filled: {r}")
            return r
        except Exception as e:
            logger.error(f"BUY fail: {e}")
            return None

    def sell(self, ticker, qty):
        if not self.is_live:
            logger.info(f"[SIM] SELL {ticker} {qty}")
            return {"simulated": True}
        try:
            r = self.upbit.sell_market_order(ticker, qty)
            logger.info(f"SELL filled: {r}")
            return r
        except Exception as e:
            logger.error(f"SELL fail: {e}")
            return None


# ─── Main Bot ────────────────────────────────────────────

class TradingBot:
    def __init__(self):
        self.trader = UpbitTrader()
        self.trade_log = []
        self.sl_price = None
        self.tp_price = None
        logger.info(f"Bot started | {TICKER} | live={self.trader.is_live}")

    def run_once(self):
        try:
            logger.info("=" * 60)
            logger.info(f"Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            df = get_market_data(TICKER, INTERVAL, CANDLE_COUNT)
            price = df["close"].iloc[-1]
            atr = df["atr"].iloc[-1]
            bal = self.trader.get_balance_info(TICKER)

            logger.info(f"P={price:,.0f} ATR={atr:,.0f} KRW={bal['krw']:,.0f} Coins={bal['coin_qty']}")

            # Stop-loss / Take-profit
            if bal["coin_qty"] > 0 and self.sl_price and self.tp_price:
                if price <= self.sl_price:
                    logger.info(f"STOP-LOSS: {price:,.0f} <= {self.sl_price:,.0f}")
                    self.trader.sell(TICKER, bal["coin_qty"])
                    self._log("sell_sl", bal["coin_qty"], {"reason": "stop-loss"})
                    self.sl_price = self.tp_price = None
                    return
                if price >= self.tp_price:
                    logger.info(f"TAKE-PROFIT: {price:,.0f} >= {self.tp_price:,.0f}")
                    self.trader.sell(TICKER, bal["coin_qty"])
                    self._log("sell_tp", bal["coin_qty"], {"reason": "take-profit"})
                    self.sl_price = self.tp_price = None
                    return

            # Multi-agent pipeline
            analysis = agent_analyst(df, TICKER)
            trade_dec = agent_trader(analysis, bal, price)
            risk_dec = agent_risk(trade_dec, analysis, bal, atr, price)

            final = risk_dec["final"]

            if final == "buy":
                ratio = risk_dec.get("size_ratio", 0.1)
                amt = bal["krw"] * ratio
                if amt >= 5000:
                    self.trader.buy(TICKER, amt)
                    sl = risk_dec.get("stop_loss_pct", 3)
                    tp = risk_dec.get("take_profit_pct", 5)
                    self.sl_price = price * (1 - sl / 100)
                    self.tp_price = price * (1 + tp / 100)
                    logger.info(f"SL={self.sl_price:,.0f} TP={self.tp_price:,.0f}")
                    self._log("buy", amt, risk_dec)
                else:
                    logger.info(f"Insufficient: {amt:,.0f} KRW")
            elif final == "sell":
                sr = risk_dec.get("sell_ratio", 1.0)
                qty = bal["coin_qty"] * sr
                if qty > 0:
                    self.trader.sell(TICKER, qty)
                    self._log("sell", qty, risk_dec)
                    if sr >= 1.0:
                        self.sl_price = self.tp_price = None
                else:
                    logger.info("No coins to sell")
            else:
                logger.info("HOLD")

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

    def _log(self, action, amount, details):
        entry = {"time": datetime.now().isoformat(), "action": action,
                 "amount": amount, "details": details}
        self.trade_log.append(entry)
        with open("trade_history.json", "w", encoding="utf-8") as f:
            json.dump(self.trade_log, f, ensure_ascii=False, indent=2)

        # Telegram 알림
        now = datetime.now().strftime("%m/%d %H:%M")
        price = pyupbit.get_current_price(TICKER) or 0
        icons = {"buy": "🟢 매수", "sell": "🔴 매도", "sell_sl": "🛑 손절", "sell_tp": "🎯 익절"}
        label = icons.get(action, action)

        if "buy" in action:
            msg = (
                f"<b>{label}</b> | {now}\n"
                f"코인: {TICKER}\n"
                f"현재가: {price:,.0f}원\n"
                f"투자금: {amount:,.0f}원\n"
                f"SL: {self.sl_price:,.0f}원 / TP: {self.tp_price:,.0f}원" if self.sl_price else ""
            )
        else:
            reason = details.get("reason", details.get("override", ""))
            msg = (
                f"<b>{label}</b> | {now}\n"
                f"코인: {TICKER}\n"
                f"현재가: {price:,.0f}원\n"
                f"수량: {amount}\n"
                f"사유: {reason}"
            )
        send_telegram(msg)

    def start(self):
        logger.info(f"Every {TRADE_INTERVAL_MIN}min | Analyst -> Trader -> Risk")
        self.run_once()
        schedule.every(TRADE_INTERVAL_MIN).minutes.do(self.run_once)
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
