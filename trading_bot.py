# -*- coding: utf-8 -*-
"""
업비트 멀티 에이전트 시스템 트레이딩 봇

┌─────────────────────────────────────────────────┐
│  Agent 1. 시장 분석가    │  Claude Sonnet 4      │
│  Agent 2. 트레이더       │  Claude Sonnet 4      │
│  Agent 3. 리스크 매니저  │  Claude Sonnet 4      │
└─────────────────────────────────────────────────┘

투자 대상: BTC, ETH, XRP, SOL, DOGE, ADA (시총 상위 6개)
- Anthropic 프롬프트 캐싱으로 비용 최적화
- 영어 시스템 프롬프트로 토큰 효율 극대화
- 15분 간격 자동 매매 / 손절·익절 자동 관리
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
TICKERS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE", "KRW-ADA"]
INTERVAL = "minute60"
CANDLE_COUNT = 200
TRADE_INTERVAL_MIN = 30
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-20250514"

# ─── Telegram ────────────────────────────────────────────

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg: str):
    if not TG_TOKEN or not TG_CHAT or "your_" in TG_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        req_lib.post(url, json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram fail: {e}")

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

# ─── Cached System Prompts ───────────────────────────────

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
    "You are a crypto portfolio trader managing multiple coins.\n"
    "Decide buy/sell/hold based on analyst report and portfolio.\n"
    "Respond ONLY in JSON:\n"
    '{"decision":"buy|sell|hold","confidence":0-100,"reason":"2-3 sentences","urgency":"immediate|wait|none"}\n'
    "Rules: 1.Enter only in trend direction 2.Sideways=hold 3.If holding,focus on hold/sell "
    "4.No coins=cannot sell 5.confidence<70=hold 6.Consider unrealized PnL "
    "7.Consider portfolio diversification across multiple coins"
)}]

RISK_SYSTEM = [{"type": "text", "cache_control": {"type": "ephemeral"}, "text": (
    "You are a risk management expert for a multi-coin portfolio.\n"
    "Review trader decision, optimize position sizing.\n"
    "Respond ONLY in JSON:\n"
    '{"final":"buy|sell|hold","size_ratio":0.0-0.5,"sell_ratio":0.0-1.0,'
    '"stop_loss_pct":number,"take_profit_pct":number,"risk":1-10,"override":null|"reason"}\n'
    "Rules: 1.Max loss/trade=2% of total 2.Max invest per coin=30% of total "
    "3.Total invested across all coins max 80% 4.SL/TP based on ATR "
    "5.Low confidence=smaller position 6.Trader hold=keep hold "
    "7.Can override if risk too high 8.Consider existing positions in other coins"
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
    logger.debug(f"Tokens in={u.input_tokens} cached={ci} new={cc} out={u.output_tokens}")
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
    logger.info(f"  [Analyst] {r.get('trend')} str={r.get('trend_strength')} | {r.get('summary','')[:80]}")
    return r

def agent_trader(analysis, balance, price, ticker, portfolio_summary):
    pnl = ((price / balance["avg_price"] - 1) * 100) if balance["avg_price"] > 0 else 0
    msg = (
        f"Coin: {ticker}\n"
        f"Trend={analysis.get('trend')} str={analysis.get('trend_strength')}/10 "
        f"Sig={','.join(analysis.get('key_signals',[]))}\n"
        f"S={analysis.get('support','?')} R={analysis.get('resistance','?')} Vol={analysis.get('volume')}\n"
        f"{analysis.get('summary','')}\n"
        f"P={price:.0f} KRW={balance['krw']:.0f} Coins={balance['coin_qty']} "
        f"Avg={balance['avg_price']:.0f} PnL={pnl:.2f}%\n"
        f"Portfolio: {portfolio_summary}"
    )
    text = call_cached(TRADER_SYSTEM, msg, 250)
    r = parse_json(text)
    if not r:
        r = {"decision": "hold", "confidence": 0, "reason": "fail", "urgency": "none"}
    logger.info(f"  [Trader] {r['decision'].upper()} conf={r.get('confidence')}% | {r.get('reason','')[:80]}")
    return r

def agent_risk(trader, analysis, balance, atr, price, ticker, total_invested_pct):
    total = balance["krw"] + balance["coin_value"]
    pnl = ((price / balance["avg_price"] - 1) * 100) if balance["avg_price"] > 0 else 0
    cash_pct = (balance["krw"] / total * 100) if total > 0 else 100
    coin_pct = (balance["coin_value"] / total * 100) if total > 0 else 0
    msg = (
        f"Coin: {ticker}\n"
        f"Dec={trader['decision']} conf={trader.get('confidence')}% urg={trader.get('urgency')}\n"
        f"Reason: {trader.get('reason')}\n"
        f"Trend={analysis.get('trend')} str={analysis.get('trend_strength')}/10\n"
        f"ATR={atr:.0f} vol%={atr/price*100:.2f}%\n"
        f"Total={total:.0f} Cash={cash_pct:.1f}% ThisCoin={coin_pct:.1f}% PnL={pnl:.2f}%\n"
        f"TotalInvestedAcrossAllCoins={total_invested_pct:.1f}%"
    )
    text = call_cached(RISK_SYSTEM, msg, 250)
    r = parse_json(text)
    if not r:
        r = {"final": "hold", "size_ratio": 0, "sell_ratio": 0,
             "stop_loss_pct": 3, "take_profit_pct": 5, "risk": 10, "override": "fail"}
    ov = r.get("override")
    if ov:
        logger.info(f"  [Risk] OVERRIDE: {ov}")
    logger.info(
        f"  [Risk] {r['final'].upper()} size={r.get('size_ratio',0)*100:.0f}% "
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

    def get_total_assets(self, tickers):
        """전체 자산 계산 (KRW + 모든 코인 평가액)"""
        if not self.is_live:
            return 1_000_000
        krw = self.upbit.get_balance("KRW") or 0
        total = krw
        for t in tickers:
            coin = t.split("-")[1]
            qty = self.upbit.get_balance(coin) or 0
            cur = pyupbit.get_current_price(t) or 0
            total += qty * cur
        return total

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
        self.sl_tp = {}  # {ticker: {"sl": price, "tp": price}}
        logger.info(f"Bot started | {len(TICKERS)} coins | live={self.trader.is_live}")

    def _get_portfolio_summary(self):
        """전체 포트폴리오 요약 문자열"""
        parts = []
        for t in TICKERS:
            bal = self.trader.get_balance_info(t)
            if bal["coin_qty"] > 0:
                cur = pyupbit.get_current_price(t) or 0
                pnl = ((cur / bal["avg_price"] - 1) * 100) if bal["avg_price"] > 0 else 0
                parts.append(f"{t.split('-')[1]}:{bal['coin_value']:.0f}KRW({pnl:+.1f}%)")
        return ", ".join(parts) if parts else "No positions"

    def _get_total_invested_pct(self):
        """전체 자산 대비 투자 비율"""
        total = self.trader.get_total_assets(TICKERS)
        if total <= 0:
            return 0
        krw = self.trader.get_balance_info(TICKERS[0])["krw"]
        return ((total - krw) / total) * 100

    def run_once(self):
        try:
            logger.info("=" * 60)
            logger.info(f"Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            portfolio_summary = self._get_portfolio_summary()
            total_invested_pct = self._get_total_invested_pct()
            logger.info(f"Portfolio: {portfolio_summary} | Invested: {total_invested_pct:.1f}%")

            cycle_results = []  # 사이클 결과 수집

            for ticker in TICKERS:
                try:
                    result = self._process_ticker(ticker, portfolio_summary, total_invested_pct)
                    if result:
                        cycle_results.append(result)
                except Exception as e:
                    logger.error(f"{ticker} error: {e}")
                time.sleep(1)  # API rate limit

            # 사이클 종료 후 종합 운용보고서 발송
            self._send_portfolio_report(cycle_results, total_invested_pct)

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)


    def _process_ticker(self, ticker, portfolio_summary, total_invested_pct):
        logger.info(f"--- {ticker} ---")
        df = get_market_data(ticker, INTERVAL, CANDLE_COUNT)
        price = df["close"].iloc[-1]
        atr = df["atr"].iloc[-1]
        bal = self.trader.get_balance_info(ticker)

        logger.info(f"  P={price:,.0f} ATR={atr:,.0f} Coins={bal['coin_qty']}")

        # Stop-loss / Take-profit check
        stp = self.sl_tp.get(ticker)
        if bal["coin_qty"] > 0 and stp:
            if price <= stp["sl"]:
                logger.info(f"  STOP-LOSS: {price:,.0f} <= {stp['sl']:,.0f}")
                self.trader.sell(ticker, bal["coin_qty"])
                self._log(ticker, "sell_sl", bal["coin_qty"], price, {"reason": "stop-loss"}, None)
                del self.sl_tp[ticker]
                return {"ticker": ticker, "price": price, "action": "sell_sl", "analysis": None, "risk": None, "bal": bal}
            if price >= stp["tp"]:
                logger.info(f"  TAKE-PROFIT: {price:,.0f} >= {stp['tp']:,.0f}")
                self.trader.sell(ticker, bal["coin_qty"])
                self._log(ticker, "sell_tp", bal["coin_qty"], price, {"reason": "take-profit"}, None)
                del self.sl_tp[ticker]
                return {"ticker": ticker, "price": price, "action": "sell_tp", "analysis": None, "risk": None, "bal": bal}

        # Multi-agent pipeline
        analysis = agent_analyst(df, ticker)
        trade_dec = agent_trader(analysis, bal, price, ticker, portfolio_summary)
        risk_dec = agent_risk(trade_dec, analysis, bal, atr, price, ticker, total_invested_pct)

        final = risk_dec["final"]

        if final == "buy":
            ratio = risk_dec.get("size_ratio", 0.1)
            amt = bal["krw"] * ratio
            if amt >= 5000:
                self.trader.buy(ticker, amt)
                sl = risk_dec.get("stop_loss_pct", 3)
                tp = risk_dec.get("take_profit_pct", 5)
                self.sl_tp[ticker] = {
                    "sl": price * (1 - sl / 100),
                    "tp": price * (1 + tp / 100),
                }
                logger.info(f"  SL={self.sl_tp[ticker]['sl']:,.0f} TP={self.sl_tp[ticker]['tp']:,.0f}")
                self._log(ticker, "buy", amt, price, risk_dec, analysis)
            else:
                logger.info(f"  Insufficient: {amt:,.0f} KRW")
        elif final == "sell":
            sr = risk_dec.get("sell_ratio", 1.0)
            qty = bal["coin_qty"] * sr
            if qty > 0:
                self.trader.sell(ticker, qty)
                self._log(ticker, "sell", qty, price, risk_dec, analysis)
                if sr >= 1.0 and ticker in self.sl_tp:
                    del self.sl_tp[ticker]
            else:
                logger.info(f"  No coins to sell")
        else:
            logger.info(f"  HOLD")

        return {
            "ticker": ticker, "price": price, "action": final,
            "analysis": analysis, "risk": risk_dec, "bal": bal,
            "trade": trade_dec,
        }

    def _log(self, ticker, action, amount, price, details, analysis):
        entry = {"time": datetime.now().isoformat(), "ticker": ticker,
                 "action": action, "amount": amount, "price": price, "details": details}
        self.trade_log.append(entry)
        with open("trade_history.json", "w", encoding="utf-8") as f:
            json.dump(self.trade_log, f, ensure_ascii=False, indent=2)

        now = datetime.now().strftime("%Y.%m.%d %H:%M")
        coin = ticker.split("-")[1]
        icons = {"buy": "🟢", "sell": "🔴", "sell_sl": "🛑", "sell_tp": "🎯"}
        labels = {"buy": "매수 체결", "sell": "매도 체결", "sell_sl": "손절 체결", "sell_tp": "익절 체결"}
        icon = icons.get(action, "📌")
        label = labels.get(action, action)

        # 트렌드 이모지
        trend = analysis.get("trend", "sideways") if analysis else "N/A"
        trend_str_val = analysis.get("trend_strength", 0) if analysis else 0
        trend_icons = {"bullish": "📈", "bearish": "📉", "sideways": "➡️"}
        trend_icon = trend_icons.get(trend, "➡️")

        # 시그널 요약
        signals = analysis.get("key_signals", []) if analysis else []
        sig_str = " / ".join(signals[:3]) if signals else "없음"

        # 리스크 정보
        risk_score = details.get("risk", 0) if isinstance(details, dict) else 0
        risk_bar = "🟩" * (10 - risk_score) + "🟥" * risk_score
        conf = details.get("confidence", 0) if isinstance(details, dict) else 0

        bal = self.trader.get_balance_info(ticker)
        total = bal["krw"] + bal["coin_value"]

        if action == "buy":
            stp = self.sl_tp.get(ticker, {})
            sl_str = f"{stp.get('sl',0):,.0f}" if stp else "?"
            tp_str = f"{stp.get('tp',0):,.0f}" if stp else "?"
            sl_pct = details.get("stop_loss_pct", 0) if isinstance(details, dict) else 0
            tp_pct = details.get("take_profit_pct", 0) if isinstance(details, dict) else 0
            ratio = details.get("size_ratio", 0) if isinstance(details, dict) else 0

            msg = (
                f"{icon} <b>━━━ 매매 체결 보고서 ━━━</b>\n"
                f"\n"
                f"📋 <b>{label} | {coin}</b>\n"
                f"🕐 {now}\n"
                f"\n"
                f"{'─' * 28}\n"
                f"💰 <b>체결 정보</b>\n"
                f"{'─' * 28}\n"
                f"  현재가: <b>{price:,.0f}</b>원\n"
                f"  투자금: <b>{amount:,.0f}</b>원\n"
                f"  비중: 보유현금의 {ratio*100:.0f}%\n"
                f"\n"
                f"{'─' * 28}\n"
                f"🛡️ <b>리스크 관리</b>\n"
                f"{'─' * 28}\n"
                f"  손절가: {sl_str}원 (-{sl_pct}%)\n"
                f"  익절가: {tp_str}원 (+{tp_pct}%)\n"
                f"  리스크: {risk_bar} {risk_score}/10\n"
                f"\n"
                f"{'─' * 28}\n"
                f"📊 <b>AI 분석 근거</b>\n"
                f"{'─' * 28}\n"
                f"  추세: {trend_icon} {trend} (강도 {trend_str_val}/10)\n"
                f"  신호: {sig_str}\n"
                f"  확신도: {conf}%\n"
                f"\n"
                f"{'─' * 28}\n"
                f"💼 <b>잔고 현황</b>\n"
                f"{'─' * 28}\n"
                f"  보유 KRW: {bal['krw']:,.0f}원\n"
                f"  총 자산: {total:,.0f}원\n"
            )
        else:
            reason = ""
            if isinstance(details, dict):
                reason = details.get("reason", details.get("override", details.get("reason", "")))
            pnl = ((price / bal["avg_price"] - 1) * 100) if bal["avg_price"] > 0 else 0
            pnl_icon = "📈" if pnl >= 0 else "📉"

            msg = (
                f"{icon} <b>━━━ 매매 체결 보고서 ━━━</b>\n"
                f"\n"
                f"📋 <b>{label} | {coin}</b>\n"
                f"🕐 {now}\n"
                f"\n"
                f"{'─' * 28}\n"
                f"💰 <b>체결 정보</b>\n"
                f"{'─' * 28}\n"
                f"  현재가: <b>{price:,.0f}</b>원\n"
                f"  매도 수량: {amount}\n"
                f"  {pnl_icon} 수익률: <b>{pnl:+.2f}%</b>\n"
                f"\n"
                f"{'─' * 28}\n"
                f"📊 <b>AI 분석 근거</b>\n"
                f"{'─' * 28}\n"
                f"  추세: {trend_icon} {trend} (강도 {trend_str_val}/10)\n"
                f"  신호: {sig_str}\n"
                f"  사유: {reason}\n"
                f"  리스크: {risk_bar} {risk_score}/10\n"
                f"\n"
                f"{'─' * 28}\n"
                f"💼 <b>잔고 현황</b>\n"
                f"{'─' * 28}\n"
                f"  보유 KRW: {bal['krw']:,.0f}원\n"
                f"  총 자산: {total:,.0f}원\n"
            )
        send_telegram(msg)

    def _send_portfolio_report(self, cycle_results, total_invested_pct):
        """사이클 종료 후 종합 운용보고서 텔레그램 발송"""
        now = datetime.now().strftime("%Y.%m.%d %H:%M")
        total_assets = self.trader.get_total_assets(TICKERS)
        krw_bal = self.trader.get_balance_info(TICKERS[0])["krw"]

        # 코인별 현황 수집
        coin_lines = []
        total_coin_value = 0
        actions_taken = []

        for r in cycle_results:
            if not r:
                continue
            ticker = r["ticker"]
            coin = ticker.split("-")[1]
            price = r["price"]
            action = r["action"]
            bal = r.get("bal", {})
            analysis = r.get("analysis")
            trade = r.get("trade")

            coin_value = bal.get("coin_value", 0)
            total_coin_value += coin_value

            # 추세 이모지
            trend = analysis.get("trend", "?") if analysis else "?"
            trend_icons = {"bullish": "📈", "bearish": "📉", "sideways": "➡️"}
            t_icon = trend_icons.get(trend, "❓")

            # 결정 이모지
            act_icons = {"buy": "🟢매수", "sell": "🔴매도", "hold": "⏸홀드",
                         "sell_sl": "🛑손절", "sell_tp": "🎯익절"}
            act_str = act_icons.get(action, action)

            # 보유 여부
            if bal.get("coin_qty", 0) > 0 and bal.get("avg_price", 0) > 0:
                pnl = ((price / bal["avg_price"] - 1) * 100)
                pnl_str = f"{pnl:+.1f}%"
                hold_str = f"💎 보유중 ({pnl_str})"
            else:
                hold_str = "🔲 미보유"

            # 신뢰도
            conf = ""
            if trade:
                conf = f" | 확신 {trade.get('confidence', '?')}%"

            coin_lines.append(
                f"  {coin:>5} | {price:>12,.0f}원 | {t_icon}{trend:>8} | {act_str}{conf}\n"
                f"         {hold_str}"
            )

            if action != "hold":
                actions_taken.append(f"{act_str} {coin}")

        # 투자 비율 바
        inv_pct = total_invested_pct
        bar_filled = int(inv_pct / 5)
        bar_empty = 20 - bar_filled
        inv_bar = "▓" * bar_filled + "░" * bar_empty

        # 액션 요약
        if actions_taken:
            action_summary = " / ".join(actions_taken)
        else:
            action_summary = "전 종목 홀드 (관망)"

        msg = (
            f"📊 <b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n"
            f"    <b>투자 운용 보고서</b>\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>\n"
            f"🕐 {now} | 주기: {TRADE_INTERVAL_MIN}분\n"
            f"\n"
            f"{'─' * 28}\n"
            f"💼 <b>포트폴리오 총괄</b>\n"
            f"{'─' * 28}\n"
            f"  총 자산: <b>{total_assets:,.0f}</b>원\n"
            f"  보유 현금: {krw_bal:,.0f}원\n"
            f"  투자 평가액: {total_coin_value:,.0f}원\n"
            f"  투자 비율: [{inv_bar}] {inv_pct:.1f}%\n"
            f"\n"
            f"{'─' * 28}\n"
            f"🪙 <b>종목별 분석 현황</b>\n"
            f"{'─' * 28}\n"
        )

        for line in coin_lines:
            msg += f"{line}\n"

        msg += (
            f"\n"
            f"{'─' * 28}\n"
            f"⚡ <b>이번 사이클 액션</b>\n"
            f"{'─' * 28}\n"
            f"  {action_summary}\n"
            f"\n"
            f"{'─' * 28}\n"
            f"🤖 <b>시스템 상태</b>\n"
            f"{'─' * 28}\n"
            f"  모델: {MODEL_SONNET}\n"
            f"  모니터링: {len(TICKERS)}개 종목\n"
            f"  다음 분석: {TRADE_INTERVAL_MIN}분 후\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
        )

        send_telegram(msg)
        logger.info("Portfolio report sent to Telegram")

    def start(self):
        coins = ", ".join(t.split("-")[1] for t in TICKERS)
        logger.info(f"Every {TRADE_INTERVAL_MIN}min | {coins} | Analyst->Trader->Risk")
        now = datetime.now().strftime("%Y.%m.%d %H:%M")
        start_msg = (
            f"🤖 <b>━━━ 트레이딩 봇 가동 ━━━</b>\n"
            f"\n"
            f"🕐 {now}\n"
            f"📡 상태: <b>ONLINE</b>\n"
            f"\n"
            f"{'─' * 28}\n"
            f"⚙️ <b>운용 설정</b>\n"
            f"{'─' * 28}\n"
            f"  모니터링: {coins}\n"
            f"  분석 주기: {TRADE_INTERVAL_MIN}분\n"
            f"  캔들: {INTERVAL} × {CANDLE_COUNT}개\n"
            f"  AI 모델: {MODEL_SONNET}\n"
            f"  에이전트: 분석가 → 트레이더 → 리스크\n"
            f"\n"
            f"{'─' * 28}\n"
            f"🛡️ <b>리스크 정책</b>\n"
            f"{'─' * 28}\n"
            f"  최대 손실/건: 2%\n"
            f"  종목당 최대 비중: 30%\n"
            f"  총 투자 한도: 80%\n"
            f"  자동 손절/익절: ✅\n"
            f"\n"
            f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</b>"
        )
        send_telegram(start_msg)
        self.run_once()
        schedule.every(TRADE_INTERVAL_MIN).minutes.do(self.run_once)
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    bot = TradingBot()
    bot.start()
