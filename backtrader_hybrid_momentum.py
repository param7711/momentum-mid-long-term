"""Portfolio-level momentum breakout system (Backtrader + yfinance).

Sections:
- data download / cleaning
- momentum ranking
- strategy class
- portfolio management
- backtest execution
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import backtrader as bt
import pandas as pd
import yfinance as yf


# =========================
# Data download / cleaning
# =========================


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df.copy())
    needed = ["open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in downloaded data")
    out = df[needed].dropna().copy()
    out = out[(out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0) & (out["close"] > 0)]
    return out


def get_nifty500_symbols(symbols_file: Optional[str] = None) -> List[str]:
    if symbols_file:
        syms = [s.strip().upper() for s in open(symbols_file, "r", encoding="utf-8") if s.strip()]
        return [s if s.endswith(".NS") else f"{s}.NS" for s in syms]

    # Lightweight fallback list; for full Nifty500 pass --symbols-file.
    base = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC", "LT",
        "SBIN", "BHARTIARTL", "KOTAKBANK", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
        "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND", "WIPRO", "NTPC",
    ]
    return [f"{s}.NS" for s in base]


def download_universe(symbols: List[str], start: str, end: str, min_bars: int = 250) -> Dict[str, pd.DataFrame]:
    data_map: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False)
            if df is None or df.empty:
                continue
            clean = _clean_ohlcv(df)
            if len(clean) < min_bars:
                continue
            data_map[sym] = clean
        except Exception:
            continue
    return data_map


# =========================
# Strategy
# =========================


@dataclass
class PositionState:
    entry_price: float = 0.0
    reached_25: bool = False
    add1_done: bool = False
    add2_done: bool = False
    partial_taken: bool = False
    ma200_breach_count: int = 0
    exit_on_rank_rotation: bool = False
    exited_by_ma200: bool = False
    reentry_watch_days_left: int = 0
    reentry_above_ma200_streak: int = 0


class MomentumBreakoutPortfolio(bt.Strategy):
    params = dict(
        max_positions=20,
        initial_alloc=0.025,
        add1_alloc=0.015,
        add2_alloc=0.010,
        stop_loss_pct=0.18,
        breakout_lookback=110,  # 22 weeks * 5 trading days
        ma_period_fast=100,
        ma_period_slow=200,
        rank_rotation_buffer=5,
        pyramid_new_highs_required=3,
        pyramid_window=20,
        ma200_leniency=0.01,
        ma200_exit_confirm_bars=2,
        reentry_above_ma200_bars=3,
        reentry_window_days=10,
    )

    def __init__(self):
        self.dnames = {d: d._name for d in self.datas}
        self.sma100 = {d: bt.ind.SMA(d.close, period=self.p.ma_period_fast) for d in self.datas}
        self.sma200 = {d: bt.ind.SMA(d.close, period=self.p.ma_period_slow) for d in self.datas}
        self.high110 = {d: bt.ind.Highest(d.close(-1), period=self.p.breakout_lookback) for d in self.datas}

        self.state: Dict[bt.feeds.PandasData, PositionState] = {d: PositionState() for d in self.datas}
        self.pending_order: Dict[bt.feeds.PandasData, Optional[bt.Order]] = {d: None for d in self.datas}

        self.rank_scores: Dict[bt.feeds.PandasData, float] = {d: -1e9 for d in self.datas}
        self.rank_order: List[bt.feeds.PandasData] = []

    def _ready(self, d) -> bool:
        return len(d) > max(self.p.ma_period_slow, 252, self.p.breakout_lookback + 5)

    def _ret(self, d, n: int) -> float:
        if len(d) <= n:
            return 0.0
        return (d.close[0] / d.close[-n]) - 1.0

    def _vol(self, d, n: int = 63) -> float:
        if len(d) <= n + 1:
            return 1e-6
        closes = list(d.close.get(size=n + 1))
        rets = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            if prev == 0:
                continue
            rets.append((closes[i] / prev) - 1.0)
        if not rets:
            return 1e-6
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        return max(math.sqrt(var), 1e-6)

    def _distance_from_breakout(self, d) -> float:
        ref = self.high110[d][0]
        if ref <= 0:
            return 0.0
        return (d.close[0] / ref) - 1.0

    def _momentum_score(self, d) -> float:
        r6 = self._ret(d, 126)
        r12 = self._ret(d, 252)
        vol = self._vol(d, 63)
        dist = self._distance_from_breakout(d)
        early_bonus = max(0.0, 0.05 - abs(dist))
        return ((0.45 * r6) + (0.45 * r12) + (0.10 * early_bonus) - (0.20 * vol))

    def _update_rankings(self):
        scored = []
        for d in self.datas:
            if not self._ready(d):
                self.rank_scores[d] = -1e9
                continue
            s = self._momentum_score(d)
            self.rank_scores[d] = s
            scored.append((d, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        self.rank_order = [d for d, _ in scored]

    def _new_high_count_recent(self, d, window: int, breakout_lookback: int) -> int:
        if len(d) < breakout_lookback + window + 2:
            return 0
        count = 0
        highs = list(d.close.get(size=breakout_lookback + window + 1))
        # count last `window` days where close made a new 110-day high vs prior day history
        for i in range(window):
            idx = -(window - i)
            curr = highs[idx]
            prior_slice = highs[idx - breakout_lookback: idx]
            if prior_slice and curr >= max(prior_slice):
                count += 1
        return count

    def _eligible_for_entry(self, d) -> bool:
        st = self.state[d]
        pos = self.getposition(d)
        if pos.size > 0 or self.pending_order[d] is not None:
            return False

        if not self._ready(d):
            return False

        in_top = d in self.rank_order[: self.p.max_positions]
        cond_breakout = d.close[0] > self.high110[d][0]
        cond_ma200 = d.close[0] > self.sma200[d][0]

        if st.reentry_watch_days_left > 0:
            # re-entry: 3 days > MA200 and then new 22-week high within 10 days
            if d.close[0] > self.sma200[d][0]:
                st.reentry_above_ma200_streak += 1
            else:
                st.reentry_above_ma200_streak = 0
            st.reentry_watch_days_left -= 1
            if st.reentry_above_ma200_streak < self.p.reentry_above_ma200_bars:
                return False
            if not cond_breakout:
                return False
        return in_top and cond_breakout and cond_ma200

    def _position_value(self, d) -> float:
        p = self.getposition(d)
        return p.size * d.close[0]

    def _portfolio_value(self) -> float:
        return self.broker.getvalue()

    def _buy_value(self, d, alloc: float):
        pv = self._portfolio_value()
        target_value = pv * alloc
        size = int(target_value / max(d.close[0], 1e-9))
        if size <= 0:
            return
        self.pending_order[d] = self.buy(data=d, size=size)

    def _process_entries(self):
        open_positions = [d for d in self.datas if self.getposition(d).size > 0]
        if len(open_positions) >= self.p.max_positions:
            return

        for d in self.rank_order:
            if len(open_positions) >= self.p.max_positions:
                break
            if self._eligible_for_entry(d):
                self._buy_value(d, self.p.initial_alloc)
                open_positions.append(d)

    def _process_rotation(self):
        holdings = [d for d in self.datas if self.getposition(d).size > 0]
        if not holdings:
            return
        if len(self.rank_order) < self.p.max_positions:
            return

        top_set = set(self.rank_order[: self.p.max_positions])
        best_candidate = self.rank_order[0] if self.rank_order else None
        weakest_holding = min(holdings, key=lambda x: self.rank_scores.get(x, -1e9))

        if weakest_holding not in top_set and best_candidate is not None and best_candidate not in holdings:
            if self.rank_scores.get(best_candidate, -1e9) > self.rank_scores.get(weakest_holding, -1e9):
                self.state[weakest_holding].exit_on_rank_rotation = True

    def _process_exits(self):
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size <= 0 or self.pending_order[d] is not None:
                continue

            st = self.state[d]
            entry = st.entry_price if st.entry_price > 0 else pos.price
            pnl = (d.close[0] / max(entry, 1e-9)) - 1.0

            # initial stop
            if d.close[0] <= entry * (1.0 - self.p.stop_loss_pct):
                self.pending_order[d] = self.close(data=d)
                continue

            # milestone and stop regime switch
            if pnl >= 0.25:
                st.reached_25 = True

            if not st.reached_25:
                if d.close[0] < self.sma100[d][0]:
                    self.pending_order[d] = self.close(data=d)
                    continue
            else:
                # breakeven floor always active after +25%
                if d.close[0] <= entry:
                    self.pending_order[d] = self.close(data=d)
                    continue

                lenient_floor = self.sma200[d][0] * (1.0 - self.p.ma200_leniency)
                if d.close[0] < lenient_floor:
                    st.ma200_breach_count += 1
                else:
                    st.ma200_breach_count = 0

                if st.ma200_breach_count >= self.p.ma200_exit_confirm_bars:
                    st.exited_by_ma200 = True
                    st.reentry_watch_days_left = self.p.reentry_window_days
                    st.reentry_above_ma200_streak = 0
                    self.pending_order[d] = self.close(data=d)
                    continue

            if st.exit_on_rank_rotation:
                self.pending_order[d] = self.close(data=d)
                continue

    def _process_pyramids(self):
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size <= 0 or self.pending_order[d] is not None:
                continue

            st = self.state[d]
            entry = st.entry_price if st.entry_price > 0 else pos.price
            pnl = (d.close[0] / max(entry, 1e-9)) - 1.0
            nh_count = self._new_high_count_recent(d, self.p.pyramid_window, self.p.breakout_lookback)

            if (not st.add1_done) and pnl >= 0.20 and nh_count >= self.p.pyramid_new_highs_required:
                self._buy_value(d, self.p.add1_alloc)
                st.add1_done = True
                continue

            if st.add1_done and (not st.add2_done) and pnl >= 0.40 and nh_count >= self.p.pyramid_new_highs_required:
                self._buy_value(d, self.p.add2_alloc)
                st.add2_done = True

    def notify_order(self, order):
        d = order.data
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                st = self.state[d]
                # weighted average entry after adds
                pos = self.getposition(d)
                st.entry_price = pos.price if pos.size > 0 else order.executed.price
            elif order.issell():
                if self.getposition(d).size <= 0:
                    st = self.state[d]
                    reentry_days = st.reentry_watch_days_left if st.exited_by_ma200 else 0
                    self.state[d] = PositionState(
                        reentry_watch_days_left=reentry_days,
                        exited_by_ma200=st.exited_by_ma200,
                    )
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.pending_order[d] = None

    def next(self):
        self._update_rankings()
        self._process_rotation()
        self._process_exits()
        self._process_pyramids()
        self._process_entries()


# =========================
# Backtest execution
# =========================


def add_data_to_cerebro(cerebro: bt.Cerebro, data_map: Dict[str, pd.DataFrame]) -> None:
    for sym, df in data_map.items():
        feed = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(feed, name=sym)


def summarize_trade_stats(strat) -> Tuple[int, int, int, float, float, float]:
    analyzer = strat.analyzers.ta.get_analysis()
    total = int(analyzer.get("total", {}).get("closed", 0) or 0)
    won = int(analyzer.get("won", {}).get("total", 0) or 0)
    lost = int(analyzer.get("lost", {}).get("total", 0) or 0)
    avg_win = float(analyzer.get("won", {}).get("pnl", {}).get("average", 0.0) or 0.0)
    avg_loss = float(analyzer.get("lost", {}).get("pnl", {}).get("average", 0.0) or 0.0)
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0
    return total, won, lost, avg_win, avg_loss, payoff


def run_backtest(symbols: List[str], start: str, end: str, initial_capital: float = 100000.0):
    data_map = download_universe(symbols, start=start, end=end, min_bars=250)
    if not data_map:
        raise RuntimeError("No valid data downloaded. Check symbols/network/date range.")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.001)

    add_data_to_cerebro(cerebro, data_map)
    cerebro.addstrategy(MomentumBreakoutPortfolio)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")

    start_val = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    end_val = cerebro.broker.getvalue()
    ret = ((end_val / start_val) - 1.0) * 100.0

    total, won, lost, avg_win, avg_loss, payoff = summarize_trade_stats(strat)

    print(f"Starting capital: {start_val:.2f}")
    print(f"Final portfolio value: {end_val:.2f}")
    print(f"Total return (%): {ret:.2f}")
    print("Trade analysis:")
    print(f"  total trades: {total}")
    print(f"  winning trades: {won}")
    print(f"  losing trades: {lost}")
    print(f"  average win: {avg_win:.2f}")
    print(f"  average loss: {avg_loss:.2f}")
    print(f"  payoff ratio: {payoff:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty momentum breakout portfolio backtest")
    parser.add_argument("--symbols-file", default="", help="Optional file with Nifty 500 symbols (one per line)")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--capital", type=float, default=100000.0)
    args = parser.parse_args()

    universe = get_nifty500_symbols(args.symbols_file or None)
    run_backtest(universe, start=args.start, end=args.end, initial_capital=args.capital)
