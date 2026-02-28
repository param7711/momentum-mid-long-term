"""Backtrader implementation of the Pine strategy:
Hybrid Trend + Swing Momentum System (Refurbished v6)
with an optional portfolio-level allocation/ranking overlay.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backtrader as bt


@dataclass
class BacktestConfig:
    cash: float = 100000.0
    percent_size: float = 4.0
    commission: float = 0.0


@dataclass
class OverlayConfig:
    enabled: bool = False
    entry_rank_threshold: int = 25
    exit_rank_threshold: int = 40
    improvement_threshold: float = 0.05
    permanent_cash_reserve: float = 0.10
    base_exposure: float = 0.90
    max_total_exposure: float = 1.20
    min_position_weight: float = 0.05
    allow_margin_on_high_breadth: bool = True


class V6SignalEngine:
    """Pure V6 signal logic. Indicators and trigger logic intentionally unchanged."""

    def __init__(self, data: bt.feeds.DataBase) -> None:
        self.data = data

        # Moving averages
        self.ma20 = bt.ind.EMA(data.close, period=20)
        self.ma50 = bt.ind.SMA(data.close, period=50)
        self.ma100 = bt.ind.SMA(data.close, period=100)
        self.ma200 = bt.ind.SMA(data.close, period=200)

        # RSI + ATR
        self.rsi = bt.ind.RSI(data.close, period=14)
        self.atr = bt.ind.ATR(data, period=14)
        self.atr_ma = bt.ind.SMA(self.atr, period=50)

        # 52-week high + recent high count
        self.high52 = bt.ind.Highest(data.close, period=252)
        self.new_high = data.close >= self.high52
        self.recent_high_count = bt.ind.SumN(self.new_high, period=15)

        # Golden cross
        self.golden_cross = bt.ind.CrossOver(self.ma50, self.ma200)

        # Highest high breakout helper
        self.highest20 = bt.ind.Highest(data.high, period=20)

    def ready(self) -> bool:
        return len(self.data) >= 252

    def early_trend_condition(self) -> bool:
        return (
            self.ma20[0] > self.ma50[0]
            and self.data.close[0] > self.ma100[0]
            and self.ma50[0] > self.ma50[-5]
        )

    def confirmation_condition(self) -> bool:
        trend_aligned = (self.ma20[0] > self.ma50[0]) and (self.ma50[0] > self.ma200[0])
        bull_market = self.data.close[0] > self.ma200[0]

        momentum_entry = trend_aligned and (self.recent_high_count[0] >= 3)
        golden_cross_entry = self.golden_cross[0] > 0

        bullish_fvg = self.data.low[-1] > self.data.high[-2]
        breakout20 = self.data.close[0] > self.highest20[-1]
        fvg_entry = bullish_fvg and breakout20 and (self.data.close[0] > self.ma50[0])

        pullback = (self.data.close[0] <= self.ma20[0]) or (self.data.close[0] <= self.ma50[0])
        rsi_reset = 40 < self.rsi[0] < 55
        reversal_trigger = self.data.close[0] > self.data.high[-1]
        swing_entry = bull_market and trend_aligned and pullback and rsi_reset and reversal_trigger

        volatility_expansion = self.atr[0] > self.atr_ma[0]

        return volatility_expansion and (momentum_entry or golden_cross_entry or fvg_entry or swing_entry)

    def entry_signals(self) -> Dict[str, bool]:
        return {
            "probe_entry": self.early_trend_condition(),
            "confirm_entry": self.confirmation_condition(),
        }

    def long_condition(self) -> bool:
        return self.confirmation_condition()

    def exit_flags(self, bars_in_trade: int, entry_price: float, proven_trend_pct: float, early_failure_bars: int) -> Tuple[bool, bool, bool]:
        profit_percent = ((self.data.close[0] - entry_price) / entry_price) * 100.0
        proven_trend = profit_percent > proven_trend_pct

        early_failure = (bars_in_trade < early_failure_bars) and (self.data.close[0] < self.ma50[0])
        medium_exit = (not proven_trend) and (self.data.close[0] < self.ma100[0])
        final_exit = proven_trend and (self.data.close[0] < self.ma200[0])
        return early_failure, medium_exit, final_exit


class HybridTrendSwingMomentum(bt.Strategy):
    """Original single-asset V6 behavior."""

    params = dict(stop_loss_pct=0.18, early_failure_bars=40, proven_trend_pct=25.0)

    def __init__(self) -> None:
        self.engine = V6SignalEngine(self.data)
        self.entry_order = None
        self.stop_order = None
        self.bars_in_trade = 0

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.entry_order = None
                self.bars_in_trade = 0
                stop_price = order.executed.price * (1.0 - self.p.stop_loss_pct)
                self.stop_order = self.sell(exectype=bt.Order.Stop, price=stop_price)
            elif order.issell():
                if order == self.stop_order:
                    self.stop_order = None
                self.entry_order = None

        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order == self.entry_order:
                self.entry_order = None
            if order == self.stop_order:
                self.stop_order = None

    def close_position(self) -> None:
        if self.stop_order is not None:
            self.cancel(self.stop_order)
            self.stop_order = None
        self.close()

    def next(self) -> None:
        if not self.engine.ready():
            return

        long_condition = self.engine.long_condition()

        if not self.position and self.entry_order is None and long_condition:
            self.entry_order = self.buy()
            return

        if self.position:
            self.bars_in_trade += 1
            early_failure, medium_exit, final_exit = self.engine.exit_flags(
                bars_in_trade=self.bars_in_trade,
                entry_price=self.position.price,
                proven_trend_pct=self.p.proven_trend_pct,
                early_failure_bars=self.p.early_failure_bars,
            )
            if early_failure or medium_exit or final_exit:
                self.close_position()


class HybridMomentumPortfolioOverlay(bt.Strategy):
    """Portfolio allocator/ranking overlay on top of unchanged V6 signals."""

    params = dict(
        stop_loss_pct=0.18,
        early_failure_bars=40,
        proven_trend_pct=25.0,
        overlay_enabled=True,
        entry_rank_threshold=25,
        exit_rank_threshold=40,
        improvement_threshold=0.05,
        permanent_cash_reserve=0.10,
        base_exposure=0.90,
        max_total_exposure=1.20,
        min_position_weight=0.05,
        allow_margin_on_high_breadth=True,
        index_data_name="",
        probe_size_fraction=0.35,
        probe_stop_atr_mult=1.2,
        confirm_stop_atr_mult=2.0,
        volatility_lookback=20,
        rs_lookback=60,
        sector_index_prefix="SECTOR_",
        relative_strength_threshold=1.05,
        max_sector_positions=4,
        max_pyramid_adds=2,
        min_bars_between_adds=10,
        pyramid_add_fraction=0.25,
        partial_profit_trigger=25.0,
        partial_profit_take_fraction=0.30,
        ma200_reentry_bars=3,
        ma200_reentry_high_period=252,
        exposure_smoothing_alpha=0.15,
    )

    def __init__(self) -> None:
        self.engines: Dict[bt.feeds.DataBase, V6SignalEngine] = {d: V6SignalEngine(d) for d in self.datas}
        self.entry_orders: Dict[bt.feeds.DataBase, Optional[bt.Order]] = {d: None for d in self.datas}
        self.stop_orders: Dict[bt.feeds.DataBase, Optional[bt.Order]] = {d: None for d in self.datas}
        self.bars_in_trade: Dict[bt.feeds.DataBase, int] = {d: 0 for d in self.datas}

        self.last_week_key: Optional[Tuple[int, int]] = None
        self.ranked_list: List[bt.feeds.DataBase] = []
        self.rank_map: Dict[bt.feeds.DataBase, int] = {}
        self.score_map: Dict[bt.feeds.DataBase, float] = {}
        self.eligible_count: int = 0
        self.allowed_exposure: float = self.p.base_exposure
        self.exited_this_bar: bool = False
        self.probe_positions: set[bt.feeds.DataBase] = set()
        self.pending_confirmation: set[bt.feeds.DataBase] = set()
        self.last_add_bar: Dict[bt.feeds.DataBase, int] = {d: -10_000 for d in self.datas}
        self.pyramid_add_count: Dict[bt.feeds.DataBase, int] = {d: 0 for d in self.datas}
        self.initial_entry_size: Dict[bt.feeds.DataBase, int] = {d: 0 for d in self.datas}
        self.partial_taken: Dict[bt.feeds.DataBase, bool] = {d: False for d in self.datas}
        self.breakeven_active: Dict[bt.feeds.DataBase, bool] = {d: False for d in self.datas}
        self.reentry_above_ma200_count: Dict[bt.feeds.DataBase, int] = {d: 0 for d in self.datas}
        self.raw_allowed_exposure: float = self.p.base_exposure
        self.prev_allowed_exposure: float = self.p.base_exposure
        self.current_day: int = 0

        self.index_data = self._resolve_index_data(self.p.index_data_name)
        self.sector_index_data: Dict[str, bt.feeds.DataBase] = {}
        self.sector_ma200: Dict[str, bt.Indicator] = {}
        for d in self.datas:
            name = d._name
            if name.startswith(self.p.sector_index_prefix):
                sector = name[len(self.p.sector_index_prefix):]
                self.sector_index_data[sector] = d
                self.sector_ma200[sector] = bt.ind.SMA(d.close, period=200)

    def _resolve_index_data(self, name: str):
        if not name:
            return None
        for d in self.datas:
            if d._name == name:
                return d
        return None

    def _is_week_rollover(self) -> bool:
        dt = self.datas[0].datetime.date(0)
        key = dt.isocalendar()[:2]
        if self.last_week_key != key:
            self.last_week_key = key
            return True
        return False

    def _ranking_score(self, data: bt.feeds.DataBase) -> float:
        e = self.engines[data]
        if len(data) < 252 or len(data) < 126:
            return -999.0

        momentum_12_26w = (data.close[0] / data.close[-60]) - 1.0
        trend_strength = (data.close[0] / e.ma200[0]) - 1.0
        volatility_penalty = e.atr[0] / max(data.close[0], 1e-9)
        return (0.6 * momentum_12_26w) + (0.4 * trend_strength) - (0.2 * volatility_penalty)

    def _structural_trend_ok(self, data: bt.feeds.DataBase) -> bool:
        e = self.engines[data]
        if len(data) < 252 or len(data) < 126:
            return False
        momentum_12_26w = (data.close[0] / data.close[-60]) - 1.0
        return data.close[0] > e.ma200[0] and momentum_12_26w > 0

    def _update_weekly_ranking_and_breadth(self) -> None:
        scored: List[Tuple[bt.feeds.DataBase, float]] = []
        eligible = 0
        candidates = [
            d for d in self.datas
            if (self.index_data is None or d != self.index_data) and d in self.probe_positions and self.getposition(d).size
        ]
        weight_map = self._compute_target_weights(candidates)
        for data in candidates:
            score = self._ranking_score(data)
            self.score_map[data] = score
            scored.append((data, score))
            if self._structural_trend_ok(data):
                eligible += 1

        scored.sort(key=lambda x: x[1], reverse=True)
        self.ranked_list = [d for d, _ in scored]
        self.rank_map = {d: idx + 1 for idx, d in enumerate(self.ranked_list)}
        self.eligible_count = eligible
        self.raw_allowed_exposure = self._compute_allowed_exposure(eligible)

    def _compute_allowed_exposure(self, eligible_count: int) -> float:
        if eligible_count > 60:
            exposure = 0.90
        elif 30 <= eligible_count <= 60:
            exposure = 0.90
        elif 15 <= eligible_count < 30:
            exposure = 0.65
        elif 8 <= eligible_count < 15:
            exposure = 0.45
        elif 3 <= eligible_count < 8:
            exposure = 0.20
        else:
            exposure = 0.0

        if self.p.allow_margin_on_high_breadth and eligible_count > 60:
            exposure = min(self.p.max_total_exposure, exposure + 0.20)

        exposure *= self._volatility_multiplier()
        exposure = min(exposure, 1.0 - self.p.permanent_cash_reserve + (self.p.max_total_exposure - 1.0))
        return max(0.0, exposure)

    def _volatility_multiplier(self) -> float:
        if self.index_data is None or len(self.index_data) < 50:
            return 1.0

        index_engine = self.engines[self.index_data]
        atr_ratio = index_engine.atr[0] / max(self.index_data.close[0], 1e-9)
        if atr_ratio < 0.015:
            return 1.05
        if atr_ratio < 0.03:
            return 1.0
        if atr_ratio < 0.05:
            return 0.85
        return 0.65

    def _target_position_count(self) -> int:
        if self.allowed_exposure <= 0:
            return 0
        return max(1, int(math.floor(self.allowed_exposure / self.p.min_position_weight)))

    def _portfolio_exposure(self) -> float:
        holdings = self._current_holdings()
        if not holdings:
            return 0.0
        portfolio_value = self.broker.getvalue()
        gross = sum(abs(self.getposition(d).size * d.close[0]) for d in holdings)
        return gross / max(portfolio_value, 1e-9)

    def _sector_of(self, data: bt.feeds.DataBase) -> str:
        if hasattr(data, "sector"):
            return str(getattr(data, "sector"))
        name = data._name
        if ":" in name:
            return name.split(":", 1)[1]
        return "UNKNOWN"

    def _sector_entry_allowed(self, data: bt.feeds.DataBase) -> bool:
        sector = self._sector_of(data)
        index_data = self.sector_index_data.get(sector)
        ma200 = self.sector_ma200.get(sector)
        if index_data is None or ma200 is None or len(index_data) < 200:
            return True
        return index_data.close[0] > ma200[0]

    def _sector_position_count(self, sector: str) -> int:
        count = 0
        for d in self._current_holdings():
            if self._sector_of(d) == sector:
                count += 1
        return count

    def _volatility(self, data: bt.feeds.DataBase) -> float:
        lookback = self.p.volatility_lookback
        if len(data) < lookback + 1:
            return 1e-6
        closes = list(data.close.get(size=lookback + 1))
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
        return max(var ** 0.5, 1e-6)

    def _relative_strength_ok(self, data: bt.feeds.DataBase) -> bool:
        sector = self._sector_of(data)
        index_data = self.sector_index_data.get(sector)
        lookback = self.p.rs_lookback
        if index_data is None or len(data) <= lookback or len(index_data) <= lookback:
            return True
        stock_ret = (data.close[0] / max(data.close[-lookback], 1e-9)) - 1.0
        sector_ret = (index_data.close[0] / max(index_data.close[-lookback], 1e-9)) - 1.0
        if sector_ret <= 0:
            return stock_ret > 0
        rs_ratio = stock_ret / sector_ret
        return rs_ratio > self.p.relative_strength_threshold

    def _compute_target_weights(self, candidates: List[bt.feeds.DataBase]) -> Dict[bt.feeds.DataBase, float]:
        if not candidates:
            return {}
        n = max(len(self.ranked_list), 1)
        raw: Dict[bt.feeds.DataBase, float] = {}
        for d in candidates:
            vol = self._volatility(d)
            risk_weight = 1.0 / vol
            rank = self.rank_map.get(d, n)
            score = (n - rank + 1) / n
            score = max(score, 1e-6)
            alloc_mult = math.sqrt(score)
            raw[d] = risk_weight * alloc_mult
        total = sum(raw.values())
        if total <= 0:
            return {d: 0.0 for d in candidates}
        return {d: (w / total) * self.allowed_exposure for d, w in raw.items()}

    def _target_shares_from_weight(self, data: bt.feeds.DataBase, weight: float) -> int:
        portfolio_value = self.broker.getvalue()
        target_value = portfolio_value * max(weight, 0.0)
        return max(1, int(target_value / max(data.close[0], 1e-9)))

    def _holding_rank_valid(self, data: bt.feeds.DataBase) -> bool:
        rank = self.rank_map.get(data, 10_000)
        return rank <= self.p.exit_rank_threshold

    def _can_enter_by_rank(self, data: bt.feeds.DataBase) -> bool:
        rank = self.rank_map.get(data, 10_000)
        return rank <= self.p.entry_rank_threshold

    def _close_position(self, data: bt.feeds.DataBase) -> None:
        if self.stop_orders[data] is not None:
            self.cancel(self.stop_orders[data])
            self.stop_orders[data] = None
        self.close(data=data)

    def notify_order(self, order: bt.Order) -> None:
        data = order.data
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.entry_orders[data] = None
                if self.getposition(data).size == order.executed.size:
                    self.bars_in_trade[data] = 0
                    self.initial_entry_size[data] = abs(order.executed.size)
                    self.pyramid_add_count[data] = 0
                    self.partial_taken[data] = False
                    self.breakeven_active[data] = False
                if self.stop_orders[data] is not None:
                    self.cancel(self.stop_orders[data])
                    self.stop_orders[data] = None

                atr_mult = self.p.probe_stop_atr_mult if data in self.probe_positions else self.p.confirm_stop_atr_mult
                if data in self.pending_confirmation:
                    atr_mult = self.p.confirm_stop_atr_mult
                    self.pending_confirmation.discard(data)
                    self.probe_positions.discard(data)

                stop_price = order.executed.price - (atr_mult * self.engines[data].atr[0])
                stop_price = max(0.01, stop_price)
                if self.breakeven_active[data]:
                    stop_price = max(stop_price, self.getposition(data).price)
                self.stop_orders[data] = self.sell(
                    data=data,
                    exectype=bt.Order.Stop,
                    price=stop_price,
                    size=self.getposition(data).size,
                )
            elif order.issell():
                if order == self.stop_orders[data]:
                    self.stop_orders[data] = None
                self.entry_orders[data] = None
                if self.getposition(data).size <= 0:
                    self.pending_confirmation.discard(data)
                    self.probe_positions.discard(data)
                    self.initial_entry_size[data] = 0
                    self.pyramid_add_count[data] = 0
                    self.partial_taken[data] = False
                    self.breakeven_active[data] = False
                    self.reentry_above_ma200_count[data] = 0

        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order == self.entry_orders[data]:
                self.entry_orders[data] = None
            if order == self.stop_orders[data]:
                self.stop_orders[data] = None

    def _process_exits(self) -> None:
        self.exited_this_bar = False
        candidates = [
            d for d in self.datas
            if (self.index_data is None or d != self.index_data) and d in self.probe_positions and self.getposition(d).size
        ]
        weight_map = self._compute_target_weights(candidates)
        for data in candidates:
            pos = self.getposition(data)
            if not pos.size:
                continue

            self.bars_in_trade[data] += 1
            e = self.engines[data]
            early_failure, medium_exit, final_exit = e.exit_flags(
                bars_in_trade=self.bars_in_trade[data],
                entry_price=pos.price,
                proven_trend_pct=self.p.proven_trend_pct,
                early_failure_bars=self.p.early_failure_bars,
            )

            if early_failure or medium_exit or final_exit:
                self._close_position(data)
                self.exited_this_bar = True

    def _weakest_holding(self):
        holdings = [d for d in self.datas if self.getposition(d).size and (self.index_data is None or d != self.index_data)]
        if not holdings:
            return None
        return min(holdings, key=lambda d: self.score_map.get(d, -999.0))

    def _candidate_list(self) -> List[bt.feeds.DataBase]:
        candidates: List[bt.feeds.DataBase] = []
        for data in self.ranked_list:
            if self.getposition(data).size:
                continue
            if self.entry_orders[data] is not None:
                continue
            if not self._can_enter_by_rank(data):
                continue
            if not self.engines[data].ready():
                continue
            if not self.engines[data].entry_signals()["probe_entry"]:
                continue
            if not self._sector_entry_allowed(data):
                continue
            if not self._relative_strength_ok(data):
                continue
            sector = self._sector_of(data)
            if self._sector_position_count(sector) >= self.p.max_sector_positions:
                continue
            if len(data) >= self.p.ma200_reentry_high_period:
                if data.close[0] > self.engines[data].ma200[0]:
                    self.reentry_above_ma200_count[data] += 1
                else:
                    self.reentry_above_ma200_count[data] = 0
                if self.reentry_above_ma200_count[data] > 0 and self.reentry_above_ma200_count[data] < self.p.ma200_reentry_bars:
                    continue
            candidates.append(data)
        return candidates

    def _current_holdings(self) -> List[bt.feeds.DataBase]:
        return [d for d in self.datas if self.getposition(d).size and (self.index_data is None or d != self.index_data)]

    def _rebalance_sizing(self) -> None:
        holdings = self._current_holdings()
        if not holdings:
            return

        target_weight = self.allowed_exposure / max(1, len(holdings))
        portfolio_value = self.broker.getvalue()

        for data in holdings:
            if data in self.probe_positions or self.entry_orders[data] is not None:
                continue
            target_value = portfolio_value * target_weight
            current_value = self.getposition(data).size * data.close[0]
            delta_value = target_value - current_value
            delta_size = int(delta_value / max(data.close[0], 1e-9))

            if delta_size > 0 and self.entry_orders[data] is None:
                self.buy(data=data, size=delta_size)
            elif delta_size < 0:
                self.sell(data=data, size=abs(delta_size))

    def _process_confirmation_scale_ins(self) -> None:
        candidates = [
            d for d in self.datas
            if (self.index_data is None or d != self.index_data) and d in self.probe_positions and self.getposition(d).size
        ]
        weight_map = self._compute_target_weights(candidates)
        for data in candidates:
            if data not in self.probe_positions:
                continue
            if self.entry_orders[data] is not None:
                continue
            pos = self.getposition(data)
            if not pos.size:
                continue
            signals = self.engines[data].entry_signals()
            if not signals["confirm_entry"]:
                continue

            target_size = self._target_shares_from_weight(data, weight_map.get(data, 0.0))
            remaining_size = max(0, target_size - pos.size)
            if remaining_size <= 0:
                self.probe_positions.discard(data)
                continue

            self.entry_orders[data] = self.buy(data=data, size=remaining_size)
            self.pending_confirmation.add(data)

    def _bootstrap_entries_if_empty(self) -> None:
        holdings = self._current_holdings()
        if holdings:
            return
        max_positions = self._target_position_count()
        if max_positions <= 0:
            return

        selected = self._candidate_list()[:max_positions]
        weight_map = self._compute_target_weights(selected)
        for data in selected:
            if self._portfolio_exposure() >= self.allowed_exposure:
                break
            target_size = self._target_shares_from_weight(data, weight_map.get(data, 0.0))
            probe_size = max(1, int(target_size * self.p.probe_size_fraction))
            self.entry_orders[data] = self.buy(data=data, size=probe_size)
            self.probe_positions.add(data)

    def _run_replacements_after_exit(self) -> None:
        if not self.exited_this_bar:
            return

        holdings = self._current_holdings()
        max_positions = self._target_position_count()
        open_slots = max(0, max_positions - len(holdings))
        if open_slots <= 0:
            return

        weakest = self._weakest_holding()
        weakest_score = self.score_map.get(weakest, -999.0) if weakest else -999.0

        entered = 0
        replacement_candidates = self._candidate_list()
        weight_map = self._compute_target_weights(replacement_candidates)
        for candidate in replacement_candidates:
            if entered >= open_slots:
                break
            candidate_score = self.score_map.get(candidate, -999.0)
            if candidate_score <= weakest_score + self.p.improvement_threshold:
                continue
            if self._portfolio_exposure() >= self.allowed_exposure:
                break
            target_size = self._target_shares_from_weight(candidate, weight_map.get(candidate, 0.0))
            probe_size = max(1, int(target_size * self.p.probe_size_fraction))
            self.entry_orders[candidate] = self.buy(data=candidate, size=probe_size)
            self.probe_positions.add(candidate)
            entered += 1

    def _process_pyramiding(self) -> None:
        for data in self._current_holdings():
            if self.entry_orders[data] is not None:
                continue
            if self.pyramid_add_count[data] >= self.p.max_pyramid_adds:
                continue
            if self.current_day - self.last_add_bar[data] < self.p.min_bars_between_adds:
                continue
            if data.close[0] <= self.engines[data].ma50[0]:
                continue
            window = 50
            if len(data) <= window:
                continue
            prev_high = max(data.high.get(size=window + 1)[:-1])
            if data.close[0] <= prev_high:
                continue

            add_size = int(max(1, self.initial_entry_size[data] * self.p.pyramid_add_fraction))
            if add_size <= 0:
                continue
            self.entry_orders[data] = self.buy(data=data, size=add_size)
            self.pyramid_add_count[data] += 1
            self.last_add_bar[data] = self.current_day

    def _process_partial_profits(self) -> None:
        for data in self._current_holdings():
            if self.partial_taken[data]:
                continue
            if self.entry_orders[data] is not None:
                continue
            pos = self.getposition(data)
            if pos.size <= 0:
                continue
            profit_pct = ((data.close[0] - pos.price) / max(pos.price, 1e-9)) * 100.0
            if profit_pct < self.p.partial_profit_trigger:
                continue
            sell_size = int(max(1, pos.size * self.p.partial_profit_take_fraction))
            if sell_size >= pos.size:
                sell_size = max(1, pos.size - 1)
            if sell_size <= 0:
                continue
            self.entry_orders[data] = self.sell(data=data, size=sell_size)
            self.partial_taken[data] = True
            self.breakeven_active[data] = True
            if self.stop_orders[data] is not None:
                self.cancel(self.stop_orders[data])
                self.stop_orders[data] = None
            stop_price = max(pos.price, data.close[0] - (self.p.confirm_stop_atr_mult * self.engines[data].atr[0]))
            self.stop_orders[data] = self.sell(data=data, exectype=bt.Order.Stop, price=max(0.01, stop_price), size=pos.size - sell_size)

    def _enforce_exposure_cap(self) -> None:
        holdings = self._current_holdings()
        if not holdings:
            return

        portfolio_value = self.broker.getvalue()
        gross_exposure = sum(abs(self.getposition(d).size * d.close[0]) for d in holdings) / max(portfolio_value, 1e-9)
        if gross_exposure <= self.allowed_exposure + 0.01:
            return

        for data in sorted(holdings, key=lambda d: self.score_map.get(d, -999.0)):
            if gross_exposure <= self.allowed_exposure:
                break
            self._close_position(data)
            gross_exposure = sum(abs(self.getposition(d).size * d.close[0]) for d in self._current_holdings()) / max(portfolio_value, 1e-9)

    def _next_overlay(self) -> None:
        if self._is_week_rollover() or not self.ranked_list:
            self._update_weekly_ranking_and_breadth()

        self.allowed_exposure = (
            (1.0 - self.p.exposure_smoothing_alpha) * self.prev_allowed_exposure
            + self.p.exposure_smoothing_alpha * self.raw_allowed_exposure
        )
        self.prev_allowed_exposure = self.allowed_exposure

        self._process_exits()
        self._process_partial_profits()
        self._process_confirmation_scale_ins()
        self._process_pyramiding()
        self._bootstrap_entries_if_empty()
        self._run_replacements_after_exit()
        self._enforce_exposure_cap()
        self._rebalance_sizing()

    def _next_v6_single_asset_fallback(self) -> None:
        data = self.datas[0]
        engine = self.engines[data]
        if not engine.ready():
            return

        pos = self.getposition(data)
        signals = engine.entry_signals()

        if not pos.size and self.entry_orders[data] is None and signals["probe_entry"]:
            self.entry_orders[data] = self.buy(data=data)
            self.probe_positions.add(data)
            return

        if pos.size and data in self.probe_positions and self.entry_orders[data] is None and signals["confirm_entry"]:
            self.entry_orders[data] = self.buy(data=data)
            self.pending_confirmation.add(data)

        if pos.size:
            self.bars_in_trade[data] += 1
            early_failure, medium_exit, final_exit = engine.exit_flags(
                bars_in_trade=self.bars_in_trade[data],
                entry_price=pos.price,
                proven_trend_pct=self.p.proven_trend_pct,
                early_failure_bars=self.p.early_failure_bars,
            )
            if early_failure or medium_exit or final_exit:
                self._close_position(data)

    def next(self) -> None:
        self.current_day += 1
        if self.p.overlay_enabled:
            self._next_overlay()
        else:
            self._next_v6_single_asset_fallback()


def _add_csv_data(cerebro: bt.Cerebro, data_path: str, name: Optional[str] = None):
    feed = bt.feeds.GenericCSVData(
        dataname=data_path,
        dtformat="%Y-%m-%d",
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        header=0,
    )
    cerebro.adddata(feed, name=name or Path(data_path).stem)


def run_backtest(data_paths: List[str], config: BacktestConfig, overlay: OverlayConfig, index_data_name: str = "") -> None:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(config.cash)
    cerebro.broker.setcommission(commission=config.commission)

    for path in data_paths:
        _add_csv_data(cerebro, path)

    if overlay.enabled:
        if overlay.exit_rank_threshold <= overlay.entry_rank_threshold:
            raise ValueError("EXIT_RANK_THRESHOLD must be larger than ENTRY_RANK_THRESHOLD")
        cerebro.addstrategy(
            HybridMomentumPortfolioOverlay,
            overlay_enabled=True,
            entry_rank_threshold=overlay.entry_rank_threshold,
            exit_rank_threshold=overlay.exit_rank_threshold,
            improvement_threshold=overlay.improvement_threshold,
            permanent_cash_reserve=overlay.permanent_cash_reserve,
            base_exposure=overlay.base_exposure,
            max_total_exposure=overlay.max_total_exposure,
            min_position_weight=overlay.min_position_weight,
            allow_margin_on_high_breadth=overlay.allow_margin_on_high_breadth,
            index_data_name=index_data_name,
        )
    else:
        if len(data_paths) != 1:
            raise ValueError("Overlay disabled mode supports exactly one data feed to mirror original V6 behavior")
        cerebro.addsizer(bt.sizers.PercentSizer, percents=config.percent_size)
        cerebro.addstrategy(HybridTrendSwingMomentum)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Trend + Swing Momentum backtest")
    parser.add_argument("--data", nargs="+", required=True, help="One or more CSV files with Date,Open,High,Low,Close,Volume")
    parser.add_argument("--cash", type=float, default=100000.0)
    parser.add_argument("--size", type=float, default=4.0, help="Position size as %% of equity (single-asset mode)")
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--overlay", action="store_true", help="Enable weekly ranking + portfolio allocation overlay")
    parser.add_argument("--entry-rank-threshold", type=int, default=25)
    parser.add_argument("--exit-rank-threshold", type=int, default=40)
    parser.add_argument("--improvement-threshold", type=float, default=0.05)
    parser.add_argument("--base-exposure", type=float, default=0.90)
    parser.add_argument("--max-total-exposure", type=float, default=1.20)
    parser.add_argument("--cash-reserve", type=float, default=0.10)
    parser.add_argument("--min-position-weight", type=float, default=0.05)
    parser.add_argument("--index-data-name", default="", help="Optional feed name (file stem) used for volatility throttle")

    args = parser.parse_args()

    cfg = BacktestConfig(cash=args.cash, percent_size=args.size, commission=args.commission)
    overlay_cfg = OverlayConfig(
        enabled=args.overlay,
        entry_rank_threshold=args.entry_rank_threshold,
        exit_rank_threshold=args.exit_rank_threshold,
        improvement_threshold=args.improvement_threshold,
        permanent_cash_reserve=args.cash_reserve,
        base_exposure=args.base_exposure,
        max_total_exposure=args.max_total_exposure,
        min_position_weight=args.min_position_weight,
    )

    run_backtest(data_paths=args.data, config=cfg, overlay=overlay_cfg, index_data_name=args.index_data_name)
