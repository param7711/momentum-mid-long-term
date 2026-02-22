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

    def long_condition(self) -> bool:
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

        self.index_data = self._resolve_index_data(self.p.index_data_name)

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
        for data in self.datas:
            if self.index_data is not None and data == self.index_data:
                continue
            score = self._ranking_score(data)
            self.score_map[data] = score
            scored.append((data, score))
            if self._structural_trend_ok(data):
                eligible += 1

        scored.sort(key=lambda x: x[1], reverse=True)
        self.ranked_list = [d for d, _ in scored]
        self.rank_map = {d: idx + 1 for idx, d in enumerate(self.ranked_list)}
        self.eligible_count = eligible
        self.allowed_exposure = self._compute_allowed_exposure(eligible)

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
                self.bars_in_trade[data] = 0
                stop_price = order.executed.price * (1.0 - self.p.stop_loss_pct)
                self.stop_orders[data] = self.sell(data=data, exectype=bt.Order.Stop, price=stop_price)
            elif order.issell():
                if order == self.stop_orders[data]:
                    self.stop_orders[data] = None
                self.entry_orders[data] = None

        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order == self.entry_orders[data]:
                self.entry_orders[data] = None
            if order == self.stop_orders[data]:
                self.stop_orders[data] = None

    def _process_exits(self) -> None:
        self.exited_this_bar = False
        for data in self.datas:
            if self.index_data is not None and data == self.index_data:
                continue
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
            if not self.engines[data].long_condition():
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
            target_value = portfolio_value * target_weight
            current_value = self.getposition(data).size * data.close[0]
            delta_value = target_value - current_value
            delta_size = int(delta_value / max(data.close[0], 1e-9))

            if delta_size > 0 and self.entry_orders[data] is None:
                self.buy(data=data, size=delta_size)
            elif delta_size < 0:
                self.sell(data=data, size=abs(delta_size))

    def _bootstrap_entries_if_empty(self) -> None:
        holdings = self._current_holdings()
        if holdings:
            return
        max_positions = self._target_position_count()
        if max_positions <= 0:
            return

        for data in self._candidate_list()[:max_positions]:
            self.entry_orders[data] = self.buy(data=data)

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
        for candidate in self._candidate_list():
            if entered >= open_slots:
                break
            candidate_score = self.score_map.get(candidate, -999.0)
            if candidate_score <= weakest_score + self.p.improvement_threshold:
                continue
            self.entry_orders[candidate] = self.buy(data=candidate)
            entered += 1

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

        self._process_exits()
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
        if not pos.size and self.entry_orders[data] is None and engine.long_condition():
            self.entry_orders[data] = self.buy(data=data)
            return

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
