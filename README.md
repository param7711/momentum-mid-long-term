"""Backtrader implementation of the Pine strategy:
Hybrid Trend + Swing Momentum System (Refurbished v6).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import backtrader as bt


@dataclass
class BacktestConfig:
    cash: float = 100000.0
    percent_size: float = 4.0
    commission: float = 0.0


class HybridTrendSwingMomentum(bt.Strategy):
    params = dict(
        stop_loss_pct=0.18,
        early_failure_bars=40,
        proven_trend_pct=25.0,
    )

    def __init__(self) -> None:
        # Moving averages
        self.ma20 = bt.ind.EMA(self.data.close, period=20)
        self.ma50 = bt.ind.SMA(self.data.close, period=50)
        self.ma100 = bt.ind.SMA(self.data.close, period=100)
        self.ma200 = bt.ind.SMA(self.data.close, period=200)

        # RSI + ATR
        self.rsi = bt.ind.RSI(self.data.close, period=14)
        self.atr = bt.ind.ATR(self.data, period=14)
        self.atr_ma = bt.ind.SMA(self.atr, period=50)

        # 52-week high + recent high count
        self.high52 = bt.ind.Highest(self.data.close, period=252)
        self.new_high = self.data.close >= self.high52
        self.recent_high_count = bt.ind.SumN(self.new_high, period=15)

        # Golden cross
        self.golden_cross = bt.ind.CrossOver(self.ma50, self.ma200)

        # Highest high breakout helper
        self.highest20 = bt.ind.Highest(self.data.high, period=20)

        # Trade/order state
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
        if len(self.data) < 252:
            return

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

        long_condition = volatility_expansion and (
            momentum_entry or golden_cross_entry or fvg_entry or swing_entry
        )

        if not self.position and self.entry_order is None and long_condition:
            self.entry_order = self.buy()
            return

        if self.position:
            self.bars_in_trade += 1

            entry_price = self.position.price
            profit_percent = ((self.data.close[0] - entry_price) / entry_price) * 100.0
            proven_trend = profit_percent > self.p.proven_trend_pct

            early_failure = (self.bars_in_trade < self.p.early_failure_bars) and (
                self.data.close[0] < self.ma50[0]
            )
            medium_exit = (not proven_trend) and (self.data.close[0] < self.ma100[0])
            final_exit = proven_trend and (self.data.close[0] < self.ma200[0])

            if early_failure or medium_exit or final_exit:
                self.close_position()


def run_backtest(data_path: str, config: BacktestConfig = BacktestConfig()) -> None:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(config.cash)
    cerebro.broker.setcommission(commission=config.commission)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=config.percent_size)
    cerebro.addstrategy(HybridTrendSwingMomentum)

    kwargs = dict(
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

    data = bt.feeds.GenericCSVData(**kwargs)
    cerebro.adddata(data)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Trend + Swing Momentum backtest")
    parser.add_argument("--data", required=True, help="CSV path with columns: Date,Open,High,Low,Close,Volume")
    parser.add_argument("--cash", type=float, default=100000.0)
    parser.add_argument("--size", type=float, default=4.0, help="Position size as % of equity")
    parser.add_argument("--commission", type=float, default=0.0)

    args = parser.parse_args()

    cfg = BacktestConfig(cash=args.cash, percent_size=args.size, commission=args.commission)
    run_backtest(data_path=args.data, config=cfg)
