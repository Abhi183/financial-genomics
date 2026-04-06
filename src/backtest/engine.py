"""Event-driven backtesting engine.

Provides:
- ``Trade`` — immutable record of a single round-trip trade
- ``BacktestResult`` — container for a full backtest run with plotting helpers
- ``BacktestEngine`` — drives signal execution against a price series
- ``WalkForwardBacktest`` — rolling train/test window validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.backtest.metrics import PerformanceReport
from src.backtest.strategy import SignalEnum

logger = logging.getLogger(__name__)

matplotlib.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Immutable record of a single round-trip trade.

    Attributes
    ----------
    entry_date:
        Date / timestamp when the position was opened.
    exit_date:
        Date / timestamp when the position was closed.
    entry_price:
        Execution price at entry (after slippage).
    exit_price:
        Execution price at exit (after slippage).
    direction:
        +1 for a long trade, -1 for a short trade.
    size:
        Notional position size in dollars at entry.
    pnl:
        Realised profit-and-loss in dollars.
    slippage:
        Total slippage cost in dollars (both legs combined).
    commission:
        Total commission cost in dollars (both legs combined).
    """

    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    direction: int  # +1 long, -1 short
    size: float  # dollars
    pnl: float
    slippage: float
    commission: float = 0.0

    @property
    def duration(self) -> Optional[pd.Timedelta]:
        """Return the duration of the trade if dates are Timestamps."""
        try:
            return pd.Timestamp(self.exit_date) - pd.Timestamp(self.entry_date)
        except (TypeError, ValueError):
            return None

    @property
    def return_pct(self) -> float:
        """Percentage return of the trade relative to notional size."""
        if self.size == 0:
            return 0.0
        return self.pnl / abs(self.size)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Container for a completed backtest run.

    Attributes
    ----------
    equity_curve:
        Portfolio value at each time step.
    trades:
        List of all completed ``Trade`` objects.
    returns:
        Period return series derived from ``equity_curve``.
    performance:
        ``PerformanceReport`` instance with aggregated metrics.
    """

    equity_curve: pd.Series
    trades: List[Trade]
    returns: pd.Series
    performance: PerformanceReport

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_equity_curve(self, title: str = "Equity Curve") -> Figure:
        """Plot the equity curve over time.

        Parameters
        ----------
        title:
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.equity_curve.index, self.equity_curve.values, linewidth=1.5, color="#2563eb")
        ax.fill_between(
            self.equity_curve.index,
            self.equity_curve.values,
            self.equity_curve.values[0],
            alpha=0.08,
            color="#2563eb",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig

    def plot_drawdown(self, title: str = "Drawdown") -> Figure:
        """Plot the underwater equity curve (drawdown over time).

        Parameters
        ----------
        title:
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        rolling_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve / rolling_max - 1.0) * 100.0

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.fill_between(drawdown.index, drawdown.values, 0, color="#ef4444", alpha=0.5)
        ax.plot(drawdown.index, drawdown.values, linewidth=0.8, color="#ef4444")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven backtesting engine.

    Iterates over a pre-computed signals DataFrame aligned with a price series,
    executing trades with configurable commission and slippage.

    Parameters
    ----------
    strategy:
        Any strategy object with a ``generate_signal`` method.  Can also
        pass ``None`` if calling ``run`` with an explicit ``signals_df``.
    initial_capital:
        Starting portfolio value in dollars.
    commission_bps:
        One-way commission in basis points applied at each entry and exit.
    slippage_bps:
        One-way slippage in basis points applied at each entry and exit.
    """

    def __init__(
        self,
        strategy: Any,
        initial_capital: float = 100_000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 1.0,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        prices_df: pd.DataFrame,
        signals_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """Execute the backtest.

        Parameters
        ----------
        prices_df:
            DataFrame with at least a ``close`` column (case-insensitive).
            Index must be a DatetimeIndex or comparable sequence.
        signals_df:
            Pre-computed signal DataFrame with a ``signal`` column containing
            ``SignalEnum`` values (or integer equivalents).  If ``None``, the
            engine calls ``self.strategy.generate_signal`` at each bar using
            ``prices_df`` up to (but not including) the current row as context.

        Returns
        -------
        BacktestResult
        """
        # Normalise column names
        prices_df = prices_df.copy()
        prices_df.columns = [c.lower() for c in prices_df.columns]
        if "close" not in prices_df.columns:
            raise KeyError("prices_df must contain a 'close' column.")

        close_prices: pd.Series = prices_df["close"].astype(float)
        portfolio_value = self.initial_capital
        equity_values: List[float] = []
        trades: List[Trade] = []

        open_trade: Optional[Dict[str, Any]] = None  # track open position

        all_dates = close_prices.index
        n = len(all_dates)

        for i, date in enumerate(all_dates):
            price = float(close_prices.iloc[i])

            # ---- Determine signal ----
            if signals_df is not None and date in signals_df.index:
                raw_signal = signals_df.loc[date, "signal"]
                signal = self._parse_signal(raw_signal)
            elif self.strategy is not None:
                context = prices_df.iloc[: i + 1]
                signal = self.strategy.generate_signal(context)
            else:
                signal = SignalEnum.HOLD

            # ---- Close open position if signal changed or end of series ----
            if open_trade is not None:
                should_close = (
                    signal != open_trade["signal"]
                    or signal == SignalEnum.HOLD
                    or i == n - 1
                )
                if should_close:
                    trade = self._close_trade(open_trade, date, price, portfolio_value)
                    portfolio_value += trade.pnl - trade.commission
                    trades.append(trade)
                    open_trade = None

            # ---- Open a new position ----
            if signal != SignalEnum.HOLD and open_trade is None:
                vol_est = self._estimate_volatility(close_prices.iloc[: i + 1])
                if hasattr(self.strategy, "compute_position_size"):
                    size = self.strategy.compute_position_size(signal, portfolio_value, vol_est)
                else:
                    size = portfolio_value * signal.value  # full notional

                exec_price = self._apply_slippage(price, signal.value)
                open_trade = {
                    "entry_date": date,
                    "entry_price": exec_price,
                    "signal": signal,
                    "size": size,
                    "commission": abs(size) * self.commission_bps * 1e-4,
                }

            equity_values.append(portfolio_value)

        equity_curve = pd.Series(equity_values, index=all_dates, name="equity")
        period_returns = equity_curve.pct_change().dropna()
        perf = PerformanceReport(
            returns_series=period_returns,
            equity_curve=equity_curve,
            trades_df=self._trades_to_df(trades) if trades else None,
        )
        perf.compute_all()

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            returns=period_returns,
            performance=perf,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_trade(
        self,
        open_trade: Dict[str, Any],
        exit_date: Any,
        price: float,
        portfolio_value: float,
    ) -> Trade:
        direction = open_trade["signal"].value  # +1 or -1
        exit_price = self._apply_slippage(price, -direction)  # opposite leg
        size = open_trade["size"]
        entry_price = open_trade["entry_price"]

        shares = size / entry_price if entry_price != 0 else 0.0
        pnl = direction * shares * (exit_price - entry_price)
        exit_commission = abs(size) * self.commission_bps * 1e-4
        total_commission = open_trade["commission"] + exit_commission

        entry_slip = abs(entry_price - price) * shares
        exit_slip = abs(exit_price - price) * shares
        total_slippage = entry_slip + exit_slip

        return Trade(
            entry_date=open_trade["entry_date"],
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            size=size,
            pnl=float(pnl),
            slippage=float(total_slippage),
            commission=float(total_commission),
        )

    def _apply_slippage(self, price: float, direction: int) -> float:
        return price * (1.0 + direction * self.slippage_bps * 1e-4)

    @staticmethod
    def _estimate_volatility(price_series: pd.Series, window: int = 20) -> float:
        """Rolling 20-bar realised volatility estimate."""
        if len(price_series) < 2:
            return 0.15  # default 15 % annualised
        rets = price_series.pct_change().dropna()
        recent = rets.iloc[-window:]
        if recent.empty:
            return 0.15
        return float(recent.std() * np.sqrt(252))

    @staticmethod
    def _parse_signal(raw: Any) -> SignalEnum:
        if isinstance(raw, SignalEnum):
            return raw
        try:
            return SignalEnum(int(raw))
        except (ValueError, KeyError):
            return SignalEnum.HOLD

    @staticmethod
    def _trades_to_df(trades: List[Trade]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "direction": t.direction,
                    "size": t.size,
                    "pnl": t.pnl,
                    "slippage": t.slippage,
                    "commission": t.commission,
                }
                for t in trades
            ]
        )


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------


class WalkForwardBacktest:
    """Rolling-window walk-forward validation.

    Splits the price history into consecutive train/test windows.  For each
    window the model trainer is re-fitted on the training set and then the
    engine runs the strategy on the out-of-sample test set.

    Parameters
    ----------
    engine:
        A configured ``BacktestEngine`` instance.
    train_window:
        Number of periods (bars) in the training window.  Defaults to 252
        (approximately one trading year of daily data).
    test_window:
        Number of periods (bars) in the out-of-sample test window.  Defaults
        to 63 (approximately one trading quarter).
    """

    def __init__(
        self,
        engine: BacktestEngine,
        train_window: int = 252,
        test_window: int = 63,
    ) -> None:
        if train_window <= 0:
            raise ValueError(f"train_window must be positive, got {train_window}")
        if test_window <= 0:
            raise ValueError(f"test_window must be positive, got {test_window}")
        self.engine = engine
        self.train_window = train_window
        self.test_window = test_window

    def run(
        self,
        prices_df: pd.DataFrame,
        model_trainer: Any,
        discretizer: Any,
    ) -> List[BacktestResult]:
        """Execute walk-forward validation.

        Parameters
        ----------
        prices_df:
            Full price DataFrame (must contain a ``close`` column).
        model_trainer:
            Model trainer with ``fit(X_train, y_train)`` and
            ``predict_proba(context)`` methods.
        discretizer:
            Discretizer with ``fit_transform(prices)`` and
            ``transform(prices)`` methods used to encode the price series.

        Returns
        -------
        list of BacktestResult
            One result per out-of-sample test window.
        """
        prices_df = prices_df.copy()
        prices_df.columns = [c.lower() for c in prices_df.columns]
        close = prices_df["close"].astype(float)
        n = len(close)
        results: List[BacktestResult] = []

        start = 0
        fold = 0
        while start + self.train_window + self.test_window <= n:
            train_end = start + self.train_window
            test_end = train_end + self.test_window

            train_prices = close.iloc[start:train_end]
            test_prices_df = prices_df.iloc[train_end:test_end]

            logger.info(
                "Walk-forward fold %d: train=[%s, %s], test=[%s, %s]",
                fold,
                train_prices.index[0],
                train_prices.index[-1],
                test_prices_df.index[0],
                test_prices_df.index[-1],
            )

            # Fit discretiser and model on training data
            try:
                train_encoded = discretizer.fit_transform(train_prices)
                model_trainer.fit(train_encoded)
            except Exception as exc:
                logger.error("Training failed on fold %d: %s — skipping fold", fold, exc)
                start += self.test_window
                fold += 1
                continue

            # Generate signals for the test window
            signals = self._generate_signals(
                test_prices_df, model_trainer, discretizer, train_prices
            )

            result = self.engine.run(test_prices_df, signals_df=signals)
            results.append(result)

            start += self.test_window
            fold += 1

        if not results:
            logger.warning("Walk-forward produced no results. Check train/test window sizes.")
        return results

    def aggregate_results(self, results: List[BacktestResult]) -> BacktestResult:
        """Concatenate multiple walk-forward results into a single BacktestResult.

        Parameters
        ----------
        results:
            List of ``BacktestResult`` objects from ``run``.

        Returns
        -------
        BacktestResult
            Combined result with full equity curve and all trades.
        """
        if not results:
            raise ValueError("Cannot aggregate an empty list of results.")

        # Re-chain equity curves: each window starts from where the last ended
        equity_parts: List[pd.Series] = []
        cumulative_start = self.engine.initial_capital
        all_trades: List[Trade] = []

        for res in results:
            scale = cumulative_start / res.equity_curve.iloc[0]
            scaled = res.equity_curve * scale
            equity_parts.append(scaled)
            cumulative_start = float(scaled.iloc[-1])
            all_trades.extend(res.trades)

        combined_equity = pd.concat(equity_parts)
        combined_equity = combined_equity[~combined_equity.index.duplicated(keep="last")]
        combined_returns = combined_equity.pct_change().dropna()

        trades_df = BacktestEngine._trades_to_df(all_trades) if all_trades else None
        perf = PerformanceReport(
            returns_series=combined_returns,
            equity_curve=combined_equity,
            trades_df=trades_df,
        )
        perf.compute_all()

        return BacktestResult(
            equity_curve=combined_equity,
            trades=all_trades,
            returns=combined_returns,
            performance=perf,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_signals(
        prices_df: pd.DataFrame,
        model_trainer: Any,
        discretizer: Any,
        context_prices: pd.Series,
    ) -> pd.DataFrame:
        """Generate a signals DataFrame for the test window."""
        signals: List[Dict[str, Any]] = []
        running_context = context_prices.copy()

        for date, row in prices_df.iterrows():
            price = float(row["close"])
            try:
                encoded = discretizer.transform(running_context)
                probs = model_trainer.predict_proba(encoded)
                probs = np.asarray(probs).ravel()
                crash_p, spike_p = float(probs[0]), float(probs[1])
                if spike_p >= 0.70 and spike_p > crash_p:
                    sig = SignalEnum.LONG
                elif crash_p >= 0.70 and crash_p > spike_p:
                    sig = SignalEnum.SHORT
                else:
                    sig = SignalEnum.HOLD
            except Exception as exc:
                logger.debug("Signal generation failed at %s: %s", date, exc)
                sig = SignalEnum.HOLD

            signals.append({"date": date, "signal": sig})
            running_context = pd.concat(
                [running_context, pd.Series([price], index=[date])]
            )

        signals_df = pd.DataFrame(signals).set_index("date")
        return signals_df
