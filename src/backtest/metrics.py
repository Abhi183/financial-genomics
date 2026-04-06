"""Complete performance metrics for trading strategies.

Provides individual metric functions and a PerformanceReport class that
aggregates all metrics for a trading strategy's return and equity series.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    annualization: int = 252,
) -> float:
    """Compute the annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Daily (or period) return series.  Values should be simple returns
        (e.g. 0.01 for +1 %), not log-returns.
    risk_free:
        Risk-free rate expressed as the *same* periodicity as ``returns``
        (e.g. daily risk-free rate).  Defaults to 0.
    annualization:
        Number of periods per year.  252 for daily equity returns.

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns ``np.nan`` when volatility is zero.
    """
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    excess = returns - risk_free
    sigma = excess.std(ddof=1)
    if sigma == 0.0:
        return np.nan
    return float((excess.mean() / sigma) * np.sqrt(annualization))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    annualization: int = 252,
) -> float:
    """Compute the annualised Sortino ratio (penalises downside volatility only).

    Parameters
    ----------
    returns:
        Daily (or period) return series.
    risk_free:
        Risk-free rate at the same periodicity as ``returns``.
    annualization:
        Periods per year.

    Returns
    -------
    float
        Annualised Sortino ratio.  Returns ``np.nan`` when downside deviation
        is zero or there are no negative excess returns.
    """
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    excess = returns - risk_free
    downside = excess[excess < 0]
    if downside.empty:
        return np.nan
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0.0:
        return np.nan
    return float((excess.mean() / downside_std) * np.sqrt(annualization))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown of an equity curve.

    Parameters
    ----------
    equity_curve:
        Portfolio value over time (dollar or index value, not returns).

    Returns
    -------
    float
        Maximum drawdown as a negative fraction (e.g. -0.20 for -20 %).
        Returns 0.0 if the equity curve is empty or monotonically increasing.
    """
    equity_curve = pd.Series(equity_curve).dropna()
    if equity_curve.empty:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """Compute the Calmar ratio: annualised return divided by |max drawdown|.

    Parameters
    ----------
    returns:
        Daily (or period) return series.
    equity_curve:
        Portfolio value over time.

    Returns
    -------
    float
        Calmar ratio.  Returns ``np.nan`` when max drawdown is zero.
    """
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0:
        return np.nan
    return float(ann_ret / abs(mdd))


def win_rate(pnl_series: pd.Series) -> float:
    """Fraction of trades (or periods) with positive PnL.

    Parameters
    ----------
    pnl_series:
        Series of per-trade or per-period profit-and-loss values.

    Returns
    -------
    float
        Win rate in [0, 1].  Returns ``np.nan`` for empty input.
    """
    pnl_series = pd.Series(pnl_series).dropna()
    if pnl_series.empty:
        return np.nan
    return float((pnl_series > 0).sum() / len(pnl_series))


def profit_factor(pnl_series: pd.Series) -> float:
    """Ratio of gross profit to gross loss.

    Parameters
    ----------
    pnl_series:
        Series of per-trade profit-and-loss values.

    Returns
    -------
    float
        Profit factor.  Returns ``np.inf`` when there are no losing trades,
        ``np.nan`` when the series is empty or there are no winning trades.
    """
    pnl_series = pd.Series(pnl_series).dropna()
    if pnl_series.empty:
        return np.nan
    gross_profit = pnl_series[pnl_series > 0].sum()
    gross_loss = pnl_series[pnl_series < 0].sum()
    if gross_loss == 0.0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / abs(gross_loss))


def annualized_return(returns: pd.Series, annualization: int = 252) -> float:
    """Compute the compound annualised return (CAGR).

    Parameters
    ----------
    returns:
        Daily (or period) simple return series.
    annualization:
        Periods per year.

    Returns
    -------
    float
        Compound annualised return.
    """
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    n_periods = len(returns)
    total_return = (1.0 + returns).prod()
    if total_return <= 0:
        return np.nan
    return float(total_return ** (annualization / n_periods) - 1.0)


def annualized_volatility(returns: pd.Series, annualization: int = 252) -> float:
    """Annualised standard deviation of returns.

    Parameters
    ----------
    returns:
        Daily (or period) return series.
    annualization:
        Periods per year.

    Returns
    -------
    float
        Annualised volatility.
    """
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(annualization))


def kelly_fraction(win_rate_val: float, avg_win: float, avg_loss: float) -> float:
    """Kelly criterion fraction of capital to risk per trade.

    Parameters
    ----------
    win_rate_val:
        Probability of a winning trade, in (0, 1).
    avg_win:
        Average dollar (or absolute) gain on winning trades.
    avg_loss:
        Average dollar (or absolute) loss on losing trades (positive value).

    Returns
    -------
    float
        Kelly fraction in [0, 1].  Clipped to [0, 1] to avoid negative or
        lever-busting values.

    Raises
    ------
    ValueError
        If ``avg_loss`` is zero.
    """
    if avg_loss == 0.0:
        raise ValueError("avg_loss must be non-zero to compute Kelly fraction.")
    loss_rate = 1.0 - win_rate_val
    odds = avg_win / avg_loss
    kelly = win_rate_val - (loss_rate / odds)
    return float(np.clip(kelly, 0.0, 1.0))


def permutation_sharpe_test(
    returns: pd.Series,
    n_permutations: int = 1000,
    risk_free: float = 0.0,
    annualization: int = 252,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute the Sharpe ratio and its permutation p-value.

    The null hypothesis is that the strategy's returns have no systematic
    edge (i.e. are exchangeable in time order).  The observed Sharpe is
    compared to the distribution of Sharpe ratios obtained by randomly
    permuting the return series.

    Parameters
    ----------
    returns:
        Strategy return series.
    n_permutations:
        Number of random permutations.
    risk_free:
        Risk-free rate at the same periodicity as ``returns``.
    annualization:
        Periods per year.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    (sharpe, p_value) : Tuple[float, float]
        Observed Sharpe ratio and one-tailed p-value (fraction of permuted
        Sharpes that are >= observed Sharpe).
    """
    returns = pd.Series(returns).dropna()
    rng = np.random.default_rng(random_state)

    observed_sharpe = sharpe_ratio(returns, risk_free=risk_free, annualization=annualization)

    if np.isnan(observed_sharpe):
        return float(observed_sharpe), np.nan

    arr = returns.to_numpy()
    permuted_sharpes = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(arr)
        perm_series = pd.Series(perm, index=returns.index)
        permuted_sharpes[i] = sharpe_ratio(
            perm_series, risk_free=risk_free, annualization=annualization
        )

    p_value = float(np.mean(permuted_sharpes >= observed_sharpe))
    return float(observed_sharpe), p_value


# ---------------------------------------------------------------------------
# PerformanceReport
# ---------------------------------------------------------------------------


class PerformanceReport:
    """Aggregate performance metrics for a trading strategy.

    Parameters
    ----------
    returns_series:
        Daily (or period) simple return series with a DatetimeIndex.
    equity_curve:
        Portfolio value over time (same index as ``returns_series``).
    trades_df:
        Optional DataFrame with a ``pnl`` column, one row per trade.
    """

    def __init__(
        self,
        returns_series: pd.Series,
        equity_curve: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.returns = pd.Series(returns_series).dropna()
        self.equity_curve = pd.Series(equity_curve).dropna()
        self.trades_df = trades_df
        self._metrics: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """Compute and return all metrics as a dictionary.

        Returns
        -------
        dict
            Keys are metric names; values are floats (or ``np.nan``).
        """
        pnl_series = (
            self.trades_df["pnl"] if self.trades_df is not None and "pnl" in self.trades_df
            else self.returns  # fall back to period returns as proxy
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sharpe = sharpe_ratio(self.returns)
            sortino = sortino_ratio(self.returns)
            mdd = max_drawdown(self.equity_curve)
            calmar = calmar_ratio(self.returns, self.equity_curve)
            ann_ret = annualized_return(self.returns)
            ann_vol = annualized_volatility(self.returns)
            wr = win_rate(pnl_series)
            pf = profit_factor(pnl_series)

            n_trades = len(self.trades_df) if self.trades_df is not None else np.nan

            if self.trades_df is not None and "pnl" in self.trades_df:
                wins = self.trades_df["pnl"][self.trades_df["pnl"] > 0]
                losses = self.trades_df["pnl"][self.trades_df["pnl"] < 0]
                avg_win = float(wins.mean()) if not wins.empty else np.nan
                avg_loss = float(losses.abs().mean()) if not losses.empty else np.nan
                try:
                    kelly = kelly_fraction(wr, avg_win, avg_loss) if not np.isnan(wr) else np.nan
                except (ValueError, TypeError):
                    kelly = np.nan
            else:
                avg_win = np.nan
                avg_loss = np.nan
                kelly = np.nan

        self._metrics = {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": mdd,
            "calmar_ratio": calmar,
            "win_rate": wr,
            "profit_factor": pf,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "kelly_fraction": kelly,
            "n_trades": n_trades,
            "total_return": float((1.0 + self.returns).prod() - 1.0) if not self.returns.empty else np.nan,
        }
        return self._metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Return metrics as a single-column DataFrame with metric names as index.

        Returns
        -------
        pd.DataFrame
            Index: metric names.  Column: ``value``.
        """
        metrics = self._metrics if self._metrics is not None else self.compute_all()
        return pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])

    def print_summary(self) -> None:
        """Print a formatted summary of all performance metrics to stdout."""
        metrics = self._metrics if self._metrics is not None else self.compute_all()
        width = 40
        print("=" * width)
        print(f"{'Performance Summary':^{width}}")
        print("=" * width)
        fmt_map = {
            "annualized_return": "{:>10.2%}",
            "annualized_volatility": "{:>10.2%}",
            "sharpe_ratio": "{:>10.4f}",
            "sortino_ratio": "{:>10.4f}",
            "max_drawdown": "{:>10.2%}",
            "calmar_ratio": "{:>10.4f}",
            "win_rate": "{:>10.2%}",
            "profit_factor": "{:>10.4f}",
            "avg_win": "{:>10.4f}",
            "avg_loss": "{:>10.4f}",
            "kelly_fraction": "{:>10.4f}",
            "n_trades": "{:>10.0f}",
            "total_return": "{:>10.2%}",
        }
        for key, value in metrics.items():
            fmt = fmt_map.get(key, "{:>10.4f}")
            label = key.replace("_", " ").title()
            try:
                val_str = fmt.format(value) if not (isinstance(value, float) and np.isnan(value)) else f"{'N/A':>10}"
            except (ValueError, TypeError):
                val_str = f"{str(value):>10}"
            print(f"  {label:<28}{val_str}")
        print("=" * width)

    def compare(
        self,
        other: "PerformanceReport",
        labels: Tuple[str, str] = ("Strategy A", "Strategy B"),
    ) -> pd.DataFrame:
        """Side-by-side comparison with another PerformanceReport.

        Parameters
        ----------
        other:
            A second ``PerformanceReport`` instance to compare against.
        labels:
            Column labels for self and other.

        Returns
        -------
        pd.DataFrame
            Metrics as rows, two strategy columns.
        """
        self_metrics = self._metrics if self._metrics is not None else self.compute_all()
        other_metrics = other._metrics if other._metrics is not None else other.compute_all()

        df = pd.DataFrame(
            {labels[0]: self_metrics, labels[1]: other_metrics}
        )
        return df
