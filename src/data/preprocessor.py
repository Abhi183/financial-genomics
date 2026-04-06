"""
Return and volatility pre-processing utilities.

These functions are intentionally stateless — they operate on pandas Series
and return new Series or scalar tuples.  They form the numerical backbone
of the Financial Genomics pipeline: raw OHLCV prices are first converted
to log-returns, then characterised by rolling volatility, and finally
summarised by global distributional statistics that drive the adaptive
discretisation step.

Biological analogy
------------------
Just as a genome sequence is derived from raw nucleotide readings, the
"genomic sequence" of market behaviour is derived from these transformed
price series.  Log-returns are the raw signal; rolling volatility is the
local expression level; and the global statistics act as the reference
genome against which each day's activity is classified.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series.

    The log return at time *t* is defined as:

    .. math::

        r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right)

    Log returns are preferred over simple returns because they are
    time-additive, approximately normally distributed for short horizons,
    and symmetric around zero — all convenient properties for the genomic
    encoding step.

    Parameters
    ----------
    prices:
        A :class:`pandas.Series` of strictly positive asset prices
        (typically the ``Close`` column of an OHLCV DataFrame) with a
        :class:`pandas.DatetimeIndex` in ascending chronological order.

    Returns
    -------
    pd.Series
        Log returns with the same index as *prices*.  The first element is
        ``NaN`` because :math:`P_{t-1}` is undefined for the first
        observation; it is dropped to keep the series aligned with
        downstream computations.

    Raises
    ------
    ValueError
        If *prices* is empty or contains non-positive values (prices must
        be strictly greater than zero for the logarithm to be defined).
    """
    if prices.empty:
        raise ValueError("prices Series is empty; cannot compute log returns.")

    if (prices <= 0).any():
        n_bad = int((prices <= 0).sum())
        raise ValueError(
            f"prices contains {n_bad} non-positive value(s). "
            "Log returns require strictly positive prices."
        )

    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()
    log_returns.name = "log_return"
    logger.debug("Computed %d log returns.", len(log_returns))
    return log_returns


def compute_rolling_volatility(
    returns: pd.Series, window: int = 20
) -> pd.Series:
    """Compute rolling standard deviation of log returns (realised volatility).

    Rolling volatility is the primary feature used by the adaptive
    discretiser to determine the *local regime* of the market.  A 20-day
    window (one calendar month) is the default, reflecting common practice
    in financial risk management.

    Parameters
    ----------
    returns:
        Log-return Series produced by :func:`compute_log_returns`.
    window:
        Look-back window in trading days.  Must be a positive integer.
        Defaults to ``20``.

    Returns
    -------
    pd.Series
        Rolling standard deviation with the same index as *returns*.
        The first ``window - 1`` entries are ``NaN`` and are dropped before
        the Series is returned.

    Raises
    ------
    ValueError
        If *returns* is empty or *window* is not a positive integer.
    """
    if returns.empty:
        raise ValueError("returns Series is empty; cannot compute rolling volatility.")

    if not isinstance(window, int) or window < 1:
        raise ValueError(
            f"window must be a positive integer, got {window!r}."
        )

    rolling_vol = returns.rolling(window=window, min_periods=window).std()
    rolling_vol = rolling_vol.dropna()
    rolling_vol.name = f"rolling_vol_{window}d"
    logger.debug(
        "Computed rolling volatility with window=%d; %d valid observations.",
        window,
        len(rolling_vol),
    )
    return rolling_vol


def compute_global_stats(
    returns: pd.Series, window: int = 20
) -> Tuple[float, float, float, float]:
    """Compute global distributional statistics of the rolling volatility series.

    These four scalars serve as the *reference genome*: they define the
    boundaries that partition the volatility distribution into biological
    "bases" (A, C, G, T).  By fitting these statistics on the *training*
    set only, the pipeline avoids look-ahead bias when encoding the test set.

    Statistics returned:

    * **sigma_global** — the standard deviation of all log returns in the
      series.  This acts as the *crash/spike* threshold: a daily return more
      extreme than ±sigma_global is a tail event.
    * **Q25** — 25th percentile of the rolling volatility distribution.
      Days with rolling volatility below Q25 are "calm" (base C).
    * **Q50** — median rolling volatility (informational only).
    * **Q75** — 75th percentile of the rolling volatility distribution.
      Days with rolling volatility between Q25 and Q75 are "growing" (base G).

    Parameters
    ----------
    returns:
        Log-return Series produced by :func:`compute_log_returns`.
    window:
        Rolling window passed to :func:`compute_rolling_volatility`.
        Defaults to ``20``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(sigma_global, Q25, Q50, Q75)`` — all non-negative floats.

    Raises
    ------
    ValueError
        If *returns* has fewer elements than *window*, making it impossible
        to compute any rolling volatility statistics.
    """
    if len(returns) < window:
        raise ValueError(
            f"returns has only {len(returns)} observations, which is fewer "
            f"than the rolling window ({window}). Cannot compute global stats."
        )

    sigma_global: float = float(returns.std(ddof=1))

    rolling_vol = compute_rolling_volatility(returns, window=window)

    q25: float = float(rolling_vol.quantile(0.25))
    q50: float = float(rolling_vol.quantile(0.50))
    q75: float = float(rolling_vol.quantile(0.75))

    logger.info(
        "Global stats — sigma_global=%.6f, Q25=%.6f, Q50=%.6f, Q75=%.6f",
        sigma_global,
        q25,
        q50,
        q75,
    )
    return sigma_global, q25, q50, q75
