"""
Volatility-Adaptive Discretizer: mapping financial returns to genomic bases.

Biological analogy
------------------
In molecular biology, DNA is encoded in an alphabet of four nucleotide bases:

    A (Adenine)   — pairs with T; often marks the *start* of transcription
    C (Cytosine)  — paired with G; structurally stable
    G (Guanine)   — paired with C; structurally stable
    T (Thymine)   — pairs with A; often marks *termination*

In the Financial Genomics framework each trading day is mapped to one of the
same four symbols, creating a "DNA strand" of market behaviour:

    A — **Crash**: a rare, severe negative return (tail event, left tail)
    T — **Spike**: a rare, severe positive return (tail event, right tail)
    C — **Calm**: the market is in a low-volatility regime
    G — **Growing**: the market is in a moderate-to-elevated volatility regime

The mapping is *adaptive* because the thresholds (sigma_global, Q25, Q75) are
estimated from the training data and then applied consistently to unseen data —
analogous to aligning new sequences against a reference genome.

Once a return series is encoded as a string such as "ACGTCCCGATGT…" it can be
analysed with the same k-mer and motif-discovery tools used in bioinformatics,
revealing recurring patterns (regulatory elements) in market dynamics.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.preprocessor import (
    compute_global_stats,
    compute_rolling_volatility,
)

logger = logging.getLogger(__name__)

# Genomic base alphabet and its integer encoding.
BASES: tuple[str, ...] = ("A", "C", "G", "T")
BASE_TO_INT: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
INT_TO_BASE: dict[int, str] = {v: k for k, v in BASE_TO_INT.items()}

# Human-readable regime names corresponding to each base.
BASE_TO_REGIME: dict[str, str] = {
    "A": "crash",
    "C": "calm",
    "G": "growing",
    "T": "spike",
}
REGIME_TO_BASE: dict[str, str] = {v: k for k, v in BASE_TO_REGIME.items()}


class VolatilityAdaptiveDiscretizer:
    """Convert a financial return series into a genomic base sequence.

    The discretiser learns four scalar thresholds from the training data:

    * **sigma_global** — global std of returns; used to detect crash / spike
      days by checking whether ``|r_t| > sigma_global``.
    * **Q25** — 25th percentile of the rolling volatility distribution.
    * **Q50** — median rolling volatility (stored for inspection; not used in
      the mapping rule itself).
    * **Q75** — 75th percentile of the rolling volatility distribution.

    Mapping rule
    ~~~~~~~~~~~~
    For each day *t* with return :math:`r_t` and rolling volatility
    :math:`v_t = \\text{std}(r_{t-w+1}, \\ldots, r_t)`:

    1. If :math:`r_t < -\\sigma_{\\text{global}}` → **A** (crash)
    2. If :math:`r_t > +\\sigma_{\\text{global}}` → **T** (spike)
    3. If :math:`v_t < Q_{25}` → **C** (calm)
    4. If :math:`Q_{25} \\le v_t \\le Q_{75}` → **G** (growing)
    5. If :math:`v_t > Q_{75}` → **G** (high-vol growing, proximity to Q75 wins)

    Step 5 intentionally maps the high-volatility tail to **G** rather than
    introducing a fifth symbol, keeping the alphabet minimal (four bases) and
    consistent with DNA encoding.

    Parameters
    ----------
    window:
        Rolling window (in trading days) for computing local volatility.
        Defaults to ``20`` (approximately one calendar month).
    vol_threshold:
        Optional manual override for *sigma_global*.  When provided, the
        fitted value is ignored in favour of this constant.
    """

    def __init__(self, window: int = 20, vol_threshold: Optional[float] = None) -> None:
        if not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive integer, got {window!r}.")
        if vol_threshold is not None and vol_threshold <= 0:
            raise ValueError(
                f"vol_threshold must be a positive float, got {vol_threshold!r}."
            )

        self.window: int = window
        self.vol_threshold: Optional[float] = vol_threshold

        # Parameters learned from training data:
        self.sigma_global_: Optional[float] = None
        self.q25_: Optional[float] = None
        self.q50_: Optional[float] = None
        self.q75_: Optional[float] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, returns_series: pd.Series) -> "VolatilityAdaptiveDiscretizer":
        """Estimate threshold parameters from a training return series.

        This is the *calibration* step.  It should be called on the training
        portion of the data only; the fitted parameters are then used to
        encode both train and test sets without information leakage.

        Parameters
        ----------
        returns_series:
            Log-return Series (output of
            :func:`~src.data.preprocessor.compute_log_returns`).

        Returns
        -------
        VolatilityAdaptiveDiscretizer
            The fitted discretiser instance (enables method chaining).

        Raises
        ------
        ValueError
            If the Series is too short to compute rolling statistics.
        """
        if returns_series.empty:
            raise ValueError("returns_series is empty; cannot fit discretizer.")

        sigma_global, q25, q50, q75 = compute_global_stats(
            returns_series, window=self.window
        )

        self.sigma_global_ = (
            self.vol_threshold if self.vol_threshold is not None else sigma_global
        )
        self.q25_ = q25
        self.q50_ = q50
        self.q75_ = q75
        self._is_fitted = True

        logger.info(
            "Discretizer fitted — sigma_global=%.6f, Q25=%.6f, Q50=%.6f, Q75=%.6f",
            self.sigma_global_,
            self.q25_,
            self.q50_,
            self.q75_,
        )
        return self

    # ------------------------------------------------------------------ #
    # Transform                                                            #
    # ------------------------------------------------------------------ #

    def transform(self, returns_series: pd.Series) -> str:
        """Encode a return series as a genomic base string.

        Each day is mapped to one of {A, C, G, T} using the fitted thresholds.
        Days within the first ``window - 1`` observations (where rolling
        volatility is undefined) are encoded using the return-magnitude rule
        only, treating them as C if the return is within ±sigma_global.

        Parameters
        ----------
        returns_series:
            Log-return Series to encode.  Can be the training or test set.

        Returns
        -------
        str
            A string of genomic bases, e.g. ``"ACGTCCCGATGT…"``, one
            character per trading day.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        ValueError
            If *returns_series* is empty.
        """
        self._check_fitted()

        if returns_series.empty:
            raise ValueError("returns_series is empty; nothing to transform.")

        rolling_vol: pd.Series = compute_rolling_volatility(
            returns_series, window=self.window
        )
        # Align on index so we can look up vol for each date.
        vol_lookup: dict = rolling_vol.to_dict()

        sequence_chars: List[str] = []
        for date, r_t in returns_series.items():
            base = self._classify(r_t, vol_lookup.get(date, None))
            sequence_chars.append(base)

        sequence = "".join(sequence_chars)
        logger.debug(
            "Encoded %d returns → sequence length %d.",
            len(returns_series),
            len(sequence),
        )
        return sequence

    def fit_transform(self, returns_series: pd.Series) -> str:
        """Fit on the series and immediately encode it.

        Equivalent to calling :meth:`fit` followed by :meth:`transform` on
        the same data.  Convenient for single-dataset workflows and unit
        tests, but should *not* be used when you need to encode a separate
        test set with the same thresholds.

        Parameters
        ----------
        returns_series:
            Log-return Series to fit on and encode.

        Returns
        -------
        str
            Encoded genomic base string.
        """
        return self.fit(returns_series).transform(returns_series)

    # ------------------------------------------------------------------ #
    # Inverse / encoding helpers                                           #
    # ------------------------------------------------------------------ #

    def inverse_transform(self, sequence: str) -> List[str]:
        """Map a base sequence back to human-readable regime names.

        Parameters
        ----------
        sequence:
            String of bases from the alphabet {A, C, G, T}.

        Returns
        -------
        list[str]
            List of regime names: ``"crash"``, ``"calm"``, ``"growing"``,
            or ``"spike"``.

        Raises
        ------
        ValueError
            If *sequence* contains a character outside the valid alphabet.
        """
        regimes: List[str] = []
        for i, base in enumerate(sequence):
            if base not in BASE_TO_REGIME:
                raise ValueError(
                    f"Invalid base {base!r} at position {i}. "
                    f"Valid bases are {BASES}."
                )
            regimes.append(BASE_TO_REGIME[base])
        return regimes

    def encode_to_int(self, sequence: str) -> List[int]:
        """Map a base string to a list of integer tokens.

        The integer encoding follows the ordering A=0, C=1, G=2, T=3, which
        matches the :class:`~src.models.lstm_model.GenomicLSTM` embedding
        layer.

        Parameters
        ----------
        sequence:
            String of bases from the alphabet {A, C, G, T}.

        Returns
        -------
        list[int]
            Integer token list, same length as *sequence*.

        Raises
        ------
        ValueError
            If *sequence* contains a character outside the valid alphabet.
        """
        tokens: List[int] = []
        for i, base in enumerate(sequence):
            if base not in BASE_TO_INT:
                raise ValueError(
                    f"Invalid base {base!r} at position {i}. "
                    f"Valid bases are {BASES}."
                )
            tokens.append(BASE_TO_INT[base])
        return tokens

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _classify(self, r_t: float, vol_t: Optional[float]) -> str:
        """Classify a single (return, volatility) pair into a base.

        Priority order:
        1. Crash  — extreme negative return
        2. Spike  — extreme positive return
        3. Calm   — low rolling volatility
        4. Growing — moderate/high rolling volatility (or vol unavailable)

        Parameters
        ----------
        r_t:
            The log return for this trading day.
        vol_t:
            The rolling volatility for this day, or ``None`` if the rolling
            window has not yet been filled (early observations).

        Returns
        -------
        str
            One of ``{"A", "C", "G", "T"}``.
        """
        # Tail events take priority regardless of the volatility regime.
        if r_t < -self.sigma_global_:  # type: ignore[operator]
            return "A"  # crash
        if r_t > self.sigma_global_:   # type: ignore[operator]
            return "T"  # spike

        # For early observations without rolling volatility, default to C.
        if vol_t is None or np.isnan(vol_t):
            return "C"

        if vol_t < self.q25_:  # type: ignore[operator]
            return "C"  # calm
        # Q25 <= vol_t: growing (includes the Q75+ high-vol regime)
        return "G"

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the discretiser has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "VolatilityAdaptiveDiscretizer has not been fitted yet. "
                "Call .fit() or .fit_transform() first."
            )

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        fitted_info = (
            f"sigma_global={self.sigma_global_:.6f}, "
            f"Q25={self.q25_:.6f}, Q75={self.q75_:.6f}"
            if self._is_fitted
            else "not fitted"
        )
        return (
            f"VolatilityAdaptiveDiscretizer("
            f"window={self.window}, "
            f"vol_threshold={self.vol_threshold}, "
            f"{fitted_info})"
        )
