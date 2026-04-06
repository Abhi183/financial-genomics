"""
Baseline models for the Financial Genomics benchmark.

Four baselines are provided so that the genomic LSTM's performance can be
contextualised against well-established time-series and symbolic approaches:

1. **ARIMABaseline** — classical AutoRegressive Integrated Moving Average
   model that captures linear temporal dependencies in the return series.

2. **GARCHBaseline** — Generalised AutoRegressive Conditional Heteroskedasticity
   model that explicitly models volatility clustering, a well-known stylised
   fact of financial returns.

3. **SAXLogRegBaseline** — Symbolic Aggregate approXimation (SAX) encoding
   followed by logistic regression, a direct non-genomic symbolic alternative
   to the DNA-alphabet approach.

4. **MatrixProfileBaseline** — time-series motif discovery via the Matrix
   Profile (Stumpy), which finds the nearest-neighbour subsequence structure
   without any symbolic encoding.

All baselines implement a common three-method interface:

    fit(...)   — learn from training data
    predict(…) — produce out-of-sample forecasts
    evaluate(…)— return a dict of performance metrics
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARIMA baseline
# ─────────────────────────────────────────────────────────────────────────────


class ARIMABaseline:
    """Classical ARIMA model for return-series forecasting.

    The fitted model is stored so that its summary and residuals can be
    inspected after calling :meth:`fit`.

    Parameters
    ----------
    None (order is passed at fit-time).

    Examples
    --------
    >>> baseline = ARIMABaseline()
    >>> baseline.fit(train_returns, order=(2, 0, 2))
    >>> preds = baseline.predict(steps=20)
    >>> metrics = baseline.evaluate(test_returns)
    """

    def __init__(self) -> None:
        self._model_fit: Any = None
        self._order: Tuple[int, int, int] = (1, 1, 1)
        self._returns_series: Optional[pd.Series] = None

    def fit(
        self,
        returns_series: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
    ) -> "ARIMABaseline":
        """Fit an ARIMA(p, d, q) model to the return series.

        Parameters
        ----------
        returns_series:
            Log-return Series with a DatetimeIndex.
        order:
            ``(p, d, q)`` ARIMA order tuple.  Defaults to ``(1, 1, 1)``.

        Returns
        -------
        ARIMABaseline
            Fitted instance (supports method chaining).

        Raises
        ------
        ValueError
            If the series is empty or the order tuple is invalid.
        """
        if returns_series.empty:
            raise ValueError("returns_series is empty; cannot fit ARIMA.")
        if len(order) != 3 or any(not isinstance(v, int) or v < 0 for v in order):
            raise ValueError(
                f"order must be a tuple of three non-negative integers, got {order!r}."
            )

        self._order = order
        self._returns_series = returns_series.copy()

        logger.info("Fitting ARIMA%s …", order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(returns_series, order=order)
            self._model_fit = model.fit()

        logger.info("ARIMA AIC=%.4f, BIC=%.4f", self._model_fit.aic, self._model_fit.bic)
        return self

    def predict(self, steps: int) -> pd.Series:
        """Produce *steps*-ahead out-of-sample return forecasts.

        Parameters
        ----------
        steps:
            Number of future periods to forecast.

        Returns
        -------
        pd.Series
            Forecast values indexed by integer step (1 to *steps*).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_fitted("predict")
        forecast = self._model_fit.forecast(steps=steps)
        if isinstance(forecast, pd.Series):
            return forecast.reset_index(drop=True)
        return pd.Series(forecast)

    def evaluate(self, test_returns: pd.Series) -> Dict[str, float]:
        """Evaluate forecast accuracy against a held-out return series.

        Parameters
        ----------
        test_returns:
            Actual log-return values for the test period.

        Returns
        -------
        dict[str, float]
            Keys: ``"mae"``, ``"rmse"``, ``"mse"``, ``"direction_accuracy"``.
            Direction accuracy measures the fraction of steps where the sign
            of the forecast matches the sign of the actual return.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If *test_returns* is empty.
        """
        self._check_fitted("evaluate")
        if test_returns.empty:
            raise ValueError("test_returns is empty.")

        n = len(test_returns)
        preds = self.predict(steps=n)
        actual = test_returns.values[:n]
        pred_vals = preds.values[:n]

        mae = float(mean_absolute_error(actual, pred_vals))
        mse = float(mean_squared_error(actual, pred_vals))
        rmse = float(np.sqrt(mse))
        direction_acc = float(
            np.mean(np.sign(actual) == np.sign(pred_vals))
        )

        metrics = {
            "mae": round(mae, 6),
            "mse": round(mse, 8),
            "rmse": round(rmse, 6),
            "direction_accuracy": round(direction_acc, 4),
        }
        logger.info("ARIMA evaluation: %s", metrics)
        return metrics

    def _check_fitted(self, context: str) -> None:
        if self._model_fit is None:
            raise RuntimeError(
                f"ARIMABaseline.{context}() called before fit(). "
                "Call .fit() first."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GARCH baseline
# ─────────────────────────────────────────────────────────────────────────────


class GARCHBaseline:
    """GARCH(p, q) model for volatility forecasting.

    Uses the ``arch`` library, which provides robust GARCH estimation with
    multiple distribution options.  The model scales returns to percent
    (×100) internally, which is the convention expected by ``arch``.

    Parameters
    ----------
    None (order is passed at fit-time).

    Examples
    --------
    >>> garch = GARCHBaseline()
    >>> garch.fit(train_returns, p=1, q=1)
    >>> vol_forecast = garch.forecast_volatility(steps=5)
    >>> metrics = garch.evaluate(test_returns)
    """

    def __init__(self) -> None:
        self._model_fit: Any = None
        self._p: int = 1
        self._q: int = 1
        self._returns_series: Optional[pd.Series] = None

    def fit(
        self,
        returns_series: pd.Series,
        p: int = 1,
        q: int = 1,
    ) -> "GARCHBaseline":
        """Fit a GARCH(p, q) model to the return series.

        Parameters
        ----------
        returns_series:
            Log-return Series.
        p:
            GARCH lag order (number of lagged variance terms).  Defaults to ``1``.
        q:
            ARCH lag order (number of lagged squared-return terms).  Defaults to ``1``.

        Returns
        -------
        GARCHBaseline
            Fitted instance.

        Raises
        ------
        ImportError
            If the ``arch`` library is not installed.
        ValueError
            If the series is empty or p/q are not positive integers.
        """
        try:
            from arch import arch_model  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'arch' library is required for GARCHBaseline. "
                "Install it with: pip install arch"
            ) from exc

        if returns_series.empty:
            raise ValueError("returns_series is empty; cannot fit GARCH.")
        if not isinstance(p, int) or p < 1:
            raise ValueError(f"p must be a positive integer, got {p!r}.")
        if not isinstance(q, int) or q < 1:
            raise ValueError(f"q must be a positive integer, got {q!r}.")

        self._p = p
        self._q = q
        self._returns_series = returns_series.copy()

        # arch expects returns in percent.
        returns_pct = returns_series * 100.0

        logger.info("Fitting GARCH(%d, %d) …", p, q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                returns_pct,
                vol="Garch",
                p=p,
                q=q,
                dist="normal",
                rescale=False,
            )
            self._model_fit = model.fit(disp="off", show_warning=False)

        logger.info(
            "GARCH(%d,%d) AIC=%.4f, BIC=%.4f",
            p,
            q,
            self._model_fit.aic,
            self._model_fit.bic,
        )
        return self

    def forecast_volatility(self, steps: int) -> pd.Series:
        """Forecast conditional volatility (annualised std) for *steps* ahead.

        Parameters
        ----------
        steps:
            Forecast horizon in trading days.

        Returns
        -------
        pd.Series
            Forecast variance (in return units, not percent) indexed 1 to
            *steps*.  Take the square root for conditional standard deviation.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        self._check_fitted("forecast_volatility")
        fc = self._model_fit.forecast(horizon=steps, reindex=False)
        variance_pct2 = fc.variance.iloc[-1].values  # shape (steps,)
        # Convert from percent² back to return² units.
        variance = variance_pct2 / 10_000.0
        vol = np.sqrt(variance)
        return pd.Series(vol, index=range(1, steps + 1), name="garch_vol_forecast")

    def evaluate(self, test_returns: pd.Series) -> Dict[str, float]:
        """Evaluate GARCH volatility forecast against realised absolute returns.

        Realised absolute daily returns are used as a noisy proxy for true
        daily volatility, which is the standard approach when tick-level data
        is unavailable.

        Parameters
        ----------
        test_returns:
            Actual log-return values for the test period.

        Returns
        -------
        dict[str, float]
            Keys: ``"vol_mae"``, ``"vol_rmse"`` (computed in return units).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If *test_returns* is empty.
        """
        self._check_fitted("evaluate")
        if test_returns.empty:
            raise ValueError("test_returns is empty.")

        n = len(test_returns)
        vol_forecast = self.forecast_volatility(steps=n)
        realised_vol = np.abs(test_returns.values[:n])
        forecast_vol = vol_forecast.values[:n]

        vol_mae = float(mean_absolute_error(realised_vol, forecast_vol))
        vol_rmse = float(np.sqrt(mean_squared_error(realised_vol, forecast_vol)))

        metrics = {
            "vol_mae": round(vol_mae, 6),
            "vol_rmse": round(vol_rmse, 6),
        }
        logger.info("GARCH evaluation: %s", metrics)
        return metrics

    def _check_fitted(self, context: str) -> None:
        if self._model_fit is None:
            raise RuntimeError(
                f"GARCHBaseline.{context}() called before fit(). "
                "Call .fit() first."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAX + Logistic Regression baseline
# ─────────────────────────────────────────────────────────────────────────────


class SAXLogRegBaseline:
    """SAX symbolic encoding followed by logistic regression classification.

    The pipeline:

    1. Normalise a sliding window of ``word_size`` returns (z-score).
    2. Map each z-scored value to one of ``n_symbols`` quantile-based bins
       (SAX letters 0, 1, …, n_symbols-1).
    3. Concatenate the SAX word into a feature vector by one-hot encoding
       each position.
    4. Train a logistic regression classifier to predict the next SAX symbol
       (which corresponds to the market regime one step ahead).

    Parameters
    ----------
    n_symbols:
        Size of the SAX alphabet.  Defaults to ``4`` to match the genomic
        alphabet {A, C, G, T}.
    word_size:
        Number of time-steps in each SAX word (feature vector).  Defaults to
        ``10``.

    Examples
    --------
    >>> sax = SAXLogRegBaseline(n_symbols=4, word_size=10)
    >>> sax.fit(train_returns)
    >>> preds = sax.predict(test_returns)
    >>> metrics = sax.evaluate(test_returns, test_targets)
    """

    def __init__(self, n_symbols: int = 4, word_size: int = 10) -> None:
        if not isinstance(n_symbols, int) or n_symbols < 2:
            raise ValueError(
                f"n_symbols must be an integer >= 2, got {n_symbols!r}."
            )
        if not isinstance(word_size, int) or word_size < 1:
            raise ValueError(
                f"word_size must be a positive integer, got {word_size!r}."
            )
        self.n_symbols = n_symbols
        self.word_size = word_size

        self._clf: Optional[LogisticRegression] = None
        self._breakpoints: Optional[np.ndarray] = None  # (n_symbols - 1,) quantile thresholds
        self._label_encoder: LabelEncoder = LabelEncoder()

    # ------------------------------------------------------------------

    def fit(self, train_returns: pd.Series) -> "SAXLogRegBaseline":
        """Fit the SAX breakpoints and logistic regression on training data.

        Parameters
        ----------
        train_returns:
            Log-return Series for the training period.

        Returns
        -------
        SAXLogRegBaseline
            Fitted instance.

        Raises
        ------
        ValueError
            If the series is too short to construct at least one training sample.
        """
        if len(train_returns) < self.word_size + 1:
            raise ValueError(
                f"train_returns has {len(train_returns)} observations, but "
                f"word_size={self.word_size} requires at least "
                f"{self.word_size + 1} observations."
            )

        arr = train_returns.values

        # Compute quantile breakpoints from training data.
        quantiles = np.linspace(0, 100, self.n_symbols + 1)[1:-1]
        self._breakpoints = np.percentile(arr, quantiles)

        X_features, y_labels = self._build_features(arr)

        self._clf = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
        )
        self._clf.fit(X_features, y_labels)
        logger.info(
            "SAXLogRegBaseline fitted on %d samples.", len(y_labels)
        )
        return self

    def predict(self, test_returns: pd.Series) -> np.ndarray:
        """Predict the next SAX symbol for each window in *test_returns*.

        Parameters
        ----------
        test_returns:
            Log-return Series for the test period.

        Returns
        -------
        np.ndarray
            Integer predictions, one per window of length ``word_size``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If the series is too short.
        """
        self._check_fitted("predict")
        if len(test_returns) < self.word_size + 1:
            raise ValueError(
                f"test_returns has only {len(test_returns)} observations; "
                f"need at least {self.word_size + 1}."
            )
        arr = test_returns.values
        X_features, _ = self._build_features(arr)
        return self._clf.predict(X_features)  # type: ignore[union-attr]

    def evaluate(
        self, test_returns: pd.Series, test_targets: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate classification performance.

        If *test_targets* is not provided, targets are derived from the
        SAX encoding of *test_returns* using the fitted breakpoints.

        Parameters
        ----------
        test_returns:
            Log-return Series for the test period.
        test_targets:
            Optional ground-truth labels aligned with
            ``predict(test_returns)``'s output.

        Returns
        -------
        dict[str, float]
            Keys: ``"accuracy"``, ``"precision"``, ``"recall"``, ``"f1"``.
        """
        self._check_fitted("evaluate")
        arr = test_returns.values
        X_features, y_derived = self._build_features(arr)
        y_true = test_targets if test_targets is not None else y_derived
        y_pred = self._clf.predict(X_features)  # type: ignore[union-attr]

        # Align lengths.
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(
                float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
            ),
            "f1": round(
                float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
            ),
        }
        logger.info("SAXLogReg evaluation: %s", metrics)
        return metrics

    # ------------------------------------------------------------------

    def _sax_encode_value(self, value: float) -> int:
        """Map a single z-scored value to a SAX symbol index."""
        return int(np.searchsorted(self._breakpoints, value))  # type: ignore[arg-type]

    def _build_features(
        self, arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct (X, y) feature matrix from a return array.

        Each sample is the SAX encoding of a length-``word_size`` window
        (one-hot per position), and the target is the SAX symbol of the
        next value.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X, y)`` with shapes ``(n_samples, word_size * n_symbols)``
            and ``(n_samples,)``.
        """
        # Z-score normalise using the window's own mean and std.
        X_list: List[np.ndarray] = []
        y_list: List[int] = []

        for i in range(len(arr) - self.word_size):
            window = arr[i : i + self.word_size]
            std = window.std()
            if std < 1e-8:
                std = 1e-8
            z_window = (window - window.mean()) / std

            # Encode each position as a SAX symbol.
            sax_word = np.array([self._sax_encode_value(v) for v in z_window])

            # One-hot encode the SAX word.
            one_hot = np.zeros(self.word_size * self.n_symbols, dtype=np.float32)
            for pos, sym in enumerate(sax_word):
                one_hot[pos * self.n_symbols + int(sym)] = 1.0

            X_list.append(one_hot)

            # Target: SAX symbol of the next return.
            next_val = arr[i + self.word_size]
            z_next = (next_val - window.mean()) / std
            y_list.append(self._sax_encode_value(z_next))

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def _check_fitted(self, context: str) -> None:
        if self._clf is None or self._breakpoints is None:
            raise RuntimeError(
                f"SAXLogRegBaseline.{context}() called before fit(). "
                "Call .fit() first."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Matrix Profile baseline
# ─────────────────────────────────────────────────────────────────────────────


class MatrixProfileBaseline:
    """Nearest-neighbour motif baseline using the Matrix Profile.

    The Matrix Profile (Yeh et al. 2016) is an all-pairs Z-normalised
    Euclidean distance structure over a time series.  Each position's
    *nearest neighbour* (the most similar subsequence elsewhere in the
    series) is stored alongside its distance.  This enables efficient
    detection of repeated patterns (motifs) and anomalies (discords).

    Prediction strategy: for a new query window of length ``window_size``,
    find the nearest neighbour in the training series and use the value that
    followed *that* window as the forecast.

    Parameters
    ----------
    window_size:
        Length of subsequences to compare.  Defaults to ``20``.

    Examples
    --------
    >>> mp = MatrixProfileBaseline(window_size=20)
    >>> mp.fit(train_returns)
    >>> pred = mp.predict_next(recent_window)
    >>> metrics = mp.evaluate(test_returns)
    """

    def __init__(self, window_size: int = 20) -> None:
        if not isinstance(window_size, int) or window_size < 2:
            raise ValueError(
                f"window_size must be an integer >= 2, got {window_size!r}."
            )
        self.window_size = window_size
        self._train_series: Optional[np.ndarray] = None
        self._matrix_profile: Optional[np.ndarray] = None  # distances
        self._mp_index: Optional[np.ndarray] = None         # nearest-neighbour indices

    def fit(self, train_returns: pd.Series) -> "MatrixProfileBaseline":
        """Compute the Matrix Profile from the training return series.

        Parameters
        ----------
        train_returns:
            Log-return Series for the training period.  Must have at least
            ``2 * window_size`` observations for the matrix profile to be
            meaningful.

        Returns
        -------
        MatrixProfileBaseline
            Fitted instance.

        Raises
        ------
        ImportError
            If ``stumpy`` is not installed.
        ValueError
            If *train_returns* is too short.
        """
        try:
            import stumpy  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'stumpy' library is required for MatrixProfileBaseline. "
                "Install it with: pip install stumpy"
            ) from exc

        if len(train_returns) < 2 * self.window_size:
            raise ValueError(
                f"train_returns has {len(train_returns)} observations, but "
                f"window_size={self.window_size} requires at least "
                f"{2 * self.window_size} observations."
            )

        arr = train_returns.values.astype(np.float64)
        self._train_series = arr

        logger.info(
            "Computing Matrix Profile (m=%d, n=%d) …",
            self.window_size,
            len(arr),
        )
        mp_result = stumpy.stump(arr, m=self.window_size)
        # stump returns an (n - m + 1, 4) array.
        # Column 0: MP distances; Column 1: MP indices.
        self._matrix_profile = mp_result[:, 0].astype(np.float64)
        self._mp_index = mp_result[:, 1].astype(np.int64)

        logger.info("Matrix Profile computed.  Min dist=%.6f.", float(self._matrix_profile.min()))
        return self

    def predict_next(self, recent_window: np.ndarray) -> float:
        """Predict the next return value using nearest-neighbour lookup.

        The query window is Z-normalised and compared against all training
        subsequences.  The value that immediately followed the closest
        training match is returned as the forecast.

        Parameters
        ----------
        recent_window:
            Array of the most recent ``window_size`` return values.

        Returns
        -------
        float
            Predicted next log return (the value that followed the nearest
            training neighbour).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``recent_window`` length does not match ``window_size`` or
            the nearest neighbour is the last position in the training series
            (no successor).
        """
        self._check_fitted("predict_next")

        if len(recent_window) != self.window_size:
            raise ValueError(
                f"recent_window length {len(recent_window)} does not match "
                f"window_size={self.window_size}."
            )

        arr = self._train_series  # type: ignore[assignment]
        n = len(arr)

        # Z-normalise the query.
        std = recent_window.std()
        if std < 1e-8:
            std = 1e-8
        query_z = (recent_window - recent_window.mean()) / std

        # Exhaustive Z-normalised Euclidean distance to all training windows.
        best_dist = np.inf
        best_idx = 0
        for i in range(n - self.window_size + 1):
            subseq = arr[i : i + self.window_size]
            sub_std = subseq.std()
            if sub_std < 1e-8:
                sub_std = 1e-8
            subseq_z = (subseq - subseq.mean()) / sub_std
            dist = float(np.linalg.norm(query_z - subseq_z))
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # The successor of the best match.
        successor_idx = best_idx + self.window_size
        if successor_idx >= n:
            logger.warning(
                "Nearest neighbour is at the end of the training series; "
                "returning 0.0 as forecast."
            )
            return 0.0

        return float(arr[successor_idx])

    def evaluate(self, test_returns: pd.Series) -> Dict[str, float]:
        """Evaluate the rolling next-step forecast on the test series.

        Parameters
        ----------
        test_returns:
            Log-return Series for the test period.  Must have at least
            ``window_size + 1`` observations so that at least one prediction
            can be made.

        Returns
        -------
        dict[str, float]
            Keys: ``"mae"``, ``"rmse"``, ``"direction_accuracy"``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If *test_returns* is too short.
        """
        self._check_fitted("evaluate")

        arr = test_returns.values.astype(np.float64)
        if len(arr) < self.window_size + 1:
            raise ValueError(
                f"test_returns has {len(arr)} observations; need at least "
                f"{self.window_size + 1}."
            )

        actuals: List[float] = []
        predictions: List[float] = []

        for i in range(self.window_size, len(arr)):
            window = arr[i - self.window_size : i]
            pred = self.predict_next(window)
            actual = float(arr[i])
            predictions.append(pred)
            actuals.append(actual)

        actuals_arr = np.array(actuals)
        preds_arr = np.array(predictions)

        mae = float(mean_absolute_error(actuals_arr, preds_arr))
        rmse = float(np.sqrt(mean_squared_error(actuals_arr, preds_arr)))
        direction_acc = float(np.mean(np.sign(actuals_arr) == np.sign(preds_arr)))

        metrics = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "direction_accuracy": round(direction_acc, 4),
        }
        logger.info("MatrixProfile evaluation: %s", metrics)
        return metrics

    def _check_fitted(self, context: str) -> None:
        if self._train_series is None:
            raise RuntimeError(
                f"MatrixProfileBaseline.{context}() called before fit(). "
                "Call .fit() first."
            )
