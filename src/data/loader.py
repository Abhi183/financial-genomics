"""
Market data loading and preprocessing utilities.

This module provides the ``MarketDataLoader`` class, which handles downloading
market data from Yahoo Finance, reading from local CSV files, validating
DataFrames, and splitting into train/test partitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Load, validate, and split financial market data.

    This class centralises all I/O concerns for the Financial Genomics
    pipeline.  It can pull fresh OHLCV data from Yahoo Finance or read from
    a pre-downloaded CSV, and it enforces a uniform schema before any
    downstream processing begins.

    Examples
    --------
    >>> loader = MarketDataLoader()
    >>> df = loader.download_spy("2015-01-01", "2023-12-31")
    >>> train, test = loader.train_test_split(df, "2022-01-01")
    """

    # Expected OHLCV column names (case-insensitive comparison is performed
    # in validate() so that data from different sources can be normalised).
    REQUIRED_COLUMNS: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def download_spy(self, start: str, end: str) -> pd.DataFrame:
        """Download SPDR S&P 500 ETF (SPY) daily OHLCV data from Yahoo Finance.

        Parameters
        ----------
        start:
            Start date string in ``YYYY-MM-DD`` format (inclusive).
        end:
            End date string in ``YYYY-MM-DD`` format (exclusive, following
            yfinance convention).

        Returns
        -------
        pd.DataFrame
            A DataFrame with a ``DatetimeIndex`` and columns
            ``["Open", "High", "Low", "Close", "Volume"]``, validated and
            de-duplicated.

        Raises
        ------
        ValueError
            If the download returns an empty DataFrame or the date range is
            logically invalid.
        RuntimeError
            If the yfinance download fails for network or API reasons.
        """
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError(
                f"start date {start!r} must be strictly before end date {end!r}."
            )

        logger.info("Downloading SPY from %s to %s via yfinance …", start, end)
        try:
            raw: pd.DataFrame = yf.download(
                "SPY",
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"yfinance download failed for SPY ({start} – {end}): {exc}"
            ) from exc

        if raw.empty:
            raise ValueError(
                f"yfinance returned no data for SPY in the range {start} – {end}."
            )

        # yfinance may return MultiIndex columns when downloading a single
        # ticker with auto_adjust=True on some versions; flatten if needed.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Retain only the canonical OHLCV columns; yfinance sometimes adds
        # "Dividends" or "Stock Splits".
        available = [c for c in self.REQUIRED_COLUMNS if c in raw.columns]
        missing = set(self.REQUIRED_COLUMNS) - set(available)
        if missing:
            raise ValueError(
                f"Downloaded DataFrame is missing required columns: {missing}"
            )
        df = raw[list(self.REQUIRED_COLUMNS)].copy()

        logger.info("Downloaded %d rows for SPY.", len(df))
        return self.validate(df)

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Load OHLCV data from a CSV file on disk.

        The CSV must have a date/datetime column that can be used as the
        index.  The loader attempts several common index-column names
        (``"Date"``, ``"Datetime"``, ``"date"``, ``"datetime"``) before
        falling back to the first column.

        Parameters
        ----------
        path:
            Filesystem path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Validated DataFrame with a ``DatetimeIndex`` and OHLCV columns.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist on the filesystem.
        ValueError
            If the CSV cannot be parsed or is missing required columns.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info("Loading data from %s …", path)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV at {path}: {exc}") from exc

        # Attempt to set the DatetimeIndex from a known column name.
        date_candidates = ["Date", "Datetime", "date", "datetime", "timestamp"]
        index_col: str | None = None
        for candidate in date_candidates:
            if candidate in df.columns:
                index_col = candidate
                break

        if index_col is None:
            # Fall back to the first column.
            index_col = df.columns[0]
            logger.warning(
                "No recognised date column found; using first column %r as index.",
                index_col,
            )

        try:
            df[index_col] = pd.to_datetime(df[index_col])
        except Exception as exc:
            raise ValueError(
                f"Could not parse column {index_col!r} as datetime: {exc}"
            ) from exc

        df = df.set_index(index_col)
        df.index.name = "Date"

        logger.info("Loaded %d rows from %s.", len(df), path)
        return self.validate(df)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean a market-data DataFrame.

        Performs the following checks and transformations in order:

        1. Ensures the index is a :class:`pandas.DatetimeIndex`; attempts
           conversion if it is not.
        2. Checks for the presence of the required OHLCV columns.
        3. Drops duplicate index entries, keeping the first occurrence.
        4. Sorts the index in ascending chronological order.
        5. Reports (but does *not* silently drop) rows with ``NaN`` values;
           forward-fills a single missing day and raises if gaps remain.

        Parameters
        ----------
        df:
            Raw DataFrame to validate.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame satisfying the pipeline schema.

        Raises
        ------
        TypeError
            If the index cannot be coerced to a ``DatetimeIndex``.
        ValueError
            If required columns are missing or irrecoverable NaNs remain
            after forward-fill.
        """
        # ---- 1. Ensure DatetimeIndex ----
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                "Index is not a DatetimeIndex (%s); attempting conversion.",
                type(df.index).__name__,
            )
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                raise TypeError(
                    f"Cannot convert index to DatetimeIndex: {exc}"
                ) from exc
        df.index.name = "Date"

        # ---- 2. Check required columns ----
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"DataFrame is missing required OHLCV columns: {missing_cols}. "
                f"Present columns: {list(df.columns)}"
            )

        # ---- 3. Drop duplicate dates ----
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            logger.warning("Dropping %d duplicate date entries.", n_dupes)
            df = df[~df.index.duplicated(keep="first")]

        # ---- 4. Sort chronologically ----
        df = df.sort_index()

        # ---- 5. Handle NaNs ----
        nan_count = df[list(self.REQUIRED_COLUMNS)].isna().sum().sum()
        if nan_count > 0:
            logger.warning(
                "%d NaN values detected; applying forward-fill (limit=1).",
                nan_count,
            )
            df[list(self.REQUIRED_COLUMNS)] = (
                df[list(self.REQUIRED_COLUMNS)].ffill(limit=1)
            )
            remaining_nans = df[list(self.REQUIRED_COLUMNS)].isna().sum().sum()
            if remaining_nans > 0:
                raise ValueError(
                    f"{remaining_nans} NaN values remain after forward-fill. "
                    "Inspect the data for large contiguous gaps."
                )

        logger.info(
            "Validation passed: %d rows, index range %s – %s.",
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )
        return df[list(self.REQUIRED_COLUMNS)]

    def train_test_split(
        self, df: pd.DataFrame, split_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split a time-series DataFrame into train and test partitions.

        The split is performed on the date index.  All rows with index
        *strictly before* ``split_date`` go into the training set; all rows
        from ``split_date`` onwards form the test set.  This preserves
        temporal ordering and prevents look-ahead bias.

        Parameters
        ----------
        df:
            Validated OHLCV DataFrame with a ``DatetimeIndex``.
        split_date:
            The first date of the test set in ``YYYY-MM-DD`` format.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(train_df, test_df)`` — two non-overlapping DataFrames.

        Raises
        ------
        ValueError
            If ``split_date`` falls outside the range of ``df``'s index,
            which would produce an empty train or test partition.
        """
        split_ts = pd.Timestamp(split_date)

        if split_ts <= df.index.min():
            raise ValueError(
                f"split_date {split_date!r} is on or before the earliest "
                f"data date {df.index.min().date()}, which would produce an "
                "empty training set."
            )
        if split_ts > df.index.max():
            raise ValueError(
                f"split_date {split_date!r} is after the latest data date "
                f"{df.index.max().date()}, which would produce an empty test set."
            )

        train_df = df[df.index < split_ts].copy()
        test_df = df[df.index >= split_ts].copy()

        logger.info(
            "Train/test split at %s: train=%d rows, test=%d rows.",
            split_date,
            len(train_df),
            len(test_df),
        )
        return train_df, test_df
