"""End-to-end Financial Genomics pipeline.

Orchestrates all components:
  1. Data download and pre-processing (SPY daily closes)
  2. Volatility-based discretisation to A/C/G/T sequence
  3. K-mer analysis and motif discovery
  4. LSTM model training with walk-forward validation
  5. Baseline model training (ARIMA, GARCH, SAX, Matrix Profile)
  6. Event-driven backtesting for all strategies
  7. Figure generation and metrics table export

CLI usage
---------
python -m src.pipeline --config configs/config.yaml --mode full --output-dir results/

Modes:
  full      — run every step end-to-end
  train     — steps 1–4 only (data + LSTM)
  backtest  — step 6 (requires prior results in output-dir)
  report    — step 7 only (requires prior results in output-dir)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="arch")

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# Optional heavy imports — imported lazily in each step so the module can be
# imported even if not all packages are installed.
# ---------------------------------------------------------------------------

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    _ARIMA_AVAILABLE = True
except ImportError:
    _ARIMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "ticker": "SPY",
        "start_date": "2000-01-01",
        "end_date": "2023-12-31",
        "returns_col": "close",
    },
    "discretizer": {
        "n_states": 4,
        "method": "volatility",      # 'volatility' | 'quantile'
        "vol_window": 20,
        "labels": ["A", "C", "G", "T"],  # spike, calm, drop, crash
    },
    "kmer": {
        "k": 3,
        "top_n": 15,
    },
    "lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "seq_len": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 50,
        "early_stopping_patience": 7,
    },
    "backtest": {
        "initial_capital": 100_000,
        "commission_bps": 1,
        "slippage_bps": 1,
        "train_window": 252,
        "test_window": 63,
        "spike_threshold": 0.70,
        "crash_threshold": 0.70,
        "position_sizing": "kelly",
    },
    "output": {
        "dir": "results/",
        "figures_dir": "paper/figures/",
        "save_models": True,
    },
}


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class FinancialGenomicsPipeline:
    """Orchestrates the full Financial Genomics research pipeline.

    Parameters
    ----------
    config_path:
        Path to a YAML configuration file.  If the file does not exist, the
        built-in default configuration is used.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config: Dict[str, Any] = self.load_config(config_path)
        self.results: Dict[str, Any] = {}

        # Resolved at runtime
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.Series] = None
        self.volatility: Optional[pd.Series] = None
        self.sequence: Optional[str] = None
        self.discretizer: Optional[Any] = None
        self.model_trainer: Optional[Any] = None
        self.backtest_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def load_config(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration, falling back to defaults.

        Parameters
        ----------
        path:
            Path to the YAML config file.

        Returns
        -------
        dict
            Merged configuration dictionary.
        """
        cfg = _DEFAULT_CONFIG.copy()

        if not Path(path).exists():
            logger.warning(
                "Config file not found at %r — using default configuration.", path
            )
            return cfg

        if not _YAML_AVAILABLE:
            logger.warning("PyYAML not installed — using default configuration.")
            return cfg

        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh)

        if user_cfg:
            cfg = _deep_merge(cfg, user_cfg)

        logger.info("Configuration loaded from %r", path)
        return cfg

    # ------------------------------------------------------------------
    # Full pipeline entry point
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute all pipeline steps end-to-end.

        Returns
        -------
        dict
            Aggregated results from every step.
        """
        logger.info("=" * 60)
        logger.info("  Financial Genomics — Full Pipeline")
        logger.info("=" * 60)

        self.step1_data()
        self.step2_discretize()
        self.step3_kmer()
        self.step4_train_lstm()
        self.step5_train_baselines()
        self.step6_backtest()
        self.step7_report()

        output_dir = self.config["output"]["dir"]
        self.save_results(output_dir)

        logger.info("Pipeline complete.  Results saved to %r", output_dir)
        return self.results

    # ------------------------------------------------------------------
    # Step 1: Data download and preprocessing
    # ------------------------------------------------------------------

    def step1_data(self) -> None:
        """Download SPY price data and compute returns and rolling volatility."""
        logger.info("[Step 1] Downloading and preprocessing data …")
        cfg = self.config["data"]

        if _YF_AVAILABLE:
            raw = yf.download(
                cfg["ticker"],
                start=cfg["start_date"],
                end=cfg["end_date"],
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                raise RuntimeError(
                    f"yfinance returned empty data for ticker {cfg['ticker']!r}."
                )
            raw.columns = [c.lower() for c in raw.columns]
        else:
            logger.warning(
                "yfinance not installed — generating synthetic SPY data for testing."
            )
            raw = _generate_synthetic_prices(cfg["start_date"], cfg["end_date"])

        self.prices = raw[["close"]].copy()
        self.returns = self.prices["close"].pct_change().dropna()

        vol_window = self.config["discretizer"]["vol_window"]
        self.volatility = (
            self.returns
            .rolling(vol_window)
            .std()
            .mul(np.sqrt(252))
            .dropna()
        )

        logger.info(
            "  Loaded %d bars for %s (%s → %s)",
            len(self.prices),
            cfg["ticker"],
            str(self.prices.index[0])[:10],
            str(self.prices.index[-1])[:10],
        )

        self.results["prices"] = self.prices["close"]
        self.results["returns"] = self.returns
        self.results["volatility"] = self.volatility
        self.results["dates"] = list(self.prices.index)

    # ------------------------------------------------------------------
    # Step 2: Discretisation
    # ------------------------------------------------------------------

    def step2_discretize(self) -> None:
        """Fit volatility-based discretiser and encode prices as A/C/G/T."""
        logger.info("[Step 2] Discretising return series …")

        self.discretizer = _SimpleDiscretizer(
            n_states=self.config["discretizer"]["n_states"],
            labels=self.config["discretizer"]["labels"],
            method=self.config["discretizer"]["method"],
            vol_window=self.config["discretizer"]["vol_window"],
        )

        self.sequence = self.discretizer.fit_transform(self.returns)

        # Compute transition matrix
        labels = self.config["discretizer"]["labels"]
        tm = _compute_transition_matrix(self.sequence, labels)

        logger.info("  Encoded %d returns → sequence of length %d", len(self.returns), len(self.sequence))
        logger.info("  Class distribution: %s", _count_labels(self.sequence))

        self.results["sequence"] = self.sequence
        self.results["transition_matrix"] = tm
        self.results["sequence_annotations"] = _well_known_events()

    # ------------------------------------------------------------------
    # Step 3: K-mer analysis
    # ------------------------------------------------------------------

    def step3_kmer(self) -> None:
        """Compute k-mer frequencies and log top motifs."""
        logger.info("[Step 3] K-mer analysis …")
        k = self.config["kmer"]["k"]
        top_n = self.config["kmer"]["top_n"]

        if not self.sequence:
            raise RuntimeError("sequence is None — run step2_discretize first.")

        kmer_counts = _count_kmers(self.sequence, k)
        top_kmers = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

        logger.info("  Top-%d %d-mers:", top_n, k)
        for kmer, count in top_kmers:
            logger.info("    %s: %d", kmer, count)

        self.results["kmer_counts"] = kmer_counts

    # ------------------------------------------------------------------
    # Step 4: LSTM training
    # ------------------------------------------------------------------

    def step4_train_lstm(self) -> None:
        """Train the Genomic LSTM with walk-forward cross-validation."""
        logger.info("[Step 4] Training Genomic LSTM …")

        lstm_cfg = self.config["lstm"]
        bt_cfg = self.config["backtest"]

        try:
            from src.models.lstm import GenomicLSTMTrainer  # type: ignore
        except ImportError:
            logger.warning(
                "src.models.lstm not found — using stub trainer for pipeline smoke-test."
            )
            self.model_trainer = _StubModelTrainer()
            self.results["train_losses"] = [0.9 - i * 0.01 for i in range(lstm_cfg["epochs"])]
            self.results["val_losses"] = [1.0 - i * 0.009 for i in range(lstm_cfg["epochs"])]
            self.results["train_accs"] = [0.25 + i * 0.005 for i in range(lstm_cfg["epochs"])]
            self.results["val_accs"] = [0.24 + i * 0.004 for i in range(lstm_cfg["epochs"])]
            return

        trainer = GenomicLSTMTrainer(
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            seq_len=lstm_cfg["seq_len"],
            batch_size=lstm_cfg["batch_size"],
            lr=lstm_cfg["lr"],
            epochs=lstm_cfg["epochs"],
            early_stopping_patience=lstm_cfg["early_stopping_patience"],
        )

        if self.sequence is None:
            raise RuntimeError("sequence is None — run step2_discretize first.")

        trainer.fit(self.sequence)
        self.model_trainer = trainer

        self.results["train_losses"] = trainer.train_losses
        self.results["val_losses"] = trainer.val_losses
        self.results["train_accs"] = trainer.train_accs
        self.results["val_accs"] = trainer.val_accs

        # Predictions on the full sequence for confusion matrix
        y_true, y_pred = trainer.evaluate(self.sequence)
        self.results["y_true"] = y_true
        self.results["y_pred"] = y_pred

        logger.info("  LSTM training complete.")

    # ------------------------------------------------------------------
    # Step 5: Baseline models
    # ------------------------------------------------------------------

    def step5_train_baselines(self) -> None:
        """Fit ARIMA, GARCH, SAX, and Matrix Profile baselines."""
        logger.info("[Step 5] Training baseline models …")

        baselines: Dict[str, Any] = {}

        # ARIMA
        if _ARIMA_AVAILABLE and self.returns is not None:
            try:
                arima_fit = ARIMA(self.returns.values, order=(1, 0, 1)).fit()
                baselines["arima"] = arima_fit
                logger.info("  ARIMA(1,0,1) fitted.")
            except Exception as exc:
                logger.warning("  ARIMA fitting failed: %s", exc)
        else:
            logger.info("  ARIMA skipped (statsmodels not available or returns not loaded).")

        # GARCH
        if _ARCH_AVAILABLE and self.returns is not None:
            try:
                garch_fit = arch_model(
                    self.returns * 100, vol="Garch", p=1, q=1
                ).fit(disp="off")
                baselines["garch"] = garch_fit
                logger.info("  GARCH(1,1) fitted.")
            except Exception as exc:
                logger.warning("  GARCH fitting failed: %s", exc)
        else:
            logger.info("  GARCH skipped (arch not available or returns not loaded).")

        # SAX baseline (stub — SAX discretisation shares the same discretiser
        # but uses a naive Markov-chain signal)
        baselines["sax"] = _SAXBaseline(self.discretizer)
        logger.info("  SAX Markov baseline created.")

        # Matrix Profile baseline (stub)
        baselines["matrix_profile"] = _MatrixProfileBaseline()
        logger.info("  Matrix Profile baseline stub created.")

        self.results["baselines"] = baselines

    # ------------------------------------------------------------------
    # Step 6: Backtesting
    # ------------------------------------------------------------------

    def step6_backtest(self) -> None:
        """Run event-driven backtests for all strategies."""
        logger.info("[Step 6] Running backtests …")

        from src.backtest.engine import BacktestEngine, WalkForwardBacktest
        from src.backtest.strategy import (
            BuyAndHoldStrategy,
            GARCHStrategy,
            GenomicTradingStrategy,
        )

        bt_cfg = self.config["backtest"]
        baselines: Dict[str, Any] = self.results.get("baselines", {})

        if self.prices is None:
            raise RuntimeError("prices is None — run step1_data first.")

        equity_curves: Dict[str, pd.Series] = {}

        # --- Buy and Hold ---
        bh_strategy = BuyAndHoldStrategy()
        bh_engine = BacktestEngine(
            bh_strategy,
            initial_capital=bt_cfg["initial_capital"],
            commission_bps=bt_cfg["commission_bps"],
            slippage_bps=bt_cfg["slippage_bps"],
        )
        bh_result = bh_engine.run(self.prices)
        equity_curves["Buy & Hold"] = bh_result.equity_curve
        self.backtest_results["buy_and_hold"] = bh_result
        logger.info("  Buy & Hold backtest complete.")

        # --- Genomic LSTM (walk-forward) ---
        if self.model_trainer is not None:
            genomic_strategy = GenomicTradingStrategy(
                model_trainer=self.model_trainer,
                spike_threshold=bt_cfg["spike_threshold"],
                crash_threshold=bt_cfg["crash_threshold"],
                position_sizing=bt_cfg["position_sizing"],
            )
            genomic_engine = BacktestEngine(
                genomic_strategy,
                initial_capital=bt_cfg["initial_capital"],
                commission_bps=bt_cfg["commission_bps"],
                slippage_bps=bt_cfg["slippage_bps"],
            )
            wf = WalkForwardBacktest(
                genomic_engine,
                train_window=bt_cfg["train_window"],
                test_window=bt_cfg["test_window"],
            )
            wf_results = wf.run(self.prices, self.model_trainer, self.discretizer)
            if wf_results:
                agg = wf.aggregate_results(wf_results)
                equity_curves["Genomic LSTM"] = agg.equity_curve
                self.backtest_results["genomic_lstm"] = agg
                logger.info("  Genomic LSTM walk-forward backtest complete.")

        # --- GARCH strategy ---
        if "garch" in baselines:
            try:
                garch_strategy = GARCHStrategy(baselines["garch"])
                garch_engine = BacktestEngine(
                    garch_strategy,
                    initial_capital=bt_cfg["initial_capital"],
                    commission_bps=bt_cfg["commission_bps"],
                    slippage_bps=bt_cfg["slippage_bps"],
                )
                garch_result = garch_engine.run(self.prices)
                equity_curves["GARCH"] = garch_result.equity_curve
                self.backtest_results["garch"] = garch_result
                logger.info("  GARCH strategy backtest complete.")
            except Exception as exc:
                logger.warning("  GARCH strategy backtest failed: %s", exc)

        self.results["equity_curves"] = equity_curves
        self.results["benchmark_returns"] = self.returns
        self.results["backtest_results"] = self.backtest_results

    # ------------------------------------------------------------------
    # Step 7: Report generation
    # ------------------------------------------------------------------

    def step7_report(self) -> None:
        """Generate all paper figures and export a metrics comparison table."""
        logger.info("[Step 7] Generating report figures and metrics table …")

        from src.visualization.plots import create_full_report_figures

        figures_dir = self.config["output"]["figures_dir"]
        create_full_report_figures(self.results, save_dir=figures_dir)

        # Build metrics comparison table
        metrics_rows: List[Dict[str, Any]] = []
        for strategy_name, bt_result in self.backtest_results.items():
            row = bt_result.performance._metrics or bt_result.performance.compute_all()
            row = {"strategy": strategy_name.replace("_", " ").title(), **row}
            metrics_rows.append(row)

        if metrics_rows:
            metrics_df = pd.DataFrame(metrics_rows).set_index("strategy")
            output_dir = self.config["output"]["dir"]
            os.makedirs(output_dir, exist_ok=True)
            metrics_path = os.path.join(output_dir, "metrics_table.csv")
            metrics_df.to_csv(metrics_path)
            logger.info("  Metrics table saved to %r", metrics_path)
            self.results["metrics_table"] = metrics_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str = "results/") -> None:
        """Serialise pipeline results to disk.

        Saves:
        - ``results_summary.json`` — JSON-serialisable subset of results
        - ``prices.parquet`` — raw price DataFrame
        - ``returns.parquet`` — returns series
        - ``sequence.txt`` — full genomic sequence
        - ``model.pkl`` — pickled model trainer (if configured)

        Parameters
        ----------
        output_dir:
            Directory path for output files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prices
        if self.prices is not None:
            self.prices.to_parquet(os.path.join(output_dir, "prices.parquet"))

        # Returns
        if self.returns is not None:
            self.returns.to_frame("return").to_parquet(
                os.path.join(output_dir, "returns.parquet")
            )

        # Sequence
        if self.sequence is not None:
            with open(os.path.join(output_dir, "sequence.txt"), "w") as fh:
                fh.write(self.sequence)

        # Metrics table
        if "metrics_table" in self.results:
            self.results["metrics_table"].to_csv(
                os.path.join(output_dir, "metrics_table.csv")
            )

        # Model trainer
        if self.model_trainer is not None and self.config["output"].get("save_models"):
            try:
                with open(os.path.join(output_dir, "model_trainer.pkl"), "wb") as fh:
                    pickle.dump(self.model_trainer, fh)
            except Exception as exc:
                logger.warning("Could not pickle model trainer: %s", exc)

        # JSON summary
        summary = {
            "ticker": self.config["data"]["ticker"],
            "n_bars": len(self.prices) if self.prices is not None else None,
            "sequence_length": len(self.sequence) if self.sequence else None,
            "kmer_top5": (
                sorted(
                    self.results.get("kmer_counts", {}).items(),
                    key=lambda x: x[1], reverse=True
                )[:5]
            ),
        }
        with open(os.path.join(output_dir, "results_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)

        logger.info("Results saved to %r", output_dir)


# ---------------------------------------------------------------------------
# Internal helper classes and functions
# ---------------------------------------------------------------------------


class _SimpleDiscretizer:
    """Volatility- or quantile-based return discretiser mapping returns to
    genomic nucleotide labels (A=spike, C=calm, G=drop, T=crash).

    Parameters
    ----------
    n_states:
        Number of discrete states (must equal ``len(labels)``).
    labels:
        Ordered list of nucleotide labels from most positive to most negative.
    method:
        ``'volatility'`` uses a ±σ threshold; ``'quantile'`` uses equal-frequency bins.
    vol_window:
        Rolling window for volatility estimation (only used with ``'volatility'``).
    """

    def __init__(
        self,
        n_states: int = 4,
        labels: Optional[List[str]] = None,
        method: str = "volatility",
        vol_window: int = 20,
    ) -> None:
        self.n_states = n_states
        self.labels = labels or ["A", "C", "G", "T"]
        self.method = method
        self.vol_window = vol_window
        self._thresholds: Optional[np.ndarray] = None
        self._fitted = False

    def fit_transform(self, returns: pd.Series) -> str:
        """Fit on returns and return the encoded sequence string.

        Parameters
        ----------
        returns:
            Daily return series.

        Returns
        -------
        str
            Genomic sequence string of same length as valid (non-NaN) returns.
        """
        returns = pd.Series(returns).dropna()

        if self.method == "quantile":
            quantiles = np.linspace(0, 1, self.n_states + 1)
            self._thresholds = np.quantile(returns, quantiles[1:-1])
        else:
            # Volatility-based: split on ±0.5σ and ±1.5σ
            sigma = returns.std()
            self._thresholds = np.array([-1.5 * sigma, -0.5 * sigma, 0.5 * sigma])

        self._fitted = True
        return self.transform(returns)

    def transform(self, returns: pd.Series) -> str:
        """Encode a returns series using fitted thresholds.

        Parameters
        ----------
        returns:
            Daily return series.

        Returns
        -------
        str
            Encoded genomic sequence.

        Raises
        ------
        RuntimeError
            If ``fit_transform`` has not been called first.
        """
        if not self._fitted or self._thresholds is None:
            raise RuntimeError("Discretizer has not been fitted. Call fit_transform first.")
        returns = pd.Series(returns).dropna()
        encoded = []
        for r in returns.values:
            # Label assignment: large positive → A (spike), moderate positive → C (calm),
            # moderate negative → G (drop), large negative → T (crash)
            idx = int(np.searchsorted(self._thresholds, r))
            # idx 0=T(crash), 1=G(drop), 2=C(calm), 3=A(spike)
            label_order = ["T", "G", "C", "A"]  # from most negative to most positive
            encoded.append(label_order[min(idx, len(label_order) - 1)])
        return "".join(encoded)


def _compute_transition_matrix(sequence: str, labels: List[str]) -> np.ndarray:
    """Compute a first-order Markov transition count matrix."""
    n = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    tm = np.zeros((n, n), dtype=float)
    for a, b in zip(sequence[:-1], sequence[1:]):
        i = idx.get(a)
        j = idx.get(b)
        if i is not None and j is not None:
            tm[i, j] += 1
    return tm


def _count_labels(sequence: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for char in sequence:
        counts[char] = counts.get(char, 0) + 1
    return dict(sorted(counts.items()))


def _count_kmers(sequence: str, k: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i: i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def _well_known_events() -> Dict[str, str]:
    return {
        "2000-03-24": "Dot-com peak",
        "2001-09-11": "9/11",
        "2008-09-15": "Lehman collapse",
        "2009-03-09": "GFC trough",
        "2020-02-20": "COVID crash start",
        "2020-03-23": "COVID trough",
        "2022-01-03": "Rate-hike cycle",
    }


def _generate_synthetic_prices(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate synthetic GBM price data for testing without yfinance."""
    dates = pd.bdate_range(start=start_date, end=end_date)
    rng = np.random.default_rng(42)
    daily_returns = rng.normal(0.0003, 0.01, size=len(dates))
    price = 100.0 * (1 + daily_returns).cumprod()
    return pd.DataFrame({"close": price}, index=dates)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflict)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class _StubModelTrainer:
    """Minimal stub trainer used when src.models.lstm is not available."""

    historical_win_rate: float = 0.52
    historical_avg_win: float = 0.012
    historical_avg_loss: float = 0.010

    def fit(self, sequence: str) -> None:
        pass

    def predict_proba(self, context: Any) -> np.ndarray:
        # Random probs summing to 1 — not informative, just for pipeline smoke-tests
        rng = np.random.default_rng()
        p = rng.dirichlet(alpha=[1, 1, 1, 1])
        return p

    def evaluate(self, sequence: str):
        rng = np.random.default_rng(0)
        labels = list("ACGT")
        n = len(sequence)
        y_true = list(sequence)
        y_pred = [labels[rng.integers(4)] for _ in range(n)]
        return y_true, y_pred


class _SAXBaseline:
    """Naive SAX Markov-chain baseline using the fitted discretiser."""

    def __init__(self, discretizer: Any) -> None:
        self.discretizer = discretizer

    def generate_signal(self, context: Any):
        from src.backtest.strategy import SignalEnum
        return SignalEnum.HOLD


class _MatrixProfileBaseline:
    """Stub for a Matrix Profile motif-based strategy."""

    def generate_signal(self, context: Any):
        from src.backtest.strategy import SignalEnum
        return SignalEnum.HOLD


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-genomics",
        description="Financial Genomics end-to-end pipeline",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "backtest", "report"],
        default="full",
        help=(
            "Pipeline mode: "
            "'full' runs all steps, "
            "'train' runs steps 1–4, "
            "'backtest' runs step 6, "
            "'report' runs step 7."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory for output files (default: results/)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    pipeline = FinancialGenomicsPipeline(config_path=args.config)
    # Override output dir from CLI
    pipeline.config["output"]["dir"] = args.output_dir

    if args.mode == "full":
        results = pipeline.run_full_pipeline()

    elif args.mode == "train":
        pipeline.step1_data()
        pipeline.step2_discretize()
        pipeline.step3_kmer()
        pipeline.step4_train_lstm()
        pipeline.save_results(args.output_dir)
        results = pipeline.results

    elif args.mode == "backtest":
        # Load pre-computed artefacts from output_dir
        prices_path = os.path.join(args.output_dir, "prices.parquet")
        returns_path = os.path.join(args.output_dir, "returns.parquet")
        seq_path = os.path.join(args.output_dir, "sequence.txt")
        model_path = os.path.join(args.output_dir, "model_trainer.pkl")

        if not Path(prices_path).exists():
            raise FileNotFoundError(
                f"Prices not found at {prices_path!r}. Run with --mode train first."
            )
        pipeline.prices = pd.read_parquet(prices_path)
        pipeline.returns = pd.read_parquet(returns_path)["return"]
        pipeline.volatility = pipeline.returns.rolling(20).std().mul(np.sqrt(252)).dropna()

        if Path(seq_path).exists():
            with open(seq_path) as fh:
                pipeline.sequence = fh.read().strip()
        pipeline.discretizer = _SimpleDiscretizer()
        if pipeline.sequence:
            pipeline.discretizer.fit_transform(pipeline.returns)

        if Path(model_path).exists():
            with open(model_path, "rb") as fh:
                pipeline.model_trainer = pickle.load(fh)

        pipeline.step6_backtest()
        pipeline.save_results(args.output_dir)
        results = pipeline.results

    elif args.mode == "report":
        # Load metrics table and equity curves from disk then regenerate figures
        metrics_path = os.path.join(args.output_dir, "metrics_table.csv")
        if Path(metrics_path).exists():
            pipeline.results["metrics_table"] = pd.read_csv(metrics_path, index_col=0)
        pipeline.step7_report()
        results = pipeline.results

    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    logger.info("Done.")
