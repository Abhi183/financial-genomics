"""
Tests for src/backtest/ modules:
  - src/backtest/metrics.py       (sharpe_ratio, max_drawdown, kelly_fraction)
  - src/backtest/engine.py        (BacktestEngine, BacktestResult)
  - src/backtest/strategies.py    (BuyAndHoldStrategy)
  - src/backtest/report.py        (PerformanceReport)
"""

import math

import numpy as np
import pytest

from src.backtest.metrics import sharpe_ratio, max_drawdown, kelly_fraction
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.strategies import BuyAndHoldStrategy
from src.backtest.report import PerformanceReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_prices():
    """100 prices with no movement."""
    return np.ones(100) * 100.0


@pytest.fixture
def rising_prices():
    """Linearly rising prices: 100, 101, ..., 199."""
    return np.linspace(100, 199, 100)


@pytest.fixture
def drawdown_equity():
    """Classic drawdown scenario: 100 -> 80 -> 90. Max DD = 0.20."""
    return np.array([100.0, 80.0, 90.0])


@pytest.fixture
def zero_returns():
    return np.zeros(252)


@pytest.fixture
def positive_returns():
    rng = np.random.default_rng(0)
    return rng.standard_normal(252) * 0.01 + 0.001


@pytest.fixture
def buy_and_hold():
    return BuyAndHoldStrategy()


INITIAL_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self, positive_returns):
        sr = sharpe_ratio(positive_returns)
        assert sr > 0, f"Expected positive Sharpe, got {sr}"

    def test_zero_returns_zero_sharpe(self, zero_returns):
        sr = sharpe_ratio(zero_returns)
        assert math.isclose(sr, 0.0, abs_tol=1e-6), (
            f"Zero returns should yield Sharpe=0, got {sr}"
        )

    def test_known_value(self):
        """Deterministic check: constant return of 0.01/day, std=0 => very high Sharpe."""
        returns = np.full(252, 0.01)
        sr = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        # std=0 -> ill-defined; implementation should return inf or very large number
        assert sr > 10 or not math.isfinite(sr), (
            f"Expected very high or inf Sharpe for zero-std returns, got {sr}"
        )

    def test_negative_returns_negative_sharpe(self):
        returns = np.full(252, -0.005)
        sr = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sr < 0, f"Expected negative Sharpe for constant negative returns, got {sr}"

    def test_annualised_by_default(self):
        """With periods_per_year=252 and daily mean=0.001, std=0.01, Sharpe ~ 1.59."""
        rng = np.random.default_rng(7)
        # Construct returns with known mean and std
        n = 10_000
        returns = rng.normal(loc=0.001, scale=0.01, size=n)
        sr = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        expected = (0.001 / 0.01) * math.sqrt(252)
        assert abs(sr - expected) < 0.3, (
            f"Sharpe ratio {sr:.3f} far from expected {expected:.3f}"
        )

    def test_returns_float(self, positive_returns):
        sr = sharpe_ratio(positive_returns)
        assert isinstance(sr, float)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_known_equity_curve(self, drawdown_equity):
        """[100, 80, 90] -> max drawdown = 0.20."""
        dd = max_drawdown(drawdown_equity)
        assert math.isclose(dd, 0.20, abs_tol=1e-6), (
            f"Expected max_drawdown=0.20, got {dd}"
        )

    def test_monotonically_rising_zero_drawdown(self, rising_prices):
        dd = max_drawdown(rising_prices)
        assert math.isclose(dd, 0.0, abs_tol=1e-9), (
            f"Rising equity should have 0 drawdown, got {dd}"
        )

    def test_flat_equity_zero_drawdown(self, flat_prices):
        dd = max_drawdown(flat_prices)
        assert math.isclose(dd, 0.0, abs_tol=1e-9)

    def test_drawdown_in_0_1_range(self):
        rng = np.random.default_rng(3)
        equity = np.cumprod(1 + rng.normal(0.0005, 0.015, 500)) * 100_000
        dd = max_drawdown(equity)
        assert 0.0 <= dd <= 1.0, f"max_drawdown={dd} out of [0, 1]"

    def test_deeper_drawdown_detected(self):
        """50% drawdown: 100 -> 50 -> 75."""
        equity = np.array([100.0, 50.0, 75.0])
        dd = max_drawdown(equity)
        assert math.isclose(dd, 0.50, abs_tol=1e-6), (
            f"Expected 0.50, got {dd}"
        )

    def test_returns_float(self, rising_prices):
        dd = max_drawdown(rising_prices)
        assert isinstance(dd, float)


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_equity_curve_same_length_as_prices(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            commission_bps=1,
            slippage_bps=1,
        )
        result = engine.run(rising_prices, buy_and_hold)
        assert len(result.equity_curve) == len(rising_prices), (
            f"Equity curve length {len(result.equity_curve)} != "
            f"prices length {len(rising_prices)}"
        )

    def test_equity_curve_starts_at_initial_capital(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = engine.run(rising_prices, buy_and_hold)
        assert math.isclose(result.equity_curve[0], INITIAL_CAPITAL, rel_tol=1e-6), (
            f"Equity curve should start at {INITIAL_CAPITAL}, "
            f"got {result.equity_curve[0]}"
        )

    def test_equity_curve_returns_backtestresult(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = engine.run(rising_prices, buy_and_hold)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_positive_for_rising_prices(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = engine.run(rising_prices, buy_and_hold)
        assert result.equity_curve[-1] > INITIAL_CAPITAL, (
            "Long position in rising market should be profitable"
        )

    def test_equity_curve_all_positive(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = engine.run(rising_prices, buy_and_hold)
        assert np.all(np.array(result.equity_curve) > 0), (
            "Equity curve should remain positive"
        )

    def test_flat_prices_equity_stays_near_initial(self, flat_prices, buy_and_hold):
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            commission_bps=0,
            slippage_bps=0,
        )
        result = engine.run(flat_prices, buy_and_hold)
        # With zero costs, flat prices should yield roughly unchanged equity
        assert abs(result.equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL < 0.01


# ---------------------------------------------------------------------------
# BuyAndHoldStrategy
# ---------------------------------------------------------------------------

class TestBuyAndHoldStrategy:
    def test_always_returns_long_signal(self):
        strategy = BuyAndHoldStrategy()
        prices = np.linspace(100, 200, 50)
        for i in range(len(prices)):
            signal = strategy.generate_signal(prices, i)
            assert signal == "LONG", (
                f"Expected LONG at index {i}, got {signal}"
            )

    def test_signal_on_single_price(self):
        strategy = BuyAndHoldStrategy()
        prices = np.array([150.0])
        signal = strategy.generate_signal(prices, 0)
        assert signal == "LONG"

    def test_signal_does_not_depend_on_price_level(self):
        strategy = BuyAndHoldStrategy()
        for price_level in [1.0, 100.0, 10_000.0, 0.001]:
            prices = np.array([price_level])
            assert strategy.generate_signal(prices, 0) == "LONG"


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_equity_curve_attribute_exists(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = engine.run(rising_prices, buy_and_hold)
        assert hasattr(result, "equity_curve")

    def test_trades_attribute_exists(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = engine.run(rising_prices, buy_and_hold)
        assert hasattr(result, "trades")

    def test_equity_curve_starts_at_initial_capital(self):
        """Direct construction test for BacktestResult."""
        equity = np.array([100_000.0, 101_000.0, 102_000.0])
        result = BacktestResult(
            equity_curve=equity,
            trades=[],
            initial_capital=100_000.0,
        )
        assert math.isclose(result.equity_curve[0], 100_000.0, rel_tol=1e-9)

    def test_total_return_positive_for_rising(self, rising_prices, buy_and_hold):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        result = engine.run(rising_prices, buy_and_hold)
        total_return = (result.equity_curve[-1] - result.equity_curve[0]) / result.equity_curve[0]
        assert total_return > 0


# ---------------------------------------------------------------------------
# PerformanceReport
# ---------------------------------------------------------------------------

EXPECTED_REPORT_KEYS = {
    "sharpe_ratio",
    "max_drawdown",
    "total_return",
    "annualized_return",
    "win_rate",
    "total_trades",
}


class TestPerformanceReport:
    def _make_result(self, rising_prices):
        engine = BacktestEngine(initial_capital=INITIAL_CAPITAL)
        strategy = BuyAndHoldStrategy()
        return engine.run(rising_prices, strategy)

    def test_compute_all_returns_expected_keys(self, rising_prices):
        result = self._make_result(rising_prices)
        report = PerformanceReport(result)
        metrics = report.compute_all()
        assert isinstance(metrics, dict)
        for key in EXPECTED_REPORT_KEYS:
            assert key in metrics, (
                f"Expected key '{key}' missing from PerformanceReport.compute_all()"
            )

    def test_sharpe_ratio_is_finite(self, rising_prices):
        result = self._make_result(rising_prices)
        report = PerformanceReport(result)
        metrics = report.compute_all()
        sr = metrics["sharpe_ratio"]
        assert math.isfinite(sr) or sr == float("inf"), (
            f"sharpe_ratio should be finite (or inf), got {sr}"
        )

    def test_max_drawdown_in_0_1(self, rising_prices):
        result = self._make_result(rising_prices)
        report = PerformanceReport(result)
        metrics = report.compute_all()
        dd = metrics["max_drawdown"]
        assert 0.0 <= dd <= 1.0, f"max_drawdown out of [0,1]: {dd}"

    def test_win_rate_in_0_1(self, rising_prices):
        result = self._make_result(rising_prices)
        report = PerformanceReport(result)
        metrics = report.compute_all()
        wr = metrics["win_rate"]
        assert 0.0 <= wr <= 1.0, f"win_rate out of [0,1]: {wr}"

    def test_total_trades_non_negative(self, rising_prices):
        result = self._make_result(rising_prices)
        report = PerformanceReport(result)
        metrics = report.compute_all()
        assert metrics["total_trades"] >= 0

    def test_total_return_matches_equity_curve(self, rising_prices):
        result = self._make_result(rising_prices)
        report = PerformanceReport(result)
        metrics = report.compute_all()
        expected = (result.equity_curve[-1] - result.equity_curve[0]) / result.equity_curve[0]
        assert math.isclose(metrics["total_return"], expected, rel_tol=1e-5), (
            f"total_return mismatch: {metrics['total_return']:.6f} vs {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def test_known_value(self):
        """Kelly: f = (p * b - q) / b where b = avg_win / avg_loss."""
        win_rate = 0.6
        avg_win = 0.02
        avg_loss = 0.01
        # b = avg_win / avg_loss = 2.0
        # f = (0.6 * 2 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.4
        expected = 0.4
        f = kelly_fraction(win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss)
        assert math.isclose(f, expected, abs_tol=1e-6), (
            f"kelly_fraction: expected {expected}, got {f}"
        )

    def test_fifty_fifty_symmetric(self):
        """50% win rate, equal win/loss -> Kelly = 0."""
        f = kelly_fraction(win_rate=0.5, avg_win=0.01, avg_loss=0.01)
        assert math.isclose(f, 0.0, abs_tol=1e-6), (
            f"50/50 symmetric should give f=0, got {f}"
        )

    def test_certain_win_returns_1(self):
        """100% win rate should return Kelly fraction of 1 (or all-in)."""
        f = kelly_fraction(win_rate=1.0, avg_win=0.01, avg_loss=0.01)
        assert f >= 0.99, f"Expected Kelly ~1.0 for certain win, got {f}"

    def test_certain_loss_returns_non_positive(self):
        """0% win rate should return 0 or negative Kelly (don't bet)."""
        f = kelly_fraction(win_rate=0.0, avg_win=0.01, avg_loss=0.01)
        assert f <= 0.0, f"0% win rate should give f<=0, got {f}"

    def test_kelly_fraction_is_float(self):
        f = kelly_fraction(win_rate=0.55, avg_win=0.015, avg_loss=0.01)
        assert isinstance(f, float)

    def test_kelly_positive_for_favorable_bet(self):
        """Win rate > loss rate with good odds should give positive Kelly."""
        f = kelly_fraction(win_rate=0.65, avg_win=0.02, avg_loss=0.01)
        assert f > 0, f"Expected positive Kelly for favorable bet, got {f}"


# ---------------------------------------------------------------------------
# Win rate = wins / total_trades
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_win_rate_definition(self):
        """Win rate must equal wins / total_trades."""
        wins = 60
        total = 100
        expected = wins / total
        result = BacktestResult(
            equity_curve=np.linspace(100_000, 110_000, 101),
            trades=[{"pnl": 1.0 if i < wins else -1.0} for i in range(total)],
            initial_capital=100_000.0,
        )
        report = PerformanceReport(result)
        metrics = report.compute_all()
        assert math.isclose(metrics["win_rate"], expected, abs_tol=1e-6), (
            f"Expected win_rate={expected}, got {metrics['win_rate']}"
        )

    def test_all_winning_trades(self):
        trades = [{"pnl": 1.0} for _ in range(10)]
        result = BacktestResult(
            equity_curve=np.linspace(100_000, 110_000, 101),
            trades=trades,
            initial_capital=100_000.0,
        )
        report = PerformanceReport(result)
        metrics = report.compute_all()
        assert math.isclose(metrics["win_rate"], 1.0, abs_tol=1e-6)

    def test_all_losing_trades(self):
        trades = [{"pnl": -1.0} for _ in range(10)]
        result = BacktestResult(
            equity_curve=np.linspace(100_000, 90_000, 101),
            trades=trades,
            initial_capital=100_000.0,
        )
        report = PerformanceReport(result)
        metrics = report.compute_all()
        assert math.isclose(metrics["win_rate"], 0.0, abs_tol=1e-6)

    def test_no_trades_win_rate_zero(self):
        result = BacktestResult(
            equity_curve=np.ones(100) * 100_000,
            trades=[],
            initial_capital=100_000.0,
        )
        report = PerformanceReport(result)
        metrics = report.compute_all()
        assert metrics["win_rate"] == 0.0
        assert metrics["total_trades"] == 0
