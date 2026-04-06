"""
Tests for src/features/discretization.py

The discretization module maps log-returns to a 4-letter genomic alphabet:
  A -> crash / large negative return
  T -> spike / large positive return
  C -> calm  / near-zero (low-vol) return
  G -> growth / moderate positive (normal trending) return
"""

import numpy as np
import pytest

from src.features.discretization import ReturnDiscretizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_returns(rng):
    """500-point series of standard-normal log-returns."""
    return rng.standard_normal(500)


@pytest.fixture
def fitted_discretizer(synthetic_returns):
    disc = ReturnDiscretizer(volatility_window=20)
    disc.fit(synthetic_returns)
    return disc


@pytest.fixture
def crash_returns():
    """Extreme negative returns guaranteed to map to A."""
    return np.array([-0.20, -0.15, -0.18, -0.25, -0.22])


@pytest.fixture
def spike_returns():
    """Extreme positive returns guaranteed to map to T."""
    return np.array([0.20, 0.15, 0.18, 0.25, 0.22])


@pytest.fixture
def calm_returns():
    """Near-zero returns guaranteed to map to C."""
    return np.array([0.0001, -0.0001, 0.0002, -0.0002, 0.0])


# ---------------------------------------------------------------------------
# Alphabet mapping tests
# ---------------------------------------------------------------------------

class TestAlphabetMapping:
    def test_A_maps_to_crash_returns(self, crash_returns):
        """Very large negative returns must map to A."""
        disc = ReturnDiscretizer(volatility_window=5)
        disc.fit(crash_returns)
        symbols = disc.transform(crash_returns)
        assert all(s == "A" for s in symbols), (
            f"Expected all A, got: {symbols}"
        )

    def test_T_maps_to_spike_returns(self, spike_returns):
        """Very large positive returns must map to T."""
        disc = ReturnDiscretizer(volatility_window=5)
        disc.fit(spike_returns)
        symbols = disc.transform(spike_returns)
        assert all(s == "T" for s in symbols), (
            f"Expected all T, got: {symbols}"
        )

    def test_C_maps_to_calm_returns(self, calm_returns):
        """Returns near zero should map to C (calm regime)."""
        # Fit on a broader distribution so thresholds are wide, then test tiny vals
        rng = np.random.default_rng(0)
        big_series = np.concatenate([
            rng.standard_normal(200) * 0.01,  # calm
            calm_returns,
        ])
        disc = ReturnDiscretizer(volatility_window=20)
        disc.fit(big_series)
        symbols = disc.transform(calm_returns)
        # At minimum majority should be C
        c_count = sum(1 for s in symbols if s == "C")
        assert c_count >= len(calm_returns) // 2, (
            f"Expected mostly C, got: {symbols}"
        )

    def test_G_maps_to_growing_returns(self):
        """Moderate positive returns in a low-vol context map to G."""
        # Construct a series where moderate positives fall in the G bucket
        rng = np.random.default_rng(7)
        base = rng.standard_normal(200) * 0.005  # very calm baseline
        moderate_pos = np.array([0.005, 0.006, 0.007, 0.005, 0.006])
        series = np.concatenate([base, moderate_pos])
        disc = ReturnDiscretizer(volatility_window=20)
        disc.fit(series)
        symbols = disc.transform(moderate_pos)
        g_count = sum(1 for s in symbols if s == "G")
        assert g_count >= len(moderate_pos) // 2, (
            f"Expected mostly G, got: {symbols}"
        )

    def test_output_alphabet_is_subset_of_ACGT(self, fitted_discretizer, synthetic_returns):
        """All symbols produced must belong to {A, C, G, T}."""
        symbols = fitted_discretizer.transform(synthetic_returns)
        allowed = {"A", "C", "G", "T"}
        invalid = [s for s in symbols if s not in allowed]
        assert len(invalid) == 0, f"Found invalid symbols: {set(invalid)}"


# ---------------------------------------------------------------------------
# Fit / transform consistency
# ---------------------------------------------------------------------------

class TestFitTransformConsistency:
    def test_transform_without_fit_raises(self, synthetic_returns):
        disc = ReturnDiscretizer(volatility_window=20)
        with pytest.raises(Exception):
            disc.transform(synthetic_returns)

    def test_fit_transform_same_length(self, fitted_discretizer, synthetic_returns):
        symbols = fitted_discretizer.transform(synthetic_returns)
        assert len(symbols) == len(synthetic_returns)

    def test_deterministic_on_same_data(self, synthetic_returns):
        disc1 = ReturnDiscretizer(volatility_window=20)
        disc1.fit(synthetic_returns)
        out1 = disc1.transform(synthetic_returns)

        disc2 = ReturnDiscretizer(volatility_window=20)
        disc2.fit(synthetic_returns)
        out2 = disc2.transform(synthetic_returns)

        assert out1 == out2

    def test_fit_transform_shorthand_equals_separate(self, synthetic_returns):
        disc = ReturnDiscretizer(volatility_window=20)
        result_ft = disc.fit_transform(synthetic_returns)

        disc2 = ReturnDiscretizer(volatility_window=20)
        disc2.fit(synthetic_returns)
        result_sep = disc2.transform(synthetic_returns)

        assert result_ft == result_sep


# ---------------------------------------------------------------------------
# Synthetic returns data (np.random.randn style)
# ---------------------------------------------------------------------------

class TestSyntheticData:
    def test_standard_normal_returns(self):
        rng = np.random.default_rng(123)
        returns = rng.standard_normal(1000)
        disc = ReturnDiscretizer(volatility_window=20)
        disc.fit(returns)
        symbols = disc.transform(returns)
        assert len(symbols) == len(returns)
        assert set(symbols).issubset({"A", "C", "G", "T"})

    def test_all_four_symbols_appear_in_long_series(self):
        rng = np.random.default_rng(99)
        returns = rng.standard_normal(5000)
        disc = ReturnDiscretizer(volatility_window=20)
        disc.fit(returns)
        symbols = disc.transform(returns)
        assert set(symbols) == {"A", "C", "G", "T"}, (
            f"Not all symbols present: {set(symbols)}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_same_value_does_not_raise(self):
        """Constant returns should not raise; all map to the same bucket."""
        returns = np.zeros(50)
        disc = ReturnDiscretizer(volatility_window=10)
        disc.fit(returns)
        symbols = disc.transform(returns)
        assert len(symbols) == len(returns)
        assert all(s in {"A", "C", "G", "T"} for s in symbols)

    def test_very_long_series(self):
        rng = np.random.default_rng(0)
        returns = rng.standard_normal(50_000)
        disc = ReturnDiscretizer(volatility_window=20)
        disc.fit(returns)
        symbols = disc.transform(returns)
        assert len(symbols) == len(returns)

    def test_minimum_length_series(self):
        """Series just longer than volatility_window should work."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02,
                            0.00, -0.03, 0.01, 0.02, -0.01,
                            0.01, -0.02, 0.03, -0.01, 0.02,
                            0.00, -0.03, 0.01, 0.02, -0.01,
                            0.01])  # 21 items, window=20
        disc = ReturnDiscretizer(volatility_window=20)
        disc.fit(returns)
        symbols = disc.transform(returns)
        assert len(symbols) == len(returns)


# ---------------------------------------------------------------------------
# Inverse transform round-trip
# ---------------------------------------------------------------------------

class TestInverseTransform:
    def test_inverse_transform_round_trip_type(self, fitted_discretizer, synthetic_returns):
        """inverse_transform must return a numpy array."""
        symbols = fitted_discretizer.transform(synthetic_returns)
        recovered = fitted_discretizer.inverse_transform(symbols)
        assert isinstance(recovered, np.ndarray)

    def test_inverse_transform_length(self, fitted_discretizer, synthetic_returns):
        symbols = fitted_discretizer.transform(synthetic_returns)
        recovered = fitted_discretizer.inverse_transform(symbols)
        assert len(recovered) == len(symbols)

    def test_inverse_transform_values_in_range(self, fitted_discretizer, synthetic_returns):
        """Recovered values should be representative float return values."""
        symbols = fitted_discretizer.transform(synthetic_returns)
        recovered = fitted_discretizer.inverse_transform(symbols)
        # Reconstructed values should be finite floats
        assert np.all(np.isfinite(recovered)), "inverse_transform produced non-finite values"

    def test_inverse_transform_ordering(self, fitted_discretizer):
        """A (crash) representative should be less than T (spike) representative."""
        recovered_A = fitted_discretizer.inverse_transform(["A"])[0]
        recovered_T = fitted_discretizer.inverse_transform(["T"])[0]
        assert recovered_A < recovered_T, (
            f"Expected A representative < T representative; "
            f"got A={recovered_A:.4f}, T={recovered_T:.4f}"
        )

    def test_inverse_transform_crash_is_negative(self, fitted_discretizer):
        recovered_A = fitted_discretizer.inverse_transform(["A"])[0]
        assert recovered_A < 0, f"A (crash) representative should be negative, got {recovered_A}"

    def test_inverse_transform_spike_is_positive(self, fitted_discretizer):
        recovered_T = fitted_discretizer.inverse_transform(["T"])[0]
        assert recovered_T > 0, f"T (spike) representative should be positive, got {recovered_T}"


# ---------------------------------------------------------------------------
# encode_to_int
# ---------------------------------------------------------------------------

class TestEncodeToInt:
    def test_encode_to_int_range(self, fitted_discretizer, synthetic_returns):
        """encode_to_int must produce values exclusively in {0, 1, 2, 3}."""
        encoded = fitted_discretizer.encode_to_int(synthetic_returns)
        valid = {0, 1, 2, 3}
        invalid = [v for v in encoded if v not in valid]
        assert len(invalid) == 0, f"Found out-of-range encoded values: {set(invalid)}"

    def test_encode_to_int_length(self, fitted_discretizer, synthetic_returns):
        encoded = fitted_discretizer.encode_to_int(synthetic_returns)
        assert len(encoded) == len(synthetic_returns)

    def test_encode_to_int_dtype_is_integer(self, fitted_discretizer, synthetic_returns):
        encoded = fitted_discretizer.encode_to_int(synthetic_returns)
        arr = np.asarray(encoded)
        assert np.issubdtype(arr.dtype, np.integer), (
            f"Expected integer dtype, got {arr.dtype}"
        )

    def test_encode_to_int_consistent_with_transform(self, fitted_discretizer, synthetic_returns):
        """encode_to_int and transform must agree on ordering."""
        symbols = fitted_discretizer.transform(synthetic_returns)
        encoded = fitted_discretizer.encode_to_int(synthetic_returns)
        mapping = fitted_discretizer.int_to_symbol  # {0: 'A', 1: 'C', 2: 'G', 3: 'T'} or similar
        for sym, code in zip(symbols, encoded):
            assert mapping[code] == sym, (
                f"Mismatch: symbol={sym} but int={code} -> {mapping[code]}"
            )

    def test_encode_to_int_all_four_values_in_long_series(self):
        rng = np.random.default_rng(17)
        returns = rng.standard_normal(5000)
        disc = ReturnDiscretizer(volatility_window=20)
        disc.fit(returns)
        encoded = disc.encode_to_int(returns)
        assert set(encoded) == {0, 1, 2, 3}, (
            f"Not all 4 integer codes present: {set(encoded)}"
        )
