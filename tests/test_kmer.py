"""
Tests for src/features/kmer_analysis.py

The k-mer analysis module treats discretized return sequences (over ACGT)
analogously to DNA sequences, extracting frequency patterns, transition
matrices, enrichment p-values, and top motifs.
"""

import math

import numpy as np
import pytest

from src.features.kmer_analysis import KmerAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KNOWN_SEQ = "ACGTACGT"   # 8 chars; 7 2-mers, 6 3-mers, etc.
LONG_RANDOM_SEQ = "".join(
    np.random.default_rng(42).choice(list("ACGT"), size=500).tolist()
)


@pytest.fixture
def analyzer_k1():
    return KmerAnalyzer(k=1)


@pytest.fixture
def analyzer_k2():
    return KmerAnalyzer(k=2)


@pytest.fixture
def analyzer_k3():
    return KmerAnalyzer(k=3)


@pytest.fixture
def analyzer_k4():
    return KmerAnalyzer(k=4)


@pytest.fixture
def long_seq():
    return LONG_RANDOM_SEQ


# ---------------------------------------------------------------------------
# K-mer extraction count
# ---------------------------------------------------------------------------

class TestKmerCount:
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_kmer_count_formula(self, k):
        """Number of k-mers must equal len(seq) - k + 1."""
        seq = KNOWN_SEQ
        analyzer = KmerAnalyzer(k=k)
        kmers = analyzer.extract_kmers(seq)
        expected = len(seq) - k + 1
        assert len(kmers) == expected, (
            f"k={k}: expected {expected} k-mers, got {len(kmers)}"
        )

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_kmer_count_long_sequence(self, long_seq, k):
        analyzer = KmerAnalyzer(k=k)
        kmers = analyzer.extract_kmers(long_seq)
        expected = len(long_seq) - k + 1
        assert len(kmers) == expected

    def test_kmer_length_each_is_k(self):
        for k in [1, 2, 3, 4]:
            analyzer = KmerAnalyzer(k=k)
            kmers = analyzer.extract_kmers(KNOWN_SEQ)
            for kmer in kmers:
                assert len(kmer) == k, f"k={k}: found kmer of length {len(kmer)}: '{kmer}'"

    def test_short_sequence_returns_empty_when_k_exceeds_length(self):
        analyzer = KmerAnalyzer(k=5)
        kmers = analyzer.extract_kmers("ACGT")  # length 4 < k=5
        assert kmers == [] or len(kmers) == 0


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:
    def test_transition_matrix_shape_is_4x4(self, analyzer_k1, long_seq):
        tm = analyzer_k1.transition_matrix(long_seq)
        assert tm.shape == (4, 4), f"Expected (4,4), got {tm.shape}"

    def test_transition_matrix_rows_sum_to_1(self, analyzer_k1, long_seq):
        tm = analyzer_k1.transition_matrix(long_seq)
        row_sums = tm.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(4), atol=1e-6,
            err_msg=f"Row sums not all 1.0: {row_sums}"
        )

    def test_transition_matrix_non_negative(self, analyzer_k1, long_seq):
        tm = analyzer_k1.transition_matrix(long_seq)
        assert np.all(tm >= 0), "Transition matrix contains negative values"

    def test_transition_matrix_values_in_0_1(self, analyzer_k1, long_seq):
        tm = analyzer_k1.transition_matrix(long_seq)
        assert np.all(tm <= 1.0 + 1e-9), "Transition matrix values exceed 1.0"

    def test_transition_matrix_known_sequence(self):
        """For 'AAAA', all transitions should be A->A with prob 1."""
        seq = "AAAA"
        analyzer = KmerAnalyzer(k=1)
        tm = analyzer.transition_matrix(seq)
        # Row for A (index 0) should be [1, 0, 0, 0]
        a_idx = analyzer.alphabet_index["A"]
        np.testing.assert_allclose(
            tm[a_idx, a_idx], 1.0, atol=1e-6,
            err_msg="A->A probability should be 1.0 for sequence 'AAAA'"
        )

    def test_transition_matrix_uniform_when_all_present(self, long_seq):
        """Sufficiently random long sequence should have no all-zero rows."""
        analyzer = KmerAnalyzer(k=1)
        tm = analyzer.transition_matrix(long_seq)
        for i in range(4):
            assert tm[i].sum() > 0, f"Row {i} is all zeros (no transitions observed)"


# ---------------------------------------------------------------------------
# Top k-mers
# ---------------------------------------------------------------------------

class TestTopKmers:
    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_top_kmers_returns_n_items(self, long_seq, n):
        analyzer = KmerAnalyzer(k=3)
        top = analyzer.top_kmers(long_seq, n=n)
        assert len(top) == n, f"Expected {n} items, got {len(top)}"

    def test_top_kmers_sorted_descending(self, long_seq):
        analyzer = KmerAnalyzer(k=3)
        top = analyzer.top_kmers(long_seq, n=10)
        counts = [count for _, count in top]
        assert counts == sorted(counts, reverse=True), (
            "top_kmers results are not sorted in descending order"
        )

    def test_top_kmers_returns_tuples(self, long_seq):
        analyzer = KmerAnalyzer(k=2)
        top = analyzer.top_kmers(long_seq, n=5)
        for item in top:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"Expected (kmer, count) tuple, got {item}"
            )

    def test_top_kmers_kmer_is_string_of_length_k(self, long_seq):
        k = 3
        analyzer = KmerAnalyzer(k=k)
        top = analyzer.top_kmers(long_seq, n=5)
        for kmer, _ in top:
            assert isinstance(kmer, str)
            assert len(kmer) == k

    def test_top_1_is_most_frequent(self, long_seq):
        analyzer = KmerAnalyzer(k=2)
        top1 = analyzer.top_kmers(long_seq, n=1)[0]
        top5 = analyzer.top_kmers(long_seq, n=5)
        # top1 must equal first element of top5
        assert top1 == top5[0]

    def test_top_n_larger_than_vocab_returns_all(self):
        """If n > total unique k-mers, return all unique ones without error."""
        seq = "ACGT"  # only one 4-mer: 'ACGT'
        analyzer = KmerAnalyzer(k=4)
        top = analyzer.top_kmers(seq, n=1000)
        assert len(top) >= 1


# ---------------------------------------------------------------------------
# Frequency table
# ---------------------------------------------------------------------------

class TestFrequencyTable:
    def test_frequency_table_known_sequence(self):
        """ACGTACGT has 2 occurrences each of AC, CG, GT, TA (for k=2)."""
        seq = "ACGTACGT"
        analyzer = KmerAnalyzer(k=2)
        freq = analyzer.frequency_table(seq)

        expected_kmers = ["AC", "CG", "GT", "TA"]
        for kmer in expected_kmers:
            assert kmer in freq, f"Expected k-mer '{kmer}' missing from frequency table"
            assert freq[kmer] == 2, (
                f"Expected count 2 for '{kmer}', got {freq[kmer]}"
            )

    def test_frequency_table_k1_known(self):
        """ACGTACGT has 2 of each A, C, G, T for k=1."""
        seq = "ACGTACGT"
        analyzer = KmerAnalyzer(k=1)
        freq = analyzer.frequency_table(seq)
        for base in "ACGT":
            assert freq.get(base, 0) == 2, (
                f"Expected 2 occurrences of '{base}', got {freq.get(base, 0)}"
            )

    def test_frequency_table_sum_equals_kmer_count(self, long_seq):
        for k in [1, 2, 3]:
            analyzer = KmerAnalyzer(k=k)
            freq = analyzer.frequency_table(long_seq)
            total = sum(freq.values())
            expected = len(long_seq) - k + 1
            assert total == expected, (
                f"k={k}: sum of frequencies {total} != expected {expected}"
            )

    def test_frequency_table_returns_dict(self, long_seq):
        analyzer = KmerAnalyzer(k=2)
        freq = analyzer.frequency_table(long_seq)
        assert isinstance(freq, dict)

    def test_frequency_table_all_keys_are_length_k(self, long_seq):
        k = 3
        analyzer = KmerAnalyzer(k=k)
        freq = analyzer.frequency_table(long_seq)
        for kmer in freq:
            assert len(kmer) == k


# ---------------------------------------------------------------------------
# Enrichment test
# ---------------------------------------------------------------------------

class TestEnrichmentTest:
    def test_enrichment_returns_p_values_in_0_1(self, long_seq):
        analyzer = KmerAnalyzer(k=2)
        p_values = analyzer.enrichment_test(long_seq, n_permutations=100)
        for kmer, pval in p_values.items():
            assert 0.0 <= pval <= 1.0, (
                f"p-value for '{kmer}' out of range [0,1]: {pval}"
            )

    def test_enrichment_returns_dict(self, long_seq):
        analyzer = KmerAnalyzer(k=2)
        p_values = analyzer.enrichment_test(long_seq, n_permutations=100)
        assert isinstance(p_values, dict)

    def test_enrichment_keys_are_valid_kmers(self, long_seq):
        k = 2
        analyzer = KmerAnalyzer(k=k)
        p_values = analyzer.enrichment_test(long_seq, n_permutations=100)
        for kmer in p_values:
            assert len(kmer) == k
            assert all(c in "ACGT" for c in kmer)

    def test_enrichment_with_very_short_sequence(self):
        seq = "ACGTACGT"
        analyzer = KmerAnalyzer(k=2)
        p_values = analyzer.enrichment_test(seq, n_permutations=50)
        assert isinstance(p_values, dict)
        for pval in p_values.values():
            assert 0.0 <= pval <= 1.0

    def test_enrichment_not_all_zero(self, long_seq):
        """p-values should not be identically 0 for all k-mers."""
        analyzer = KmerAnalyzer(k=2)
        p_values = analyzer.enrichment_test(long_seq, n_permutations=200)
        assert any(p > 0 for p in p_values.values()), (
            "All enrichment p-values are 0, which is unexpected"
        )


# ---------------------------------------------------------------------------
# K values 1, 2, 3, 4 all work
# ---------------------------------------------------------------------------

class TestKValues:
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_extract_kmers_works(self, k, long_seq):
        analyzer = KmerAnalyzer(k=k)
        kmers = analyzer.extract_kmers(long_seq)
        assert len(kmers) == len(long_seq) - k + 1

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_frequency_table_works(self, k, long_seq):
        analyzer = KmerAnalyzer(k=k)
        freq = analyzer.frequency_table(long_seq)
        assert isinstance(freq, dict)
        assert len(freq) > 0

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_top_kmers_works(self, k, long_seq):
        analyzer = KmerAnalyzer(k=k)
        top = analyzer.top_kmers(long_seq, n=5)
        assert len(top) <= 5  # may be fewer if vocab is small

    @pytest.mark.parametrize("k", [1, 2])
    def test_transition_matrix_works(self, k, long_seq):
        analyzer = KmerAnalyzer(k=k)
        tm = analyzer.transition_matrix(long_seq)
        assert tm.shape == (4, 4)


# ---------------------------------------------------------------------------
# Very short sequences
# ---------------------------------------------------------------------------

class TestVeryShortSequences:
    def test_single_char_k1(self):
        seq = "A"
        analyzer = KmerAnalyzer(k=1)
        kmers = analyzer.extract_kmers(seq)
        assert kmers == ["A"]

    def test_two_chars_k2(self):
        seq = "AC"
        analyzer = KmerAnalyzer(k=2)
        kmers = analyzer.extract_kmers(seq)
        assert kmers == ["AC"]

    def test_k_equals_length(self):
        seq = "ACGT"
        analyzer = KmerAnalyzer(k=4)
        kmers = analyzer.extract_kmers(seq)
        assert kmers == ["ACGT"]

    def test_frequency_table_single_char(self):
        seq = "AAAA"
        analyzer = KmerAnalyzer(k=1)
        freq = analyzer.frequency_table(seq)
        assert freq.get("A", 0) == 4

    def test_transition_matrix_two_symbol_sequence(self):
        seq = "ACAC"
        analyzer = KmerAnalyzer(k=1)
        tm = analyzer.transition_matrix(seq)
        assert tm.shape == (4, 4)
        # A->C transition should be present
        a_idx = analyzer.alphabet_index["A"]
        c_idx = analyzer.alphabet_index["C"]
        assert tm[a_idx, c_idx] > 0
