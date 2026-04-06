"""
K-mer analysis of financial genomic sequences.

Biological analogy
------------------
In genomics, a *k-mer* is a contiguous sub-sequence of length *k*.  K-mer
analysis is used to:

* Measure how often short patterns appear in a genome (frequency tables).
* Model which base tends to follow a given base (transition matrices).
* Identify over-represented patterns that may have functional significance
  (enrichment tests).
* Find "stop codons" — triplets that strongly predict what comes next, much
  like the biological stop codons that terminate protein translation.

Applied to financial sequences the same tools reveal:

* Which three-day patterns (e.g. "AGT", "CCC") are most common.
* How likely the market is to transition from a crash day (A) to a calm day (C).
* Which patterns appear far more often than chance would predict.
* Which k-mers act as strong predictive signals for the immediately following
  market regime.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The four genomic bases used in this framework.
BASES: Tuple[str, ...] = ("A", "C", "G", "T")
BASE_INDEX: Dict[str, int] = {b: i for i, b in enumerate(BASES)}


class KmerAnalyzer:
    """Analyse k-mer statistics in a financial genomic sequence.

    Parameters
    ----------
    k:
        Length of k-mer sub-sequences.  Defaults to ``3`` (equivalent to a
        biological codon).

    Examples
    --------
    >>> analyzer = KmerAnalyzer(k=3)
    >>> kmers = analyzer.extract_kmers("ACGTACGT")
    >>> freq = analyzer.frequency_table("ACGTACGTACGT")
    >>> tm = analyzer.transition_matrix("ACGTACGTCCC")
    """

    def __init__(self, k: int = 3) -> None:
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}.")
        self.k: int = k

    # ------------------------------------------------------------------ #
    # K-mer extraction                                                     #
    # ------------------------------------------------------------------ #

    def extract_kmers(self, sequence_str: str) -> List[str]:
        """Extract all k-mers from a sequence using a sliding window.

        The window advances one character at a time, so consecutive k-mers
        overlap by ``k - 1`` characters — the same convention used in
        genome assembly.

        Parameters
        ----------
        sequence_str:
            Genomic sequence string composed of characters from {A, C, G, T}.

        Returns
        -------
        list[str]
            List of all k-mers in order.  Length is
            ``max(0, len(sequence_str) - k + 1)``.

        Raises
        ------
        ValueError
            If ``sequence_str`` contains characters outside {A, C, G, T}.
        """
        self._validate_sequence(sequence_str)
        if len(sequence_str) < self.k:
            logger.warning(
                "Sequence length %d is shorter than k=%d; returning empty list.",
                len(sequence_str),
                self.k,
            )
            return []

        kmers = [
            sequence_str[i : i + self.k]
            for i in range(len(sequence_str) - self.k + 1)
        ]
        logger.debug("Extracted %d k-mers (k=%d).", len(kmers), self.k)
        return kmers

    # ------------------------------------------------------------------ #
    # Frequency table                                                      #
    # ------------------------------------------------------------------ #

    def frequency_table(self, sequence_str: str) -> Counter:
        """Count the occurrence of every k-mer in the sequence.

        Parameters
        ----------
        sequence_str:
            Genomic sequence string.

        Returns
        -------
        collections.Counter
            Mapping from k-mer string to occurrence count, sorted by
            frequency (descending) when iterated.
        """
        kmers = self.extract_kmers(sequence_str)
        freq = Counter(kmers)
        logger.debug(
            "Frequency table: %d unique k-mers from %d total k-mers.",
            len(freq),
            len(kmers),
        )
        return freq

    # ------------------------------------------------------------------ #
    # Transition matrix                                                    #
    # ------------------------------------------------------------------ #

    def transition_matrix(self, sequence_str: str) -> np.ndarray:
        """Compute the 4×4 first-order transition probability matrix.

        Entry ``M[i, j]`` is the empirical probability
        :math:`P(\\text{base}_j \\mid \\text{base}_i)`, estimated by counting
        consecutive bi-grams and normalising each row to sum to 1.

        Rows and columns are indexed in alphabetical order: A=0, C=1, G=2, T=3.

        Parameters
        ----------
        sequence_str:
            Genomic sequence string of length ≥ 2.

        Returns
        -------
        np.ndarray
            Shape ``(4, 4)`` float64 array.  Rows with zero counts (a base
            never appears as the *from* state) are set to uniform probability
            ``0.25`` to avoid division-by-zero and maintain a valid stochastic
            matrix.

        Raises
        ------
        ValueError
            If the sequence is shorter than 2 characters.
        """
        self._validate_sequence(sequence_str)
        if len(sequence_str) < 2:
            raise ValueError(
                "sequence_str must have at least 2 characters to compute "
                "a transition matrix."
            )

        counts = np.zeros((4, 4), dtype=np.float64)
        for i in range(len(sequence_str) - 1):
            from_base = sequence_str[i]
            to_base = sequence_str[i + 1]
            if from_base in BASE_INDEX and to_base in BASE_INDEX:
                counts[BASE_INDEX[from_base], BASE_INDEX[to_base]] += 1

        # Row-normalise; rows with zero counts get uniform distribution.
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        matrix = counts / row_sums
        # Fix rows that were zero: set to uniform 0.25
        zero_rows = (counts.sum(axis=1) == 0)
        matrix[zero_rows] = 0.25

        return matrix

    # ------------------------------------------------------------------ #
    # Enrichment test                                                      #
    # ------------------------------------------------------------------ #

    def enrichment_test(
        self,
        observed_freq: Counter,
        total_kmers: int,
        n_permutations: int = 1000,
    ) -> Dict[str, Tuple[int, float]]:
        """Permutation test for over-represented k-mers.

        Under the null hypothesis the bases are drawn i.i.d. from the
        marginal frequencies observed in the sequence.  We permute the
        sequence ``n_permutations`` times, compute k-mer counts for each
        permutation, and estimate the p-value as the fraction of permutations
        in which the null count equals or exceeds the observed count.

        Parameters
        ----------
        observed_freq:
            Counter of observed k-mer frequencies (from :meth:`frequency_table`).
        total_kmers:
            Total number of k-mers extracted (``sum(observed_freq.values())``
            if all k-mers are included).
        n_permutations:
            Number of random permutations used to build the null distribution.
            Higher values yield more accurate p-values.  Defaults to ``1000``.

        Returns
        -------
        dict[str, tuple[int, float]]
            Mapping from k-mer to ``(observed_count, p_value)``.  p-values
            are not multiple-testing corrected; apply Bonferroni or BH
            correction externally if desired.

        Raises
        ------
        ValueError
            If ``total_kmers`` ≤ 0.
        """
        if total_kmers <= 0:
            raise ValueError(
                f"total_kmers must be positive, got {total_kmers}."
            )

        if not observed_freq:
            return {}

        # Reconstruct a representative sequence for permutation.
        # We reconstruct the bases by expanding the frequency table back into
        # a flat list and shuffling it.
        base_pool: List[str] = []
        for kmer, count in observed_freq.items():
            base_pool.extend(list(kmer[0]) * count)
        # Ensure we cover the last (k-1) bases.
        for kmer in list(observed_freq.keys())[:1]:
            base_pool.extend(list(kmer[1:]))

        seq_len = total_kmers + self.k - 1
        # Pad or truncate to the correct length.
        if len(base_pool) < seq_len:
            base_pool.extend(
                random.choices(BASES, k=seq_len - len(base_pool))
            )
        base_pool = base_pool[:seq_len]

        # Build null distribution via permutation.
        null_counts: Dict[str, List[int]] = {km: [] for km in observed_freq}

        for _ in range(n_permutations):
            perm = base_pool[:]
            random.shuffle(perm)
            perm_seq = "".join(perm)
            perm_freq = Counter(
                perm_seq[i : i + self.k]
                for i in range(len(perm_seq) - self.k + 1)
            )
            for kmer in observed_freq:
                null_counts[kmer].append(perm_freq.get(kmer, 0))

        # Compute p-values.
        results: Dict[str, Tuple[int, float]] = {}
        for kmer, obs_count in observed_freq.items():
            null_dist = null_counts[kmer]
            p_value = float(
                sum(1 for nc in null_dist if nc >= obs_count) / n_permutations
            )
            results[kmer] = (obs_count, p_value)

        logger.info(
            "Enrichment test: tested %d k-mers with %d permutations.",
            len(results),
            n_permutations,
        )
        return results

    # ------------------------------------------------------------------ #
    # Top k-mers                                                           #
    # ------------------------------------------------------------------ #

    def top_kmers(self, sequence_str: str, n: int = 10) -> List[Tuple[str, int]]:
        """Return the top *n* most frequent k-mers with their counts.

        Parameters
        ----------
        sequence_str:
            Genomic sequence string.
        n:
            Number of top k-mers to return.  Defaults to ``10``.

        Returns
        -------
        list[tuple[str, int]]
            List of ``(kmer, count)`` pairs sorted by descending frequency.
            If fewer than *n* distinct k-mers exist the full list is returned.
        """
        freq = self.frequency_table(sequence_str)
        top = freq.most_common(n)
        logger.debug("Top %d k-mers: %s", n, top)
        return top

    # ------------------------------------------------------------------ #
    # Stop codons                                                          #
    # ------------------------------------------------------------------ #

    def get_stop_codons(
        self, sequence_str: str, threshold: float = 0.8
    ) -> List[Dict]:
        """Identify k-mers that strongly predict the immediately following base.

        In molecular biology a *stop codon* is a three-nucleotide sequence
        that signals the end of a protein-coding region.  In this framework
        we define a "financial stop codon" as a k-mer after which the next
        base can be predicted with high confidence — i.e., the conditional
        distribution :math:`P(\\text{next} \\mid \\text{kmer})` is highly
        concentrated (low entropy, high max-probability).

        Parameters
        ----------
        sequence_str:
            Genomic sequence string.
        threshold:
            Minimum conditional probability for the most likely next base to
            qualify as a stop codon.  Defaults to ``0.8``.

        Returns
        -------
        list[dict]
            Each entry contains:

            * ``"kmer"`` — the k-mer string
            * ``"count"`` — number of occurrences
            * ``"next_base"`` — the most likely following base
            * ``"probability"`` — conditional probability of that base
            * ``"entropy"`` — Shannon entropy of the conditional distribution

        Raises
        ------
        ValueError
            If the sequence is too short to extract any k-mer plus one more base.
        """
        self._validate_sequence(sequence_str)
        if len(sequence_str) < self.k + 1:
            raise ValueError(
                f"sequence_str must have at least k+1={self.k + 1} characters "
                "to compute stop codons."
            )

        # Count occurrences of each (kmer, next_base) pair.
        kmer_next: Dict[str, Counter] = {}
        for i in range(len(sequence_str) - self.k):
            kmer = sequence_str[i : i + self.k]
            next_base = sequence_str[i + self.k]
            if kmer not in kmer_next:
                kmer_next[kmer] = Counter()
            kmer_next[kmer][next_base] += 1

        stop_codons = []
        for kmer, next_counts in kmer_next.items():
            total = sum(next_counts.values())
            most_likely_base, most_likely_count = next_counts.most_common(1)[0]
            prob = most_likely_count / total

            # Shannon entropy of the conditional distribution.
            probs = np.array([next_counts.get(b, 0) / total for b in BASES])
            # Avoid log(0).
            probs_safe = np.where(probs > 0, probs, 1e-12)
            entropy = float(-np.sum(probs * np.log2(probs_safe)))

            if prob >= threshold:
                stop_codons.append(
                    {
                        "kmer": kmer,
                        "count": total,
                        "next_base": most_likely_base,
                        "probability": round(prob, 4),
                        "entropy": round(entropy, 4),
                    }
                )

        # Sort by probability descending.
        stop_codons.sort(key=lambda x: x["probability"], reverse=True)
        logger.info(
            "Found %d stop codons at threshold=%.2f.", len(stop_codons), threshold
        )
        return stop_codons

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_sequence(sequence_str: str) -> None:
        """Raise ValueError if the sequence contains invalid characters."""
        if not sequence_str:
            raise ValueError("sequence_str must be a non-empty string.")
        invalid = set(sequence_str) - set(BASES)
        if invalid:
            raise ValueError(
                f"sequence_str contains invalid characters: {invalid}. "
                f"Only {{A, C, G, T}} are permitted."
            )

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"KmerAnalyzer(k={self.k})"
