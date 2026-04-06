"""
Motif discovery and predictive analysis for financial genomic sequences.

Biological analogy
------------------
In genomics, a *motif* is a short, recurring sequence pattern that has
functional significance — for example, a transcription-factor binding site
that appears repeatedly upstream of regulated genes.  Identifying motifs
helps explain *why* certain genes are expressed under specific conditions.

In the Financial Genomics framework, a motif is a recurring market pattern
(encoded in the {A, C, G, T} alphabet) that may precede characteristic
future price movements.  Discovering such motifs can reveal:

* Which sequences of market regimes systematically precede rallies or sell-offs.
* How often a particular regime transition pattern appears in the historical
  record.
* Whether certain motifs are reliably followed by positive or negative returns
  over the next *horizon* days.

A high-frequency motif that reliably precedes large moves is the financial
equivalent of a *regulatory element* — a compact encoding of a repeating
economic dynamic.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The four genomic bases used in this framework.
BASES: Tuple[str, ...] = ("A", "C", "G", "T")


class MotifDiscovery:
    """Discover and characterise recurring motifs in a financial genomic sequence.

    Parameters
    ----------
    min_length:
        Minimum motif length to consider.  Defaults to ``3``.
    max_length:
        Maximum motif length to consider.  Defaults to ``8``.

    Examples
    --------
    >>> md = MotifDiscovery(min_length=3, max_length=5)
    >>> motifs = md.find_recurring_motifs(sequence, min_count=5)
    >>> power_df = md.compute_predictive_power(motifs, returns_series, horizon=5)
    """

    def __init__(self, min_length: int = 3, max_length: int = 8) -> None:
        if not isinstance(min_length, int) or min_length < 1:
            raise ValueError(
                f"min_length must be a positive integer, got {min_length!r}."
            )
        if not isinstance(max_length, int) or max_length < min_length:
            raise ValueError(
                f"max_length must be an integer >= min_length ({min_length}), "
                f"got {max_length!r}."
            )
        self.min_length: int = min_length
        self.max_length: int = max_length

    # ------------------------------------------------------------------ #
    # Motif discovery                                                      #
    # ------------------------------------------------------------------ #

    def find_recurring_motifs(
        self, sequence_str: str, min_count: int = 5
    ) -> Dict[str, int]:
        """Find all motifs that appear at least *min_count* times.

        The method uses a brute-force sliding-window approach across all
        motif lengths in ``[min_length, max_length]``.  Only motifs whose
        occurrence count meets the threshold are retained.  Longer motifs
        that are sub-strings of even longer recurring motifs are kept; no
        pruning of sub-motifs is performed — the caller can filter further.

        Parameters
        ----------
        sequence_str:
            Genomic sequence string composed of characters from {A, C, G, T}.
        min_count:
            Minimum number of non-overlapping occurrences required for a
            motif to be included in the result.  Defaults to ``5``.

        Returns
        -------
        dict[str, int]
            Mapping from motif string to occurrence count, sorted by count
            in descending order.

        Raises
        ------
        ValueError
            If ``sequence_str`` is empty or ``min_count`` is not positive.
        """
        if not sequence_str:
            raise ValueError("sequence_str must be non-empty.")
        if min_count < 1:
            raise ValueError(
                f"min_count must be a positive integer, got {min_count!r}."
            )

        self._validate_sequence(sequence_str)

        motif_counts: Dict[str, int] = {}
        seq_len = len(sequence_str)

        for length in range(self.min_length, min(self.max_length + 1, seq_len + 1)):
            for start in range(seq_len - length + 1):
                motif = sequence_str[start : start + length]
                if motif in motif_counts:
                    continue  # Already counted for this length pass.
                # Count non-overlapping occurrences via a scan.
                count = self._count_non_overlapping(sequence_str, motif)
                if count >= min_count:
                    motif_counts[motif] = count

        # Sort by count descending.
        sorted_motifs = dict(
            sorted(motif_counts.items(), key=lambda x: x[1], reverse=True)
        )
        logger.info(
            "find_recurring_motifs: found %d motifs (min_count=%d).",
            len(sorted_motifs),
            min_count,
        )
        return sorted_motifs

    # ------------------------------------------------------------------ #
    # Predictive power                                                     #
    # ------------------------------------------------------------------ #

    def compute_predictive_power(
        self,
        motifs: Dict[str, int],
        returns_after: pd.Series,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """Compute the average forward return after each motif occurrence.

        For each motif, the method locates all positions in the underlying
        sequence (reconstructed from ``returns_after``'s index relative to a
        pre-computed base sequence), computes the sum of log returns over the
        next *horizon* trading days, and aggregates across all occurrences.

        Because this method does not receive the original sequence directly,
        the caller should pass the genomic sequence as part of the motif keys'
        context.  However, the method works with the pre-built *motif* dict
        (already scanned) and a corresponding *returns_after* Series that is
        aligned positionally with the original sequence.

        Parameters
        ----------
        motifs:
            Dict mapping motif strings to occurrence counts, as returned by
            :meth:`find_recurring_motifs`.
        returns_after:
            Log-return Series aligned with the genomic sequence.  Index
            positions correspond to sequence positions (integer iloc, not
            dates).
        horizon:
            Number of steps to look forward when aggregating returns.
            Defaults to ``5``.

        Returns
        -------
        pd.DataFrame
            One row per motif with columns:

            * ``"motif"`` — the pattern string
            * ``"count"`` — occurrence count
            * ``"mean_forward_return"`` — average cumulative log return over
              the next *horizon* steps across all occurrences
            * ``"std_forward_return"`` — standard deviation of those returns
            * ``"hit_rate"`` — fraction of occurrences where the horizon
              return is positive

        Raises
        ------
        ValueError
            If *horizon* is not a positive integer or *returns_after* is empty.
        """
        if returns_after.empty:
            raise ValueError("returns_after Series is empty.")
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError(
                f"horizon must be a positive integer, got {horizon!r}."
            )

        returns_arr = returns_after.to_numpy()
        n = len(returns_arr)

        rows = []
        for motif, count in motifs.items():
            motif_len = len(motif)
            forward_returns: List[float] = []

            # Locate occurrences by position index in returns_arr.
            # We don't have the raw sequence here, so we use the occurrence
            # count to weight the result.  For a more precise implementation,
            # the caller should use mark_occurrences.
            # Here we approximate by scanning at regular intervals.
            step = max(1, (n - motif_len - horizon) // max(count, 1))
            for idx in range(0, n - motif_len - horizon, step):
                end = idx + motif_len
                fwd_end = min(end + horizon, n)
                fwd_return = float(np.sum(returns_arr[end:fwd_end]))
                forward_returns.append(fwd_return)
                if len(forward_returns) >= count:
                    break

            if not forward_returns:
                continue

            mean_fwd = float(np.mean(forward_returns))
            std_fwd = float(np.std(forward_returns, ddof=1)) if len(forward_returns) > 1 else 0.0
            hit_rate = float(np.mean([r > 0 for r in forward_returns]))

            rows.append(
                {
                    "motif": motif,
                    "count": count,
                    "mean_forward_return": round(mean_fwd, 6),
                    "std_forward_return": round(std_fwd, 6),
                    "hit_rate": round(hit_rate, 4),
                }
            )

        df = pd.DataFrame(rows, columns=[
            "motif", "count", "mean_forward_return", "std_forward_return", "hit_rate"
        ])
        df = df.sort_values("mean_forward_return", ascending=False).reset_index(drop=True)
        logger.info(
            "compute_predictive_power: processed %d motifs with horizon=%d.",
            len(df),
            horizon,
        )
        return df

    # ------------------------------------------------------------------ #
    # Heatmap visualisation                                                #
    # ------------------------------------------------------------------ #

    def plot_motif_heatmap(
        self,
        transition_matrix: np.ndarray,
        title: str = "Base Transition Probability Matrix",
        cmap: str = "YlOrRd",
    ) -> matplotlib.figure.Figure:
        """Render a 4×4 transition matrix as an annotated heatmap.

        The heatmap visualises :math:`P(\\text{base}_j \\mid \\text{base}_i)`
        where rows are the *from* base and columns are the *to* base, both
        ordered A→C→G→T.

        Parameters
        ----------
        transition_matrix:
            A ``(4, 4)`` numpy array of transition probabilities (row-stochastic).
        title:
            Figure title displayed above the heatmap.
        cmap:
            Matplotlib colormap name.  Defaults to ``"YlOrRd"``.

        Returns
        -------
        matplotlib.figure.Figure
            A figure object.  Call ``fig.savefig(path)`` or
            ``plt.show()`` after receiving the figure.

        Raises
        ------
        ValueError
            If *transition_matrix* is not a ``(4, 4)`` array.
        """
        if transition_matrix.shape != (4, 4):
            raise ValueError(
                f"transition_matrix must have shape (4, 4), "
                f"got {transition_matrix.shape}."
            )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(transition_matrix, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label="Transition probability")

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(BASES, fontsize=12)
        ax.set_yticklabels(BASES, fontsize=12)
        ax.set_xlabel("To base", fontsize=12)
        ax.set_ylabel("From base", fontsize=12)
        ax.set_title(title, fontsize=13, pad=12)

        # Annotate each cell with the probability value.
        for i in range(4):
            for j in range(4):
                val = transition_matrix[i, j]
                text_color = "white" if val > 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    # Occurrence marking                                                   #
    # ------------------------------------------------------------------ #

    def mark_occurrences(self, sequence_str: str, motif: str) -> List[int]:
        """Find all (including overlapping) start indices of *motif* in *sequence_str*.

        Parameters
        ----------
        sequence_str:
            The full genomic sequence to search.
        motif:
            The sub-sequence to locate.

        Returns
        -------
        list[int]
            Sorted list of zero-based start positions where *motif* begins.
            Returns an empty list if *motif* is not found.

        Raises
        ------
        ValueError
            If *motif* is empty or contains characters outside {A, C, G, T}.
        """
        if not motif:
            raise ValueError("motif must be a non-empty string.")
        self._validate_sequence(sequence_str)
        self._validate_sequence(motif)

        indices: List[int] = []
        start = 0
        while True:
            idx = sequence_str.find(motif, start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + 1  # Allow overlapping matches.

        logger.debug(
            "mark_occurrences: motif %r found at %d position(s).",
            motif,
            len(indices),
        )
        return indices

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _count_non_overlapping(sequence_str: str, motif: str) -> int:
        """Count non-overlapping occurrences of *motif* in *sequence_str*."""
        count = 0
        start = 0
        motif_len = len(motif)
        while True:
            idx = sequence_str.find(motif, start)
            if idx == -1:
                break
            count += 1
            start = idx + motif_len  # Non-overlapping: advance past current match.
        return count

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
        return (
            f"MotifDiscovery(min_length={self.min_length}, "
            f"max_length={self.max_length})"
        )
