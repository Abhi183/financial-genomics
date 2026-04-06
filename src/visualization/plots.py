"""Publication-quality visualizations for the Financial Genomics project.

All public functions return a ``matplotlib.figure.Figure`` and optionally
save the figure to *save_path* as a PDF (or other format inferred from the
extension).  Figures are styled with a clean, editorial aesthetic suitable
for academic papers.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_GENOMIC_PALETTE = {
    "A": "#e74c3c",  # red
    "C": "#3498db",  # blue
    "G": "#2ecc71",  # green
    "T": "#f39c12",  # amber
}

_EQUITY_PALETTE = [
    "#2563eb",  # genomic LSTM
    "#16a34a",  # buy and hold
    "#dc2626",  # GARCH
    "#9333ea",  # ARIMA
    "#ea580c",  # SAX
    "#0891b2",  # matrix profile
]

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }
)


def _save(fig: Figure, save_path: Optional[str]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)


# ---------------------------------------------------------------------------
# Figure 1 — Discretisation pipeline
# ---------------------------------------------------------------------------


def plot_discretization_pipeline(
    prices: pd.Series,
    returns: pd.Series,
    volatility: pd.Series,
    sequence: str,
    save_path: Optional[str] = None,
) -> Figure:
    """Four-panel plot showing the discretisation pipeline.

    Panels (top to bottom):
    1. Price series
    2. Daily returns (bar chart)
    3. Rolling volatility
    4. Encoded genomic sequence as colour blocks

    Parameters
    ----------
    prices:
        Close price series with DatetimeIndex.
    returns:
        Daily returns aligned with prices.
    volatility:
        Rolling realised volatility series aligned with prices.
    sequence:
        Genomic sequence string (e.g. ``"ACGTACGT..."``).  Length should
        match the price series length.
    save_path:
        If given, the figure is saved to this path.

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Financial Genomics — Discretisation Pipeline", fontweight="bold", y=0.98)

    common_index = prices.index

    # Panel 1: price series
    ax0 = axes[0]
    ax0.plot(common_index, prices.values, linewidth=1.0, color="#1e293b")
    ax0.fill_between(common_index, prices.values, prices.min(), alpha=0.08, color="#1e293b")
    ax0.set_ylabel("Price ($)")
    ax0.set_title("Close Price")
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Panel 2: returns
    ax1 = axes[1]
    colors = ["#16a34a" if r >= 0 else "#dc2626" for r in returns.values]
    ax1.bar(common_index, returns.values, color=colors, width=1, alpha=0.8)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Return")
    ax1.set_title("Daily Returns")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

    # Panel 3: volatility
    ax2 = axes[2]
    ax2.plot(common_index, volatility.values, linewidth=1.0, color="#7c3aed")
    ax2.fill_between(common_index, volatility.values, alpha=0.15, color="#7c3aed")
    ax2.set_ylabel("Vol (ann.)")
    ax2.set_title("Rolling Volatility")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Panel 4: genomic sequence as colour blocks
    ax3 = axes[3]
    n = min(len(sequence), len(common_index))
    for i, char in enumerate(sequence[:n]):
        color = _GENOMIC_PALETTE.get(char.upper(), "#94a3b8")
        ax3.axvspan(i, i + 1, facecolor=color, alpha=0.9)
    ax3.set_xlim(0, n)
    ax3.set_yticks([])
    ax3.set_xlabel("Bar index")
    ax3.set_title("Genomic Sequence Encoding")
    legend_patches = [
        mpatches.Patch(color=v, label=f"{k} = {_label_for(k)}")
        for k, v in _GENOMIC_PALETTE.items()
    ]
    ax3.legend(handles=legend_patches, loc="upper right", ncol=4, fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, save_path)
    return fig


def _label_for(letter: str) -> str:
    mapping = {"A": "Spike", "C": "Calm", "G": "Drop", "T": "Crash"}
    return mapping.get(letter, letter)


# ---------------------------------------------------------------------------
# Figure 2 — Example genomic sequence
# ---------------------------------------------------------------------------


def plot_example_sequence(
    sequence_str: str,
    dates: Sequence,
    annotations: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Genomic sequence as coloured blocks with optional event annotations.

    Parameters
    ----------
    sequence_str:
        Genomic sequence (e.g. ``"AACGTTTCGA..."``).
    dates:
        Date labels for each position (same length as ``sequence_str``).
    annotations:
        Mapping of date label (as string) to annotation text,
        e.g. ``{"2020-03-16": "COVID crash"}``.
    save_path:
        Output path for saving.

    Returns
    -------
    Figure
    """
    n = len(sequence_str)
    dates = list(dates)[:n]
    annotations = annotations or {}

    fig, ax = plt.subplots(figsize=(max(12, n * 0.18), 3))

    for i, char in enumerate(sequence_str):
        color = _GENOMIC_PALETTE.get(char.upper(), "#94a3b8")
        rect = mpatches.FancyBboxPatch(
            (i, 0.1), 1, 0.8,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, char.upper(), ha="center", va="center",
                fontsize=max(5, min(9, 160 // n)), color="white", fontweight="bold")

    # Annotations
    for ann_date, ann_text in annotations.items():
        try:
            idx = [str(d)[:10] for d in dates].index(str(ann_date)[:10])
        except ValueError:
            continue
        ax.annotate(
            ann_text,
            xy=(idx + 0.5, 0.9), xytext=(idx + 0.5, 1.35),
            ha="center", fontsize=7, color="#1e293b",
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=0.8),
        )

    # X-axis: sample dates
    step = max(1, n // 12)
    tick_positions = list(range(0, n, step))
    tick_labels = [str(dates[i])[:10] if i < len(dates) else "" for i in tick_positions]
    ax.set_xticks([p + 0.5 for p in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([])
    ax.set_title("Genomic Sequence Encoding of Market Returns", fontweight="bold")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=v, label=f"{k} — {_label_for(k)}")
        for k, v in _GENOMIC_PALETTE.items()
    ]
    ax.legend(handles=legend_patches, loc="upper left", ncol=4, fontsize=7,
              bbox_to_anchor=(0, 1.55), frameon=False)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — LSTM architecture diagram
# ---------------------------------------------------------------------------


def plot_lstm_architecture(save_path: Optional[str] = None) -> Figure:
    """Matplotlib-drawn architecture diagram for the Genomic LSTM.

    Renders a schematic showing:
    Input embedding → LSTM layers → Dropout → Dense → Softmax output

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Genomic LSTM Architecture", fontweight="bold", fontsize=12, pad=10)

    layer_specs = [
        (1.0, "Input\n(seq_len × 4)", "#dbeafe", "#2563eb"),
        (3.0, "Embedding\n(d_model=64)", "#dcfce7", "#16a34a"),
        (5.5, "LSTM Layer 1\n(hidden=128)", "#fef9c3", "#ca8a04"),
        (7.5, "LSTM Layer 2\n(hidden=64)", "#fef9c3", "#b45309"),
        (9.5, "Dropout\n(p=0.3)", "#f3e8ff", "#9333ea"),
        (11.0, "Dense\n(64→4)", "#fee2e2", "#dc2626"),
        (13.0, "Softmax\nOutput (4)", "#e0f2fe", "#0284c7"),
    ]

    box_w, box_h = 1.5, 1.2
    cy = 2.5

    for x_center, label, face, edge in layer_specs:
        rect = mpatches.FancyBboxPatch(
            (x_center - box_w / 2, cy - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=face, edgecolor=edge, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x_center, cy, label, ha="center", va="center",
                fontsize=8, fontweight="medium", color="#1e293b")

    # Arrows between layers
    for i in range(len(layer_specs) - 1):
        x1 = layer_specs[i][0] + box_w / 2
        x2 = layer_specs[i + 1][0] - box_w / 2
        ax.annotate(
            "", xy=(x2, cy), xytext=(x1, cy),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.2),
        )

    # Input labels A/C/G/T
    for j, letter in enumerate(["A", "C", "G", "T"]):
        bx = 0.2 + j * 0.18
        ax.text(bx, cy + 0.6 - j * 0.25, letter, fontsize=8,
                color=_GENOMIC_PALETTE[letter], fontweight="bold")

    # Output labels
    out_labels = ["Spike\n(A)", "Calm\n(C)", "Drop\n(G)", "Crash\n(T)"]
    for j, lbl in enumerate(out_labels):
        bx = 13.7
        by = 1.2 + j * 0.65
        ax.text(bx, by, lbl, fontsize=6.5, color=list(_GENOMIC_PALETTE.values())[j],
                va="center")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Transition heatmap
# ---------------------------------------------------------------------------


def plot_transition_heatmap(
    transition_matrix: np.ndarray,
    save_path: Optional[str] = None,
) -> Figure:
    """4×4 heatmap of genomic state transition probabilities.

    Parameters
    ----------
    transition_matrix:
        (4, 4) array where ``transition_matrix[i, j]`` is the probability
        of transitioning from state *i* to state *j*.
    save_path:
        Output path.

    Returns
    -------
    Figure
    """
    labels = ["A", "C", "G", "T"]
    if transition_matrix.shape != (4, 4):
        raise ValueError(
            f"transition_matrix must be (4, 4), got {transition_matrix.shape}"
        )

    # Normalise rows to sum to 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1, row_sums)
    tm_norm = transition_matrix / safe_sums

    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    im = ax.imshow(tm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Transition Probability")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"{l}\n({_label_for(l)})" for l in labels], fontsize=9)
    ax.set_yticklabels([f"{l}\n({_label_for(l)})" for l in labels], fontsize=9)
    ax.set_xlabel("Next State", fontsize=10)
    ax.set_ylabel("Current State", fontsize=10)
    ax.set_title("Genomic State Transition Matrix", fontweight="bold")

    # Annotate cells
    for i in range(4):
        for j in range(4):
            val = tm_norm[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="medium")

    # Colour cell borders to genomic palette
    for i, lbl in enumerate(labels):
        color = _GENOMIC_PALETTE[lbl]
        ax.spines["left"].set_visible(False)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Equity curves
# ---------------------------------------------------------------------------


def plot_equity_curves(
    results_dict: Dict[str, pd.Series],
    benchmark_returns: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Overlaid equity curves for multiple strategies.

    Parameters
    ----------
    results_dict:
        Mapping of strategy name to equity curve ``pd.Series``.
    benchmark_returns:
        Optional buy-and-hold daily return series.  Will be converted to
        an equity curve starting at 100.
    save_path:
        Output path.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    all_curves: Dict[str, pd.Series] = {}

    if benchmark_returns is not None:
        bh_equity = (1.0 + benchmark_returns.fillna(0)).cumprod() * 100.0
        all_curves["Buy & Hold"] = bh_equity

    all_curves.update(results_dict)

    for idx, (name, equity) in enumerate(all_curves.items()):
        color = _EQUITY_PALETTE[idx % len(_EQUITY_PALETTE)]
        lw = 2.0 if "genomic" in name.lower() or "lstm" in name.lower() else 1.2
        ls = "--" if name == "Buy & Hold" else "-"
        ax.plot(equity.index, equity.values, label=name, color=color,
                linewidth=lw, linestyle=ls, alpha=0.9)

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (indexed to 100)")
    ax.set_title("Strategy Equity Curves", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: List[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Annotated confusion matrix.

    Parameters
    ----------
    y_true:
        Ground-truth class labels.
    y_pred:
        Predicted class labels.
    labels:
        Class label names.  Defaults to ``['A', 'C', 'G', 'T']``.
    save_path:
        Output path.

    Returns
    -------
    Figure
    """
    if labels is None:
        labels = ["A", "C", "G", "T"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(labels)

    # Build confusion matrix manually (no sklearn dependency)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    for yt, yp in zip(y_true, y_pred):
        i = label_to_idx.get(str(yt))
        j = label_to_idx.get(str(yp))
        if i is not None and j is not None:
            cm[i, j] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Normalised Count")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Prediction Confusion Matrix", fontweight="bold")

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            txt_color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({val:.1%})", ha="center", va="center",
                    fontsize=8, color=txt_color)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# K-mer frequency bar chart
# ---------------------------------------------------------------------------


def plot_kmer_frequency_bar(
    kmer_counts: Dict[str, int],
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> Figure:
    """Horizontal bar chart of the most frequent k-mers.

    Parameters
    ----------
    kmer_counts:
        Mapping of k-mer string to occurrence count.
    top_n:
        Number of top k-mers to display.
    save_path:
        Output path.

    Returns
    -------
    Figure
    """
    if not kmer_counts:
        raise ValueError("kmer_counts must not be empty.")

    sorted_items = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    kmers, counts = zip(*sorted_items)
    total = sum(kmer_counts.values())
    freqs = [c / total for c in counts]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.38)))

    bar_colors = []
    for kmer in kmers:
        first_letter = kmer[0].upper()
        bar_colors.append(_GENOMIC_PALETTE.get(first_letter, "#64748b"))

    bars = ax.barh(range(len(kmers)), freqs, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(kmers)))
    ax.set_yticklabels(kmers, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Relative Frequency")
    ax.set_title(f"Top-{top_n} K-mer Frequencies", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=2))

    for bar, freq, count in zip(bars, freqs, counts):
        ax.text(
            bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"  {count:,}",
            va="center", fontsize=8, color="#374151",
        )

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------


def plot_learning_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    save_path: Optional[str] = None,
) -> Figure:
    """Two-panel figure showing training loss and accuracy curves.

    Parameters
    ----------
    train_losses:
        Per-epoch training cross-entropy loss.
    val_losses:
        Per-epoch validation cross-entropy loss.
    train_accs:
        Per-epoch training accuracy (0–1).
    val_accs:
        Per-epoch validation accuracy (0–1).
    save_path:
        Output path.

    Returns
    -------
    Figure
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("LSTM Training Curves", fontweight="bold")

    # Loss panel
    ax_loss.plot(epochs, train_losses, label="Train", color="#2563eb", linewidth=1.5)
    ax_loss.plot(epochs, val_losses, label="Validation", color="#dc2626",
                 linewidth=1.5, linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(axis="y", linestyle="--", alpha=0.4)

    # Accuracy panel
    ax_acc.plot(epochs, train_accs, label="Train", color="#2563eb", linewidth=1.5)
    ax_acc.plot(epochs, val_accs, label="Validation", color="#dc2626",
                linewidth=1.5, linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_acc.legend()
    ax_acc.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Drawdown comparison
# ---------------------------------------------------------------------------


def plot_drawdown_comparison(
    equity_curves_dict: Dict[str, pd.Series],
    save_path: Optional[str] = None,
) -> Figure:
    """Stacked or overlaid drawdown curves for multiple strategies.

    Parameters
    ----------
    equity_curves_dict:
        Mapping of strategy name to equity curve series.
    save_path:
        Output path.

    Returns
    -------
    Figure
    """
    n = len(equity_curves_dict)
    if n == 0:
        raise ValueError("equity_curves_dict must not be empty.")

    fig, ax = plt.subplots(figsize=(13, 4))

    for idx, (name, equity) in enumerate(equity_curves_dict.items()):
        rolling_max = equity.cummax()
        dd = (equity / rolling_max - 1.0) * 100.0
        color = _EQUITY_PALETTE[idx % len(_EQUITY_PALETTE)]
        ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.25)
        ax.plot(dd.index, dd.values, label=name, color=color, linewidth=1.0)

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Strategy Drawdown Comparison", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.legend(loc="lower left", ncol=2, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Full report generator
# ---------------------------------------------------------------------------


def create_full_report_figures(
    results_dict: Dict[str, object],
    save_dir: str = "paper/figures/",
) -> None:
    """Generate all paper figures and save as PDFs.

    Parameters
    ----------
    results_dict:
        Dictionary produced by ``FinancialGenomicsPipeline.run_full_pipeline()``.
        Expected keys (all optional; missing keys are gracefully skipped):

        ``prices``, ``returns``, ``volatility``, ``sequence``, ``dates``,
        ``sequence_annotations``, ``transition_matrix``,
        ``equity_curves``, ``benchmark_returns``,
        ``y_true``, ``y_pred``,
        ``kmer_counts``,
        ``train_losses``, ``val_losses``, ``train_accs``, ``val_accs``.
    save_dir:
        Directory where PDFs are written.  Created if it does not exist.
    """
    os.makedirs(save_dir, exist_ok=True)
    generated: List[str] = []

    def _path(name: str) -> str:
        return os.path.join(save_dir, name)

    def _has(*keys: str) -> bool:
        return all(k in results_dict and results_dict[k] is not None for k in keys)

    # Figure 1 — pipeline
    if _has("prices", "returns", "volatility", "sequence"):
        fig = plot_discretization_pipeline(
            prices=results_dict["prices"],
            returns=results_dict["returns"],
            volatility=results_dict["volatility"],
            sequence=results_dict["sequence"],
            save_path=_path("fig1_pipeline.pdf"),
        )
        plt.close(fig)
        generated.append("fig1_pipeline.pdf")

    # Figure 2 — example sequence
    if _has("sequence", "dates"):
        fig = plot_example_sequence(
            sequence_str=results_dict["sequence"],
            dates=results_dict["dates"],
            annotations=results_dict.get("sequence_annotations"),
            save_path=_path("fig2_sequence.pdf"),
        )
        plt.close(fig)
        generated.append("fig2_sequence.pdf")

    # Figure 3 — LSTM architecture
    fig = plot_lstm_architecture(save_path=_path("fig3_architecture.pdf"))
    plt.close(fig)
    generated.append("fig3_architecture.pdf")

    # Figure 4 — transition heatmap
    if _has("transition_matrix"):
        fig = plot_transition_heatmap(
            transition_matrix=np.asarray(results_dict["transition_matrix"]),
            save_path=_path("fig4_transitions.pdf"),
        )
        plt.close(fig)
        generated.append("fig4_transitions.pdf")

    # Figure 5 — equity curves
    if _has("equity_curves"):
        fig = plot_equity_curves(
            results_dict=results_dict["equity_curves"],
            benchmark_returns=results_dict.get("benchmark_returns"),
            save_path=_path("fig5_equity.pdf"),
        )
        plt.close(fig)
        generated.append("fig5_equity.pdf")

    # Confusion matrix
    if _has("y_true", "y_pred"):
        fig = plot_confusion_matrix(
            y_true=results_dict["y_true"],
            y_pred=results_dict["y_pred"],
            save_path=_path("fig6_confusion.pdf"),
        )
        plt.close(fig)
        generated.append("fig6_confusion.pdf")

    # K-mer frequencies
    if _has("kmer_counts"):
        fig = plot_kmer_frequency_bar(
            kmer_counts=results_dict["kmer_counts"],
            save_path=_path("fig7_kmers.pdf"),
        )
        plt.close(fig)
        generated.append("fig7_kmers.pdf")

    # Learning curves
    if _has("train_losses", "val_losses", "train_accs", "val_accs"):
        fig = plot_learning_curves(
            train_losses=results_dict["train_losses"],
            val_losses=results_dict["val_losses"],
            train_accs=results_dict["train_accs"],
            val_accs=results_dict["val_accs"],
            save_path=_path("fig8_learning.pdf"),
        )
        plt.close(fig)
        generated.append("fig8_learning.pdf")

    # Drawdown comparison
    if _has("equity_curves"):
        fig = plot_drawdown_comparison(
            equity_curves_dict=results_dict["equity_curves"],
            save_path=_path("fig9_drawdown.pdf"),
        )
        plt.close(fig)
        generated.append("fig9_drawdown.pdf")

    print(f"[create_full_report_figures] Saved {len(generated)} figure(s) to {save_dir!r}:")
    for fname in generated:
        print(f"  • {fname}")
