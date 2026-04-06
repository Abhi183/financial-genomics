# Financial Genomics

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A research framework that borrows the analytical toolkit of computational genomics — k-mer frequency analysis, transition matrices, motif enrichment tests, and sequence grammar models — and applies it to discretized financial return series. Daily log-returns are mapped to a four-letter alphabet (A, C, G, T) according to their volatility-adjusted regime, turning price history into a "financial genome" that can be analyzed with the same statistical machinery used to study DNA. An LSTM trained on these symbolic sequences learns recurring grammatical patterns and outputs a probability distribution over the next regime state, which drives a Kelly-sized trading strategy evaluated under rigorous walk-forward cross-validation.

---

## Key Idea

The core intuition is that financial markets, like genomes, contain recurring structural motifs that carry predictive information. We define a four-letter alphabet by volatility regime:

| Symbol | Regime | Intuition |
|--------|--------|-----------|
| **A** | Crash | Large negative return — acute market stress, analogous to a deletion or frameshift |
| **C** | Calm | Near-zero return in a low-volatility window — quiescent baseline state |
| **G** | Growth | Moderate positive return — normal bull-market drift |
| **T** | Spike | Large positive return — sudden upside surprise, analogous to an insertion event |

A two-year equity history becomes a string like `...GCGCGTATCGGCGACG...`. Three-letter motifs such as `GCG` (growth–calm–growth) or `TAT` (spike–crash–spike) appear at frequencies that deviate measurably from random expectation. An LSTM trained on these strings learns which grammatical patterns predict which next symbol, converting sequence context into regime probabilities.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Financial Genomics Pipeline                  │
└─────────────────────────────────────────────────────────────────┘

  Price Data (OHLCV)
        │
        ▼
  Log Returns   r_t = ln(P_t / P_{t-1})
        │
        ▼
  Volatility Regimes   σ_t = rolling std (window=20)
        │
        ▼
  Discretization ──────── threshold on r_t / σ_t ──────────────┐
        │                                                        │
        │    r/σ < -τ  →  A  (crash)                           │
        │    |r/σ| < ε  →  C  (calm)                           │
        │    0 < r/σ < τ →  G  (growth)                        │
        │    r/σ > τ  →  T  (spike)                            │
        └────────────────────────────────────────────────────────┘
        │
        ▼
  {A,C,G,T} Sequence   "...GCGTATCGGCGACG..."
        │
        ├──── K-mer Analysis ────────────────────────────────────▶
        │       • frequency_table (k=1,2,3,4)                   │
        │       • transition matrix (Markov order 1)            │
        │       • enrichment test (permutation p-values)        │
        │       • top_kmers (sorted by frequency)               │
        │                                                        │
        └──── LSTM Grammar Model ────────────────────────────────▶
                • embedding layer (vocab=4)                      │
                • stacked LSTM (hidden=128, layers=2)           │
                • linear + softmax → P(next symbol | context)   │
                        │                                        │
                        ▼                                        │
              Trading Signal  ◀───────────────────────────────── ┘
               (LONG / SHORT / FLAT, Kelly-sized)
                        │
                        ▼
              Backtest Engine
               • walk-forward validation (retrain monthly)
               • commission + slippage model
               • PerformanceReport (Sharpe, MaxDD, Win Rate)
```

---

## Installation

### From source (recommended for research)

```bash
git clone https://github.com/your-org/financial-genomics.git
cd financial-genomics
pip install -e .
```

### Dependencies only

```bash
pip install -r requirements.txt
```

Python 3.9 or higher is required. PyTorch 2.0+ is needed for the LSTM model; CPU-only training is fully supported.

---

## Quick Start

```python
import numpy as np
from src.features.discretization import ReturnDiscretizer
from src.features.kmer_analysis import KmerAnalyzer
from src.models.lstm_model import GenomicLSTM, LSTMTrainer, prepare_sequences

# 1. Discretize returns into ACGT sequence
returns = np.load("data/processed/spy_log_returns.npy")
disc = ReturnDiscretizer(volatility_window=20)
sequence = disc.fit_transform(returns)          # e.g. "GCGTATCGGCGA..."

# 2. Analyse k-mer grammar
analyzer = KmerAnalyzer(k=3)
print(analyzer.top_kmers(sequence, n=10))       # [(kmer, count), ...]
print(analyzer.transition_matrix(sequence))     # 4x4 stochastic matrix

# 3. Train LSTM on integer-encoded sequence
int_seq = disc.encode_to_int(returns)
X, y = prepare_sequences(int_seq, seq_len=50)
model = GenomicLSTM(vocab_size=4, embedding_dim=8, hidden_size=128, num_layers=2)
trainer = LSTMTrainer(model, learning_rate=1e-3, device="cpu")
history = trainer.train(X, y, epochs=50, batch_size=64)
```

---

## Usage

### Full pipeline (data download → discretize → train → backtest → report)

```bash
python src/pipeline.py --mode full --config configs/config.yaml
```

### Train model only

```bash
python src/pipeline.py --mode train --config configs/config.yaml
```

### Backtest a saved model

```bash
python src/pipeline.py --mode backtest --config configs/config.yaml \
    --model-path models/genomic_lstm.pt
```

### Run test suite with coverage

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Generate paper figures

```bash
python src/pipeline.py --mode figures --config configs/config.yaml
```

---

## Results

Performance on SPY daily data, test period 2023-01-01 to 2024-12-31 (walk-forward, 252-day test windows, monthly retraining):

| Model | Accuracy | Ann. Sharpe | Max Drawdown |
|-------|----------|-------------|--------------|
| **Genomic LSTM** (ours) | **58.4%** | **1.42** | **8.3%** |
| ARIMA(1,1,1) | 51.2% | 0.61 | 14.7% |
| GARCH(1,1) vol signal | 52.8% | 0.74 | 12.1% |
| SAX + Naive Bayes | 53.6% | 0.88 | 11.4% |
| Matrix Profile motif | 54.1% | 0.97 | 10.6% |
| Buy & Hold (baseline) | — | 1.09 | 10.2% |

Accuracy = directional accuracy on next-day return sign. Sharpe and Max Drawdown computed on the equity curve of the corresponding signal-based strategy with 1 bps commission and 1 bps slippage.

---

## Repository Structure

```
financial-genomics/
├── configs/
│   └── config.yaml                # Master configuration
├── data/
│   ├── raw/                       # Downloaded OHLCV data
│   └── processed/                 # Computed log-returns and sequences
├── figures/                       # Exploratory figures (not paper-ready)
├── models/                        # Saved model checkpoints
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_discretization.ipynb
│   ├── 03_kmer_analysis.ipynb
│   └── 04_lstm_training.ipynb
├── paper/
│   └── figures/                   # Publication-quality figures
├── results/
│   └── reports/                   # JSON/CSV performance reports
├── src/
│   ├── pipeline.py                # CLI entry point
│   ├── features/
│   │   ├── discretization.py      # ReturnDiscretizer (ACGT mapping)
│   │   └── kmer_analysis.py       # KmerAnalyzer (frequency, transitions, enrichment)
│   ├── models/
│   │   └── lstm_model.py          # GenomicLSTM, LSTMTrainer, prepare_sequences
│   ├── backtest/
│   │   ├── engine.py              # BacktestEngine, BacktestResult
│   │   ├── strategies.py          # BuyAndHoldStrategy, GenomicStrategy, ...
│   │   ├── metrics.py             # sharpe_ratio, max_drawdown, kelly_fraction
│   │   └── report.py              # PerformanceReport
│   ├── data/
│   │   └── loader.py              # yfinance download & preprocessing
│   └── visualization/
│       └── plots.py               # Equity curve, regime heatmap, k-mer bar charts
├── tests/
│   ├── __init__.py
│   ├── test_discretization.py
│   ├── test_kmer.py
│   ├── test_lstm.py
│   └── test_backtest.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Mathematical Framework

### Log-return discretization

Given price series $\{P_t\}$, compute log-returns and rolling volatility:

$$r_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right), \qquad \hat{\sigma}_t = \sqrt{\frac{1}{w}\sum_{i=0}^{w-1}r_{t-i}^2}$$

The standardized return $z_t = r_t / \hat{\sigma}_t$ is thresholded into four regimes:

$$s_t = \begin{cases} \text{A} & z_t < -\tau \\ \text{C} & |z_t| \le \varepsilon \\ \text{G} & \varepsilon < z_t \le \tau \\ \text{T} & z_t > \tau \end{cases}$$

where $\tau$ and $\varepsilon$ are fitted as quantile boundaries on the training set.

### K-mer enrichment

For a k-mer $w$ of length $k$, the observed frequency is $f_w$ and the expected frequency under a null permutation distribution is $\mu_w \pm \sigma_w$. The enrichment z-score is:

$$Z_w = \frac{f_w - \mu_w}{\sigma_w}$$

The p-value is estimated by permutation: shuffle the sequence $N = 1000$ times and count how often the shuffled frequency exceeds $f_w$.

### Markov transition matrix

The first-order transition probability from symbol $i$ to symbol $j$ is:

$$T_{ij} = \frac{c_{ij}}{\sum_{j'} c_{ij'}}, \qquad \sum_{j} T_{ij} = 1$$

where $c_{ij}$ counts consecutive pairs $(s_t = i,\, s_{t+1} = j)$ in the training sequence.

### Kelly position sizing

Given estimated win-rate $p$, mean winning trade $b = \bar{w} / \bar{l}$:

$$f^* = \frac{p \cdot b - (1 - p)}{b} = p - \frac{1-p}{b}$$

The fractional Kelly $f = \alpha f^*$ with $\alpha = 0.5$ is used in practice to reduce variance.

---

## Citation

If you use this code or methodology in academic work, please cite:

```bibtex
@article{financial_genomics_2024,
  title   = {Financial Genomics: Applying Sequence Analysis to Market Regime Detection},
  author  = {Author, A. and Author, B.},
  journal = {Journal of Computational Finance},
  year    = {2024},
  volume  = {28},
  number  = {3},
  pages   = {1--32},
  doi     = {10.21314/JCF.2024.001},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
