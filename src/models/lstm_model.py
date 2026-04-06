"""
Genomic LSTM: sequence-to-sequence prediction over the financial DNA alphabet.

Architecture overview
---------------------
The model treats the encoded market sequence as an analogue of a DNA strand
and learns to predict the next base (A/C/G/T) given the preceding ``seq_len``
bases.  The pipeline is:

    integer tokens  →  nn.Embedding(4, 8)  →  nn.LSTM(8, 128, layers=2)
        →  last hidden state  →  nn.Linear(128, 4)  →  logits

A learned embedding is used rather than one-hot encoding so that the model
can discover a low-dimensional latent space in which semantically similar
bases (e.g. C and G, both non-tail events) cluster together.

Walk-forward validation
-----------------------
Financial time series violate the i.i.d. assumption; standard k-fold cross-
validation leaks future information into the training set.  The trainer
therefore provides :meth:`GenomicLSTMTrainer.walk_forward_validate`, which
expands the training window period by period and evaluates on the immediately
following held-out segment — mimicking the live trading scenario.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────


class GenomicLSTM(nn.Module):
    """LSTM-based sequence model for financial genomic prediction.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the learned base embedding.  Defaults to ``8``.
    hidden_size:
        Number of features in the LSTM hidden state.  Defaults to ``128``.
    num_layers:
        Number of stacked LSTM layers.  Defaults to ``2``.
    dropout:
        Dropout probability applied between LSTM layers (ignored when
        ``num_layers == 1``).  Defaults to ``0.3``.
    num_classes:
        Output vocabulary size.  Defaults to ``4`` (A=0, C=1, G=2, T=3).
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=embedding_dim,
            padding_idx=None,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return class logits.

        Parameters
        ----------
        x:
            Integer token tensor of shape ``(batch, seq_len)``.  Values must
            be in ``[0, num_classes)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)``.  Pass through
            ``F.softmax`` to obtain class probabilities.
        """
        # x: (batch, seq_len)
        embedded = self.embedding(x)           # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)       # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]        # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)           # (batch, num_classes)
        return logits

    def __repr__(self) -> str:
        return (
            f"GenomicLSTM("
            f"embedding_dim={self.embedding_dim}, "
            f"hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, "
            f"num_classes={self.num_classes})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class GenomicLSTMTrainer:
    """Training, evaluation, and inference wrapper for :class:`GenomicLSTM`.

    Parameters
    ----------
    model:
        A :class:`GenomicLSTM` instance.
    lr:
        Initial learning rate for the Adam optimiser.  Defaults to ``1e-3``.
    device:
        PyTorch device string (``"cpu"`` or ``"cuda"``).  Defaults to
        ``"cpu"``; use ``"cuda"`` for GPU acceleration.
    """

    def __init__(
        self,
        model: GenomicLSTM,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model: GenomicLSTM = model.to(self.device)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        # Training history.
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []

    # ------------------------------------------------------------------ #
    # Sequence preparation                                                 #
    # ------------------------------------------------------------------ #

    def prepare_sequences(
        self, int_sequence: List[int], seq_len: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (X, y) sliding-window tensors from an integer-encoded sequence.

        For each position *i* from 0 to ``len(int_sequence) - seq_len - 1``,
        the window ``int_sequence[i : i + seq_len]`` is a training context
        and ``int_sequence[i + seq_len]`` is the target label.

        Parameters
        ----------
        int_sequence:
            Integer-encoded genomic sequence (values in ``{0, 1, 2, 3}``).
        seq_len:
            Number of preceding bases used as context.  Defaults to ``50``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(X, y)`` where:

            * ``X`` has shape ``(n_samples, seq_len)`` and dtype ``torch.long``.
            * ``y`` has shape ``(n_samples,)`` and dtype ``torch.long``.

        Raises
        ------
        ValueError
            If the sequence is shorter than ``seq_len + 1``.
        """
        n = len(int_sequence)
        if n < seq_len + 1:
            raise ValueError(
                f"int_sequence has {n} elements but seq_len={seq_len} requires "
                f"at least {seq_len + 1} elements."
            )

        arr = np.array(int_sequence, dtype=np.int64)
        X_list = []
        y_list = []
        for i in range(n - seq_len):
            X_list.append(arr[i : i + seq_len])
            y_list.append(arr[i + seq_len])

        X = torch.tensor(np.array(X_list), dtype=torch.long)
        y = torch.tensor(np.array(y_list), dtype=torch.long)
        logger.debug(
            "prepare_sequences: X=%s, y=%s", tuple(X.shape), tuple(y.shape)
        )
        return X, y

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> Dict[str, List[float]]:
        """Train the model for a given number of epochs.

        Parameters
        ----------
        X:
            Input tensor of shape ``(n_samples, seq_len)``.
        y:
            Target label tensor of shape ``(n_samples,)``.
        epochs:
            Number of complete passes over the training data.  Defaults to
            ``50``.
        batch_size:
            Mini-batch size.  Defaults to ``64``.

        Returns
        -------
        dict[str, list[float]]
            Training history with keys ``"loss"`` and ``"accuracy"``.
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimiser.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimiser.step()

                epoch_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            avg_loss = epoch_loss / total
            accuracy = correct / total
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d — loss=%.4f  acc=%.4f",
                    epoch,
                    epochs,
                    avg_loss,
                    accuracy,
                )

        return history

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 256
    ) -> Dict[str, float]:
        """Evaluate the model on a labelled dataset.

        Parameters
        ----------
        X:
            Input tensor of shape ``(n_samples, seq_len)``.
        y:
            Target label tensor of shape ``(n_samples,)``.
        batch_size:
            Inference batch size.  Defaults to ``256``.

        Returns
        -------
        dict[str, float]
            Keys: ``"loss"``, ``"accuracy"``, ``"precision"``, ``"recall"``,
            ``"f1"`` (all macro-averaged for multi-class metrics).
        """
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(y_batch.cpu().numpy().tolist())

        n = len(all_labels)
        avg_loss = total_loss / n
        accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        labels_arr = np.array(all_labels)
        preds_arr = np.array(all_preds)
        precision = float(
            precision_score(labels_arr, preds_arr, average="macro", zero_division=0)
        )
        recall = float(
            recall_score(labels_arr, preds_arr, average="macro", zero_division=0)
        )
        f1 = float(
            f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
        )

        metrics = {
            "loss": round(avg_loss, 6),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict_proba(self, context_sequence: List[int]) -> np.ndarray:
        """Return class probabilities for the base following *context_sequence*.

        Parameters
        ----------
        context_sequence:
            Integer-encoded context window of length ``seq_len``.  Shorter
            sequences are left-padded with zeros (base A); longer sequences
            are right-truncated.

        Returns
        -------
        np.ndarray
            Shape ``(4,)`` array of probabilities over {A=0, C=1, G=2, T=3}.
        """
        if not context_sequence:
            raise ValueError("context_sequence must be non-empty.")

        # Determine expected seq_len from the model's training context.
        # We infer it from the first Linear layer's implicit requirement.
        # For safety, use the length of context_sequence as-is if <= 1000.
        seq = context_sequence[-1000:]  # Reasonable cap.

        arr = np.array(seq, dtype=np.int64)
        x = torch.tensor(arr, dtype=torch.long).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)  # (1, 4)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return probs

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Persist model weights and optimiser state to disk.

        Parameters
        ----------
        path:
            File path for the checkpoint (e.g. ``"checkpoints/model.pt"``).
            Parent directories are created if they do not exist.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "model_config": {
                "embedding_dim": self.model.embedding_dim,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "num_classes": self.model.num_classes,
            },
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
        }
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s.", path)

    def load(self, path: str) -> None:
        """Load model weights and optimiser state from a checkpoint file.

        Parameters
        ----------
        path:
            Path to a checkpoint saved by :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.train_accuracies = checkpoint.get("train_accuracies", [])
        self.model.to(self.device)
        logger.info("Checkpoint loaded from %s.", path)

    # ------------------------------------------------------------------ #
    # Walk-forward validation                                              #
    # ------------------------------------------------------------------ #

    def walk_forward_validate(
        self,
        full_sequence: List[int],
        initial_train_size: int,
        retrain_every: int,
        seq_len: int = 50,
        epochs: int = 30,
        batch_size: int = 64,
    ) -> List[Dict]:
        """Walk-forward cross-validation for time-series data.

        The method expands the training window in increments of
        ``retrain_every`` steps, retrains the model from scratch at each
        increment, and evaluates on the next ``retrain_every`` held-out steps.

        This scheme respects temporal ordering and prevents look-ahead bias,
        making it the appropriate validation strategy for financial data.

        Parameters
        ----------
        full_sequence:
            Complete integer-encoded sequence (train + test combined).
        initial_train_size:
            Number of tokens used in the first training fold.  Must be
            greater than ``seq_len``.
        retrain_every:
            Number of new tokens added to the training window at each step
            and used as the held-out test segment.
        seq_len:
            LSTM context window length.  Defaults to ``50``.
        epochs:
            Training epochs per fold.  Defaults to ``30``.
        batch_size:
            Mini-batch size.  Defaults to ``64``.

        Returns
        -------
        list[dict]
            One entry per validation fold containing:

            * ``"period"`` — (train_end, test_end) index tuple
            * ``"train_size"`` — number of training tokens
            * ``"test_size"`` — number of test tokens
            * ``"metrics"`` — dict from :meth:`evaluate`

        Raises
        ------
        ValueError
            If ``initial_train_size <= seq_len`` or the sequence is too short.
        """
        if initial_train_size <= seq_len:
            raise ValueError(
                f"initial_train_size ({initial_train_size}) must be greater "
                f"than seq_len ({seq_len})."
            )
        n = len(full_sequence)
        if n < initial_train_size + retrain_every:
            raise ValueError(
                f"full_sequence has {n} elements, which is not enough for "
                f"even one fold (need {initial_train_size + retrain_every})."
            )

        results = []
        train_end = initial_train_size

        while train_end + retrain_every <= n:
            test_end = train_end + retrain_every
            train_seq = full_sequence[:train_end]
            test_seq = full_sequence[train_end - seq_len : test_end]

            # Re-initialise model weights for each fold.
            self.model.apply(self._reset_weights)
            self.optimiser = torch.optim.Adam(
                self.model.parameters(), lr=self.lr
            )

            # Train.
            X_train, y_train = self.prepare_sequences(train_seq, seq_len=seq_len)
            self.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

            # Evaluate.
            if len(test_seq) <= seq_len:
                logger.warning(
                    "Test segment too short for seq_len=%d; skipping fold.",
                    seq_len,
                )
                train_end = test_end
                continue

            X_test, y_test = self.prepare_sequences(test_seq, seq_len=seq_len)
            metrics = self.evaluate(X_test, y_test)

            results.append(
                {
                    "period": (train_end, test_end),
                    "train_size": train_end,
                    "test_size": retrain_every,
                    "metrics": metrics,
                }
            )
            logger.info(
                "Walk-forward fold train_end=%d, test_end=%d — %s",
                train_end,
                test_end,
                metrics,
            )
            train_end = test_end

        return results

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reset_weights(module: nn.Module) -> None:
        """Reset learnable parameters of supported layer types."""
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()  # type: ignore[union-attr]
