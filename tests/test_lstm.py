"""
Tests for src/models/lstm_model.py

The LSTM model treats discretized return sequences as a genomic grammar and
predicts the probability distribution over the next regime symbol (A/C/G/T).

All tests run on CPU and use a tiny architecture (hidden=32, layers=1) for
speed.  We mock CUDA-dependent paths where necessary.
"""

import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import guard — skip if torch is not installed
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")

from src.models.lstm_model import GenomicLSTM, LSTMTrainer, prepare_sequences


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 4       # A=0, C=1, G=2, T=3
SEQ_LEN = 10
BATCH_SIZE = 8
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EMBEDDING_DIM = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """Minimal GenomicLSTM on CPU."""
    model = GenomicLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_int_sequence(rng):
    """500-length sequence of integer codes in {0,1,2,3}."""
    return rng.integers(0, 4, size=500).tolist()


@pytest.fixture
def trainer(tiny_model):
    return LSTMTrainer(
        model=tiny_model,
        learning_rate=1e-3,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self, tiny_model):
        """Input (batch, seq_len) -> output (batch, 4)."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        with torch.no_grad():
            out = tiny_model(x)
        assert out.shape == (BATCH_SIZE, VOCAB_SIZE), (
            f"Expected output shape ({BATCH_SIZE}, {VOCAB_SIZE}), got {out.shape}"
        )

    def test_output_sums_to_1_after_softmax(self, tiny_model):
        """Probabilities must sum to 1 along class dimension."""
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        with torch.no_grad():
            logits = tiny_model(x)
            probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        np.testing.assert_allclose(
            sums.numpy(), np.ones(BATCH_SIZE), atol=1e-5,
            err_msg="Softmax probabilities do not sum to 1"
        )

    def test_output_probabilities_non_negative(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        with torch.no_grad():
            logits = tiny_model(x)
            probs = torch.softmax(logits, dim=-1)
        assert torch.all(probs >= 0), "Negative probabilities detected"

    def test_single_sample_forward(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        with torch.no_grad():
            out = tiny_model(x)
        assert out.shape == (1, VOCAB_SIZE)

    def test_output_finite(self, tiny_model):
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        with torch.no_grad():
            out = tiny_model(x)
        assert torch.all(torch.isfinite(out)), "Non-finite values in model output"

    def test_different_seq_lengths_work(self, tiny_model):
        for seq_len in [5, 10, 20, 50]:
            x = torch.randint(0, VOCAB_SIZE, (4, seq_len))
            with torch.no_grad():
                out = tiny_model(x)
            assert out.shape == (4, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# prepare_sequences
# ---------------------------------------------------------------------------

class TestPrepareSequences:
    def test_X_shape(self, synthetic_int_sequence):
        seq_len = 20
        X, y = prepare_sequences(synthetic_int_sequence, seq_len=seq_len)
        n = len(synthetic_int_sequence)
        expected_samples = n - seq_len
        assert X.shape == (expected_samples, seq_len), (
            f"Expected X shape ({expected_samples}, {seq_len}), got {X.shape}"
        )

    def test_y_shape(self, synthetic_int_sequence):
        seq_len = 20
        X, y = prepare_sequences(synthetic_int_sequence, seq_len=seq_len)
        n = len(synthetic_int_sequence)
        expected_samples = n - seq_len
        assert y.shape == (expected_samples,), (
            f"Expected y shape ({expected_samples},), got {y.shape}"
        )

    def test_X_y_aligned(self, synthetic_int_sequence):
        """y[i] must equal seq[seq_len + i] (the token right after each window)."""
        seq_len = 10
        X, y = prepare_sequences(synthetic_int_sequence, seq_len=seq_len)
        for i in range(min(20, len(y))):
            expected_label = synthetic_int_sequence[seq_len + i]
            assert y[i] == expected_label, (
                f"y[{i}]={y[i]} but expected {expected_label}"
            )

    def test_X_window_content(self, synthetic_int_sequence):
        """X[i] must equal seq[i : i + seq_len]."""
        seq_len = 10
        X, y = prepare_sequences(synthetic_int_sequence, seq_len=seq_len)
        for i in range(min(10, len(X))):
            expected_window = synthetic_int_sequence[i: i + seq_len]
            np.testing.assert_array_equal(
                X[i], expected_window,
                err_msg=f"Window mismatch at index {i}"
            )

    @pytest.mark.parametrize("seq_len", [5, 10, 20, 50])
    def test_various_seq_lengths(self, synthetic_int_sequence, seq_len):
        X, y = prepare_sequences(synthetic_int_sequence, seq_len=seq_len)
        n = len(synthetic_int_sequence)
        assert X.shape[0] == n - seq_len
        assert X.shape[1] == seq_len

    def test_y_values_in_vocab(self, synthetic_int_sequence):
        X, y = prepare_sequences(synthetic_int_sequence, seq_len=10)
        assert np.all((y >= 0) & (y < VOCAB_SIZE)), "y contains out-of-vocab labels"

    def test_returns_numpy_arrays(self, synthetic_int_sequence):
        X, y = prepare_sequences(synthetic_int_sequence, seq_len=10)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)


# ---------------------------------------------------------------------------
# Trainer: loss decreases
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_train_reduces_loss(self, trainer, rng):
        """Loss after training should be lower than initial loss on a tiny dataset."""
        seq = rng.integers(0, 4, size=300).tolist()
        X, y = prepare_sequences(seq, seq_len=10)

        history = trainer.train(X, y, epochs=10, batch_size=32)

        assert "train_loss" in history, "train() must return dict with 'train_loss'"
        losses = history["train_loss"]
        assert len(losses) == 10, f"Expected 10 loss values, got {len(losses)}"
        # Loss should decrease (or at least not monotonically increase)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_train_returns_dict_with_expected_keys(self, trainer, rng):
        seq = rng.integers(0, 4, size=200).tolist()
        X, y = prepare_sequences(seq, seq_len=10)
        history = trainer.train(X, y, epochs=3, batch_size=32)
        assert isinstance(history, dict)
        assert "train_loss" in history

    def test_train_loss_values_are_finite(self, trainer, rng):
        seq = rng.integers(0, 4, size=200).tolist()
        X, y = prepare_sequences(seq, seq_len=10)
        history = trainer.train(X, y, epochs=5, batch_size=32)
        for loss in history["train_loss"]:
            assert np.isfinite(loss), f"Non-finite loss value: {loss}"

    def test_train_loss_positive(self, trainer, rng):
        seq = rng.integers(0, 4, size=200).tolist()
        X, y = prepare_sequences(seq, seq_len=10)
        history = trainer.train(X, y, epochs=3, batch_size=32)
        for loss in history["train_loss"]:
            assert loss > 0, f"Loss should be positive, got {loss}"


# ---------------------------------------------------------------------------
# Save / load produces same predictions
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_load_same_predictions(self, tiny_model):
        """After save/load, model must produce identical outputs."""
        x = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
        with torch.no_grad():
            original_out = tiny_model(x).numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            tiny_model.save(path)

            loaded_model = GenomicLSTM.load(
                path,
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMBEDDING_DIM,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=0.0,
            )
            loaded_model.eval()
            with torch.no_grad():
                loaded_out = loaded_model(x).numpy()

        np.testing.assert_allclose(
            original_out, loaded_out, atol=1e-6,
            err_msg="Save/load produced different predictions"
        )

    def test_save_creates_file(self, tiny_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            tiny_model.save(path)
            assert os.path.exists(path), "Save did not create file"

    def test_load_model_type(self, tiny_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            tiny_model.save(path)
            loaded = GenomicLSTM.load(
                path,
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMBEDDING_DIM,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=0.0,
            )
            assert isinstance(loaded, GenomicLSTM)


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

class TestWalkForwardValidate:
    EXPECTED_KEYS = {"fold", "train_loss", "val_accuracy", "val_loss"}

    def test_returns_list(self, trainer, rng):
        seq = rng.integers(0, 4, size=400).tolist()
        X, y = prepare_sequences(seq, seq_len=10)
        results = trainer.walk_forward_validate(
            X, y,
            initial_train_size=200,
            test_window=50,
            retrain_every=50,
            epochs=2,
        )
        assert isinstance(results, list), "walk_forward_validate must return a list"

    def test_each_item_has_expected_keys(self, trainer, rng):
        seq = rng.integers(0, 4, size=400).tolist()
        X, y = prepare_sequences(seq, seq_len=10)
        results = trainer.walk_forward_validate(
            X, y,
            initial_train_size=200,
            test_window=50,
            retrain_every=50,
            epochs=2,
        )
        assert len(results) > 0, "walk_forward_validate returned empty list"
        for record in results:
            for key in self.EXPECTED_KEYS:
                assert key in record, (
                    f"Expected key '{key}' missing from result dict: {record.keys()}"
                )

    def test_fold_indices_are_sequential(self, trainer, rng):
        seq = rng.integers(0, 4, size=400).tolist()
        X, y = prepare_sequences(seq, seq_len=10)
        results = trainer.walk_forward_validate(
            X, y,
            initial_train_size=200,
            test_window=50,
            retrain_every=50,
            epochs=2,
        )
        folds = [r["fold"] for r in results]
        assert folds == list(range(len(results))), (
            f"Fold indices not sequential: {folds}"
        )

    def test_val_accuracy_in_0_1(self, trainer, rng):
        seq = rng.integers(0, 4, size=400).tolist()
        X, y = prepare_sequences(seq, seq_len=10)
        results = trainer.walk_forward_validate(
            X, y,
            initial_train_size=200,
            test_window=50,
            retrain_every=50,
            epochs=2,
        )
        for record in results:
            acc = record["val_accuracy"]
            assert 0.0 <= acc <= 1.0, f"val_accuracy out of [0,1]: {acc}"


# ---------------------------------------------------------------------------
# CPU-only: model runs without CUDA
# ---------------------------------------------------------------------------

class TestCPUOnly:
    def test_model_on_cpu(self):
        model = GenomicLSTM(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=0.0,
        ).to("cpu")
        x = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
        with torch.no_grad():
            out = model(x)
        assert out.device.type == "cpu"

    def test_trainer_device_cpu(self):
        model = GenomicLSTM(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=0.0,
        )
        trainer = LSTMTrainer(model=model, learning_rate=1e-3, device="cpu")
        assert trainer.device == "cpu" or str(trainer.device) == "cpu"
