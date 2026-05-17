"""
4_ann.py
Artificial Neural Network — fully from scratch using NumPy only.

Architecture:
    Embedding  (lookup + average-pool)
    Dense(128) + ReLU
    Dropout(0.3)
    Dense(64)  + ReLU
    Dropout(0.3)
    Dense(3)   + Softmax

Training: Mini-batch SGD with momentum, forward + manual backprop.
No keras, no torch, no sklearn.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import load_features
from utils.metrics import classification_report, cross_entropy_loss

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR     = os.path.join("data", "processed", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")

# ─────────────────────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────
LR           = 0.001
MOMENTUM     = 0.9
BATCH_SIZE   = 32
EPOCHS       = 5
PATIENCE     = 5          # early stopping
DROPOUT_RATE = 0.3
SEED         = 42


# ─────────────────────────────────────────────────────────────
# ACTIVATIONS & THEIR DERIVATIVES
# ─────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 where x > 0, else 0."""
    return (x > 0).astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────
# DROPOUT
# ─────────────────────────────────────────────────────────────

def dropout_forward(x: np.ndarray, rate: float,
                    training: bool, rng) -> tuple:
    """
    Inverted dropout:
        - During training: zero out rate% of neurons, scale by 1/(1-rate)
        - During inference: pass through unchanged
    Returns (output, mask)
    """
    if not training or rate == 0.0:
        return x, np.ones_like(x)
    mask = (rng.random(x.shape) > rate).astype(np.float32) / (1.0 - rate)
    return x * mask, mask

def dropout_backward(dout: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return dout * mask


# ─────────────────────────────────────────────────────────────
# EMBEDDING LAYER (lookup + mean pooling)
# ─────────────────────────────────────────────────────────────

def embedding_forward(seq: np.ndarray,
                      E: np.ndarray) -> tuple:
    """
    seq : (N, T)  integer indices
    E   : (V, D)  embedding matrix

    Returns:
        out    : (N, D)  mean-pooled embedding
        cache  : (seq, mask) for backprop
    """
    N, T = seq.shape
    D    = E.shape[1]

    out  = np.zeros((N, D), dtype=np.float32)
    mask = (seq != 0)                       # True where not PAD

    for i in range(N):
        valid = seq[i][mask[i]]             # non-PAD indices
        if len(valid) > 0:
            out[i] = E[valid].mean(axis=0)
        # else: zero vector (all PAD — shouldn't happen)

    return out, (seq, mask)


def embedding_backward(dout: np.ndarray,
                       cache: tuple,
                       V: int) -> np.ndarray:
    """
    dout : (N, D)
    Returns dE : (V, D)
    """
    seq, mask = cache
    N, T      = seq.shape
    D         = dout.shape[1]
    dE        = np.zeros((V, D), dtype=np.float32)

    for i in range(N):
        valid    = seq[i][mask[i]]
        n_valid  = max(len(valid), 1)
        grad     = dout[i] / n_valid        # from mean pooling
        for idx in valid:
            dE[idx] += grad

    return dE


# ─────────────────────────────────────────────────────────────
# DENSE LAYER
# ─────────────────────────────────────────────────────────────

class DenseLayer:
    """Fully-connected layer: out = x @ W + b"""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng     = np.random.default_rng(seed)
        # He initialization (good for ReLU)
        scale   = np.sqrt(2.0 / in_dim)
        self.W  = (rng.standard_normal((in_dim, out_dim)) * scale
                   ).astype(np.float32)
        self.b  = np.zeros(out_dim, dtype=np.float32)

        # Momentum buffers
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return x @ self.W + self.b          # (N, out_dim)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x    = self._cache
        self.dW = x.T @ dout               # (in_dim, out_dim)
        self.db = dout.sum(axis=0)         # (out_dim,)
        dx   = dout @ self.W.T             # (N, in_dim)
        return dx

    def update(self, lr: float, momentum: float) -> None:
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# ─────────────────────────────────────────────────────────────
# ANN MODEL
# ─────────────────────────────────────────────────────────────

class ANN:
    """
    Embedding → Dense(128,ReLU) → Dropout → Dense(64,ReLU) → Dropout → Dense(C,Softmax)
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden1: int, hidden2: int, num_classes: int,
                 embed_matrix: np.ndarray = None,
                 dropout_rate: float = DROPOUT_RATE,
                 seed: int = SEED):
        self.vocab_size    = vocab_size
        self.embed_dim     = embed_dim
        self.dropout_rate  = dropout_rate
        self.rng           = np.random.default_rng(seed)

        # Embedding matrix
        if embed_matrix is not None:
            self.E = embed_matrix.astype(np.float32).copy()
        else:
            rng2   = np.random.default_rng(seed)
            self.E = (rng2.normal(0, 0.1, (vocab_size, embed_dim))
                      .astype(np.float32))
            self.E[0] = 0.0   # PAD

        # Dense layers
        self.d1 = DenseLayer(embed_dim, hidden1, seed=seed)
        self.d2 = DenseLayer(hidden1,  hidden2, seed=seed+1)
        self.d3 = DenseLayer(hidden2,  num_classes, seed=seed+2)

        # Caches for backprop
        self._emb_cache = None
        self._h1 = self._h1_pre = None
        self._m1 = self._m2     = None
        self._h2 = self._h2_pre = None

    # ── FORWARD ────────────────────────────────────────────────

    def forward(self, seq: np.ndarray,
                training: bool = True) -> np.ndarray:
        """
        seq : (N, T)  padded integer indices
        Returns (N, C) softmax probabilities
        """
        # Embedding (lookup + mean pool)
        emb, self._emb_cache = embedding_forward(seq, self.E)

        # Dense 1 + ReLU
        self._h1_pre = self.d1.forward(emb)
        h1           = relu(self._h1_pre)

        # Dropout 1
        h1, self._m1 = dropout_forward(h1, self.dropout_rate,
                                        training, self.rng)
        self._h1 = h1

        # Dense 2 + ReLU
        self._h2_pre = self.d2.forward(h1)
        h2           = relu(self._h2_pre)

        # Dropout 2
        h2, self._m2 = dropout_forward(h2, self.dropout_rate,
                                        training, self.rng)
        self._h2 = h2

        # Output layer + Softmax
        logits = self.d3.forward(h2)
        probs  = softmax(logits)
        return probs

    # ── BACKWARD ───────────────────────────────────────────────

    def backward(self, probs: np.ndarray,
                 y_onehot: np.ndarray) -> None:
        """
        Combined softmax + cross-entropy gradient:
            dL/d_logits = probs - y_onehot   (clean closed-form)
        """
        N = probs.shape[0]

        # Gradient of loss w.r.t. logits (softmax + cross-entropy combined)
        d_logits = (probs - y_onehot) / N        # (N, C)

        # Dense 3 backward
        dh2 = self.d3.backward(d_logits)

        # Dropout 2
        dh2 = dropout_backward(dh2, self._m2)

        # ReLU 2 backward
        dh2_pre = dh2 * relu_grad(self._h2_pre)

        # Dense 2 backward
        dh1 = self.d2.backward(dh2_pre)

        # Dropout 1
        dh1 = dropout_backward(dh1, self._m1)

        # ReLU 1 backward
        dh1_pre = dh1 * relu_grad(self._h1_pre)

        # Dense 1 backward
        demb = self.d1.backward(dh1_pre)

        # Embedding backward (update E)
        dE = embedding_backward(demb, self._emb_cache, self.vocab_size)
        self._dE = dE

    # ── PARAMETER UPDATE ───────────────────────────────────────

    def update(self, lr: float = LR, momentum: float = MOMENTUM) -> None:
        self.d1.update(lr, momentum)
        self.d2.update(lr, momentum)
        self.d3.update(lr, momentum)
        # Embedding gradient update (with momentum would need extra buffer;
        # use simple SGD for embeddings)
        self.E -= lr * self._dE
        self.E[0] = 0.0          # keep PAD row zero

    # ── PREDICT ────────────────────────────────────────────────

    def predict(self, seq: np.ndarray) -> np.ndarray:
        probs = self.forward(seq, training=False)
        return np.argmax(probs, axis=1)

    # ── SAVE / LOAD ────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(path,
                 E   = self.E,
                 W1  = self.d1.W, b1 = self.d1.b,
                 W2  = self.d2.W, b2 = self.d2.b,
                 W3  = self.d3.W, b3 = self.d3.b,
                 cfg = np.array([self.vocab_size, self.embed_dim,
                                  self.d1.W.shape[1], self.d2.W.shape[1],
                                  self.d3.W.shape[1]]))
        print(f"  Model saved: {path}.npz")

    @classmethod
    def load(cls, path: str,
             dropout_rate: float = 0.0) -> "ANN":
        data = np.load(path + ".npz")
        cfg  = data["cfg"].astype(int)
        vocab_size, embed_dim, h1, h2, num_cls = cfg
        model = cls(vocab_size, embed_dim, h1, h2, num_cls,
                    dropout_rate=dropout_rate)
        model.E    = data["E"]
        model.d1.W = data["W1"]; model.d1.b = data["b1"]
        model.d2.W = data["W2"]; model.d2.b = data["b2"]
        model.d3.W = data["W3"]; model.d3.b = data["b3"]
        return model


# ─────────────────────────────────────────────────────────────
# DATA BATCHING
# ─────────────────────────────────────────────────────────────

def iter_batches(X: np.ndarray, y: np.ndarray,
                 batch_size: int, rng):
    N   = X.shape[0]
    idx = rng.permutation(N)
    for start in range(0, N, batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], y[batch]


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_ann(model: ANN,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val:   np.ndarray, y_val:   np.ndarray,
              num_classes: int,
              epochs: int = EPOCHS,
              batch_size: int = BATCH_SIZE,
              lr: float = LR) -> dict:

    rng      = np.random.default_rng(SEED)
    history  = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  []}

    best_val_loss = np.inf
    patience_cnt  = 0
    best_weights  = None

    print(f"\n  Training ANN for up to {epochs} epochs "
          f"(early stop patience={PATIENCE}) …\n")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>10}  "
          f"{'ValLoss':>10}  {'ValAcc':>10}")
    print("  " + "-" * 55)

    for epoch in range(1, epochs + 1):

        # ── Mini-batch training ──
        model_losses, model_accs = [], []
        for X_b, y_b in iter_batches(X_train, y_train, batch_size, rng):
            y_oh   = np.eye(num_classes, dtype=np.float32)[y_b]
            probs  = model.forward(X_b, training=True)
            loss   = cross_entropy_loss(y_oh, probs)
            preds  = np.argmax(probs, axis=1)
            acc    = np.mean(preds == y_b)

            model.backward(probs, y_oh)
            model.update(lr=lr)

            model_losses.append(loss)
            model_accs.append(acc)

        train_loss = np.mean(model_losses)
        train_acc  = np.mean(model_accs)

        # ── Validation ──
        y_val_oh   = np.eye(num_classes, dtype=np.float32)[y_val]
        val_probs  = model.forward(X_val, training=False)
        val_loss   = cross_entropy_loss(y_val_oh, val_probs)
        val_acc    = np.mean(np.argmax(val_probs, axis=1) == y_val)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc*100:>9.2f}%  "
              f"{val_loss:>10.4f}  {val_acc*100:>9.2f}%")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            # Save best weight copies
            best_weights  = {
                "E":  model.E.copy(),
                "W1": model.d1.W.copy(), "b1": model.d1.b.copy(),
                "W2": model.d2.W.copy(), "b2": model.d2.b.copy(),
                "W3": model.d3.W.copy(), "b3": model.d3.b.copy(),
            }
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    # Restore best weights
    if best_weights:
        model.E    = best_weights["E"]
        model.d1.W = best_weights["W1"]; model.d1.b = best_weights["b1"]
        model.d2.W = best_weights["W2"]; model.d2.b = best_weights["b2"]
        model.d3.W = best_weights["W3"]; model.d3.b = best_weights["b3"]
        print("  Best weights restored.")

    return history


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 4: ANN — ARTIFICIAL NEURAL NETWORK (from scratch)")
    print("=" * 55)

    # Load data
    print("\n[1/4] Loading features …")
    feat = load_features()

    X_train = feat["seq_X_train"]
    X_val   = feat["seq_X_val"]
    X_test  = feat["seq_X_test"]
    y_train = feat["y_train"]
    y_val   = feat["y_val"]
    y_test  = feat["y_test"]

    embed_matrix = feat["embed_matrix"]
    vocab_size   = int(feat["vocab_size"])
    embed_dim    = int(feat["embed_dim"])
    num_classes  = int(feat["num_classes"])

    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"  Vocab: {vocab_size}  EmbedDim: {embed_dim}  Classes: {num_classes}")

    # Build model
    print("\n[2/4] Building ANN …")
    model = ANN(
        vocab_size   = vocab_size,
        embed_dim    = embed_dim,
        hidden1      = 128,
        hidden2      = 64,
        num_classes  = num_classes,
        embed_matrix = embed_matrix,
        dropout_rate = DROPOUT_RATE,
        seed         = SEED,
    )

    # Print architecture summary
    params = (vocab_size * embed_dim
              + embed_dim * 128 + 128
              + 128 * 64 + 64
              + 64 * num_classes + num_classes)
    print(f"\n  Architecture:")
    print(f"    Embedding  : ({vocab_size}, {embed_dim})")
    print(f"    Dense 1    : ({embed_dim} → 128) + ReLU + Dropout({DROPOUT_RATE})")
    print(f"    Dense 2    : (128 → 64) + ReLU + Dropout({DROPOUT_RATE})")
    print(f"    Output     : (64 → {num_classes}) + Softmax")
    print(f"  Total trainable parameters: {params:,}\n")

    # Train
    print("[3/4] Training …")
    history = train_ann(
        model, X_train, y_train, X_val, y_val,
        num_classes=num_classes, epochs=EPOCHS,
        batch_size=BATCH_SIZE, lr=LR,
    )

    # Save model
    model.save(os.path.join(MODEL_DIR, "ann"))

    # Evaluate
    print("\n[4/4] Evaluating on test set …")
    y_pred = model.predict(X_test)

    y_true_labels = [classes[i] for i in y_test]
    y_pred_labels = [classes[i] for i in y_pred]

    results = classification_report(
        y_true_labels, y_pred_labels, classes, model_name="ANN"
    )

    # Save results + history
    out_path = os.path.join(MODEL_DIR, "ann_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "accuracy":  results["accuracy"],
            "macro_f1":  results["macro"]["f1"],
            "history":   history,
        }, f, indent=2)
    print(f"  Results saved: {out_path}")

    print("\nANN complete. Run cnn.py next.\n")
    return results


if __name__ == "__main__":
    main()
