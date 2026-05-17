"""
5_cnn.py
Convolutional Neural Network — fully from scratch using NumPy only.

Architecture:
    Embedding  (lookup table)
    Conv1D     (128 filters, kernel=5, ReLU)
    GlobalMaxPool1D
    Dense(64)  + ReLU
    Dropout(0.3)
    Dense(3)   + Softmax

Training: Mini-batch SGD with momentum, manual backprop.
No keras, no torch, no sklearn.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import load_features
from utils.metrics import classification_report, cross_entropy_loss
from ann import (DenseLayer, relu, relu_grad, softmax,
                    dropout_forward, dropout_backward, iter_batches)

PROCESSED_DIR  = os.path.join("data", "processed")
MODEL_DIR      = os.path.join("data", "processed", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")

# ─────────────────────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────
LR           = 0.001
MOMENTUM     = 0.9
BATCH_SIZE   = 32
EPOCHS       = 30
PATIENCE     = 5
DROPOUT_RATE = 0.3
SEED         = 42
NUM_FILTERS  = 64
KERNEL_SIZE  = 5


# ─────────────────────────────────────────────────────────────
# EMBEDDING LAYER  (full lookup, not mean-pooled — needed for Conv1D)
# ─────────────────────────────────────────────────────────────

def embedding_forward_full(seq: np.ndarray,
                            E: np.ndarray) -> tuple:
    """
    seq : (N, T)  integer indices
    E   : (V, D)  embedding matrix

    Returns:
        out   : (N, T, D)   — full sequence of embeddings
        cache : seq
    """
    out = E[seq]          # fancy indexing → (N, T, D)
    return out.astype(np.float32), seq


def embedding_backward_full(dout: np.ndarray,
                             cache: np.ndarray,
                             V: int) -> np.ndarray:
    """
    dout  : (N, T, D)
    cache : seq  (N, T)
    Returns dE : (V, D)
    """
    N, T, D = dout.shape
    dE      = np.zeros((V, D), dtype=np.float32)
    np.add.at(dE, cache, dout)     # accumulate gradients
    return dE


# ─────────────────────────────────────────────────────────────
# CONV1D LAYER  (from scratch)
# ─────────────────────────────────────────────────────────────

class Conv1DLayer:
    """
    1-D convolution over sequence dimension.

    Input  : (N, T, D)   — batch of sequences
    Output : (N, T-K+1, F) — after convolution (valid padding)
    """

    def __init__(self, in_channels: int, num_filters: int,
                 kernel_size: int, seed: int = 0):
        rng     = np.random.default_rng(seed)
        # He init: scale by sqrt(2 / fan_in)
        fan_in  = in_channels * kernel_size
        scale   = np.sqrt(2.0 / fan_in)
        # W shape: (F, K, D)
        self.W  = (rng.standard_normal((num_filters, kernel_size, in_channels))
                   * scale).astype(np.float32)
        self.b  = np.zeros(num_filters, dtype=np.float32)

        # Momentum
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.K  = kernel_size
        self.F  = num_filters
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (N, T, D)
        out: (N, T-K+1, F)
        """
        N, T, D = x.shape
        K, F    = self.K, self.F
        T_out   = T - K + 1

        out = np.zeros((N, T_out, F), dtype=np.float32)

        for f in range(F):
            for k in range(K):
                # x[:, k:k+T_out, :] → (N, T_out, D)
                # self.W[f, k, :]    → (D,)
                out[:, :, f] += x[:, k:k + T_out, :] @ self.W[f, k, :]

        out += self.b[np.newaxis, np.newaxis, :]   # bias broadcast
        self._cache = x
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout : (N, T-K+1, F)
        Returns dx : (N, T, D)
        """
        x       = self._cache
        N, T, D = x.shape
        K, F    = self.K, self.F
        T_out   = dout.shape[1]

        dx  = np.zeros_like(x)
        dW  = np.zeros_like(self.W)
        db  = dout.sum(axis=(0, 1))          # (F,)

        for f in range(F):
            for k in range(K):
                # dW[f,k,:] += sum over N,T_out of dout[:,:,f] * x[:,k:k+T_out,:]
                # dout[:,:,f] shape (N,T_out); x[:,k:k+T_out,:] shape (N,T_out,D)
                dW[f, k, :] += (
                    dout[:, :, f][:, :, np.newaxis] *
                    x[:, k:k + T_out, :]
                ).sum(axis=(0, 1))

                dx[:, k:k + T_out, :] += (
                    dout[:, :, f][:, :, np.newaxis] *
                    self.W[f, k, :][np.newaxis, np.newaxis, :]
                )

        self.dW = dW
        self.db = db
        return dx

    def update(self, lr: float, momentum: float) -> None:
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# ─────────────────────────────────────────────────────────────
# GLOBAL MAX POOLING 1D
# ─────────────────────────────────────────────────────────────

def global_max_pool_forward(x: np.ndarray) -> tuple:
    """
    x   : (N, T, F)
    out : (N, F)   — max over time dimension
    cache : argmax positions for backprop
    """
    out     = x.max(axis=1)                     # (N, F)
    argmax  = x.argmax(axis=1)                  # (N, F) — position of max
    return out, (x.shape, argmax)


def global_max_pool_backward(dout: np.ndarray,
                              cache: tuple) -> np.ndarray:
    """
    dout  : (N, F)
    Returns dx : (N, T, F)
    """
    shape, argmax = cache
    N, T, F       = shape
    dx            = np.zeros(shape, dtype=np.float32)

    for f in range(F):
        for i in range(N):
            dx[i, argmax[i, f], f] = dout[i, f]

    return dx


# ─────────────────────────────────────────────────────────────
# CNN MODEL
# ─────────────────────────────────────────────────────────────

class CNN:
    """
    Embedding → Conv1D(F,K,ReLU) → GlobalMaxPool1D
              → Dense(64,ReLU) → Dropout → Dense(C,Softmax)
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 num_filters: int, kernel_size: int,
                 hidden_dim: int, num_classes: int,
                 embed_matrix: np.ndarray = None,
                 dropout_rate: float = DROPOUT_RATE,
                 seed: int = SEED):

        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.dropout_rate = dropout_rate
        self.rng          = np.random.default_rng(seed)

        # Embedding
        if embed_matrix is not None:
            self.E = embed_matrix.astype(np.float32).copy()
        else:
            rng2   = np.random.default_rng(seed)
            self.E = (rng2.normal(0, 0.1, (vocab_size, embed_dim))
                      .astype(np.float32))
            self.E[0] = 0.0

        # Conv1D
        self.conv = Conv1DLayer(embed_dim, num_filters, kernel_size, seed=seed)

        # Dense layers
        self.d1 = DenseLayer(num_filters, hidden_dim, seed=seed+1)
        self.d2 = DenseLayer(hidden_dim, num_classes, seed=seed+2)

        # Caches
        self._emb_out = self._emb_cache = None
        self._conv_out = self._conv_pre  = None
        self._pool_out = self._pool_cache = None
        self._h1 = self._h1_pre = self._drop_mask = None

    def forward(self, seq: np.ndarray,
                training: bool = True) -> np.ndarray:
        # Embedding lookup (full sequence)
        emb, self._emb_cache = embedding_forward_full(seq, self.E)
        self._emb_out = emb

        # Conv1D + ReLU
        self._conv_pre = self.conv.forward(emb)       # (N, T', F)
        conv_relu      = relu(self._conv_pre)
        self._conv_out = conv_relu

        # Global Max Pooling
        self._pool_out, self._pool_cache = global_max_pool_forward(conv_relu)

        # Dense 1 + ReLU
        self._h1_pre = self.d1.forward(self._pool_out)
        h1           = relu(self._h1_pre)

        # Dropout
        h1, self._drop_mask = dropout_forward(h1, self.dropout_rate,
                                               training, self.rng)
        self._h1 = h1

        # Output + Softmax
        logits = self.d2.forward(h1)
        return softmax(logits)

    def backward(self, probs: np.ndarray,
                 y_onehot: np.ndarray) -> None:
        N = probs.shape[0]

        # Softmax + CE gradient
        d_logits = (probs - y_onehot) / N

        # Dense 2
        dh1 = self.d2.backward(d_logits)

        # Dropout
        dh1 = dropout_backward(dh1, self._drop_mask)

        # ReLU 1
        dh1_pre = dh1 * relu_grad(self._h1_pre)

        # Dense 1
        dpool = self.d1.backward(dh1_pre)

        # Global Max Pool backward
        dconv_relu = global_max_pool_backward(dpool, self._pool_cache)

        # ReLU conv backward
        dconv_pre = dconv_relu * relu_grad(self._conv_pre)

        # Conv1D backward
        demb = self.conv.backward(dconv_pre)

        # Embedding backward
        self._dE = embedding_backward_full(demb, self._emb_cache,
                                            self.vocab_size)

    def update(self, lr: float = LR,
               momentum: float = MOMENTUM) -> None:
        self.conv.update(lr, momentum)
        self.d1.update(lr, momentum)
        self.d2.update(lr, momentum)
        self.E -= lr * self._dE
        self.E[0] = 0.0

    def predict(self, seq: np.ndarray) -> np.ndarray:
        probs = self.forward(seq, training=False)
        return np.argmax(probs, axis=1)

    def save(self, path: str) -> None:
        np.savez(path,
                 E      = self.E,
                 conv_W = self.conv.W, conv_b = self.conv.b,
                 W1     = self.d1.W,   b1     = self.d1.b,
                 W2     = self.d2.W,   b2     = self.d2.b,
                 cfg    = np.array([self.vocab_size, self.embed_dim,
                                     self.conv.F, self.conv.K,
                                     self.d1.W.shape[1],
                                     self.d2.W.shape[1]]))
        print(f"  Model saved: {path}.npz")

    @classmethod
    def load(cls, path: str, dropout_rate: float = 0.0) -> "CNN":
        data = np.load(path + ".npz")
        cfg  = data["cfg"].astype(int)
        v, d, F, K, h, c = cfg
        model = cls(v, d, F, K, h, c, dropout_rate=dropout_rate)
        model.E        = data["E"]
        model.conv.W   = data["conv_W"]; model.conv.b = data["conv_b"]
        model.d1.W     = data["W1"];     model.d1.b   = data["b1"]
        model.d2.W     = data["W2"];     model.d2.b   = data["b2"]
        return model


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_cnn(model: CNN,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val:   np.ndarray, y_val:   np.ndarray,
              num_classes: int,
              epochs: int = EPOCHS,
              batch_size: int = BATCH_SIZE,
              lr: float = LR) -> dict:

    rng     = np.random.default_rng(SEED)
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}

    best_val_loss = np.inf
    patience_cnt  = 0
    best_weights  = None

    print(f"\n  Training CNN for up to {epochs} epochs "
          f"(early stop patience={PATIENCE}) …\n")
    print(f"  {'Epoch':>5}  {'TrainLoss':>10}  {'TrainAcc':>10}  "
          f"{'ValLoss':>10}  {'ValAcc':>10}")
    print("  " + "-" * 55)

    for epoch in range(1, epochs + 1):

        batch_losses, batch_accs = [], []
        for X_b, y_b in iter_batches(X_train, y_train, batch_size, rng):
            y_oh  = np.eye(num_classes, dtype=np.float32)[y_b]
            probs = model.forward(X_b, training=True)
            loss  = cross_entropy_loss(y_oh, probs)
            preds = np.argmax(probs, axis=1)
            acc   = np.mean(preds == y_b)

            model.backward(probs, y_oh)
            model.update(lr=lr)

            batch_losses.append(loss)
            batch_accs.append(acc)

        train_loss = np.mean(batch_losses)
        train_acc  = np.mean(batch_accs)

        y_val_oh  = np.eye(num_classes, dtype=np.float32)[y_val]
        val_probs = model.forward(X_val, training=False)
        val_loss  = cross_entropy_loss(y_val_oh, val_probs)
        val_acc   = np.mean(np.argmax(val_probs, axis=1) == y_val)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc*100:>9.2f}%  "
              f"{val_loss:>10.4f}  {val_acc*100:>9.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_weights  = {
                "E":      model.E.copy(),
                "conv_W": model.conv.W.copy(),
                "conv_b": model.conv.b.copy(),
                "W1":     model.d1.W.copy(), "b1": model.d1.b.copy(),
                "W2":     model.d2.W.copy(), "b2": model.d2.b.copy(),
            }
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    if best_weights:
        model.E        = best_weights["E"]
        model.conv.W   = best_weights["conv_W"]
        model.conv.b   = best_weights["conv_b"]
        model.d1.W     = best_weights["W1"]; model.d1.b = best_weights["b1"]
        model.d2.W     = best_weights["W2"]; model.d2.b = best_weights["b2"]
        print("  Best weights restored.")

    return history


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 5: CNN — 1D CONVOLUTIONAL NETWORK (from scratch)")
    print("=" * 55)

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

    print("\n[2/4] Building CNN …")
    model = CNN(
        vocab_size   = vocab_size,
        embed_dim    = embed_dim,
        num_filters  = NUM_FILTERS,
        kernel_size  = KERNEL_SIZE,
        hidden_dim   = 64,
        num_classes  = num_classes,
        embed_matrix = embed_matrix,
        dropout_rate = DROPOUT_RATE,
        seed         = SEED,
    )

    T_out  = 200 - KERNEL_SIZE + 1
    params = (vocab_size * embed_dim
              + NUM_FILTERS * KERNEL_SIZE * embed_dim + NUM_FILTERS
              + NUM_FILTERS * 64 + 64
              + 64 * num_classes + num_classes)
    print(f"\n  Architecture:")
    print(f"    Embedding  : ({vocab_size}, {embed_dim})")
    print(f"    Conv1D     : filters={NUM_FILTERS}, kernel={KERNEL_SIZE} → (N,{T_out},{NUM_FILTERS})")
    print(f"    MaxPool    : (N,{T_out},{NUM_FILTERS}) → (N,{NUM_FILTERS})")
    print(f"    Dense 1    : ({NUM_FILTERS} → 64) + ReLU + Dropout({DROPOUT_RATE})")
    print(f"    Output     : (64 → {num_classes}) + Softmax")
    print(f"  Total trainable parameters: {params:,}\n")

    print("[3/4] Training …")
    history = train_cnn(
        model, X_train, y_train, X_val, y_val,
        num_classes=num_classes, epochs=EPOCHS,
        batch_size=BATCH_SIZE, lr=LR,
    )

    model.save(os.path.join(MODEL_DIR, "cnn"))

    print("\n[4/4] Evaluating on test set …")
    y_pred = model.predict(X_test)

    y_true_labels = [classes[i] for i in y_test]
    y_pred_labels = [classes[i] for i in y_pred]

    results = classification_report(
        y_true_labels, y_pred_labels, classes, model_name="CNN"
    )

    out_path = os.path.join(MODEL_DIR, "cnn_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": results["accuracy"],
            "macro_f1": results["macro"]["f1"],
            "history":  history,
        }, f, indent=2)
    print(f"  Results saved: {out_path}")

    print("\nCNN complete. Run train_test.py next.\n")
    return results


if __name__ == "__main__":
    main()
