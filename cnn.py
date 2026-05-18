"""
5_cnn.py
TextCNN — fully from scratch using NumPy only.
Target accuracy: 80%+  |  Fast: vectorized Conv1D (no Python loops)

Architecture (TextCNN — multi-kernel):
    Embedding  (lookup table, 100-dim)
    4 parallel Conv1D branches: kernel=2,3,4,5
    Each branch: Conv1D(128 filters) → ReLU → GlobalMaxPool1D
    Concatenate all branch outputs  → (N, 512)
    LayerNorm
    Dense(256) + ReLU + Dropout(0.5)
    Dense(128) + ReLU + Dropout(0.3)
    Dense(3)   + Softmax

Speed fix: Conv1D uses im2col (matrix reshape trick) — no Python loops.
Training:  Adam optimizer + cosine LR decay + early stopping
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import load_features
from utils.metrics import classification_report, cross_entropy_loss

PROCESSED_DIR  = os.path.join("data", "processed")
MODEL_DIR      = os.path.join("data", "processed", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")

# ─────────────────────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────────────────────
LR           = 0.001
BETA1        = 0.9
BETA2        = 0.999
EPSILON      = 1e-8
L2_LAMBDA    = 1e-4
BATCH_SIZE   = 64
EPOCHS       = 50
PATIENCE     = 8
DROPOUT_1    = 0.5
DROPOUT_2    = 0.3
SEED         = 42
NUM_FILTERS  = 128
KERNEL_SIZES = [2, 3, 4, 5]
EMBED_DIM    = 100


# ─────────────────────────────────────────────────────────────
# ACTIVATIONS
# ─────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────
# DROPOUT
# ─────────────────────────────────────────────────────────────

def dropout_forward(x, rate, training, rng):
    if not training or rate == 0.0:
        return x, np.ones_like(x)
    mask = (rng.random(x.shape) > rate).astype(np.float32) / (1.0 - rate)
    return x * mask, mask

def dropout_backward(dout, mask):
    return dout * mask


# ─────────────────────────────────────────────────────────────
# ADAM OPTIMIZER
# ─────────────────────────────────────────────────────────────

class AdamState:
    def __init__(self, shape):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def update(self, param, grad, lr, l2=0.0):
        self.t += 1
        grad   = grad + l2 * param
        self.m = BETA1 * self.m + (1 - BETA1) * grad
        self.v = BETA2 * self.v + (1 - BETA2) * grad ** 2
        m_hat  = self.m / (1 - BETA1 ** self.t)
        v_hat  = self.v / (1 - BETA2 ** self.t)
        return param - lr * m_hat / (np.sqrt(v_hat) + EPSILON)


# ─────────────────────────────────────────────────────────────
# DENSE LAYER
# ─────────────────────────────────────────────────────────────

class DenseLayer:
    def __init__(self, in_dim, out_dim, seed=0):
        rng    = np.random.default_rng(seed)
        scale  = np.sqrt(2.0 / in_dim)
        self.W = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.adamW = AdamState(self.W.shape)
        self.adamb = AdamState(self.b.shape)
        self._cache = None

    def forward(self, x):
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._cache.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def update(self, lr, l2=L2_LAMBDA):
        self.W = self.adamW.update(self.W, self.dW, lr, l2=l2)
        self.b = self.adamb.update(self.b, self.db, lr, l2=0.0)


# ─────────────────────────────────────────────────────────────
# LAYER NORMALIZATION
# ─────────────────────────────────────────────────────────────

class LayerNorm:
    def __init__(self, dim):
        self.gamma  = np.ones(dim,  dtype=np.float32)
        self.beta   = np.zeros(dim, dtype=np.float32)
        self._cache = None

    def forward(self, x, training=True):
        mu   = x.mean(axis=1, keepdims=True)
        var  = x.var(axis=1,  keepdims=True) + 1e-8
        xhat = (x - mu) / np.sqrt(var)
        out  = self.gamma * xhat + self.beta
        self._cache = (xhat, var)
        return out

    def backward(self, dout):
        xhat, var   = self._cache
        self.dgamma = (dout * xhat).sum(axis=0)
        self.dbeta  = dout.sum(axis=0)
        return dout * self.gamma / np.sqrt(var)

    def update(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta


# ─────────────────────────────────────────────────────────────
# BATCH ITERATOR
# ─────────────────────────────────────────────────────────────

def iter_batches(X, y, batch_size, rng):
    idx = rng.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


# ─────────────────────────────────────────────────────────────
# EMBEDDING LAYER
# ─────────────────────────────────────────────────────────────

def embedding_forward(seq, E):
    return E[seq].astype(np.float32), seq

def embedding_backward(dout, cache, V):
    dE = np.zeros((V, dout.shape[2]), dtype=np.float32)
    np.add.at(dE, cache, dout)
    return dE


# ─────────────────────────────────────────────────────────────
# IM2COL  —  the key speed trick
# ─────────────────────────────────────────────────────────────
# Instead of looping over kernel positions, we reshape the input
# so that one big matrix multiply does ALL positions at once.
#
# x        : (N, T, D)
# im2col   : (N, T_out, K*D)   — each row = one window flattened
# W_flat   : (K*D, F)          — filters reshaped
# output   : (N, T_out, F)     — single matmul, no loops!

def im2col(x, K):
    """
    x   : (N, T, D)
    K   : kernel size
    out : (N, T-K+1, K*D)
    """
    N, T, D = x.shape
    T_out   = T - K + 1
    # Use stride tricks for zero-copy view
    shape   = (N, T_out, K, D)
    strides = (x.strides[0], x.strides[1], x.strides[1], x.strides[2])
    windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    # Reshape to (N, T_out, K*D)
    return windows.reshape(N, T_out, K * D).astype(np.float32)


def col2im(dcol, x_shape, K):
    """
    Reverse of im2col — scatter gradients back.
    dcol    : (N, T_out, K*D)
    x_shape : (N, T, D)
    returns   (N, T, D)
    """
    N, T, D = x_shape
    T_out   = T - K + 1
    dx      = np.zeros(x_shape, dtype=np.float32)
    dcol_   = dcol.reshape(N, T_out, K, D)
    for k in range(K):
        dx[:, k:k + T_out, :] += dcol_[:, :, k, :]
    return dx


# ─────────────────────────────────────────────────────────────
# CONV1D LAYER  —  vectorized, no Python loops
# ─────────────────────────────────────────────────────────────

class Conv1DLayer:
    """
    Fast Conv1D using im2col trick.
    Forward:  one matmul instead of F×K loops.
    Backward: two matmuls instead of F×K loops.
    ~20-50x faster than the loop version.
    """

    def __init__(self, in_channels, num_filters, kernel_size, seed=0):
        rng      = np.random.default_rng(seed)
        fan_in   = in_channels * kernel_size
        scale    = np.sqrt(2.0 / fan_in)
        # W shape: (K*D, F)  — already flattened for fast matmul
        self.W   = (rng.standard_normal((kernel_size * in_channels, num_filters))
                    * scale).astype(np.float32)
        self.b   = np.zeros(num_filters, dtype=np.float32)
        self.adamW = AdamState(self.W.shape)
        self.adamb = AdamState(self.b.shape)
        self.K   = kernel_size
        self.F   = num_filters
        self.D   = in_channels
        self._col_cache  = None
        self._x_shape    = None

    def forward(self, x):
        """
        x   : (N, T, D)
        out : (N, T-K+1, F)

        All computation = one matmul: col @ W + b
        """
        col          = im2col(x, self.K)        # (N, T_out, K*D)
        self._col_cache = col
        self._x_shape   = x.shape

        N, T_out, KD = col.shape
        # Reshape to (N*T_out, K*D) for batched matmul
        col_2d = col.reshape(N * T_out, KD)
        out_2d = col_2d @ self.W + self.b       # (N*T_out, F)
        return out_2d.reshape(N, T_out, self.F)

    def backward(self, dout):
        """
        dout : (N, T-K+1, F)
        Returns dx : (N, T, D)

        All computation = two matmuls.
        """
        col   = self._col_cache
        N, T_out, KD = col.shape

        dout_2d = dout.reshape(N * T_out, self.F)   # (N*T_out, F)
        col_2d  = col.reshape(N * T_out, KD)        # (N*T_out, K*D)

        # Gradient w.r.t. weights: (K*D, N*T_out) @ (N*T_out, F) = (K*D, F)
        self.dW = col_2d.T @ dout_2d
        self.db = dout_2d.sum(axis=0)

        # Gradient w.r.t. input columns: (N*T_out, F) @ (F, K*D) = (N*T_out, K*D)
        dcol_2d = dout_2d @ self.W.T
        dcol    = dcol_2d.reshape(N, T_out, KD)

        # Convert column gradients back to input shape
        dx = col2im(dcol, self._x_shape, self.K)
        return dx

    def update(self, lr):
        self.W = self.adamW.update(self.W, self.dW, lr, l2=0.0)
        self.b = self.adamb.update(self.b, self.db, lr, l2=0.0)


# ─────────────────────────────────────────────────────────────
# GLOBAL MAX POOLING 1D
# ─────────────────────────────────────────────────────────────

def global_max_pool_forward(x):
    out    = x.max(axis=1)                  # (N, F)
    argmax = x.argmax(axis=1)              # (N, F)
    return out, (x.shape, argmax)

def global_max_pool_backward(dout, cache):
    """Vectorized — no Python loops."""
    shape, argmax = cache
    N, T, F = shape
    # Build flat index and scatter
    dx     = np.zeros(shape, dtype=np.float32)
    n_idx  = np.arange(N)[:, None]         # (N, 1)
    f_idx  = np.arange(F)[None, :]         # (1, F)
    dx[n_idx, argmax, f_idx] = dout        # vectorized scatter
    return dx


# ─────────────────────────────────────────────────────────────
# TEXT CNN
# ─────────────────────────────────────────────────────────────

class TextCNN:
    def __init__(self, vocab_size, embed_dim, num_filters,
                 kernel_sizes, num_classes, embed_matrix=None, seed=SEED):

        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.num_filters  = num_filters
        self.kernel_sizes = kernel_sizes
        self.rng          = np.random.default_rng(seed)

        # Embedding
        rng2   = np.random.default_rng(seed)
        self.E = (rng2.normal(0, 0.05, (vocab_size, embed_dim))
                  .astype(np.float32))
        self.E[0] = 0.0
        self.adamE = AdamState(self.E.shape)

        # Conv branches
        self.convs = [
            Conv1DLayer(embed_dim, num_filters, k, seed=seed + i)
            for i, k in enumerate(kernel_sizes)
        ]

        concat_dim = num_filters * len(kernel_sizes)
        self.ln = LayerNorm(concat_dim)
        self.d1 = DenseLayer(concat_dim, 256, seed=seed + 10)
        self.d2 = DenseLayer(256,        128, seed=seed + 11)
        self.d3 = DenseLayer(128, num_classes, seed=seed + 12)

        self._emb_cache   = None
        self._conv_pres   = []
        self._pool_caches = []
        self._concat      = None
        self._ln_out      = None
        self._h1_pre = self._h1 = self._m1 = None
        self._h2_pre = self._h2 = self._m2 = None

    def forward(self, seq, training=True):
        emb, self._emb_cache = embedding_forward(seq, self.E)

        branch_pools      = []
        self._conv_pres   = []
        self._pool_caches = []

        for conv in self.convs:
            conv_pre  = conv.forward(emb)
            conv_relu = relu(conv_pre)
            pool_out, pool_cache = global_max_pool_forward(conv_relu)
            self._conv_pres.append(conv_pre)
            self._pool_caches.append(pool_cache)
            branch_pools.append(pool_out)

        self._concat = np.concatenate(branch_pools, axis=1)
        self._ln_out = self.ln.forward(self._concat, training=training)

        self._h1_pre = self.d1.forward(self._ln_out)
        h1           = relu(self._h1_pre)
        h1, self._m1 = dropout_forward(h1, DROPOUT_1, training, self.rng)
        self._h1     = h1

        self._h2_pre = self.d2.forward(h1)
        h2           = relu(self._h2_pre)
        h2, self._m2 = dropout_forward(h2, DROPOUT_2, training, self.rng)
        self._h2     = h2

        return softmax(self.d3.forward(h2))

    def backward(self, probs, y_onehot):
        N        = probs.shape[0]
        d_logits = (probs - y_onehot) / N

        dh2      = self.d3.backward(d_logits)
        dh2      = dropout_backward(dh2, self._m2)
        dh2_pre  = dh2 * relu_grad(self._h2_pre)

        dh1      = self.d2.backward(dh2_pre)
        dh1      = dropout_backward(dh1, self._m1)
        dh1_pre  = dh1 * relu_grad(self._h1_pre)

        d_ln     = self.d1.backward(dh1_pre)
        d_concat = self.ln.backward(d_ln)

        F          = self.num_filters
        emb_shape  = self.E[self._emb_cache].shape
        demb_total = np.zeros(emb_shape, dtype=np.float32)

        for i, conv in enumerate(self.convs):
            d_branch    = d_concat[:, i * F:(i + 1) * F]
            d_pool      = global_max_pool_backward(d_branch, self._pool_caches[i])
            d_conv_relu = d_pool * relu_grad(self._conv_pres[i])
            demb_i      = conv.backward(d_conv_relu)
            demb_total += demb_i

        self._dE = embedding_backward(demb_total, self._emb_cache, self.vocab_size)

    def update(self, lr):
        for conv in self.convs:
            conv.update(lr)
        self.ln.update(lr)
        self.d1.update(lr)
        self.d2.update(lr)
        self.d3.update(lr)
        self.E = self.adamE.update(self.E, self._dE, lr, l2=0.0)
        self.E[0] = 0.0

    def predict(self, seq):
        return np.argmax(self.forward(seq, training=False), axis=1)

    def save(self, path):
        save_dict = {
            "E": self.E,
            "ln_gamma": self.ln.gamma,
            "ln_beta":  self.ln.beta,
        }
        for i, conv in enumerate(self.convs):
            save_dict[f"conv{i}_W"] = conv.W
            save_dict[f"conv{i}_b"] = conv.b
        save_dict.update({
            "W1": self.d1.W, "b1": self.d1.b,
            "W2": self.d2.W, "b2": self.d2.b,
            "W3": self.d3.W, "b3": self.d3.b,
            "cfg": np.array([self.vocab_size, self.embed_dim,
                              self.num_filters, self.d3.W.shape[1],
                              len(self.kernel_sizes)] + self.kernel_sizes),
        })
        np.savez(path, **save_dict)
        print(f"  Model saved: {path}.npz")

    @classmethod
    def load(cls, path):
        data = np.load(path + ".npz")
        cfg  = data["cfg"].astype(int)
        vocab_size, embed_dim, num_filters, num_classes, n_kernels = cfg[:5]
        ks   = list(cfg[5:5 + n_kernels])
        model = cls(vocab_size, embed_dim, num_filters, ks, num_classes)
        model.E          = data["E"]
        model.ln.gamma   = data["ln_gamma"]
        model.ln.beta    = data["ln_beta"]
        for i in range(n_kernels):
            model.convs[i].W = data[f"conv{i}_W"]
            model.convs[i].b = data[f"conv{i}_b"]
        model.d1.W = data["W1"]; model.d1.b = data["b1"]
        model.d2.W = data["W2"]; model.d2.b = data["b2"]
        model.d3.W = data["W3"]; model.d3.b = data["b3"]
        return model


# ─────────────────────────────────────────────────────────────
# COSINE LR SCHEDULE
# ─────────────────────────────────────────────────────────────

def cosine_lr(epoch, total_epochs, lr_max, lr_min=1e-6, warmup=3):
    if epoch <= warmup:
        return lr_max * epoch / warmup
    progress = (epoch - warmup) / (total_epochs - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_cnn(model, X_train, y_train, X_val, y_val,
              num_classes, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):

    rng           = np.random.default_rng(SEED)
    history       = {"train_loss": [], "val_loss": [],
                     "train_acc":  [], "val_acc":  []}
    best_val_acc  = 0.0
    patience_cnt  = 0
    best_weights  = None

    print(f"\n  Training TextCNN — Adam + Cosine LR + im2col Conv")
    print(f"  Epochs={epochs}  Batch={batch_size}  "
          f"Filters={NUM_FILTERS}  Kernels={KERNEL_SIZES}")
    print(f"  Patience={PATIENCE}  L2={L2_LAMBDA}\n")
    print(f"  {'Epoch':>5}  {'LR':>8}  {'TrainLoss':>10}  "
          f"{'TrainAcc':>9}  {'ValLoss':>10}  {'ValAcc':>9}")
    print("  " + "-" * 65)

    for epoch in range(1, epochs + 1):

        current_lr   = cosine_lr(epoch, epochs, lr)
        batch_losses = []
        batch_accs   = []

        for X_b, y_b in iter_batches(X_train, y_train, batch_size, rng):
            y_oh  = np.eye(num_classes, dtype=np.float32)[y_b]
            probs = model.forward(X_b, training=True)
            loss  = cross_entropy_loss(y_oh, probs)
            acc   = np.mean(np.argmax(probs, axis=1) == y_b)
            model.backward(probs, y_oh)
            model.update(lr=current_lr)
            batch_losses.append(loss)
            batch_accs.append(acc)

        train_loss = float(np.mean(batch_losses))
        train_acc  = float(np.mean(batch_accs))

        # Validation in mini-batches
        val_probs_list = []
        for start in range(0, len(X_val), 512):
            val_probs_list.append(
                model.forward(X_val[start:start + 512], training=False)
            )
        val_probs = np.vstack(val_probs_list)
        y_val_oh  = np.eye(num_classes, dtype=np.float32)[y_val]
        val_loss  = float(cross_entropy_loss(y_val_oh, val_probs))
        val_acc   = float(np.mean(np.argmax(val_probs, axis=1) == y_val))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  {epoch:>5}  {current_lr:>8.6f}  {train_loss:>10.4f}  "
              f"{train_acc*100:>8.2f}%  {val_loss:>10.4f}  {val_acc*100:>8.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            best_weights = {
                "E":        model.E.copy(),
                "ln_gamma": model.ln.gamma.copy(),
                "ln_beta":  model.ln.beta.copy(),
                "W1": model.d1.W.copy(), "b1": model.d1.b.copy(),
                "W2": model.d2.W.copy(), "b2": model.d2.b.copy(),
                "W3": model.d3.W.copy(), "b3": model.d3.b.copy(),
            }
            for i, conv in enumerate(model.convs):
                best_weights[f"conv{i}_W"] = conv.W.copy()
                best_weights[f"conv{i}_b"] = conv.b.copy()
            print(f"  ★ New best val_acc: {val_acc*100:.2f}%")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. "
                      f"Best val_acc={best_val_acc*100:.2f}%")
                break

    # Restore best weights
    if best_weights:
        model.E        = best_weights["E"]
        model.ln.gamma = best_weights["ln_gamma"]
        model.ln.beta  = best_weights["ln_beta"]
        model.d1.W = best_weights["W1"]; model.d1.b = best_weights["b1"]
        model.d2.W = best_weights["W2"]; model.d2.b = best_weights["b2"]
        model.d3.W = best_weights["W3"]; model.d3.b = best_weights["b3"]
        for i, conv in enumerate(model.convs):
            conv.W = best_weights[f"conv{i}_W"]
            conv.b = best_weights[f"conv{i}_b"]
        print(f"  Best weights restored (val_acc={best_val_acc*100:.2f}%)")

    return history


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 3: TextCNN — VECTORIZED (im2col, no loops)")
    print("  Optimizer: Adam | Schedule: Cosine LR | L2 Reg")
    print("=" * 60)

    print("\n[1/4] Loading features …")
    feat = load_features()

    X_train      = feat["seq_X_train"]
    X_val        = feat["seq_X_val"]
    X_test       = feat["seq_X_test"]
    y_train      = feat["y_train"]
    y_val        = feat["y_val"]
    y_test       = feat["y_test"]
    vocab_size   = int(feat["vocab_size"])
    num_classes  = int(feat["num_classes"])

    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    print("\n[2/4] Building TextCNN …")
    model = TextCNN(
        vocab_size   = vocab_size,
        embed_dim    = EMBED_DIM,
        num_filters  = NUM_FILTERS,
        kernel_sizes = KERNEL_SIZES,
        num_classes  = num_classes,
        seed         = SEED,
    )

    concat_dim = NUM_FILTERS * len(KERNEL_SIZES)
    print(f"\n  Architecture:")
    print(f"    Embedding    : ({vocab_size:,}, {EMBED_DIM})  [trained from scratch]")
    for k in KERNEL_SIZES:
        print(f"    Conv1D(k={k})  : (N,200,{EMBED_DIM}) → (N,{200-k+1},{NUM_FILTERS}) "
              f"→ MaxPool → (N,{NUM_FILTERS})  [im2col]")
    print(f"    Concatenate  : (N, {concat_dim})")
    print(f"    LayerNorm    : (N, {concat_dim})")
    print(f"    Dense 1      : {concat_dim} → 256  + ReLU + Dropout({DROPOUT_1})")
    print(f"    Dense 2      : 256 → 128  + ReLU + Dropout({DROPOUT_2})")
    print(f"    Output       : 128 → {num_classes}  + Softmax")
    print(f"\n  Speed: im2col replaces all Python loops in Conv1D (~20-50x faster)")
    print(f"  LR={LR}  Batch={BATCH_SIZE}  Epochs={EPOCHS}  Patience={PATIENCE}\n")

    print("[3/4] Training …")
    history = train_cnn(
        model, X_train, y_train, X_val, y_val,
        num_classes=num_classes, epochs=EPOCHS,
        batch_size=BATCH_SIZE, lr=LR,
    )

    model.save(os.path.join(MODEL_DIR, "cnn"))

    print("\n[4/4] Evaluating on test set …")
    preds_list = []
    for start in range(0, len(X_test), 512):
        preds_list.append(model.predict(X_test[start:start + 512]))
    y_pred = np.concatenate(preds_list)

    y_true_labels = [classes[i] for i in y_test]
    y_pred_labels = [classes[i] for i in y_pred]

    results = classification_report(
        y_true_labels, y_pred_labels, classes, model_name="TextCNN"
    )

    out_path = os.path.join(MODEL_DIR, "cnn_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "accuracy": results["accuracy"],
            "macro_f1": results["macro"]["f1"],
            "history":  history,
        }, f, indent=2)
    print(f"  Results saved: {out_path}")
    return results


if __name__ == "__main__":
    main()