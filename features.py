"""
2_features.py
Feature extraction from scratch:
  - Vocabulary builder
  - Bag-of-Words (BoW) with TF-IDF weighting
  - Word Embedding matrix (random init, trained later inside ANN/CNN)
  - Train/test split (stratified)

No sklearn, no keras — pure Python + NumPy only.
"""

import os
import sys
import json
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_processed, PROCESSED_DIR
# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
FEATURES_OUT  = os.path.join(PROCESSED_DIR, "features.npz")
VOCAB_OUT     = os.path.join(PROCESSED_DIR, "vocab.json")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
MAX_VOCAB  = 10_000   # keep top-N most frequent words
MAX_LEN    = 200      # max tokens per sample (for embedding input)
EMBED_DIM  = 300      # embedding vector size
TEST_RATIO = 0.20
VAL_RATIO  = 0.10     # of training set
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────
# 1. VOCABULARY
# ─────────────────────────────────────────────────────────────

class Vocabulary:
    """Build and manage word-to-index mappings."""

    PAD_TOKEN = "<PAD>"   # index 0
    UNK_TOKEN = "<OOV>"   # index 1

    def __init__(self, max_size: int = MAX_VOCAB):
        self.max_size = max_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self._built   = False

    def build(self, tokenized_texts: list) -> None:
        """Count frequencies and assign indices to top-N words."""
        for tokens in tokenized_texts:
            self.word_freq.update(tokens)

        # Special tokens first
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        # Add top-N frequent words
        for word, _ in self.word_freq.most_common(self.max_size - 2):
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._built   = True
        print(f"  Vocabulary size : {len(self.word2idx)} tokens")

    def encode(self, tokens: list) -> list:
        """Convert token list to index list; unknown tokens → UNK."""
        assert self._built, "Call build() first."
        return [self.word2idx.get(t, 1) for t in tokens]   # 1 = UNK

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx,
                       "idx2word": {str(k): v for k, v in self.idx2word.items()}},
                      f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.word_freq = Counter()
        vocab._built   = True
        return vocab

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────
# 2. BAG-OF-WORDS WITH TF-IDF  (from scratch)
# ─────────────────────────────────────────────────────────────

def build_tfidf(tokenized_texts: list,
                vocab: Vocabulary) -> np.ndarray:
    """
    Compute TF-IDF matrix from scratch.

    TF(t,d)  = count(t in d) / total_tokens(d)
    IDF(t)   = log( N / (1 + df(t)) )   [smoothed]
    TFIDF    = TF * IDF

    Returns: (N, V) float32 matrix — N samples, V vocab size
    """
    V = len(vocab)
    N = len(tokenized_texts)

    # ── Term-frequency matrix ──
    tf_matrix = np.zeros((N, V), dtype=np.float32)

    for i, tokens in enumerate(tokenized_texts):
        indices = vocab.encode(tokens)
        total   = max(len(indices), 1)
        for idx in indices:
            tf_matrix[i, idx] += 1.0
        tf_matrix[i] /= total          # normalize by doc length

    # ── Document frequency ──
    df = (tf_matrix > 0).sum(axis=0).astype(np.float32)  # shape (V,)

    # ── IDF (smoothed) ──
    idf = np.log((N + 1) / (df + 1)) + 1.0   # shape (V,)

    # ── TF-IDF ──
    tfidf = tf_matrix * idf[np.newaxis, :]    # broadcast

    # ── L2 normalize each row ──
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf /= norms

    print(f"  TF-IDF matrix   : {tfidf.shape}")
    return tfidf


# ─────────────────────────────────────────────────────────────
# 3. SEQUENCE PADDING  (for ANN / CNN embedding input)
# ─────────────────────────────────────────────────────────────

def texts_to_padded_sequences(tokenized_texts: list,
                               vocab: Vocabulary,
                               max_len: int = MAX_LEN) -> np.ndarray:
    """
    Encode each text as integer indices then pad/truncate to max_len.
    Padding token index = 0 (PAD_TOKEN).

    Returns: (N, max_len) int32 array
    """
    N   = len(tokenized_texts)
    out = np.zeros((N, max_len), dtype=np.int32)

    for i, tokens in enumerate(tokenized_texts):
        indices = vocab.encode(tokens)[:max_len]   # truncate
        out[i, :len(indices)] = indices            # left-fill; rest stays 0

    print(f"  Sequence matrix : {out.shape}")
    return out


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / VAL / TEST SPLIT  (stratified, from scratch)
# ─────────────────────────────────────────────────────────────

def stratified_split(X, y, test_ratio: float = TEST_RATIO,
                     val_ratio: float = VAL_RATIO,
                     seed: int = RANDOM_SEED) -> dict:
    """
    Stratified split into train / val / test.
    Works on any X (array rows) and y (integer labels).

    Returns dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(y)

    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        idx  = np.where(y == cls)[0]
        idx  = rng.permutation(idx)

        n_test = max(1, int(len(idx) * test_ratio))
        n_val  = max(1, int(len(idx) * val_ratio))

        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    # Shuffle within each split
    train_idx = list(rng.permutation(train_idx))
    val_idx   = list(rng.permutation(val_idx))
    test_idx  = list(rng.permutation(test_idx))

    def _index(arr, idx):
        if isinstance(arr, np.ndarray):
            return arr[idx]
        return [arr[i] for i in idx]

    return {
        "X_train": _index(X, train_idx),
        "X_val":   _index(X, val_idx),
        "X_test":  _index(X, test_idx),
        "y_train": y[train_idx],
        "y_val":   y[val_idx],
        "y_test":  y[test_idx],
    }


# ─────────────────────────────────────────────────────────────
# 5. ONE-HOT ENCODING
# ─────────────────────────────────────────────────────────────

def to_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer label array to one-hot matrix."""
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


# ─────────────────────────────────────────────────────────────
# 6. RANDOM EMBEDDING MATRIX
# ─────────────────────────────────────────────────────────────

def init_embedding_matrix(vocab_size: int,
                           embed_dim: int = EMBED_DIM,
                           seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Initialize embedding matrix with small random values.
    Row 0 (PAD) = zeros.
    Shape: (vocab_size, embed_dim)
    """
    rng = np.random.default_rng(seed)
    E   = rng.normal(0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
    E[0] = 0.0   # PAD token → zero vector
    print(f"  Embedding matrix: {E.shape}")
    return E


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 2: FEATURE EXTRACTION")
    print("=" * 55)

    # Load preprocessed data
    print("\n[1/5] Loading processed data …")
    texts, label_ints, classes = load_processed()
    num_classes = len(classes)
    print(f"  Samples  : {len(texts)}")
    print(f"  Classes  : {classes}")

    # Tokenize (texts are already normalized strings)
    tokenized = [t.split() for t in texts]

    # Build vocabulary
    print("\n[2/5] Building vocabulary …")
    vocab = Vocabulary(max_size=MAX_VOCAB)
    vocab.build(tokenized)
    vocab.save(VOCAB_OUT)
    print(f"  Saved: {VOCAB_OUT}")

    # TF-IDF features (for Naive Bayes)
    print("\n[3/5] Computing TF-IDF features …")
    tfidf_matrix = build_tfidf(tokenized, vocab)

    # Padded sequences (for ANN / CNN)
    print("\n[4/5] Building padded sequences …")
    seq_matrix = texts_to_padded_sequences(tokenized, vocab, max_len=MAX_LEN)

    # Embedding matrix (random init — updated during ANN/CNN training)
    embed_matrix = init_embedding_matrix(len(vocab), embed_dim=EMBED_DIM)

    # Stratified split
    print("\n[5/5] Splitting data …")
    label_arr = np.array(label_ints, dtype=np.int32)

    splits_tfidf = stratified_split(tfidf_matrix, label_arr)
    splits_seq   = stratified_split(seq_matrix,   label_arr)

    def _sizes(sp):
        return (len(sp["y_train"]), len(sp["y_val"]), len(sp["y_test"]))
    print(f"  Train/Val/Test  : {_sizes(splits_tfidf)}")

    # Save everything
    np.savez(FEATURES_OUT,
             # TF-IDF splits
             tfidf_X_train = splits_tfidf["X_train"],
             tfidf_X_val   = splits_tfidf["X_val"],
             tfidf_X_test  = splits_tfidf["X_test"],
             # Sequence splits
             seq_X_train   = splits_seq["X_train"],
             seq_X_val     = splits_seq["X_val"],
             seq_X_test    = splits_seq["X_test"],
             # Labels (same split order for both)
             y_train        = splits_tfidf["y_train"],
             y_val          = splits_tfidf["y_val"],
             y_test         = splits_tfidf["y_test"],
             # Embedding matrix
             embed_matrix   = embed_matrix,
             # Meta
             num_classes    = np.array(num_classes),
             vocab_size     = np.array(len(vocab)),
             embed_dim      = np.array(EMBED_DIM),
             max_len        = np.array(MAX_LEN),
             )

    print(f"\nSaved features: {FEATURES_OUT}")
    print("\nFeature extraction complete. Run naive_bayes.py next.\n")


def load_features():
    """Helper used by later scripts to load feature matrices."""
    data = np.load(FEATURES_OUT, allow_pickle=True)
    return data


if __name__ == "__main__":
    main()
