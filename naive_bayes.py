"""
3_naive_bayes.py
Multinomial Naive Bayes — implemented fully from scratch using NumPy.

  P(class | text) ∝ P(class) × ∏ P(word | class)

With Laplace (add-α) smoothing.
No sklearn.
"""


import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import load_features
from utils.metrics import classification_report

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR     = os.path.join("data", "processed", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP_PATH = os.path.join(PROCESSED_DIR, "label_map.json")


# ─────────────────────────────────────────────────────────────
# MULTINOMIAL NAIVE BAYES — FROM SCRATCH
# ─────────────────────────────────────────────────────────────

class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes with Laplace smoothing.

    Operates on raw count / TF-IDF feature vectors.
    All computations in log-space to avoid underflow.
    """

    def __init__(self, alpha: float = 1.0):
        """
        alpha : Laplace smoothing parameter (1.0 = standard Laplace)
        """
        self.alpha          = alpha
        self.log_class_prior = None   # shape (C,)
        self.log_likelihood  = None   # shape (C, V)
        self.classes_        = None
        self.num_classes_    = None
        self.vocab_size_     = None

    # ── TRAINING ────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesClassifier":
        """
        X : (N, V) TF-IDF or count matrix  (non-negative)
        y : (N,)   integer class labels
        """
        N, V          = X.shape
        self.classes_ = np.unique(y)
        C             = len(self.classes_)
        self.num_classes_ = C
        self.vocab_size_  = V

        # ── Log prior: log P(c) = log(count_c / N) ──────────────
        log_prior = np.zeros(C, dtype=np.float64)
        for i, cls in enumerate(self.classes_):
            count_c       = np.sum(y == cls)
            log_prior[i]  = np.log(count_c / N)
        self.log_class_prior = log_prior

        # ── Log likelihood: log P(w | c) with Laplace smoothing ─
        # For each class, sum all feature values of docs in that class
        log_lik = np.zeros((C, V), dtype=np.float64)
        for i, cls in enumerate(self.classes_):
            X_cls        = X[y == cls]            # docs of class c
            word_counts  = X_cls.sum(axis=0) + self.alpha   # (V,) smoothed
            total_words  = word_counts.sum()
            log_lik[i]   = np.log(word_counts / total_words)

        self.log_likelihood = log_lik
        return self

    # ── INFERENCE ───────────────────────────────────────────────

    def _log_posterior(self, X: np.ndarray) -> np.ndarray:
        """
        Compute unnormalized log P(c|x) for each sample.
        log P(c|x) ∝ log P(c) + x · log P(w|c)

        X : (N, V)
        Returns : (N, C)
        """
        # X @ log_likelihood.T  →  (N, V) x (V, C)  =  (N, C)
        return X @ self.log_likelihood.T + self.log_class_prior

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class index for each sample."""
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Approximate class probabilities via softmax over log-posteriors.
        Returns (N, C).
        """
        log_post = self._log_posterior(X)
        # Numerically stable softmax
        log_post -= log_post.max(axis=1, keepdims=True)
        exp_vals  = np.exp(log_post)
        return exp_vals / exp_vals.sum(axis=1, keepdims=True)

    # ── PERSISTENCE ────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(path,
                 log_class_prior = self.log_class_prior,
                 log_likelihood  = self.log_likelihood,
                 classes         = self.classes_,
                 alpha           = np.array([self.alpha]))
        print(f"  Model saved: {path}")

    @classmethod
    def load(cls, path: str) -> "NaiveBayesClassifier":
        data  = np.load(path + ".npz")
        model = cls(alpha=float(data["alpha"][0]))
        model.log_class_prior = data["log_class_prior"]
        model.log_likelihood  = data["log_likelihood"]
        model.classes_        = data["classes"]
        model.num_classes_    = len(model.classes_)
        model.vocab_size_     = model.log_likelihood.shape[1]
        return model


# ─────────────────────────────────────────────────────────────
# HYPER-PARAMETER SEARCH (alpha)
# ─────────────────────────────────────────────────────────────

def tune_alpha(X_train, y_train, X_val, y_val,
               alphas=(0.01, 0.1, 0.5, 1.0, 2.0, 5.0)) -> float:
    """Grid-search best Laplace smoothing alpha on validation set."""
    best_alpha, best_acc = 1.0, 0.0
    print("  Alpha grid search:")
    for a in alphas:
        model = NaiveBayesClassifier(alpha=a)
        model.fit(X_train, y_train)
        preds  = model.predict(X_val)
        acc    = np.mean(preds == y_val)
        print(f"    α={a:6.3f}  val_acc={acc*100:.2f}%")
        if acc > best_acc:
            best_acc   = acc
            best_alpha = a
    print(f"  Best α = {best_alpha}  (val_acc = {best_acc*100:.2f}%)")
    return best_alpha


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 3: NAIVE BAYES (from scratch)")
    print("=" * 55)

    # Load features
    print("\n[1/4] Loading features …")
    feat = load_features()

    X_train = feat["tfidf_X_train"].astype(np.float64)
    X_val   = feat["tfidf_X_val"].astype(np.float64)
    X_test  = feat["tfidf_X_test"].astype(np.float64)
    y_train = feat["y_train"]
    y_val   = feat["y_val"]
    y_test  = feat["y_test"]

    with open(LABEL_MAP_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Alpha tuning
    print("\n[2/4] Tuning Laplace smoothing (α) …")
    best_alpha = tune_alpha(X_train, y_train, X_val, y_val)

    # Final training on train+val
    print("\n[3/4] Training final model on train+val …")
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    model = NaiveBayesClassifier(alpha=best_alpha)
    model.fit(X_full, y_full)
    model.save(os.path.join(MODEL_DIR, "naive_bayes"))

    # Evaluation on test set
    print("\n[4/4] Evaluating on test set …")
    y_pred = model.predict(X_test)

    y_true_labels = [classes[i] for i in y_test]
    y_pred_labels = [classes[i] for i in y_pred]

    results = classification_report(
        y_true_labels, y_pred_labels, classes, model_name="Naive Bayes"
    )

    # Save results
    results_path = os.path.join(MODEL_DIR, "nb_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "accuracy": results["accuracy"],
            "macro_f1": results["macro"]["f1"],
            "alpha":    best_alpha,
        }, f, indent=2)
    print(f"  Results saved: {results_path}")

    print("\nNaive Bayes complete. Run ann.py next.\n")
    return results


if __name__ == "__main__":
    main()
