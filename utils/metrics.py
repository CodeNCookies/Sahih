"""
utils/metrics.py
Accuracy, Precision, Recall, F1 — implemented from scratch using pure Python.
No sklearn, no external libraries.
"""

import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────

def accuracy(y_true: list, y_pred: list) -> float:
    """
    Accuracy = correct predictions / total predictions
    """
    assert len(y_true) == len(y_pred), "Length mismatch"
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true: list, y_pred: list, classes: list) -> np.ndarray:
    """
    Build confusion matrix from scratch.
    Rows = true labels, Columns = predicted labels.

    Returns: np.ndarray of shape (n_classes, n_classes)
    """
    n = len(classes)
    class_idx = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[class_idx[t]][class_idx[p]] += 1

    return matrix


def precision_recall_f1(y_true: list, y_pred: list,
                        classes: list) -> dict:
    """
    Per-class Precision, Recall, F1-score — computed from scratch.

    Precision_i = TP_i / (TP_i + FP_i)
    Recall_i    = TP_i / (TP_i + FN_i)
    F1_i        = 2 * P_i * R_i / (P_i + R_i)

    Returns dict: {class_name: {"precision", "recall", "f1", "support"}}
    """
    cm = confusion_matrix(y_true, y_pred, classes)
    results = {}

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp    # predicted as i but not i
        fn = cm[i, :].sum() - tp    # actually i but predicted as something else
        support = cm[i, :].sum()

        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = (2 * prec * recall / (prec + recall)
                  if (prec + recall) > 0 else 0.0)

        results[cls] = {
            "precision": prec,
            "recall":    recall,
            "f1":        f1,
            "support":   int(support),
        }

    return results


def macro_avg(per_class: dict) -> dict:
    """Macro average: unweighted mean across classes."""
    keys = ["precision", "recall", "f1"]
    return {k: np.mean([per_class[c][k] for c in per_class]) for k in keys}


def weighted_avg(per_class: dict) -> dict:
    """Weighted average: weighted by support (number of true samples)."""
    total = sum(per_class[c]["support"] for c in per_class)
    keys  = ["precision", "recall", "f1"]
    result = {}
    for k in keys:
        result[k] = (
            sum(per_class[c][k] * per_class[c]["support"] for c in per_class)
            / total if total > 0 else 0.0
        )
    return result


# ─────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────

def classification_report(y_true: list, y_pred: list,
                           classes: list, model_name: str = "") -> None:
    """Print a formatted classification report."""
    acc      = accuracy(y_true, y_pred)
    per_cls  = precision_recall_f1(y_true, y_pred, classes)
    macro    = macro_avg(per_cls)
    weighted = weighted_avg(per_cls)
    cm       = confusion_matrix(y_true, y_pred, classes)

    header = f"  Classification Report — {model_name}  " if model_name else ""
    print("\n" + "=" * 60)
    if header:
        print(header)
        print("=" * 60)

    # Per-class table
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 57)
    for cls in classes:
        m = per_cls[cls]
        print(f"{cls:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}")

    print("-" * 57)
    print(f"{'Macro avg':<15} {macro['precision']:>10.4f} {macro['recall']:>10.4f} "
          f"{macro['f1']:>10.4f}")
    print(f"{'Weighted avg':<15} {weighted['precision']:>10.4f} {weighted['recall']:>10.4f} "
          f"{weighted['f1']:>10.4f}")

    print(f"\n  Accuracy : {acc * 100:.2f}%  ({sum(t==p for t,p in zip(y_true,y_pred))}"
          f" / {len(y_true)} correct)")

    # Confusion matrix
    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    header_row = " " * 15 + "".join(f"{c:>12}" for c in classes)
    print(header_row)
    for i, cls in enumerate(classes):
        row = f"  {cls:<13}" + "".join(f"{cm[i,j]:>12}" for j in range(len(classes)))
        print(row)
    print("=" * 60 + "\n")

    return {
        "accuracy":     acc,
        "per_class":    per_cls,
        "macro":        macro,
        "weighted":     weighted,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────
# LOSS FUNCTIONS (used by ANN / CNN trainers)
# ─────────────────────────────────────────────────────────────

def cross_entropy_loss(y_true_onehot: np.ndarray,
                        y_pred_probs: np.ndarray,
                        eps: float = 1e-12) -> float:
    """
    Categorical cross-entropy loss.
    y_true_onehot : (N, C) one-hot
    y_pred_probs  : (N, C) softmax probabilities
    """
    y_pred_probs = np.clip(y_pred_probs, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_probs), axis=1))


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    classes  = ["Real", "Biased", "Fake"]
    y_true   = ["Real", "Fake", "Biased", "Real", "Fake",
                "Real", "Biased", "Fake", "Real", "Biased"]
    y_pred   = ["Real", "Fake", "Real",   "Real", "Fake",
                "Biased", "Biased", "Real",  "Real", "Biased"]

    classification_report(y_true, y_pred, classes, model_name="Demo")
