"""
7_compare.py
Load saved results from all three models and produce:
  1. ASCII bar charts in the terminal (no matplotlib needed)
  2. Per-class comparison table
  3. Training curves (loss + accuracy) printed as ASCII
  4. Optional: matplotlib charts saved as PNG if available

Run after 6_train_test.py has completed.
"""

import os
import sys
import json
import numpy as np

MODEL_DIR = os.path.join("data", "processed", "models")


# ─────────────────────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────────────────────

def load_results() -> dict:
    summary_path = os.path.join(MODEL_DIR, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            "summary.json not found. Run 6_train_test.py first."
        )
    with open(summary_path) as f:
        summary = json.load(f)

    # Load per-model detail files
    for key, fname in [("naive_bayes", "nb_results.json"),
                        ("ann",         "ann_results.json"),
                        ("cnn",         "cnn_results.json")]:
        detail_path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(detail_path):
            with open(detail_path) as f:
                summary[key]["detail"] = json.load(f)

    return summary


# ─────────────────────────────────────────────────────────────
# ASCII CHART HELPERS
# ─────────────────────────────────────────────────────────────

def ascii_bar(value: float, width: int = 40,
              symbol: str = "█") -> str:
    filled = int(round(value * width))
    return symbol * filled + "░" * (width - filled)


def print_accuracy_chart(summary: dict) -> None:
    models = [
        ("Naive Bayes (baseline)", summary["naive_bayes"]["accuracy"]),
        ("ANN        (main)     ", summary["ann"]["accuracy"]),
        ("CNN        (compare)  ", summary["cnn"]["accuracy"]),
    ]

    print("\n  ── Accuracy Comparison ────────────────────────────────")
    best_acc = max(a for _, a in models)
    for name, acc in models:
        bar  = ascii_bar(acc)
        star = " ◀ BEST" if abs(acc - best_acc) < 1e-6 else ""
        print(f"  {name}  {acc*100:5.2f}%  {bar}{star}")
    print()


def print_f1_chart(summary: dict) -> None:
    models = [
        ("Naive Bayes", summary["naive_bayes"]["macro_f1"]),
        ("ANN        ", summary["ann"]["macro_f1"]),
        ("CNN        ", summary["cnn"]["macro_f1"]),
    ]

    print("  ── Macro F1 Comparison ────────────────────────────────")
    best = max(f for _, f in models)
    for name, f1 in models:
        bar  = ascii_bar(f1)
        star = " ◀ BEST" if abs(f1 - best) < 1e-6 else ""
        print(f"  {name}  {f1:.4f}  {bar}{star}")
    print()


# ─────────────────────────────────────────────────────────────
# TRAINING CURVE  (ASCII sparkline)
# ─────────────────────────────────────────────────────────────

SPARK_CHARS = " ▁▂▃▄▅▆▇█"

def sparkline(values: list) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo if hi != lo else 1.0
    idxs = [int((v - lo) / rng * (len(SPARK_CHARS) - 1)) for v in values]
    return "".join(SPARK_CHARS[i] for i in idxs)


def print_training_curves(summary: dict) -> None:
    print("  ── Training Curves (sparklines) ──────────────────────")
    for model_key, label in [("ann", "ANN"), ("cnn", "CNN")]:
        detail = summary.get(model_key, {}).get("detail", {})
        hist   = detail.get("history", {})
        if not hist:
            continue
        tl = hist.get("train_loss", [])
        vl = hist.get("val_loss", [])
        ta = hist.get("train_acc", [])
        va = hist.get("val_acc", [])
        n_epochs = len(tl)
        final_acc = va[-1] * 100 if va else 0.0
        print(f"\n  {label}  ({n_epochs} epochs, final val_acc={final_acc:.2f}%)")
        print(f"    Train Loss : {sparkline(tl)}  "
              f"{tl[0]:.4f}→{tl[-1]:.4f}")
        print(f"    Val   Loss : {sparkline(vl)}  "
              f"{vl[0]:.4f}→{vl[-1]:.4f}")
        print(f"    Train Acc  : {sparkline(ta)}  "
              f"{ta[0]*100:.1f}%→{ta[-1]*100:.1f}%")
        print(f"    Val   Acc  : {sparkline(va)}  "
              f"{va[0]*100:.1f}%→{va[-1]*100:.1f}%")
    print()


# ─────────────────────────────────────────────────────────────
# FULL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────

def print_full_table(summary: dict) -> None:
    models = [
        ("Naive Bayes", "naive_bayes"),
        ("ANN",         "ann"),
        ("CNN",         "cnn"),
    ]

    print("  ── Full Metrics Table ─────────────────────────────────")
    header = f"  {'Model':<14} {'Accuracy':>10} {'Macro F1':>10} {'TrainTime':>10}"
    print(header)
    print("  " + "-" * 48)

    for label, key in models:
        d    = summary[key]
        acc  = d["accuracy"] * 100
        f1   = d["macro_f1"]
        t    = d.get("train_time_s", 0)
        print(f"  {label:<14} {acc:>9.2f}% {f1:>10.4f} {t:>8.1f}s")

    print()

    # Winner
    best_model = max(
        [("Naive Bayes", "naive_bayes"), ("ANN", "ann"), ("CNN", "cnn")],
        key=lambda x: summary[x[1]]["accuracy"]
    )
    print(f"  🏆  Winner by accuracy : {best_model[0]}")
    total = summary.get("total_time_s", 0)
    print(f"  ⏱  Total pipeline time : {total:.1f}s\n")


# ─────────────────────────────────────────────────────────────
# OPTIONAL: MATPLOTLIB CHARTS
# ─────────────────────────────────────────────────────────────

def try_matplotlib_charts(summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping PNG charts)")
        return

    CLASSES = ["Real", "Biased", "Fake"]

    # ── Accuracy bar chart ──
    labels = ["Naive Bayes", "ANN (main)", "CNN (compare)"]
    accs   = [summary["naive_bayes"]["accuracy"] * 100,
              summary["ann"]["accuracy"] * 100,
              summary["cnn"]["accuracy"] * 100]
    f1s    = [summary["naive_bayes"]["macro_f1"],
              summary["ann"]["macro_f1"],
              summary["cnn"]["macro_f1"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4CAF50", "#2196F3", "#FF9800"]

    # Accuracy
    bars = axes[0].bar(labels, accs, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.2f}%", ha="center", fontsize=10)
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Model Accuracy Comparison")

    # Macro F1
    bars2 = axes[1].bar(labels, f1s, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars2, f1s):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", fontsize=10)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Macro F1 Score")
    axes[1].set_title("Macro F1 Comparison")

    plt.suptitle("Urdu Fake News Classification — Model Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "comparison_chart.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📊 Chart saved: {out_path}")

    # ── Training curves ──
    for model_key, label in [("ann", "ANN"), ("cnn", "CNN")]:
        detail = summary.get(model_key, {}).get("detail", {})
        hist   = detail.get("history", {})
        if not hist:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = range(1, len(hist["train_loss"]) + 1)

        axes[0].plot(epochs, hist["train_loss"], label="Train")
        axes[0].plot(epochs, hist["val_loss"],   label="Val")
        axes[0].set_title(f"{label} — Loss")
        axes[0].set_xlabel("Epoch"); axes[0].legend()

        axes[1].plot(epochs, [v*100 for v in hist["train_acc"]], label="Train")
        axes[1].plot(epochs, [v*100 for v in hist["val_acc"]],   label="Val")
        axes[1].set_title(f"{label} — Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)"); axes[1].legend()

        plt.tight_layout()
        curve_path = os.path.join(MODEL_DIR, f"curve_{label.lower()}.png")
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"  📈 Curve saved: {curve_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("   STEP 7: COMPARISON REPORT")
    print("═" * 60)

    summary = load_results()

    print_accuracy_chart(summary)
    print_f1_chart(summary)
    print_training_curves(summary)
    print_full_table(summary)
    try_matplotlib_charts(summary)

    print("Comparison complete.\n")


if __name__ == "__main__":
    main()
