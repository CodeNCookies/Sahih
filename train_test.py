"""
6_train_test.py
Orchestrator — runs the full pipeline end-to-end:
    1_preprocess → 2_features → 3_naive_bayes → 4_ann → 5_cnn

Collects all results and saves a summary JSON.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

MODEL_DIR = os.path.join("data", "processed", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(MODEL_DIR, "summary.json")


def run_step(label: str, fn):
    """Run one pipeline step, measure time, return results."""
    print("\n" + "━" * 60)
    print(f"{label}")
    print("━" * 60)
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    return result, elapsed


def main():
    print("\n" + "═" * 60)
    print("   URDU FAKE NEWS CLASSIFIER — FULL PIPELINE")
    print("═" * 60)

    timings = {}

    # ── Step 1: Preprocess ────────────────────────────────────
    from preprocess import main as preprocess_main
    _, timings["preprocess"] = run_step("STEP 1: Preprocessing", preprocess_main)

    # ── Step 2: Features ──────────────────────────────────────
    from features import main as features_main
    _, timings["features"] = run_step("STEP 2: Feature Extraction", features_main)

    # ── Step 3: Naive Bayes ───────────────────────────────────
    from naive_bayes import main as nb_main
    nb_results, timings["naive_bayes"] = run_step("STEP 3: Naive Bayes", nb_main)

    # ── Step 4: ANN ───────────────────────────────────────────
    from ann import main as ann_main
    ann_results, timings["ann"] = run_step("STEP 4: ANN", ann_main)

    # ── Step 5: CNN ───────────────────────────────────────────
    from cnn import main as cnn_main
    cnn_results, timings["cnn"] = run_step("STEP 5: CNN", cnn_main)

    # ── Compile summary ───────────────────────────────────────
    summary = {
        "naive_bayes": {
            "accuracy": nb_results["accuracy"],
            "macro_f1": nb_results["macro"]["f1"],
            "train_time_s": timings["naive_bayes"],
        },
        "ann": {
            "accuracy": ann_results["accuracy"],
            "macro_f1": ann_results["macro"]["f1"],
            "train_time_s": timings["ann"],
        },
        "cnn": {
            "accuracy": cnn_results["accuracy"],
            "macro_f1": cnn_results["macro"]["f1"],
            "train_time_s": timings["cnn"],
        },
        "total_time_s": sum(timings.values()),
    }

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print final table ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("   PIPELINE COMPLETE — RESULTS SUMMARY")
    print("═" * 60)
    print(f"\n  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'Time':>10}")
    print("  " + "-" * 52)
    for name, d in [("Naive Bayes", summary["naive_bayes"]),
                    ("ANN (main)", summary["ann"]),
                    ("CNN (comparison)", summary["cnn"])]:
        print(f"  {name:<20} {d['accuracy']*100:>9.2f}% "
              f"{d['macro_f1']:>10.4f} {d['train_time_s']:>8.1f}s")

    best = max(summary, key=lambda k: summary[k].get("accuracy", -1)
               if k != "total_time_s" else -1)
    print(f"\nBest model: {best.upper()}")
    print(f"Summary saved: {SUMMARY_PATH}")
    print("\nRun compare.py to see charts and detailed comparison.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
