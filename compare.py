"""
7_compare.py

Compares:
- Naive Bayes
- CNN

Reads:
- nb_results.json
- cnn_results.json

Creates:
- accuracy_comparison.png
- f1_comparison.png

Saved inside:
data/processed/models/
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join("data", "processed", "models")


def load_json(filename):

    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        print(f"Missing: {filename}")
        return None

    with open(path, "r") as f:
        return json.load(f)


def main():

    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)

    nb = load_json("nb_results.json")
    cnn = load_json("cnn_results.json")

    models=[]

    if nb:
        models.append((
            "Naive Bayes",
            nb["accuracy"]*100,
            nb["macro_f1"]
        ))

    if cnn:
        models.append((
            "CNN",
            cnn["accuracy"]*100,
            cnn["macro_f1"]
        ))

    if len(models)==0:
        print("No results found")
        return


    print()

    print(
        f"{'Model':<20}"
        f"{'Accuracy':<15}"
        f"{'Macro F1'}"
    )

    print("-"*45)

    for name,acc,f1 in models:

        print(
            f"{name:<20}"
            f"{acc:>7.2f}%"
            f"{f1:>12.4f}"
        )


    winner=max(models,key=lambda x:x[1])

    print("\nWinner:",winner[0])
    print(f"Accuracy: {winner[1]:.2f}%")


    # Accuracy graph

    labels=[x[0] for x in models]
    accs=[x[1] for x in models]

    plt.figure(figsize=(8,5))

    bars=plt.bar(labels,accs)

    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0,100)

    for b,val in zip(bars,accs):

        plt.text(
            b.get_x()+b.get_width()/2,
            val+1,
            f"{val:.2f}%",
            ha='center'
        )

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "accuracy_comparison.png"
        ),
        dpi=150
    )

    plt.close()


    # F1 graph

    f1s=[x[2] for x in models]

    plt.figure(figsize=(8,5))

    bars=plt.bar(labels,f1s)

    plt.title("Macro F1 Comparison")
    plt.ylabel("F1 Score")
    plt.ylim(0,1)

    for b,val in zip(bars,f1s):

        plt.text(
            b.get_x()+b.get_width()/2,
            val+0.02,
            f"{val:.4f}",
            ha='center'
        )

    plt.savefig(
        os.path.join(
            MODEL_DIR,
            "f1_comparison.png"
        ),
        dpi=150
    )

    plt.close()

    print("\nSaved:")
    print("accuracy_comparison.png")
    print("f1_comparison.png")

    print("\nDone")


if __name__=="__main__":
    main()