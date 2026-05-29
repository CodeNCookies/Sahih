# Urdu Fake News Classifier — Built From Scratch

## Project Structure

Urdu_Fake_News_Project/
│
├── preprocess.py          # Step 1: Clean and normalize Urdu text
├── features.py            # Step 2: Build vocabulary, TF-IDF, sequences
├── naive_bayes.py         # Step 3: Multinomial Naive Bayes classifier            
├── cnn.py                 # Step 5: 1D Convolutional Neural Network          
├── compare.py             # Step 7: Compare models, generate charts
│
├── data/
│   ├── raw/               # Put your dataset.csv here
│   └── processed/         # Auto-generated: features, models, results
│
|__templates/
|   |__ index.html
|   |
└── utils/
    ├── urdu_normalizer.py # Urdu character normalization + stopwords
    └── metrics.py         # Accuracy, Precision, Recall, F1 (from scratch)
```
```

## How to Run

### All at once: python preprocess.py; python features.py; python naive_bayes.py; python cnn.py; python compare;


## Your Dataset Format
| text | label |
|------|-------|
| "حکومت نے نیا قانون منظور کر لیا" | Real |
| "مخالف جماعت نے جھوٹے الزامات لگائے" | Biased |
| "واٹس ایپ پر افواہ پھیل رہی ہے" | Fake |


## What Each File Does 

### `preprocess.py` — Step 1: Clean the Text

Takes raw Urdu news headlines and makes them ready for the computer to understand.

**What happens:**
1. **Reads** your CSV file from `data/raw/`
2. **Removes** emojis, special symbols, URLs, HTML code
3. **Normalizes** Urdu characters (so آ and ا are treated as the same letter)
4. **Removes** stopwords (common words like "ہے", "کا", "میں" that don't add meaning)
5. **Saves** clean texts and labels to `data/processed/dataset.npz`

**Example transformation:**
```
Input:  "حکومت نے نیا قانون پاس کیا ہے!!!"
Output: "حکومت نیا قانون پاس کیا"
```

---

### `features.py` — Step 2: Turn Words Into Numbers

Computers don't understand words — they understand numbers. This file converts text into a format a machine learning model can process.

**What happens:**
1. **Builds a vocabulary** — assigns a unique number to every word (like a dictionary)
2. **Creates TF-IDF features** — measures how important each word is in each headline
   - *TF* = how often a word appears in one headline
   - *IDF* = how rare that word is across ALL headlines (rare words are more valuable for detection)
3. **Creates padded sequences** — makes all headlines the same length (200 words) by adding zeros at the end
4. **Creates a word embedding matrix** — gives each word a 300-dimensional vector (random at first, the model will learn to adjust these)
5. **Splits data** into train (70%), validation (10%), and test (20%) sets
6. **Saves** everything to `data/processed/features.npz`

**Why three different formats?**
- **TF-IDF** — for Naive Bayes (works well with word frequencies)
- **Sequences + Embeddings** — for ANN and CNN (deep learning needs this)

---

### `naive_bayes.py` — Step 3: Naive Bayes Classifier

The simplest model — uses word probabilities to predict if news is Real, Biased, or Fake.

**How it works (intuition):**
1. **Counts words:** "The word 'افواہ' (rumor) appears in 80% of Fake news but only 5% of Real news"
2. **Calculates probabilities:** If a headline contains 'افواہ', it's probably Fake
3. **Combines evidence:** Looks at ALL words in the headline and multiplies their probabilities together
4. **Uses Laplace smoothing:** If a word was never seen during training, it doesn't break the model (gives it a small probability instead of zero)
5. **Tries different smoothing values** and picks the best one

**Why "Naive"?** It assumes all words are independent (which isn't true — "نیا" and "قانون" often appear together). But it still works surprisingly well!

**What you'll see:**
- Grid search over alpha values (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
- Best alpha chosen based on validation accuracy
- Final accuracy, precision, recall, F1 score on test data
- Confusion matrix showing which classes get confused

---

### `cnn.py` — Step 5: Convolutional Neural Network

Like the ANN, but better at finding local patterns — it can detect phrases like "جھوٹی خبر" (fake news) regardless of where they appear.

**Architecture:**
```
Input Words -> Embedding -> Conv1D(128 filters) -> GlobalMaxPool -> Dense(64) -> Output
                |              |                      |                |           |
           Words->vectors  Scan for patterns    Keep strongest     Second      Prediction
                           (like "fake news")   detection         layer
```

**Key difference from ANN:**
- **Conv1D layer:** Slides a "window" of 5 words across the headline, looking for suspicious patterns (like a magnifying glass scanning text)
- **128 filters:** Learns 128 different patterns to look for (phrases indicating fake, biased, or real news)
- **GlobalMaxPool:** For each pattern, keeps only the STRONGEST match — "was this pattern found anywhere in the text?"

**Example of what a filter might learn:**
```
Filter #1: Detects "افواہ" + "پھیل" -> strong signal for Fake
Filter #2: Detects "حکومت" + "اعلان" -> strong signal for Real
Filter #3: Detects "مخالف" + "الزام" -> strong signal for Biased
```

**What you'll see:**
- Same training loop as ANN
- Typically better accuracy than both Naive Bayes and ANN
- 277K total parameters (the model has to learn)

---

### `compare.py` — Step 7: Results Comparison

Shows you which model performed best with visual charts.

**What you'll see:**
- **Bar chart:** Accuracy comparison (ASCII + PNG)
- **Table:** All metrics side by side
- **Training curves:** How ANN and CNN learned over time (loss going down = good)
- **Best model:** Automatically identified

**Saved outputs:**
- `comparison_chart.png` — accuracy + F1 bar chart
- `curve_ann.png` — ANN training progress
- `curve_cnn.png` — CNN training progress

---

### `utils/urdu_normalizer.py` — Urdu Text Helper

Handles all the quirks of Urdu script.

**What it fixes:**
- **Character variants:** آ/أ/إ -> ا (all Alef forms become standard Alef)
- **Yeh problem:** ي/ئ/ى -> ی (Arabic Yeh becomes Urdu Yeh)
- **Kaf problem:** ك -> ک (Arabic Kaf becomes Urdu Kaf)
- **Digits:** 123 -> ۱۲۳ (Latin to Urdu, or vice versa)
- **Diacritics:** Optional removal of zabar/zer/pesh marks
- **75+ stopwords:** Common words like "ہے", "کا", "میں", "اور" that don't carry meaning

---

### `utils/metrics.py` — Performance Measurement

Calculates how good your model is — all formulas implemented from scratch.

**Metrics explained:**
- **Accuracy:** What % of predictions were correct? (3/5 = 60%)
- **Precision:** When the model says "Fake", how often is it right? (avoid false alarms)
- **Recall:** Of all actual Fake news, how many did the model catch? (don't miss fakes)
- **F1 Score:** Balance between Precision and Recall (harmonic mean)
- **Confusion Matrix:** Shows exactly which classes get mixed up

---

## Expected Results (With 1000 Samples)

| Model | Accuracy | Why? |
|-------|----------|------|
| **Naive Bayes** | 60-68% | Simple, fast, assumes word independence |
| **ANN** | 68-75% | Learns word relationships, better than NB |
| **CNN** | 70-78% | Best at finding patterns anywhere in text |

--

## Summary

You now have a complete, working fake news detection system that:
- Handles real Urdu text (normalizes characters, removes noise)
- Converts text to numbers (TF-IDF + word embeddings)
- Trains 2 different models (simple to complex)
- Compares results with charts
- Everything is built from scratch — no black boxes
- Well-commented so you can learn from it