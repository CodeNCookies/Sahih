"""
1_preprocess.py
Load raw Urdu texts, clean + normalize, save to data/processed/.

Input  : data/raw/   — CSV file(s) with columns: text, label
Output : data/processed/dataset.npz  (texts array + labels array)
         data/processed/label_map.json
"""

import os
os.system('cls')
import sys
import json
import csv
import re
import numpy as np

# Allow imports from utils/ when run from project root
sys.path.insert(0, os.path.dirname(__file__))
from utils.urdu_normalizer import (normalize, tokenize,
                                    remove_stopwords, URDU_STOPWORDS)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

DATASET_OUT   = os.path.join(PROCESSED_DIR, "dataset.npz")
LABEL_MAP_OUT = os.path.join(PROCESSED_DIR, "label_map.json")

# ─────────────────────────────────────────────────────────────
# VALID CLASS LABELS
# ─────────────────────────────────────────────────────────────
CLASSES      = ["Real", "Biased", "Fake"]
LABEL_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INT_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}


# ─────────────────────────────────────────────────────────────
# EMOJI & SPECIAL CHARACTER REMOVAL
# ─────────────────────────────────────────────────────────────

def remove_emojis_and_special_chars(text: str) -> str:
    """
    Remove emojis, emoticons, and other special characters from text.
    Keeps Urdu characters, basic Latin, Arabic script, and essential punctuation.
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text without emojis and special characters
    """
    # Emoji pattern - covers most emojis including flags, symbols, etc.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended-A
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U0001F000-\U0001F02F"  # mahjong tiles
        "\U0001F0A0-\U0001F0FF"  # playing cards
        "\U0001F100-\U0001F64F"  # enclosed alphanumeric supplement
        "\U0001F680-\U0001F6FF"  # transport and map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes extended
        "\U0001F800-\U0001F8FF"  # supplemental arrows-C
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U0001F004-\U0001F0CF"  # additional emoticons
        "\U0001F200-\U0001F251"  # enclosed ideographic supplement
        "]+", 
        flags=re.UNICODE
    )
    
    # Remove emojis
    text = emoji_pattern.sub('', text)
    
    # Remove common emoticons like :), :D, <3, etc.
    emoticon_pattern = re.compile(
        r'[:;=]'  # eyes
        r'[-~^]?'  # optional nose
        r'[)D\]\(/\\|pP*3]'  # mouth
        r'|'  # or
        r'<3|</3'  # hearts
        r'|'  # or
        r'XD|xd|:P|:p|:O|:o|:3'  # common combinations
    )
    text = emoticon_pattern.sub('', text)
    
    # Remove HTML entities like &#1234; and &amp;
    text = re.sub(r'&\w+;', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    
    # Remove multiple spaces, tabs, and newlines
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# (Used when no real CSV exists - replace with your own data)
# ─────────────────────────────────────────────────────────────

def _make_synthetic_dataset(n: int = 1000) -> tuple:
    """Generate synthetic Urdu-like dataset for testing."""
    np.random.seed(42)

    vocab = [
        "خبر", "سیاست", "حکومت", "ملک", "عوام", "میڈیا", "اطلاع",
        "رپورٹ", "الزام", "دعویٰ", "تصدیق", "جھوٹ", "سچ", "واقعہ",
        "بیان", "تحقیق", "صحافت", "افواہ", "پروپیگنڈا", "انتخاب",
        "وزیر", "پارلیمنٹ", "کرپشن", "احتجاج", "آزادی", "معیشت",
        "تعلیم", "صحت", "وائرل", "سوشل", "ذریعہ", "اخبار", "چینل",
        "ویڈیو", "تصویر", "ثبوت", "گواہ", "قانون", "عدالت", "انصاف",
        "ووٹ", "جماعت", "لیڈر", "تقریر", "جلسہ", "احتساب", "غریب",
        "امیر", "سرمایہ", "قرض", "بجٹ", "ٹیکس", "اصلاح", "اعلان",
    ]

    def sentence(length=None):
        n_words = length or int(np.random.randint(8, 25))
        return " ".join(np.random.choice(vocab, size=n_words, replace=True))

    texts  = [sentence() for _ in range(n)]
    labels = list(
        np.random.choice(CLASSES, size=n, p=[0.40, 0.30, 0.30])
    )
    return texts, labels


# ─────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────

def load_raw(raw_dir: str) -> tuple:
    """
    Try to load CSV files from raw_dir.
    Expected CSV columns: 'text', 'label'
    Falls back to synthetic data if directory is empty.
    """
    csv_files = [f for f in os.listdir(raw_dir)
                 if f.lower().endswith(".csv")]

    if not csv_files:
        print("No CSV found in data/raw/ - using synthetic data.")
        print("    To use your own data: place a CSV with 'text' and 'label'")
        print("    columns in data/raw/ and re-run.\n")
        return _make_synthetic_dataset(n=1000)

    texts, labels = [], []
    for fname in csv_files:
        fpath = os.path.join(raw_dir, fname)
        print(f"  Loading: {fpath}")
        with open(fpath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text  = row.get("text", "").strip()
                label = row.get("label", "").strip()
                if text and label in CLASSES:
                    texts.append(text)
                    labels.append(label)

    print(f"  Loaded {len(texts)} samples from {len(csv_files)} file(s).")
    return texts, labels


# ─────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess_text(text: str,
                    keep_stopwords: bool = False) -> str:
    """
    Clean text: remove emojis/special chars -> normalize -> tokenize -> 
    (optional stopword removal) -> rejoin.
    
    Args:
        text: Input text string
        keep_stopwords: If True, stopwords are not removed
    
    Returns:
        Cleaned string (space-separated tokens)
    """
    # Step 0: Remove emojis and special characters first
    text = remove_emojis_and_special_chars(text)
    
    # Step 1: Urdu text normalization
    normalized = normalize(text)
    
    # Step 2: Tokenize
    tokens = tokenize(normalized)

    # Step 3: Optional stopword removal
    if not keep_stopwords:
        tokens = remove_stopwords(tokens)

    # Step 4: Drop very short tokens (single character) and pure digits
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
    
    # Step 5: Remove any remaining non-Urdu/non-Arabic script characters
    # Keep Urdu (0600-06FF), Arabic (0600-06FF), basic punctuation
    cleaned_tokens = []
    for token in tokens:
        # Keep if contains at least one Urdu/Arabic character or is a valid word
        if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', token) or token.isalpha():
            cleaned_tokens.append(token)
    
    return " ".join(cleaned_tokens) if cleaned_tokens else ""


def preprocess_dataset(texts: list, labels: list,
                        keep_stopwords: bool = False) -> tuple:
    """Apply preprocessing to every sample."""
    cleaned_texts  = []
    cleaned_labels = []
    skipped        = 0

    for idx, (text, label) in enumerate(zip(texts, labels)):
        clean = preprocess_text(text, keep_stopwords=keep_stopwords)
        if clean.strip():  # skip if empty after cleaning
            cleaned_texts.append(clean)
            cleaned_labels.append(label)
        else:
            skipped += 1
            if skipped <= 5:  # Show first few skipped examples
                print(f"  Warning: Sample {idx} became empty after cleaning")
                print(f"    Original: {text[:100]}...")

    print(f"  Preprocessing done. Kept {len(cleaned_texts)} / "
          f"{len(texts)} samples. Skipped {skipped} empty.")
    return cleaned_texts, cleaned_labels


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────

def print_stats(texts: list, labels: list) -> None:
    """Print basic dataset statistics."""
    from collections import Counter
    lengths = [len(t.split()) for t in texts]
    dist    = Counter(labels)

    print("\n── Dataset Statistics ──────────────────────────")
    print(f"  Total samples     : {len(texts)}")
    if lengths:
        print(f"  Avg tokens/sample : {sum(lengths)/len(lengths):.1f}")
        print(f"  Min tokens        : {min(lengths)}")
        print(f"  Max tokens        : {max(lengths)}")
    print("  Class distribution:")
    for cls in CLASSES:
        count = dist.get(cls, 0)
        pct   = count / len(labels) * 100 if labels else 0
        bar   = "#" * int(pct // 2)
        print(f"    {cls:<10}: {count:>5}  ({pct:5.1f}%)  {bar}")
    print("─" * 50 + "\n")


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────

def save_processed(texts: list, labels: list) -> None:
    """Save processed texts and integer-encoded labels to .npz."""
    label_ints = np.array([LABEL_TO_INT[l] for l in labels], dtype=np.int32)
    texts_arr  = np.array(texts, dtype=object)   # variable-length strings

    np.savez(DATASET_OUT,
             texts=texts_arr,
             labels=label_ints)

    with open(LABEL_MAP_OUT, "w", encoding="utf-8") as f:
        json.dump({"label_to_int": LABEL_TO_INT,
                   "int_to_label": {str(k): v for k, v in INT_TO_LABEL.items()},
                   "classes": CLASSES}, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {DATASET_OUT}")
    print(f"  Saved: {LABEL_MAP_OUT}")


def load_processed() -> tuple:
    """Load processed dataset from .npz. Returns (texts, label_ints, classes)."""
    data       = np.load(DATASET_OUT, allow_pickle=True)
    texts      = list(data["texts"])
    label_ints = data["labels"]

    with open(LABEL_MAP_OUT, encoding="utf-8") as f:
        meta = json.load(f)

    return texts, label_ints, meta["classes"]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 1: PREPROCESSING")
    print("=" * 55)

    # 1. Load raw data
    print("\n[1/4] Loading raw data ...")
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_texts, raw_labels = load_raw(RAW_DIR)

    # 2. Preprocess
    print("\n[2/4] Preprocessing texts ...")
    clean_texts, clean_labels = preprocess_dataset(
        raw_texts, raw_labels, keep_stopwords=False
    )

    # 3. Statistics
    print("[3/4] Dataset statistics:")
    print_stats(clean_texts, clean_labels)

    # 4. Save
    print("[4/4] Saving processed data ...")
    save_processed(clean_texts, clean_labels)

    print("\nPreprocessing complete. Run features.py next.\n")


if __name__ == "__main__":
    main()