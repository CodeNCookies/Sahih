"""
utils/urdu_normalizer.py
Urdu character normalization — pure Python, no external NLP libraries.
"""



import re
import unicodedata


# ─────────────────────────────────────────────────────────────
# CHARACTER MAPPING TABLES
# ─────────────────────────────────────────────────────────────

# Normalize visually similar / alternate Unicode Urdu characters to one form
URDU_CHAR_MAP = {
    # Alef variants → plain Alef
    "\u0622": "\u0627",   # آ → ا
    "\u0623": "\u0627",   # أ → ا
    "\u0625": "\u0627",   # إ → ا
    "\u0671": "\u0627",   # ٱ → ا

    # Hamza variants
    "\u0624": "\u0648",   # ؤ → و
    "\u0626": "\u06CC",   # ئ → ی

    # Ye variants → Urdu Ye
    "\u064A": "\u06CC",   # ي (Arabic) → ی (Urdu)
    "\u0649": "\u06CC",   # ى → ی

    # Ke / Kaf variants
    "\u0643": "\u06A9",   # ك (Arabic Kaf) → ک (Urdu Kaf)

    # He variants
    "\u0629": "\u06C1",   # ة (Ta Marbuta) → ہ

    # Digits: Arabic-Indic → ASCII
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3",
    "\u0664": "4", "\u0665": "5", "\u0666": "6", "\u0667": "7",
    "\u0668": "8", "\u0669": "9",

    # Extended Arabic-Indic digits
    "\u06F0": "0", "\u06F1": "1", "\u06F2": "2", "\u06F3": "3",
    "\u06F4": "4", "\u06F5": "5", "\u06F6": "6", "\u06F7": "7",
    "\u06F8": "8", "\u06F9": "9",
}

# Urdu diacritics to strip (harakat)
DIACRITICS = set([
    "\u064B",  # Fathatan
    "\u064C",  # Dammatan
    "\u064D",  # Kasratan
    "\u064E",  # Fatha
    "\u064F",  # Damma
    "\u0650",  # Kasra
    "\u0651",  # Shadda
    "\u0652",  # Sukun
    "\u0653",  # Maddah
    "\u0654",  # Hamza above
    "\u0655",  # Hamza below
    "\u0670",  # Superscript Alef
    "\u06E1",  # Small High Dotless Head of Khah
])

# Punctuation and special chars to remove
PUNCT_PATTERN = re.compile(
    r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؟؛»«…٪×÷]'
)

# Repeated whitespace
WHITESPACE_PATTERN = re.compile(r'\s+')


# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def normalize_characters(text: str) -> str:
    """Apply character-level normalization using URDU_CHAR_MAP."""
    return "".join(URDU_CHAR_MAP.get(ch, ch) for ch in text)


def remove_diacritics(text: str) -> str:
    """Strip Urdu diacritical marks (harakat)."""
    return "".join(ch for ch in text if ch not in DIACRITICS)


def remove_punctuation(text: str) -> str:
    """Remove punctuation and special symbols."""
    return PUNCT_PATTERN.sub(" ", text)


def remove_non_urdu(text: str) -> str:
    """
    Keep only Urdu/Arabic Unicode block characters and spaces.
    Removes English letters, numbers, and other scripts.
    """
    # Urdu/Arabic range: U+0600–U+06FF + U+FB50–U+FDFF + U+FE70–U+FEFF
    filtered = []
    for ch in text:
        cp = ord(ch)
        if (0x0600 <= cp <= 0x06FF or
                0xFB50 <= cp <= 0xFDFF or
                0xFE70 <= cp <= 0xFEFF or
                ch == " "):
            filtered.append(ch)
    return "".join(filtered)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize(text: str,
              remove_diacritics_flag: bool = True,
              remove_non_urdu_flag: bool = True) -> str:
    """
    Full normalization pipeline:
        1. Unicode NFC normalization
        2. Character mapping (alef variants, ye, kaf, digits …)
        3. Diacritic removal
        4. Punctuation removal
        5. Optional: remove non-Urdu characters
        6. Whitespace normalization
    """
    if not isinstance(text, str):
        return ""

    # Step 1: NFC
    text = unicodedata.normalize("NFC", text)

    # Step 2: character map
    text = normalize_characters(text)

    # Step 3: diacritics
    if remove_diacritics_flag:
        text = remove_diacritics(text)

    # Step 4: punctuation
    text = remove_punctuation(text)

    # Step 5: non-Urdu
    if remove_non_urdu_flag:
        text = remove_non_urdu(text)

    # Step 6: whitespace
    text = normalize_whitespace(text)

    return text


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer for Urdu."""
    return text.split()


def normalize_and_tokenize(text: str) -> list:
    """Normalize then tokenize in one call."""
    return tokenize(normalize(text))


# ─────────────────────────────────────────────────────────────
# STOPWORDS  (common Urdu function words)
# ─────────────────────────────────────────────────────────────

URDU_STOPWORDS = set([
    "کا", "کی", "کے", "میں", "سے", "پر", "کو", "نے", "ہے", "ہیں",
    "تھا", "تھی", "تھے", "اور", "یا", "بھی", "تو", "یہ", "وہ",
    "اس", "ان", "اک", "ایک", "جو", "جس", "جن", "لیے", "لئے",
    "ہم", "آپ", "تم", "مجھے", "ہمیں", "انہیں", "اسے", "جب", "تب",
    "اب", "پھر", "مگر", "لیکن", "کہ", "تاکہ", "ہو", "ہوا", "ہوئی",
    "ہوئے", "کر", "کرنا", "کیا", "کیے", "کیں", "نہیں", "نہ", "مت",
    "ہاں", "جی", "بس", "تک", "ساتھ", "بعد", "پہلے", "اگر", "ورنہ",
    "سب", "کچھ", "کوئی", "کسی", "ہر", "جہاں", "کہاں", "کیوں", "کیسے",
])


def remove_stopwords(tokens: list) -> list:
    """Remove Urdu stopwords from a token list."""
    return [t for t in tokens if t not in URDU_STOPWORDS]


# ─────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "حکومتِ پاكستان نے عوام كے لئے نئی پالیسی کا اعلان کیا۔",
        "ي ك ة — Arabic chars that should normalize to Urdu equivalents",
        "یہ خبر بالکل غلط ہے! #FakeNews @user 2024",
    ]
    for s in samples:
        norm = normalize(s)
        tokens = tokenize(norm)
        clean_tokens = remove_stopwords(tokens)
        print(f"Original : {s}")
        print(f"Normalized: {norm}")
        print(f"Tokens    : {tokens}")
        print(f"No stops  : {clean_tokens}")
        print()
