"""
Text normalization, suffix stripping, and feature tokenization.
Robust to open country sets (US, India, France, etc.).
"""

import re
import unicodedata

# Common legal suffixes across US, India, and France
LEGAL_SUFFIXES = [
    r"\bprivate limited\b",
    r"\bpvt ltd\b",
    r"\bpvt\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\binc\b",
    r"\bincorporated\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\bsarl\b",
    r"\bsas\b",
    r"\beurl\b",
    r"\bsa\b",
    r"\bsci\b",
]
LEGAL_REGEX = re.compile("|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

# Standard address replacements
ADDRESS_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bflr\b": "floor",
    r"\bapt\b": "apartment",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
}


def strip_accents(text: str) -> str:
    """Normalize accented characters (e.g., French accents: é, è, ê -> e)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Basic text sanitization: accents, symbols, whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = strip_accents(text.lower())
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_business_name(name: str) -> str:
    """Clean business name and strip legal company suffixes."""
    text = clean_text(name)
    text = LEGAL_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_address(address: str) -> str:
    """Standardize common address abbreviations and formats."""
    text = clean_text(address)
    for pattern, replacement in ADDRESS_MAP.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text: str) -> set:
    """Extract set of numeric tokens (building numbers, postal codes)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def extract_tokens(text: str, min_len: int = 3) -> list:
    """Extract filtered tokens of minimum length."""
    tokens = clean_text(text).split()
    return [t for t in tokens if len(t) >= min_len]
