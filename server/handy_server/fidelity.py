"""Safety net: reject edits that alter numerical values or protected terms.

It cannot prove semantic equivalence. Keep both raw and edited variants.
"""
from collections import Counter
import re
import unicodedata


# Collect digit groups; punctuation changes are fine but numeric digits must not
# be inserted, dropped or reordered within a group. Unicode digits normalize.
NUMBER = re.compile(r"\d+(?:[.,٫٬/:\-]\d+)*", re.UNICODE)


def numeric_tokens(text: str) -> Counter[str]:
    def normalize(match: re.Match[str]) -> str:
        raw = match.group()
        digits = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in raw)
        return digits.replace("٬", ",").replace("٫", ".")

    return Counter(normalize(m) for m in NUMBER.finditer(text))


def protect_source(raw: str, edited: str, protected_terms: list[str]) -> list[str]:
    warnings = []
    if numeric_tokens(raw) != numeric_tokens(edited):
        warnings.append("Numeric values changed; original text returned")
    for term in protected_terms:
        if term.strip() and term.casefold() in raw.casefold() and term.casefold() not in edited.casefold():
            warnings.append("Protected term missing; original text returned")
            break
    if not edited.strip():
        warnings.append("Model returned empty text; original text returned")
    return warnings
