"""
Module 1: KashmiriOrthographyNormalizer
Canonical Unicode (NFC) and Kashmiri Perso-Arabic letter normalization, using the same
convention as the Kashmiri dictionary dataset so OCR output can be matched against it.
"""

import re
import unicodedata
from typing import List, Dict, Optional

class KashmiriOrthographyNormalizer:
    """
    Enforces deterministic canonical Unicode sequences for printed and handwritten Kashmiri.
    Resolves double-diacritic code-point ordering chaos and maps legacy non-standard glyphs.
    """

    # Canonical glyph replacements. Same convention as the Kashmiri dictionary dataset
    # (scripts/ks_normalize.py in github.com/Rizvonium/Kashmiri_dictionary) and the dual-lookup
    # guide, section 6.3: Arabic-keyboard letters are mapped, Kashmiri letters and vowel signs
    # (ٲ ۄ ؠ ۍ ێ, ٕ ٔ ٚ ٛ ٟ ...) are always kept. Earlier versions turned ۍ into ؠ and ٛ into a
    # sukun, which produced spellings the dictionary never contains.
    CANONICAL_REPLACEMENTS: Dict[str, str] = {
        '\u0643': '\u06A9',  # ك -> ک (keheh)
        '\u064A': '\u06CC',  # ي -> ی (farsi yeh)
        '\u0649': '\u06CC',  # ى alef maksura -> ی
        '\u0647': '\u06C1',  # ه -> ہ (heh goal)
        '\u06C0': '\u06C2',  # ۀ -> ۂ
        '\u06C5': '\u06C4',  # ۅ (Kirghiz waw) -> ۄ (Kashmiri waw with ring)
        '\u06ED': '\u065F',  # legacy small low meem -> wavy hamza below
        '\u0673': '\u0627\u065F',  # deprecated ٳ -> اٟ
        '\u0640': '',  # tatweel
    }

    def __init__(self):
        # Regular expression for Perso-Arabic diacritics
        self.diacritic_re = re.compile(r'[\u064B-\u065F\u0670]')
        # Non-joining consonants in Perso-Arabic
        self.non_joiners = set('\u0627\u0622\u062F\u0688\u0630\u0631\u0691\u0632\u0698\u0648\u06C4')

    def normalize(self, text: str) -> str:
        """
        Main normalization entry point:
        1. Unicode NFC Normalization
        2. Strip invisible control characters (except ZWNJ/ZWJ where structurally needed)
        3. Presentation forms and legacy code points
        4. Canonical Kashmiri character mapping (dictionary convention)
        5. ۆ -> وٚ
        6. Duplicate marks collapsed, NFC (canonical order of vowel signs)
        7. Inter-word whitespace regularization
        """
        if not text:
            return ""

        # Step 1: NFC Normalization
        text = unicodedata.normalize('NFC', text)

        # Step 2: Strip stray zero-width chars and soft hyphens
        text = text.replace('\u200e', '').replace('\u200f', '').replace('\u00ad', '')

        # Step 3: Arabic presentation forms -> base letters; legacy Kashmiri yeh (ٮ۪) -> ؠ
        text = "".join(unicodedata.normalize("NFKC", c) if "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff" else c
                       for c in text).replace("\u066e\u06ea", "\u0620")

        # Step 4: Direct canonical glyph remapping
        for src, dst in self.CANONICAL_REPLACEMENTS.items():
            text = text.replace(src, dst)

        # Step 5: ۆ -> وٚ (ێ is a Kashmiri letter in the dataset convention and is kept)
        text = text.replace('\u06C6', '\u0648\u065A')

        # Step 6: collapse repeated identical marks, then NFC fixes the order of vowel signs
        # (Unicode canonical order; the dual-lookup validator requires NFC).
        text = re.sub(r'([\u064B-\u065F\u0670])\1+', r'\1', text)
        text = unicodedata.normalize('NFC', text)

        # Step 6: Regularize whitespace (preserve single space, strip trailing)
        text = re.sub(r'[ \t]+', ' ', text)
        text = '\n'.join(line.strip() for line in text.splitlines())

        return text

    @staticmethod
    def _is_perso_arabic_letter(ch: str) -> bool:
        """Returns True if character is a Perso-Arabic base consonant or vowel letter."""
        code = ord(ch)
        return (0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFF) and not (0x064B <= code <= 0x065F)


# Global singleton instance for fast imports
kashmiri_normalizer = KashmiriOrthographyNormalizer()

def normalize_kashmiri(text: str) -> str:
    """Helper wrapper for quick normalization."""
    return kashmiri_normalizer.normalize(text)
