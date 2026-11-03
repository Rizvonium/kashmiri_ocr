"""
Module 1: KashmiriOrthographyNormalizer
Implements canonical Unicode normalization (NFC) followed by Kashmiri Perso-Arabic
canonical decomposition, grapheme remapping, and deterministic combining mark ordering.
"""

import re
import unicodedata
from typing import List, Dict, Optional

class KashmiriOrthographyNormalizer:
    """
    Enforces deterministic canonical Unicode sequences for printed and handwritten Kashmiri.
    Resolves double-diacritic code-point ordering chaos and maps legacy non-standard glyphs.
    """

    # Combining Mark Priority Ranks (lowest emitted first)
    # 1. Medial Cluster Sukun / Inverted-V (U+0652, U+065B)
    # 2. Shadda / Tashdid (U+0651)
    # 3. Primary Vowel Diacritics (Fatha U+064E, Damma U+064F, Kasra U+0650, Hamza Below U+0655,
    #    Small V Above U+065A, Subscript Alef U+0656, Wavy Dagger U+065F, etc.)
    # 4. Nasalization (Tanwin U+064B..U+064D, Sukun on Noon)
    MARK_PRIORITIES = {
        # Sukun / Jazm (onset cluster indicator in Kashmiri, e.g. ترٛومُت)
        '\u0652': 10, # Arabic Sukun
        '\u065B': 10, # Arabic Vowel Sign Inverted Small V (mapped to 0652)
        
        # Shadda
        '\u0651': 20, # Arabic Shaddah (gemination)
        
        # Vowels & Vocalic Modifiers
        '\u0655': 30, # Arabic Hamza Below (Kashmiri short /ɨ/)
        '\u0654': 31, # Arabic Hamza Above (Kashmiri /ə/)
        '\u065A': 32, # Arabic Small V Above (Kashmiri /e/ and /o/)
        '\u065F': 33, # Arabic Wavy Dagger Below (Kashmiri long /ɨː/)
        '\u064E': 34, # Fatha
        '\u064F': 35, # Damma
        '\u0650': 36, # Kasra
        '\u0656': 37, # Subscript Alef
        '\u0657': 38, # Inverted Damma
        '\u0670': 39, # Superscript Alef
        
        # Tanwin / Nasals
        '\u064B': 40, # Fathatan
        '\u064C': 41, # Dammatan
        '\u064D': 42, # Kasratan
    }

    # Canonical Glyphs Replacement Table
    CANONICAL_REPLACEMENTS: Dict[str, str] = {
        # Arabic Kaf to Persian/Kashmiri Keheh
        '\u0643': '\u06A9',  # ك -> ک
        
        # Arabic Ya with 2 dots to Farsi/Urdu Yeh
        '\u064A': '\u06CC',  # ي -> ی
        
        # Arabic Ya with Hamza to Yeh with Hamza Above
        '\u0626': '\u0626',  # ئ
        
        # Kirghiz Waw to Kashmiri Waw with Ring /ɔ/
        '\u06C5': '\u06C4',  # ۅ -> ۄ
        
        # Non-standard Yeh with tail / old palatal to Kashmiri Yeh (ؠ)
        '\u06CD': '\u0620',  # ۍ -> ؠ
        '\u06D0': '\u0620',  # ے with 2 dots below -> ؠ
        
        # Deprecated isolated Kashmiri Alef with Wavy Dagger (ٳ) -> اٟ
        '\u0673': '\u0627\u065F', # ٳ -> اٟ
        
        # Non-standard African Vowel Sign Inverted V -> Standard Sukun / Inverted-V
        '\u065B': '\u0652',
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
        3. Canonical Kashmiri character mapping
        4. Context-aware positional vowel expansion (ێ, ۆ)
        5. Syllable combining diacritic re-ordering
        6. Inter-word whitespace regularization
        """
        if not text:
            return ""

        # Step 1: NFC Normalization
        text = unicodedata.normalize('NFC', text)

        # Step 2: Strip stray zero-width chars and soft hyphens
        text = text.replace('\u200e', '').replace('\u200f', '').replace('\u00ad', '')

        # Step 3: Direct canonical glyph remapping
        for src, dst in self.CANONICAL_REPLACEMENTS.items():
            text = text.replace(src, dst)

        # Step 4: Positional decomposition of composite vowels
        text = self._normalize_positional_vowels(text)

        # Step 5: Syllabic combining mark re-ordering
        text = self._reorder_syllable_diacritics(text)

        # Step 6: Regularize whitespace (preserve single space, strip trailing)
        text = re.sub(r'[ \t]+', ' ', text)
        text = '\n'.join(line.strip() for line in text.splitlines())

        return text

    def _normalize_positional_vowels(self, text: str) -> str:
        """
        Handles Kashmiri vowels that have distinct medial vs final representations:
        - ۆ (U+06C6, short mid-back /o/) -> وٚ (U+0648 + U+065A)
        - ێ (U+06CE, short mid-front /e/):
          - Medially -> یٚ (U+06CC + U+065A)
          - Finally -> ےٚ (U+06D2 + U+065A)
        """
        # Replace ۆ unconditionally with וٚ
        text = text.replace('\u06C6', '\u0648\u065A')

        # Replace ێ based on whether it is followed by a letter or boundary
        # If at word end or followed by space/punctuation -> ےٚ
        # Otherwise medially -> یٚ
        def replace_yeh_vowel(match):
            pos = match.start()
            # Look ahead: if next char is end of string or non-Arabic letter
            if match.end() == len(text):
                return '\u06D2\u065A'
            next_char = text[match.end()]
            if not self._is_perso_arabic_letter(next_char):
                return '\u06D2\u065A' # Final position
            return '\u06CC\u065A'     # Medial position

        text = re.sub(r'\u06CE', replace_yeh_vowel, text)
        return text

    def _reorder_syllable_diacritics(self, text: str) -> str:
        """
        Per-syllable deterministic re-ordering:
        Groups base consonant + following sequence of combining marks,
        and sorts marks by canonical Kashmiri priority.
        """
        result = []
        i = 0
        n = len(text)

        while i < n:
            char = text[i]
            # If combining mark without preceding base, keep as is
            if self.diacritic_re.match(char):
                result.append(char)
                i += 1
                continue

            result.append(char)
            i += 1

            # Collect subsequent combining marks on this base character
            marks = []
            while i < n and self.diacritic_re.match(text[i]):
                marks.append(text[i])
                i += 1

            if len(marks) > 1:
                # Sort marks by priority key
                marks.sort(key=lambda m: self.MARK_PRIORITIES.get(m, 99))

            result.extend(marks)

        return "".join(result)

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
