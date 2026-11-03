import pytest
from python_engine.normalizer import KashmiriOrthographyNormalizer, normalize_kashmiri

def test_canonical_glyph_mappings():
    normalizer = KashmiriOrthographyNormalizer()
    
    # Arabic Kaf to Persian Keheh
    assert normalizer.normalize("كتاب") == "کتاب"
    
    # Arabic Ya to Farsi/Urdu Yeh
    assert normalizer.normalize("علي") == "علی"
    
    # Kirghiz Waw to Kashmiri Waw with Ring
    assert normalizer.normalize("ۅ") == "ۄ"
    
    # Kashmiri letters are kept as they are (dictionary / dual-lookup convention)
    assert normalizer.normalize("ۍ") == "ۍ"
    assert normalizer.normalize("آسمٲنۍ") == "آسمٲنۍ"
    assert normalizer.normalize("مَنٛز") == "مَنٛز"  # U+065B kept, not turned into sukun

    # Arabic heh -> heh goal, as in the dictionary dataset
    assert normalizer.normalize("ه") == "ہ"

def test_positional_vowel_decomposition():
    # ۆ (U+06C6) -> وٚ (U+0648 + U+065A)
    assert normalize_kashmiri("ۆ") == "\u0648\u065A"
    
    # Final ێ (U+06CE) -> ےٚ (U+06D2 + U+065A)
    assert normalize_kashmiri("مےٚ") == "مےٚ"
    
    # ێ (U+06CE) is a Kashmiri letter in the dictionary convention and is kept
    assert normalize_kashmiri("مێم") == "مێم"

def test_deterministic_syllable_mark_ordering():
    # Vowel signs come out in Unicode canonical (NFC) order whatever order they were typed in
    import unicodedata
    for raw in ("ت\u064E\u0652", "ت\u0652\u064E"):
        normalized = normalize_kashmiri(raw)
        assert normalized == unicodedata.normalize("NFC", normalized)
    assert normalize_kashmiri("ت\u064E\u0652") == normalize_kashmiri("ت\u0652\u064E")
    # repeated identical marks collapse
    assert normalize_kashmiri("ت\u064E\u064E") == "ت\u064E"


def test_matches_dictionary_normalizer_on_shared_cases():
    cases = {"اَسنہ ٕ دٕہان": "اَسنہٕ دٕہان", "اَمہ ِکِس": "اَمہِ کِس", "كتاب": "کتاب", "اڈلنۆک": "اڈلنوٚک", "سۭتۍ": "سٟتۍ", "پٮ۪ٹھ": "پؠٹھ"}
    for raw, want in cases.items():
        assert normalize_kashmiri(raw) == want
