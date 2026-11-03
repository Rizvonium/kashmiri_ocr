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
    
    # Non-standard Yeh with tail to Kashmiri Yeh (ؠ)
    assert normalizer.normalize("ۍ") == "ؠ"

def test_positional_vowel_decomposition():
    # ۆ (U+06C6) -> وٚ (U+0648 + U+065A)
    assert normalize_kashmiri("ۆ") == "\u0648\u065A"
    
    # Final ێ (U+06CE) -> ےٚ (U+06D2 + U+065A)
    assert normalize_kashmiri("مےٚ") == "مےٚ"
    
    # Medial ێ (U+06CE) -> یٚ (U+06CC + U+065A)
    assert normalize_kashmiri("مێم") == "میٚم"

def test_deterministic_syllable_mark_ordering():
    # Sukun (U+0652) must precede Fatha (U+064E)
    # Input with Fatha then Sukun
    raw = "ت\u064E\u0652"
    normalized = normalize_kashmiri(raw)
    
    codepoints = [ord(c) for c in normalized]
    assert codepoints == [0x062A, 0x0652, 0x064E]
