from python_engine.paw import paws, paw_body


def test_kashmiri_word_splits_at_non_joining_letters():
    assert paws("کٲشُر") == ["کٲ", "شُر"]
    assert paws("زَبان") == ["زَ", "با", "ن"]
    assert paws("کِتاب") == ["کِتا", "ب"]


def test_joined_words_stay_one_piece():
    assert paws("مَنٛز") == ["مَنٛز"]
    assert paws("پؠٹھ") == ["پؠٹھ"]


def test_vowel_signs_stay_with_their_letter():
    assert paws("سۄمبرٲوِتھ") == ["سۄ", "مبر", "ٲ", "وِ", "تھ"]
    assert paw_body("شُر") == "شر"


def test_hamza_and_zwnj_break_pieces():
    assert paws("جاء") == ["جا", "ء"]
    assert paws("می‌خواہ") == ["می", "خو", "ا", "ہ"]
