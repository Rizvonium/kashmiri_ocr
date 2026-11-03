#!/usr/bin/env python3
"""Build the compact OCR lexicon from the Kashmiri dictionary dataset.

Usage:
    python3 scripts/build_lexicon.py ../Kashmiri_dictionary public/lexicon/ks-lexicon.json [--corpus FILE ...]

Reads data/kashmiri_dictionary.jsonl and data/kashmiri_roman_pairs.tsv from the
dictionary repository (github.com/Rizvonium/Kashmiri_dictionary) and writes every
attested Perso-Arabic word form with a weight, a short English gloss, part of speech
and valency. The OCR post-corrector only ever substitutes forms from this list
(the "attested forms only" rule of the dual-lookup guide).

Weights (how much we trust a spelling):
    3  native modern-orthography headword
    2  word attested in AI4Bharat Aksharantar (mined from real Kashmiri text)
    1  word inside a multi-word native headword, or Grierson's 1932 spelling
    0.5 auto-transliterated / unverified headword
Corpus words (--corpus, e.g. data/corpora/*.txt from scripts/fetch_corpora.py) add running-text
attestation: seen >= 5 times -> 2, 2-4 times -> 1, once -> 0.5 (singletons are often typos).
Every form also carries its corpus count, used to sample realistic words for synthetic OCR data.
"""
import json
import os
import unicodedata
import re
import sys
from collections import defaultdict

import argparse
from collections import Counter

_ap = argparse.ArgumentParser()
_ap.add_argument("dict_dir", nargs="?", default="../Kashmiri_dictionary")
_ap.add_argument("out", nargs="?", default="public/lexicon/ks-lexicon.json")
_ap.add_argument("--corpus", nargs="*", default=[])
_ap.add_argument("--min-corpus-count", type=int, default=2, help="corpus words seen fewer times are left out")
_args = _ap.parse_args()
DICT_DIR, OUT = _args.dict_dir, _args.out
sys.path.insert(0, os.path.join(DICT_DIR, "scripts"))
from ks_normalize import normalize_ks  # noqa: E402

QUALITY_WEIGHT = {"native": 3.0, "source-perso-arabic (1932)": 1.0, "auto-transliterated": 0.5, "unverified": 0.5}
PUNCT = re.compile(r"[\s،؛؟۔,.;:!?()\[\]\"'«»…/\-–—]+")
ARABIC_LETTER = re.compile(r"[ؠ-يٮ-ۓۺ-ۿ]")


def valency_of(entry):
    pos = entry.get("pos", "")
    if "transitive" in pos:
        return "intransitive" if "intransitive" in pos else "transitive"
    d = entry.get("definition_en", "")[:80]
    if re.search(r"\bv\.\s?(t|tr|a)\b|\btrans\.", d):
        return "transitive"
    if re.search(r"\bv\.\s?i\b|\bintrans\.", d):
        return "intransitive"
    if re.search(r"\bcaus\.", d):
        return "causative"
    return ""


def short_gloss(entry):
    g = entry.get("gloss_en") or entry.get("definition_en") or ""
    g = re.sub(r"\s+", " ", g.split(" ¶ ")[0]).strip()
    return g[:70] + ("…" if len(g) > 70 else "")


_adding_corpus = False
forms = defaultdict(lambda: {"w": 0.0, "g": [], "p": set(), "v": set(), "n": 0, "dict": False})


def add(form, weight, gloss="", pos="", val=""):
    form = normalize_ks(form)
    if not form or not ARABIC_LETTER.search(form) or len(form) > 40:
        return
    if unicodedata.category(form[0]).startswith("M"):  # starts with a stray vowel sign: a data error
        return
    f = forms[form]
    f["w"] = max(f["w"], weight)
    f["dict"] = f["dict"] or not _adding_corpus
    for g in gloss.split("; ") if gloss else []:
        if g and g not in f["g"] and len(f["g"]) < 4:
            f["g"].append(g)
    if pos:
        f["p"].add(pos.split(" (")[0])
    if val:
        f["v"].add(val)


with open(os.path.join(DICT_DIR, "data/kashmiri_dictionary.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        e = json.loads(line)
        ks = e["headword_ks"]
        if not ks:
            continue
        w = QUALITY_WEIGHT.get(e["script_quality"], 0.5)
        words = [t for t in PUNCT.split(ks) if t]
        if len(words) == 1:
            add(words[0], w, short_gloss(e), e["pos"], valency_of(e))
        else:
            for t in words:  # words inside phrases are attested, but carry no gloss of their own
                add(t, min(w, 1.0))

with open(os.path.join(DICT_DIR, "data/kashmiri_roman_pairs.tsv"), encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        ks = line.split("\t")[0]
        for t in PUNCT.split(ks):
            if t:
                add(t, 2.0)

import unicodedata as _ud

# Running-text attestation from corpora
corpus_counts = Counter()
WORD = re.compile(r"[\u0620-\u065F\u0670-\u06D3\u06D5-\u06FF\u200c]+")
for path in _args.corpus:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            for t in WORD.findall(normalize_ks(line)):
                t = t.strip("\u200c")
                # a vowel sign separated from its letter by a space is a typo, not a word
                if t and not _ud.category(t[0]).startswith("M"):
                    corpus_counts[t] += 1
_adding_corpus = True
for t, c in corpus_counts.items():
    if t in forms:
        forms[t]["n"] = c
        if c >= 5:
            forms[t]["w"] = max(forms[t]["w"], 2.0)
    elif c >= _args.min_corpus_count:
        add(t, 2.0 if c >= 5 else 1.0)
        if normalize_ks(t) in forms:  # add() skips non-words; do not create empty entries
            forms[normalize_ks(t)]["n"] = c

# A corpus-only spelling that differs from a dictionary spelling only by Kashmiri look-alike letters
# or missing vowel signs (آزأدی vs آزٲدی, خوشحالی vs خۄشحٲلی) and is rarer than it in the corpora is
# a variant, not an authority: low trust, so OCR correction still prefers the dictionary spelling.
_LOOSE = str.maketrans({"ٲ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", "ۄ": "و", "ؤ": "و", "ؠ": "ی", "ێ": "ی", "ۍ": "ی",
                        "ئ": "ی", "ے": "ی", "ۂ": "ہ", "ۃ": "ہ", "ة": "ہ", "ں": "ن"})
import unicodedata as _ud


def _loose(t):
    return _ud.normalize("NFC", "".join(c for c in _ud.normalize("NFD", t) if _ud.category(c) != "Mn")).translate(_LOOSE)


best_dict_count = defaultdict(int)
for k, v in forms.items():
    if v["dict"]:
        best_dict_count[_loose(k)] = max(best_dict_count[_loose(k)], v["n"], 1)
demoted = 0
for k, v in forms.items():
    if not v["dict"] and v["n"] < best_dict_count.get(_loose(k), 0):
        v["w"] = 0.5
        demoted += 1
print(f"{demoted} corpus-only spellings demoted as variants of dictionary spellings")

rows = sorted(forms.items(), key=lambda kv: (-kv[1]["w"], kv[0]))
out = {
    "version": 1,
    "source": "github.com/Rizvonium/Kashmiri_dictionary (data/kashmiri_dictionary.jsonl + kashmiri_roman_pairs.tsv)"
              + ("; corpus counts: GAASH-Lab/Bolbosh (CC-BY-4.0), nawabhussain/Kashmiri-Language-Corpus (Apache-2.0)" if _args.corpus else ""),
    "fields": ["form", "weight", "gloss", "pos", "valency", "corpus_count"],
    "forms": [
        [k, v["w"], "; ".join(v["g"]), "/".join(sorted(v["p"])), "/".join(sorted(v["v"])), v["n"]] for k, v in rows
    ],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
print(f"{len(rows)} forms -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
