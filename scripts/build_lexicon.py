#!/usr/bin/env python3
"""Build the compact OCR lexicon from the Kashmiri dictionary dataset.

Usage:
    python3 scripts/build_lexicon.py ../Kashmiri_dictionary public/lexicon/ks-lexicon.json

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
"""
import json
import os
import re
import sys
from collections import defaultdict

DICT_DIR = sys.argv[1] if len(sys.argv) > 1 else "../Kashmiri_dictionary"
OUT = sys.argv[2] if len(sys.argv) > 2 else "public/lexicon/ks-lexicon.json"
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


forms = defaultdict(lambda: {"w": 0.0, "g": [], "p": set(), "v": set()})


def add(form, weight, gloss="", pos="", val=""):
    form = normalize_ks(form)
    if not form or not ARABIC_LETTER.search(form) or len(form) > 40:
        return
    f = forms[form]
    f["w"] = max(f["w"], weight)
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

rows = sorted(forms.items(), key=lambda kv: (-kv[1]["w"], kv[0]))
out = {
    "version": 1,
    "source": "github.com/Rizvonium/Kashmiri_dictionary (data/kashmiri_dictionary.jsonl + kashmiri_roman_pairs.tsv)",
    "fields": ["form", "weight", "gloss", "pos", "valency"],
    "forms": [
        [k, v["w"], "; ".join(v["g"]), "/".join(sorted(v["p"])), "/".join(sorted(v["v"]))] for k, v in rows
    ],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
print(f"{len(rows)} forms -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
