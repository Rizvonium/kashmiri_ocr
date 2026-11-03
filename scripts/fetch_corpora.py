#!/usr/bin/env python3
"""Download open Kashmiri Unicode text used to extend the lexicon and the synthetic OCR data.

    python3 scripts/fetch_corpora.py data/corpora

* GAASH-Lab/Bolbosh (Hugging Face, CC-BY-4.0): ~40,000 vowel-marked sentences (the TTS
  transcripts only, fetched through the dataset viewer API; the 12.7 GB of audio is skipped).
* nawabhussain/Kashmiri-Language-Corpus (Hugging Face, Apache-2.0): ~47,000 literary/news lines.

Both need the normalizer (Arabic ك/ي/ه and legacy ٮ۪ are common in them). The raw files
are not committed; scripts/build_lexicon.py --corpus reads them and keeps word counts.
"""
import concurrent.futures as cf
import csv
import json
import os
import sys
import time
import urllib.request

out = sys.argv[1] if len(sys.argv) > 1 else "data/corpora"
os.makedirs(out, exist_ok=True)


def fetch(url, tries=5):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"failed: {url}")


csv_bytes = fetch("https://huggingface.co/datasets/nawabhussain/Kashmiri-Language-Corpus/resolve/main/Kashmiri-Language-Corpus.csv")
rows = list(csv.DictReader(csv_bytes.decode("utf-8").splitlines()))
with open(os.path.join(out, "kashmiri_language_corpus.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(r["sentence"].replace("\n", " ") for r in rows))
print(f"Kashmiri-Language-Corpus: {len(rows)} lines")

splits = {"train": 33182, "validation": 4542, "test": 2272}
jobs = [(s, o) for s, n in splits.items() for o in range(0, n, 100)]


def rows_of(job):
    s, o = job
    url = f"https://datasets-server.huggingface.co/rows?dataset=GAASH-Lab/Bolbosh&config=default&split={s}&offset={o}&length=100"
    return [r["row"]["text"] for r in json.loads(fetch(url))["rows"]]


with cf.ThreadPoolExecutor(8) as ex:
    texts = [t for batch in ex.map(rows_of, jobs) for t in batch]
with open(os.path.join(out, "bolbosh.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(t.replace("\n", " ") for t in texts))
print(f"Bolbosh: {len(texts)} sentences")
