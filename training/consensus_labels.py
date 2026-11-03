#!/usr/bin/env python3
"""Narrow noisy word labels down to the most likely Unicode, using many examples at once.

    python3 training/consensus_labels.py occurrences.jsonl out.jsonl [--min-margin 2]

Each input line is one word image with candidate transcriptions from any number of sources:

    {"image": "crops/p12_w7.png", "group": "kytaab",
     "candidates": {"book_label": "کتاب", "model": "سٕتاپیع", "unlimited_ocr": "کِتاب", "roman_dict": ["کِتاب", "کتاب"]}}

`group` ties together images of the same word (e.g. the book's Roman spelling, or a
normalised label); leave it out for a word seen once.

For every group, every candidate spelling gets a score:
  image evidence   --image-weight x sum over ALL images in the group of the word model's CTC
                   log-likelihood of that spelling (how well it explains each image)
  agreement        +3 x the votes it received (each source has ONE vote per image, split evenly
                   over the candidates it proposes), +1 x votes for the same letters with other signs
  attestation      +1 if the Kashmiri dictionary lexicon contains it (trust weight >= 2)
The winner becomes every member's label; `margin` (score lead over the runner-up, per image)
is the confidence. Keep only labels with a healthy margin for retraining.

Tested on a real scanned page (52 words, verified by hand) with a noisy label source that is wrong
30% of the time plus dictionary spellings of the book's Roman column: noisy labels alone 36/52
correct; consensus 38-39/52; keeping only margin >= 1 gives 31-34 labels at ~85-89% precision
(vs 69%). Consensus removes random noise, but a model that misreads the same letter everywhere
(an unseen font) cannot out-vote itself, so independent sources (another OCR engine, the book's
own text, a human) matter more than many copies; keep --image-weight low until the model has
been trained on the book's font.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import unicodedata

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from python_engine.normalizer import normalize_kashmiri  # noqa: E402
from python_engine.paw import binarize  # noqa: E402

SESSION = ort.InferenceSession(os.path.join(ROOT, "public/models/kashmiri_word_ocr_int8.onnx"))
UNITS = json.load(open(os.path.join(ROOT, "public/models/kashmiri_word_ocr.json"), encoding="utf-8"))["units"]
INDEX = {u: i for i, u in enumerate(UNITS)}


def bare(s: str) -> str:
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"))


def logits(path: str) -> np.ndarray:
    g = np.asarray(Image.open(path).convert("L"))
    ys, xs = np.nonzero(binarize(g))
    if len(ys):
        p = max(2, int(0.12 * (ys.max() - ys.min() + 1)))
        g = g[max(0, ys.min() - p):ys.max() + p + 1, max(0, xs.min() - p):xs.max() + p + 1]
    im = Image.fromarray(g)
    w = max(8, min(512, round(im.width * 48 / im.height)))
    a = 1 - np.asarray(im.resize((w, 48), Image.BILINEAR), dtype=np.float32) / 255
    return SESSION.run(None, {"image": a[:, ::-1].copy()[None, None]})[0][0]


def ctc_loglik(lg: np.ndarray, text: str) -> float:
    ids = [INDEX[c] for c in text if c in INDEX]
    if not ids or len(ids) > lg.shape[0]:
        return -1e4
    lp = torch.from_numpy(lg).log_softmax(-1)[:, None, :]
    return -F.ctc_loss(lp, torch.tensor([ids]), torch.tensor([lg.shape[0]]), torch.tensor([len(ids)]),
                       blank=0, reduction="sum", zero_infinity=True).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("occurrences")
    ap.add_argument("out")
    ap.add_argument("--min-margin", type=float, default=1.0, help="only write labels at least this confident")
    ap.add_argument("--image-weight", type=float, default=0.1,
                    help="how much the word model's reading of the image counts against the sources; raise it once "
                         "the model has been trained on the book's font (on an unseen font 0.1 works best)")
    ap.add_argument("--lexicon", default=os.path.join(ROOT, "public/lexicon/ks-lexicon.json"))
    a = ap.parse_args()
    lexicon = {normalize_kashmiri(r[0]) for r in json.load(open(a.lexicon, encoding="utf-8"))["forms"] if r[1] >= 2}

    occ = [json.loads(l) for l in open(a.occurrences, encoding="utf-8") if l.strip()]
    groups = collections.defaultdict(list)
    for i, o in enumerate(occ):
        cands = {}  # candidate -> {source: share of that source's single vote}
        for src, v in o.get("candidates", {}).items():
            vals = [normalize_kashmiri(str(c)).strip() for c in (v if isinstance(v, list) else [v])]
            vals = [c for c in dict.fromkeys(vals) if c]
            for c in vals:
                cands.setdefault(c, {})[src] = 1 / len(vals)  # a source proposing a list splits its vote
        o["_cands"], o["_lg"] = cands, logits(o["image"])
        groups[o.get("group") or f"#{i}"].append(o)

    kept = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for key, members in groups.items():
            pool = set().union(*[m["_cands"] for m in members])
            votes = collections.defaultdict(float)  # candidate -> summed vote shares over the group
            srcs_of = collections.defaultdict(set)
            for m in members:
                for c, shares in m["_cands"].items():
                    votes[c] += sum(shares.values())
                    srcs_of[c] |= set(shares)
            score = {}
            for c in pool:
                near = sum(v for d, v in votes.items() if d != c and bare(d) == bare(c))  # same letters, other signs
                score[c] = (a.image_weight * sum(ctc_loglik(m["_lg"], c) for m in members)
                            + 3 * votes[c] + 1 * near + (1 if c in lexicon else 0))
            ranked = sorted(score, key=lambda c: -score[c])
            margin = (score[ranked[0]] - score[ranked[1]]) / len(members) if len(ranked) > 1 else 99.0
            for m in members:
                if margin < a.min_margin:
                    continue
                kept += 1
                fh.write(json.dumps({"image": m["image"], "label": ranked[0], "margin": round(margin, 2),
                                     "group": key, "group_size": len(members),
                                     "sources": sorted(srcs_of[ranked[0]])}, ensure_ascii=False) + "\n")
    print(f"{kept}/{len(occ)} words labelled with margin >= {a.min_margin}")


if __name__ == "__main__":
    main()
