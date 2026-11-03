#!/usr/bin/env python3
"""Convert an external word-image OCR dataset into this project's training layout.

    python3 training/prepare_external.py SOURCE OUT [--name NAME]

SOURCE is a .zip or a folder. The labels are found automatically:
  * labels/<id>.txt next to word_images/<id>.png   (Kaggle nawabhussaen/kashmiri-ocr)
  * a CSV with an image column and a text column   (e.g. HF Omarrran/40K..., labels.csv)
  * a JSONL with image and text keys               (e.g. HF Omarrran/600k... "JSONL" format)
  * a TXT with "image<TAB or space>text" per line  (CRNN-style labels.txt)
OUT gets labels/<i>.txt, word_images/<i>.png and meta.jsonl (source, original name), with
every label in the dictionary's Unicode convention (python_engine/normalizer.py):
Arabic ك/ي/ه mapped, legacy ٮ۪ -> ؠ, NFC, ZWJ/tatweel removed, trailing spaces dropped.
Labels that are empty or contain no Perso-Arabic letter are skipped and counted.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from python_engine.normalizer import normalize_kashmiri  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
IMG_COLS = ("file_name", "filename", "file", "image", "image_path", "img", "path", "image_file", "image file", "image id", "id")
TXT_COLS = ("text", "label", "transcript", "transcription", "word", "ground_truth", "gt")
KS_LETTER = re.compile(r"[ؠ-يٮ-ۓۺ-ۿ]")


class Source:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path) if path.endswith(".zip") else None
        self.root = path
        if self.zip:
            self.names = [n for n in self.zip.namelist() if not n.endswith("/")]
        else:
            self.names = [os.path.relpath(os.path.join(d, f), path) for d, _, fs in os.walk(path) for f in fs]
        self.by_base = {}
        for n in self.names:
            if n.lower().endswith(IMG_EXT):
                b = os.path.basename(n)
                self.by_base.setdefault(b, n)
                self.by_base.setdefault(os.path.splitext(b)[0], n)

    def read(self, name) -> bytes:
        if self.zip:
            return self.zip.read(name)
        with open(os.path.join(self.root, name), "rb") as fh:
            return fh.read()

    def find_image(self, ref: str):
        ref = ref.strip().replace("\\", "/")
        if ref in self.names:
            return ref
        b = os.path.basename(ref)
        return self.by_base.get(b) or self.by_base.get(os.path.splitext(b)[0]) or self.by_base.get(b.zfill(6) + ".png")


def pairs_from(src: Source):
    """Yield (image_name_in_source, raw_label) pairs and describe the format found."""
    label_files = [n for n in src.names if n.startswith("labels/") or "/labels/" in n]
    txt_labels = [n for n in label_files if n.endswith(".txt")]
    if txt_labels and any("word_images/" in n for n in src.names):
        yield "per-file labels (labels/<id>.txt)"
        for n in txt_labels:
            i = os.path.splitext(os.path.basename(n))[0]
            img = src.find_image(f"{i}.png")
            if img:
                yield img, src.read(n).decode("utf-8", "replace")
        return
    tables = sorted((n for n in src.names if n.lower().endswith((".csv", ".tsv", ".jsonl", ".txt"))
                     and not n.lower().endswith(("readme.txt", "license.txt"))),
                    key=lambda n: (0 if re.search(r"label|gt|annotation|train|data", n, re.I) else 1, len(n)))
    for t in tables:
        raw = src.read(t).decode("utf-8-sig", "replace")
        lines = [l for l in raw.splitlines() if l.strip()]
        if not lines:
            continue
        if t.lower().endswith(".jsonl"):
            first = json.loads(lines[0])
            ik = next((k for k in first if k.lower() in IMG_COLS), None)
            tk = next((k for k in first if k.lower() in TXT_COLS), None)
            if ik and tk:
                yield f"JSONL {t} (image={ik!r}, text={tk!r})"
                for l in lines:
                    r = json.loads(l)
                    img = src.find_image(str(r[ik]))
                    if img:
                        yield img, str(r[tk])
                return
            continue
        delim = "\t" if "\t" in lines[0] else ","
        if t.lower().endswith((".csv", ".tsv")) or delim == "\t":
            rows = list(csv.reader(io.StringIO("\n".join(lines)), delimiter=delim))
            head = [h.strip().lower() for h in rows[0]]
            ii = next((k for k, h in enumerate(head) if h in IMG_COLS), None)
            ti = next((k for k, h in enumerate(head) if h in TXT_COLS), None)
            body = rows[1:]
            if ii is None or ti is None:  # no header: guess column roles from the first row
                r0 = rows[0]
                ii = next((k for k, v in enumerate(r0) if src.find_image(v)), None)
                ti = next((k for k, v in enumerate(r0) if KS_LETTER.search(v)), None)
                body = rows
            if ii is not None and ti is not None:
                yield f"table {t} (image column {ii}, text column {ti})"
                for r in body:
                    if len(r) > max(ii, ti):
                        img = src.find_image(r[ii])
                        if img:
                            yield img, r[ti]
                return
        if t.lower().endswith(".txt"):  # "image text" per line
            m = re.match(r"^(\S+?\.(?:png|jpe?g|bmp|tiff?|webp))\s+(.+)$", lines[0], re.I)
            if m:
                yield f"text list {t} (image path + text per line)"
                for l in lines:
                    m = re.match(r"^(\S+?\.(?:png|jpe?g|bmp|tiff?|webp))\s+(.+)$", l, re.I)
                    if m and (img := src.find_image(m.group(1))):
                        yield img, m.group(2)
                return
    raise SystemExit("could not find a labels file this script understands; please open an issue with the layout")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--name", default=None, help="source name stored in meta.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    src = Source(a.source)
    name = a.name or os.path.splitext(os.path.basename(os.path.normpath(a.source)))[0]
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "word_images"), exist_ok=True)
    gen = pairs_from(src)
    print("format:", next(gen))
    stats = Counter()
    n = len(os.listdir(os.path.join(a.out, "labels")))  # append to an existing output folder
    with open(os.path.join(a.out, "meta.jsonl"), "a", encoding="utf-8") as meta:
        for img, raw in gen:
            if a.limit and stats["kept"] >= a.limit:
                break
            label = normalize_kashmiri(raw.replace("‍", "").strip())
            stats["changed by normalizer"] += label != raw.strip()
            if not label or not KS_LETTER.search(label):
                stats["skipped: no Kashmiri letters"] += 1
                continue
            data = src.read(img)
            ext = os.path.splitext(img)[1].lower()
            if ext != ".png":
                from PIL import Image

                b = io.BytesIO()
                Image.open(io.BytesIO(data)).save(b, "PNG")
                data = b.getvalue()
            with open(os.path.join(a.out, "word_images", f"{n}.png"), "wb") as fh:
                fh.write(data)
            with open(os.path.join(a.out, "labels", f"{n}.txt"), "w", encoding="utf-8") as fh:
                fh.write(label)
            meta.write(json.dumps({"id": n, "label": label, "source": name, "original": img}, ensure_ascii=False) + "\n")
            n += 1
            stats["kept"] += 1
    print(dict(stats))


if __name__ == "__main__":
    main()
