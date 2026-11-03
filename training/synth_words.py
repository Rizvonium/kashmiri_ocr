#!/usr/bin/env python3
"""Render Kashmiri dictionary words as training images (word-level OCR).

    python3 training/synth_words.py --fonts fonts/ --out data/synth --n 60000

Writes the same layout as the Kaggle "Kashmiri OCR" dataset (nawabhussaen/kashmiri-ocr):
    out/labels/<i>.txt        Unicode label (NFC, dictionary convention)
    out/word_images/<i>.png   grayscale word image
plus out/meta.jsonl with the font and augmentation used, so results can be split by font.

Words come from public/lexicon/ks-lexicon.json (the Kashmiri dictionary export), sampled
with their trust weight. A font is only used for a word if it has a glyph for every
character (Gulzar, for example, has no ٲ or ؠ) - otherwise the image would teach the
model a wrong shape. Needs Pillow with libraqm for Arabic shaping, and fontTools.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from python_engine.normalizer import normalize_kashmiri  # noqa: E402

NASTALIQ_HINTS = ("nastaliq", "nastaleeq", "gulzar", "mehr", "jameel", "awami")


def font_charset(path: str) -> set[str]:
    from fontTools.ttLib import TTFont

    f = TTFont(path, lazy=True)
    return {chr(cp) for cp in f.getBestCmap()}


def augment(img: Image.Image, rng: random.Random, size: int = 40) -> tuple[Image.Image, list[str]]:
    """Camera / print damage, kept mild so labels stay readable (milder on small text)."""
    done = []
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=True, fillcolor=255)
        done.append("rotate")
    if rng.random() < 0.25:
        # MinFilter thickens ink; MaxFilter thins it, which erases small text, so only on large text
        thin = size >= 40 and rng.random() < 0.5
        img = img.filter(ImageFilter.MaxFilter(3) if thin else ImageFilter.MinFilter(3))
        done.append("thin-ink" if thin else "bold-ink")
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.3)))
        done.append("blur")
    a = np.asarray(img).astype(np.float32)
    if rng.random() < 0.4:  # uneven lighting / paper tone
        h, w = a.shape
        grad = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.85, 1.0), w)[None, :]
        a = 255 - (255 - a) * rng.uniform(0.6, 1.0)
        a = a * grad
        done.append("light")
    if rng.random() < 0.4:
        a = a + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(4, 14), a.shape)
        done.append("noise")
    if rng.random() < 0.3:  # downscale-upscale = low-resolution camera
        img2 = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        s = rng.uniform(0.55 if size < 36 else 0.4, 0.75)
        small = img2.resize((max(8, int(img2.width * s)), max(8, int(img2.height * s))), Image.BILINEAR)
        a = np.asarray(small.resize(img2.size, Image.BILINEAR)).astype(np.float32)
        done.append("lowres")
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), done


def render(word: str, font: ImageFont.FreeTypeFont, pad: int) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), word, font=font, direction="rtl", language="ks")
    img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), word, font=font, fill=0, direction="rtl", language="ks")
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fonts", required=True, help="folder of .ttf/.otf fonts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--lexicon", default=os.path.join(ROOT, "public/lexicon/ks-lexicon.json"))
    ap.add_argument("--nastaliq-share", type=float, default=0.7, help="share of images drawn in Nastaliq fonts")
    ap.add_argument("--min-weight", type=float, default=1.0, help="skip low-trust spellings")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if not features.check("raqm"):
        sys.exit("Pillow was built without libraqm: Arabic letters would not join. Install libraqm.")

    rng = random.Random(a.seed)
    rows = json.load(open(a.lexicon, encoding="utf-8"))["forms"]
    words, weights = [], []
    for form, w, *_ in rows:
        form = normalize_kashmiri(form)
        if w >= a.min_weight and form and " " not in form and len(form) <= 24:
            words.append(form)
            weights.append(w)

    fonts = []
    for f in sorted(os.listdir(a.fonts)):
        if f.lower().endswith((".ttf", ".otf")):
            p = os.path.join(a.fonts, f)
            try:
                chars = font_charset(p)
            except Exception as e:  # e.g. a failed download saved under a .ttf name
                print(f"skipping {f}: not a usable font ({e})")
                continue
            fonts.append({"name": os.path.splitext(f)[0], "path": p, "chars": chars,
                          "nastaliq": any(h in f.lower() for h in NASTALIQ_HINTS)})
    if not fonts:
        sys.exit("no fonts found")
    print("fonts:", ", ".join(f"{f['name']}{' (nastaliq)' if f['nastaliq'] else ''}" for f in fonts))

    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "word_images"), exist_ok=True)
    cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
    skipped = 0
    with open(os.path.join(a.out, "meta.jsonl"), "w", encoding="utf-8") as meta:
        i = 0
        while i < a.n:
            word = rng.choices(words, weights)[0]
            want_nastaliq = rng.random() < a.nastaliq_share
            usable = [f for f in fonts if set(word) <= f["chars"] or not f["chars"]]
            pool = [f for f in usable if f["nastaliq"] == want_nastaliq] or usable
            if not pool:
                skipped += 1
                continue
            f = rng.choice(pool)
            size = rng.choice([28, 32, 36, 40, 48])
            key = (f["path"], size)
            if key not in cache:
                cache[key] = ImageFont.truetype(f["path"], size, layout_engine=ImageFont.Layout.RAQM)
            img = render(word, cache[key], pad=rng.randint(4, 14))
            img, aug = augment(img, rng, size)
            img.save(os.path.join(a.out, "word_images", f"{i}.png"))
            with open(os.path.join(a.out, "labels", f"{i}.txt"), "w", encoding="utf-8") as fh:
                fh.write(word)
            meta.write(json.dumps({"id": i, "label": word, "font": f["name"], "nastaliq": f["nastaliq"],
                                   "size": size, "aug": aug}, ensure_ascii=False) + "\n")
            i += 1
    print(f"wrote {a.n} images to {a.out} ({skipped} words skipped: no font had all their glyphs)")


if __name__ == "__main__":
    main()
