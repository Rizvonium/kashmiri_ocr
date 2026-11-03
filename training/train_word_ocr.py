#!/usr/bin/env python3
"""Train a Kashmiri word recognizer (CNN + BiLSTM + CTC) and compare unit spaces.

    python3 training/train_word_ocr.py --data /kaggle/input/kashmiri-ocr data/synth \
        --units paws --epochs 30 --out runs/paws

Data folders use the layout of the Kaggle "Kashmiri OCR" dataset (nawabhussaen/kashmiri-ocr):
labels/<id>.txt + word_images/<id>.png, optionally meta.jsonl (from synth_words.py) with
the font of each image.

--units chars   one output per character (letters and vowel signs, ~65 units)
--units paws    the expanded space: every PAW (piece of a word between non-joining letters,
                e.g. کٲشُر = کٲ + شُر) seen at least --min-paw-count times in training is
                its own unit; rarer PAWs fall back to characters, so any word can still be
                written. Compare both on the same split.

Train / validation / test are split by the label text (hash), so a test word never
appears in training in any font. Reports character and word error rates per source and
for Nastaliq vs Naskh fonts, with and without dictionary correction, and exports ONNX.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from python_engine.normalizer import normalize_kashmiri  # noqa: E402
from python_engine.paw import paws  # noqa: E402

H = 48
MAX_W = 512


# --------------------------------------------------------------------------- data

def split_of(label: str) -> str:
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if h < 5 else "test" if h < 10 else "train"


def load_items(folders: list[str], limit: int | None) -> list[dict]:
    items = []
    for folder in folders:
        src = os.path.basename(os.path.normpath(folder))
        meta = {}
        mp = os.path.join(folder, "meta.jsonl")
        if os.path.exists(mp):
            for line in open(mp, encoding="utf-8"):
                m = json.loads(line)
                meta[str(m["id"])] = m
        lab_dir = os.path.join(folder, "labels")
        names = sorted(os.listdir(lab_dir), key=lambda n: (len(n), n))
        if limit:
            names = names[:limit]
        for n in names:
            i = n[:-4]
            label = normalize_kashmiri(open(os.path.join(lab_dir, n), encoding="utf-8").read().strip())
            img = os.path.join(folder, "word_images", f"{i}.png")
            if not label or not os.path.exists(img):
                continue
            m = meta.get(i, {})
            items.append({"img": img, "label": label, "source": src, "font": m.get("font", "unknown"),
                          "nastaliq": m.get("nastaliq"), "split": split_of(label)})
    return items


def load_image(path: str, train: bool, rng: random.Random) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA"):  # transparent background -> white
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert("RGBA"))
    im = im.convert("L")
    w = max(1, round(im.width * H / im.height))
    if train:
        w = max(8, round(w * rng.uniform(0.85, 1.15)))
    im = im.resize((min(w, MAX_W), H), Image.BILINEAR)
    a = 1.0 - np.asarray(im, dtype=np.float32) / 255.0  # ink = 1
    return a[:, ::-1].copy()  # flip: right-to-left text becomes left-to-right in time


class Words(torch.utils.data.Dataset):
    def __init__(self, items, tok, train):
        self.items, self.tok, self.train = items, tok, train
        self.rng = random.Random(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        it = self.items[k]
        return load_image(it["img"], self.train, self.rng), self.tok.encode(it["label"]), k


def collate(batch):
    imgs, labels, idx = zip(*batch)
    width = max(i.shape[1] for i in imgs)
    width = int(math.ceil(width / 4) * 4)
    x = torch.zeros(len(imgs), 1, H, width)
    for b, i in enumerate(imgs):
        x[b, 0, :, : i.shape[1]] = torch.from_numpy(i)
    widths = torch.tensor([i.shape[1] for i in imgs])
    targets = torch.tensor([t for l in labels for t in l], dtype=torch.long)
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return x, widths, targets, lengths, list(idx)


# --------------------------------------------------------------------------- units

class Tokenizer:
    def __init__(self, units: list[str], mode: str):
        self.mode = mode
        self.units = units  # index 0 is the CTC blank
        self.index = {u: i for i, u in enumerate(units)}
        self.multi = {u for u in units if len(u) > 1}

    @classmethod
    def build(cls, labels: list[str], mode: str, min_paw: int, max_paw: int):
        chars = sorted({c for l in labels for c in l})
        if mode == "chars":
            return cls(["<blank>"] + chars, mode)
        counts = Counter(p for l in labels for p in paws(l) if len(p) > 1)
        frequent = [p for p, c in counts.most_common(max_paw) if c >= min_paw]
        return cls(["<blank>"] + chars + frequent, mode)

    def pieces(self, label: str) -> list[str]:
        if self.mode == "chars":
            return list(label)
        out = []
        for p in paws(label):
            out.extend([p] if p in self.multi else list(p))
        return out

    def encode(self, label: str) -> list[int]:
        return [self.index[u] for u in self.pieces(label) if u in self.index]

    def decode(self, ids: list[int]) -> str:
        out, prev = [], 0
        for i in ids:
            if i != prev and i != 0:
                out.append(self.units[i])
            prev = i
        return "".join(out)


# --------------------------------------------------------------------------- model

class CRNN(nn.Module):
    def __init__(self, n_out: int, hidden: int = 256):
        super().__init__()

        def block(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)] + (
                [nn.MaxPool2d(pool)] if pool else [])

        self.cnn = nn.Sequential(
            *block(1, 64, (2, 2)), *block(64, 128, (2, 2)), *block(128, 256, None), *block(256, 256, (2, 1)),
            *block(256, 384, None), *block(384, 384, (2, 1)), nn.Dropout2d(0.1))
        self.rnn = nn.LSTM(384 * 3, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, n_out)

    def forward(self, x):  # x: B,1,48,W  ->  B,T,C (T = W/4)
        f = self.cnn(x)  # B,384,3,W/4
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        y, _ = self.rnn(f)
        return self.fc(y)


# --------------------------------------------------------------------------- metrics

def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


class Lexicon:
    """Dictionary correction: replace a word not in the lexicon by its single best close match."""

    def __init__(self, words: set[str]):
        self.words = words
        self.by_len = defaultdict(list)
        for w in words:
            self.by_len[len(w)].append(w)

    def correct(self, w: str, cutoff: float = 0.82) -> str:
        if w in self.words or len(w) < 3:
            return w
        pool = [x for n in range(len(w) - 2, len(w) + 3) for x in self.by_len.get(n, ())]
        best = difflib.get_close_matches(w, pool, n=2, cutoff=cutoff)
        if len(best) == 1 or (len(best) == 2 and difflib.SequenceMatcher(None, w, best[0]).ratio()
                              - difflib.SequenceMatcher(None, w, best[1]).ratio() >= 0.04):
            return best[0]
        return w


def evaluate(model, loader, tok, items, device, lexicon: Lexicon | None = None):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, widths, _, _, idx in loader:
            out = model(x.to(device)).argmax(-1).cpu()
            for b, k in enumerate(idx):
                t = int(math.ceil(widths[b].item() / 4))
                rows.append((k, tok.decode(out[b, :t].tolist())))
    groups = defaultdict(lambda: {"n": 0, "chars": 0, "cerr": 0, "werr": 0, "werr_dict": 0, "in_lex": 0})
    for k, pred in rows:
        it = items[k]
        fixed = lexicon.correct(pred) if lexicon else pred
        keys = ["all", f"source:{it['source']}"]
        if it["nastaliq"] is not None:
            keys.append("nastaliq" if it["nastaliq"] else "naskh-style")
        if lexicon:
            keys.append("label-in-dictionary" if it["label"] in lexicon.words else "label-not-in-dictionary")
        for key in keys:
            g = groups[key]
            g["n"] += 1
            g["chars"] += len(it["label"])
            g["cerr"] += lev(pred, it["label"])
            g["werr"] += pred != it["label"]
            g["werr_dict"] += fixed != it["label"]
    return {k: {"n": g["n"], "CER": g["cerr"] / max(1, g["chars"]), "WER": g["werr"] / g["n"],
                "WER_with_dictionary": g["werr_dict"] / g["n"] if lexicon else None} for k, g in groups.items()}


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--units", choices=["chars", "paws"], default="chars")
    ap.add_argument("--min-paw-count", type=int, default=25)
    ap.add_argument("--max-paw-units", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="max items per data folder (quick tests)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (smoke tests)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lexicon", default=os.path.join(ROOT, "public/lexicon/ks-lexicon.json"))
    ap.add_argument("--out", default="runs/latest")
    a = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    items = load_items(a.data, a.limit)
    split = defaultdict(list)
    for k, it in enumerate(items):
        split[it["split"]].append(k)
    train_labels = [items[k]["label"] for k in split["train"]]
    tok = Tokenizer.build(train_labels, a.units, a.min_paw_count, a.max_paw_units)
    covered = sum(len(tok.pieces(l)) for l in train_labels) / max(1, sum(len(l) for l in train_labels))
    print(f"{len(items)} images | train {len(split['train'])} val {len(split['val'])} test {len(split['test'])} | "
          f"units={a.units}: {len(tok.units) - 1} outputs, {covered:.2f} units per character | device {device}")

    def loader(keys, train):
        ds = Words(items, tok, train)
        sub = torch.utils.data.Subset(ds, keys)
        return torch.utils.data.DataLoader(sub, batch_size=a.batch, shuffle=train, num_workers=a.workers,
                                           collate_fn=collate, drop_last=False)

    tr, va, te = loader(split["train"], True), loader(split["val"], False), loader(split["test"], False)
    model = CRNN(len(tok.units)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(tr)
    if a.max_steps:
        steps = min(steps, a.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1), pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best, step, history = 1e9, 0, []
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for x, widths, targets, lengths, _ in tr:
            x = x.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(x)
            logp = logits.float().log_softmax(-1).permute(1, 0, 2)  # T,B,C
            in_len = torch.clamp(torch.ceil(widths / 4).long(), max=logp.shape[0])
            loss = ctc(logp, targets, in_len, lengths)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        val = evaluate(model, va, tok, items, device)["all"]
        history.append({"epoch": epoch + 1, "loss": total / max(1, len(tr)), **val, "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch + 1}: loss {history[-1]['loss']:.3f} | val CER {val['CER']:.3f} WER {val['WER']:.3f} "
              f"| {history[-1]['minutes']:.1f} min", flush=True)
        if val["CER"] < best:
            best = val["CER"]
            torch.save({"model": model.state_dict(), "units": tok.units, "mode": tok.mode}, os.path.join(a.out, "best.pt"))
        if a.max_steps and step >= a.max_steps:
            break

    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    lex_words = set(train_labels)
    if os.path.exists(a.lexicon):
        for form, w, *_ in json.load(open(a.lexicon, encoding="utf-8"))["forms"]:
            if w >= 1:
                lex_words.add(normalize_kashmiri(form))
    test = evaluate(model, te, tok, items, device, Lexicon(lex_words))
    report = {"units": a.units, "n_outputs": len(tok.units) - 1, "data": a.data, "history": history, "test": test}
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), ensure_ascii=False, indent=1)
    print("\nTEST (unseen words):")
    for k, v in sorted(test.items()):
        d = f" | WER with dictionary {v['WER_with_dictionary']:.3f}" if v["WER_with_dictionary"] is not None else ""
        print(f"  {k:28s} n={v['n']:6d}  CER {v['CER']:.3f}  WER {v['WER']:.3f}{d}")

    # ONNX export for the web app (onnxruntime-web): input 1x1x48xW, ink=1, flipped left-right
    model.eval().cpu()
    dummy = torch.zeros(1, 1, H, 128)
    onnx_path = os.path.join(a.out, "kashmiri_word_ocr.onnx")
    try:
        torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "time"}},
                          opset_version=17, dynamo=False)
        json.dump({"units": tok.units, "mode": tok.mode, "height": H, "flip_horizontal": True, "ink_is_one": True},
                  open(os.path.join(a.out, "kashmiri_word_ocr.json"), "w"), ensure_ascii=False)
        print(f"exported {onnx_path}")
    except Exception as e:  # export is a convenience; training results stand without it
        print(f"ONNX export failed: {e}")


if __name__ == "__main__":
    main()
