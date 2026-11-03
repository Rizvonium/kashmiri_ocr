"""
PAW (piece of Arabic word / ligature) tools for Kashmiri Nastaliq.

A Perso-Arabic word breaks into PAWs wherever a letter does not join to the next
one (ا ٲ د ڈ ر ڑ ز ژ و ۄ ے ۍ ...). کٲشُر is two PAWs, کٲ + شُر, written with no
space between them. Each PAW is drawn as one connected stroke plus separate dots
and vowel signs, and its shape is very different from the letters it is made of.
That makes the PAW a natural recognition unit (Urdu ligature OCR, e.g. UPTI/CLE).

Text side (`paws`): deterministic from Unicode, so ground-truth text gives PAW
labels for free.
Image side (`segment_word_image`): connected components, dots/vowel signs attached
to the main stroke they overlap, ordered right to left. When the number of image
segments equals the number of text PAWs, image and label align automatically, which
is how a PAW training set can be built from any (image, Unicode text) dataset.
"""
from __future__ import annotations

import unicodedata
from collections import deque
from dataclasses import dataclass

import numpy as np

# Letters that join only to the preceding letter (Unicode Joining_Type R), as used in Kashmiri.
RIGHT_JOINING = set("اآأإٱٲٳؤدڈذرڑزژوۄۅۆۇےۓۍةۀ")
# Letters that join to nothing (Joining_Type U).
NON_JOINING = set("ء")
ZWNJ = "‌"


def is_mark(ch: str) -> bool:
    return unicodedata.category(ch) == "Mn"


def paws(word: str, keep_marks: bool = True) -> list[str]:
    """Split one word into PAWs.  paws("کٲشُر") -> ["کٲ", "شُر"]."""
    word = unicodedata.normalize("NFC", word)
    out: list[str] = []
    cur = ""
    for ch in word:
        if ch.isspace():
            if cur:
                out.append(cur)
            cur = ""
            continue
        if ch == ZWNJ:
            if cur:
                out.append(cur)
            cur = ""
            continue
        if is_mark(ch):
            if keep_marks:
                if not cur and out:  # a mark after a PAW break belongs to the previous letter
                    out[-1] += ch
                else:
                    cur += ch
            continue
        if ch in NON_JOINING:
            if cur:
                out.append(cur)
            out.append(ch)
            cur = ""
            continue
        cur += ch
        if ch in RIGHT_JOINING:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    # marks that came right after a right-joining letter were placed on the next PAW above;
    # move leading marks back to the PAW they belong to
    fixed: list[str] = []
    for p in out:
        lead = ""
        while p and is_mark(p[0]):
            lead += p[0]
            p = p[1:]
        if lead and fixed:
            fixed[-1] += lead
        if p:
            fixed.append(p)
    return fixed


def paw_body(paw: str) -> str:
    """The PAW without vowel signs (its main stroke + dots)."""
    return "".join(c for c in unicodedata.normalize("NFD", paw) if not is_mark(c))


# --------------------------------------------------------------------------- image side

@dataclass
class Segment:
    pixels: np.ndarray  # boolean mask, same size as the image
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    n_marks: int  # dots / vowel signs attached
    body_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # the main stroke only, without dots / signs


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold; returns True for ink (dark on light)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / (total - w_b)
        between = w_b * (total - w_b) * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    return gray <= thr


def stroke_width(ink: np.ndarray) -> float:
    """Typical pen width: median length of horizontal and vertical ink runs."""
    runs = []
    for arr in (ink, ink.T):
        for row in arr:
            if not row.any():
                continue
            d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
            starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            runs.extend((ends - starts).tolist())
    return float(np.median(runs)) if runs else 1.0


def components(ink: np.ndarray) -> list[np.ndarray]:
    """8-connected components as arrays of (y, x) coordinates."""
    h, w = ink.shape
    seen = np.zeros_like(ink, dtype=bool)
    comps = []
    for y0, x0 in zip(*np.nonzero(ink)):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        pts = []
        while q:
            y, x = q.popleft()
            pts.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and ink[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        comps.append(np.array(pts))
    return comps


def segment_word_image(gray: np.ndarray, mark_ratio: float = 0.12, upper_band: float = 0.5, dot_size: float = 4.0, min_px: int = 4) -> list[Segment]:
    """Split a word image into PAW segments, right to left.

    Components smaller than `mark_ratio` x the largest one, or lying entirely in the upper
    `upper_band` of the word, are treated as dots or vowel signs and attached to the main
    stroke they overlap most horizontally (else the nearest one).
    """
    ink = binarize(gray)
    comps = [c for c in components(ink) if len(c) >= min_px]
    if not comps:
        return []
    biggest = max(len(c) for c in comps)
    top = min(c[:, 0].min() for c in comps)
    bottom = max(c[:, 0].max() for c in comps)
    band = top + upper_band * (bottom - top)

    sw = stroke_width(ink)

    def is_mark(c):
        # small; or floating entirely in the upper part of the word (dots, ٲ's hamza, ُ ...);
        # or compact, only a few stroke widths across in both directions (dots below پ ی ...)
        h = c[:, 0].max() - c[:, 0].min() + 1
        w = c[:, 1].max() - c[:, 1].min() + 1
        compact = h <= dot_size * sw and w <= dot_size * sw and len(c) < 0.5 * biggest
        return len(c) < mark_ratio * biggest or c[:, 0].max() < band or compact

    mains = [c for c in comps if not is_mark(c)] or [max(comps, key=len)]
    marks = [c for c in comps if is_mark(c) and not any(c is m for m in mains)]
    groups = [[m] for m in mains]
    spans = [(m[:, 1].min(), m[:, 1].max(), m[:, 1].mean(), m[:, 0].mean()) for m in mains]
    for mk in marks:
        x0, x1, cx, cy = mk[:, 1].min(), mk[:, 1].max(), mk[:, 1].mean(), mk[:, 0].mean()

        def score(i):
            a0, a1, acx, acy = spans[i]
            overlap = max(0, min(x1, a1) - max(x0, a0))
            return (overlap, -abs(cx - acx) - 0.3 * abs(cy - acy))

        groups[max(range(len(mains)), key=score)].append(mk)
    segs = []
    for g in groups:
        pts = np.concatenate(g)
        mask = np.zeros(ink.shape, dtype=bool)
        mask[pts[:, 0], pts[:, 1]] = True
        ys, xs = pts[:, 0], pts[:, 1]
        body = g[0]
        segs.append(Segment(mask, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), len(g) - 1,
                            (int(body[:, 1].min()), int(body[:, 0].min()), int(body[:, 1].max()), int(body[:, 0].max()))))
    # Nastaliq reads right to left; PAWs also step down, so order by the right edge.
    segs.sort(key=lambda s: -s.bbox[2])
    return segs
