"""Annotate a Kashmiri text column of a scanned page.

    KS_FONTS=fonts python3 training/annotate_column.py page.png X0 X1 Y0 Y1 N_LINES [--word-gap 0.5] [--roman roman.txt]

Draws, on a 3x enlargement of the column: curved line separators (least-ink seams, red/blue),
word boxes (green/orange), part (PAW) boxes (purple), a heat map with shadows and show-through
removed, and the Kashmiri word model's reading next to the book's Roman spelling. Also writes
the boxes and readings as JSON. Ink shapes touching the column's left/right edge, thin rules,
and specks far from any line are excluded (the "touching the box = from outside" rule).
"""
import os, sys, json, numpy as np, onnxruntime as ort
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); sys.path.insert(0, ROOT)
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from python_engine.paw import binarize, components, segment_word_image
from python_engine.normalizer import normalize_kashmiri
S = os.environ.get('KS_WORK', '.')
F = os.environ.get('KS_FONTS', 'fonts')
sess = ort.InferenceSession(os.path.join(ROOT, 'public/models/kashmiri_word_ocr_int8.onnx'))
units = json.load(open(os.path.join(ROOT, 'public/models/kashmiri_word_ocr.json'))['units']

def dec(ids):
    o, p = [], 0
    for i in ids:
        if i != p and i: o.append(units[i])
        p = i
    return ''.join(o)

def read(gray):
    ys, xs = np.nonzero(gray < 128)
    if not len(ys): return ''
    p = max(2, int(0.12 * (ys.max() - ys.min() + 1)))
    g = gray[max(0, ys.min() - p):ys.max() + p + 1, max(0, xs.min() - p):xs.max() + p + 1]
    i = Image.fromarray(g); w = max(8, min(512, round(i.width * 48 / i.height)))
    a = 1 - np.asarray(i.resize((w, 48), Image.BILINEAR), dtype=np.float32) / 255
    return dec(sess.run(None, {'image': a[:, ::-1].copy()[None, None]})[0][0].argmax(-1))

def annotate(page_path, box, out_path, roman=None, n_lines=None, min_sep=None, word_gap=0.15):
    im = Image.open(page_path).convert('RGB'); G = np.asarray(im.convert('L')).astype(np.float32)
    X0, X1, Y0, Y1 = box
    # 1. heat map clean-up: divide by a heavily blurred background (shadow, paper tone, show-through)
    bg = np.asarray(Image.fromarray(G.astype(np.uint8)).filter(ImageFilter.GaussianBlur(18))).astype(np.float32)
    norm = np.clip(G / (bg + 1) * 255, 0, 255)[Y0:Y1, X0:X1].astype(np.uint8)
    ink = binarize(norm)
    comps = [c for c in components(ink) if len(c) >= 6]
    h, w = ink.shape
    # edge rule at column level: shapes touching the column's left/right edge come from the next column;
    # thin shapes far taller than letters are rules or scan marks
    heights = np.array([c[:, 0].max() - c[:, 0].min() + 1 for c in comps])
    med_h = float(np.median(heights[heights > 4])) if (heights > 4).any() else 10.0
    def keep(c):
        x0, x1 = c[:, 1].min(), c[:, 1].max(); hh = c[:, 0].max() - c[:, 0].min() + 1; ww = x1 - x0 + 1
        if x0 <= 0 or x1 >= w - 1: return False
        if hh > 3.5 * med_h and ww < 0.5 * hh: return False
        return True
    comps = [c for c in comps if keep(c)]
    ink = np.zeros_like(ink)
    for c in comps: ink[c[:, 0], c[:, 1]] = True
    # 2. line centres: peaks of the smoothed ink profile of main strokes
    prof = ink.sum(1).astype(float); k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); sm = np.convolve(prof, k / k.sum(), 'same')
    sep = min_sep or 18
    peaks = []
    for y in np.argsort(-sm):
        if sm[y] < 0.15 * sm.max(): break
        if all(abs(y - q) >= sep for q in peaks): peaks.append(int(y))
    peaks.sort()
    if n_lines: peaks = sorted(sorted(peaks, key=lambda y: -sm[y])[:n_lines])
    # 3. seams: least-ink path between consecutive centres (dynamic programming, |dy| <= 1 per column)
    cost = ink.astype(float) * 50 + 1
    seams = []
    for a, b in zip(peaks, peaks[1:]):
        lo, hi = a + 2, b - 1
        band = cost[lo:hi]; H = band.shape[0]
        acc = band[:, 0].copy(); back = np.zeros((H, w), int)
        for x in range(1, w):
            prev = np.stack([np.r_[np.inf, acc[:-1]], acc, np.r_[acc[1:], np.inf]])
            j = prev.argmin(0); acc = prev[j, np.arange(H)] + band[:, x]; back[:, x] = np.arange(H) + j - 1
        y = int(acc.argmin()); path = [0] * w
        for x in range(w - 1, -1, -1): path[x] = y + lo; y = back[y, x]
        sm_path = np.convolve(np.r_[[path[0]] * 4, path, [path[-1]] * 4], np.ones(9) / 9, 'valid')  # smooth: spline-like
        seams.append(sm_path)
    def line_of(c):
        cy, cx = c[:, 0].mean(), int(c[:, 1].mean())
        return sum(cy > s[min(cx, w - 1)] for s in seams)
    lines = [[] for _ in peaks]
    for c in comps:
        li = line_of(c)
        # shapes far from their line's centre (specks of show-through in the gaps) belong to no line
        if abs(c[:, 0].mean() - peaks[li]) <= 2.2 * med_h:
            lines[li].append(c)
    results = []
    for li, cs in enumerate(lines):
        if not cs: continue
        m = np.zeros_like(ink)
        for c in cs: m[c[:, 0], c[:, 1]] = True
        ys, xs = np.nonzero(m); y0, y1 = ys.min(), ys.max() + 1
        lineimg = np.where(m, norm, 255).astype(np.uint8)[y0:y1]
        # words: gaps wider than 0.15 x line height between ink columns
        cols = (lineimg < 128).any(0); spans = []; st = None
        for x, v in enumerate(cols):
            if v and st is None: st = x
            if not v and st is not None: spans.append([st, x]); st = None
        if st is not None: spans.append([st, len(cols)])
        words = [spans[0]]
        for sp in spans[1:]:
            if sp[0] - words[-1][1] > word_gap * med_h: words.append(sp)
            else: words[-1][1] = sp[1]
        words = words[::-1]
        word_out = []
        for (a, b) in words:
            wimg = lineimg[:, max(0, a - 2):b + 2]
            parts = []
            for sgm in segment_word_image(wimg):
                px0, py0, px1, py1 = sgm.bbox
                gx0, gy0, gx1, gy1 = px0 + max(0, a - 2), py0 + y0, px1 + max(0, a - 2), py1 + y0
                # edge rule: shapes touching this part box from outside are excluded
                touching = 0
                for c in cs:
                    cy0, cy1, cx0, cx1 = c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()
                    inside = gx0 <= cx0 and cx1 <= gx1 and gy0 <= cy0 and cy1 <= gy1
                    overlaps = not (cx1 < gx0 or cx0 > gx1 or cy1 < gy0 or cy0 > gy1)
                    touching += overlaps and not inside
                parts.append({'box': [int(gx0), int(gy0), int(gx1), int(gy1)], 'excluded_shapes': int(touching)})
            word_out.append({'box': [int(a), int(y0), int(b), int(y1)], 'text': normalize_kashmiri(read(wimg)), 'parts': parts})
        results.append({'line': len(results) + 1, 'words': word_out, 'text': ' '.join(x['text'] for x in word_out)})
    # 4. draw: original (x3) with seams / words / parts, heat map, readings
    Z = 3
    orig = im.crop((X0, Y0, X1, Y1)).resize((w * Z, h * Z), Image.LANCZOS)
    d = ImageDraw.Draw(orig)
    for i, s in enumerate(seams):
        d.line([(x * Z, s[x] * Z) for x in range(0, w, 2)], fill=(220, 30, 30) if i % 2 == 0 else (30, 80, 230), width=3)
    for r in results:
        for j, wd in enumerate(r['words']):
            x0, y0, x1, y1 = wd['box']; d.rectangle((x0 * Z - 3, y0 * Z - 3, x1 * Z + 3, y1 * Z + 3), outline=(20, 160, 60) if j % 2 == 0 else (240, 140, 0), width=3)
            for pt in wd['parts']:
                a0, b0, a1, b1 = pt['box']; d.rectangle((a0 * Z, b0 * Z, a1 * Z, b1 * Z), outline=(150, 40, 200), width=1)
    # heat map panel (inferno-like ramp)
    t = 1 - norm.astype(np.float32) / 255
    ramp = np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1) * 0.6 + 0.1 * (t > 0.05)], -1)
    heat = Image.fromarray((ramp * 255).astype(np.uint8)).resize((w * Z, h * Z), Image.NEAREST)
    # readings column
    fk = ImageFont.truetype(f'{F}/NotoNaskhArabic.ttf', 26, layout_engine=ImageFont.Layout.RAQM)
    fl = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18) if __import__('os').path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf') else ImageFont.load_default()
    panel = Image.new('RGB', (620, h * Z), 'white'); pd = ImageDraw.Draw(panel)
    for i, r in enumerate(results):
        ymid = np.mean([wd['box'][1] + wd['box'][3] for wd in r['words']]) / 2 * Z
        pd.text((610, ymid), r['text'], font=fk, fill=(0, 0, 0), anchor='rm', direction='rtl', language='ks')
        pd.text((8, ymid), f"{i + 1}. {roman[i] if roman and i < len(roman) else ''}", font=fl, fill=(90, 90, 90), anchor='lm')
    W = orig.width + heat.width + panel.width + 40
    canvas = Image.new('RGB', (W, h * Z + 50), 'white'); cd = ImageDraw.Draw(canvas)
    canvas.paste(orig, (0, 50)); canvas.paste(heat, (orig.width + 20, 50)); canvas.paste(panel, (orig.width + heat.width + 40, 50))
    cd.text((5, 6), 'scan: line seams (red/blue)', font=fl, fill=(0, 0, 0))
    cd.text((5, 27), 'words (green/orange), parts (purple)', font=fl, fill=(0, 0, 0))
    cd.text((orig.width + 25, 6), 'heat map', font=fl, fill=(0, 0, 0))
    cd.text((orig.width + 25, 27), '(shadows removed)', font=fl, fill=(0, 0, 0))
    cd.text((orig.width + heat.width + 45, 6), 'book Roman  |  model reading', font=fl, fill=(0, 0, 0))
    canvas.save(out_path)
    return {'lines': results, 'n_seams': len(seams), 'centres': peaks}

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('page'); ap.add_argument('x0', type=int); ap.add_argument('x1', type=int)
    ap.add_argument('y0', type=int); ap.add_argument('y1', type=int); ap.add_argument('n_lines', type=int)
    ap.add_argument('--word-gap', type=float, default=0.5, help='word gap in letter heights (large = one word per line)')
    ap.add_argument('--roman', help='text file with the book\'s Roman spelling, one line per Kashmiri line')
    ap.add_argument('--out', default='annotated.png')
    a = ap.parse_args()
    roman = open(a.roman, encoding='utf-8').read().splitlines() if a.roman else None
    out = annotate(a.page, (a.x0, a.x1, a.y0, a.y1), a.out, roman, n_lines=a.n_lines, min_sep=14, word_gap=a.word_gap)
    json.dump(out, open(os.path.splitext(a.out)[0] + '.json', 'w'), ensure_ascii=False, indent=1)
    for r in out['lines']: print(r['line'], r['text'])
