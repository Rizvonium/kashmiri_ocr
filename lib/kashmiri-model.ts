/**
 * Kashmiri word recognizer (CNN + BiLSTM + CTC) running in the browser with onnxruntime-web.
 *
 * Trained with training/train_word_ocr.py on Kaggle (v2, character units): 69,434 word images
 * from Kaggle nawabhussaen/kashmiri-ocr + 120,000 synthetic Nastaliq/Naskh words drawn from the
 * Kashmiri dictionary and corpora. On unseen, computer-rendered words: 2.7% word errors (Nastaliq
 * 4.7%, Naskh 1.6%). Not yet measured on real scans.
 *
 * Input contract (public/models/kashmiri_word_ocr.json): one word, cropped to its ink, grayscale
 * resized to height 48, ink = 1, flipped left-right (so reading order runs left to right in time).
 */
import type { InferenceSession, Tensor } from "onnxruntime-web";

const ORT_VERSION = "1.31.1"; // keep in step with package.json
const MODEL_URL = "/models/kashmiri_word_ocr_int8.onnx";
const META_URL = "/models/kashmiri_word_ocr.json";
const H = 48;
const MAX_W = 512;

interface Loaded {
  session: InferenceSession;
  units: string[];
  TensorCtor: typeof Tensor;
}

let loading: Promise<Loaded> | null = null;

export function loadKashmiriModel(): Promise<Loaded> {
  if (!loading) {
    loading = (async () => {
      const ort = await import("onnxruntime-web/wasm");
      ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
      ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which this site does not use
      const [meta, model] = await Promise.all([
        fetch(META_URL).then((r) => {
          if (!r.ok) throw new Error(`model description missing (HTTP ${r.status})`);
          return r.json();
        }),
        fetch(MODEL_URL).then((r) => {
          if (!r.ok) throw new Error(`model download failed (HTTP ${r.status})`);
          return r.arrayBuffer();
        }),
      ]);
      const session = await ort.InferenceSession.create(new Uint8Array(model), { executionProviders: ["wasm"] });
      return { session, units: meta.units as string[], TensorCtor: ort.Tensor };
    })();
    loading.catch(() => (loading = null));
  }
  return loading;
}

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Otsu threshold on 0..255 gray values. */
function otsu(gray: Uint8ClampedArray): number {
  const hist = new Array(256).fill(0);
  for (const g of gray) hist[g]++;
  const total = gray.length;
  let sumAll = 0;
  for (let t = 0; t < 256; t++) sumAll += t * hist[t];
  let wB = 0, sumB = 0, best = -1, thr = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB || wB === total) continue;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sumAll - sumB) / (total - wB);
    const between = wB * (total - wB) * (mB - mF) ** 2;
    if (between > best) {
      best = between;
      thr = t;
    }
  }
  return thr;
}

/**
 * Keeps only the ink shapes that lie fully inside the crop. A shape that touches the crop's edge
 * reaches in from outside (a neighbouring word's stroke, a dot or vowel sign from the next line)
 * and is erased, so the model sees only this word. Returns 1 for kept ink, 0 elsewhere.
 */
function dropBorderShapes(
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  thr: number,
  pageEdge = { left: false, top: false, right: false, bottom: false } // crop sides that are the image border
): Uint8Array {
  const keep = new Uint8Array(w * h);
  const seen = new Uint8Array(w * h);
  const stack: number[] = [];
  const shape: number[] = [];
  for (let start = 0; start < w * h; start++) {
    if (seen[start] || gray[start] > thr) continue;
    let touchesEdge = false;
    shape.length = 0;
    stack.push(start);
    seen[start] = 1;
    while (stack.length) {
      const i = stack.pop()!;
      shape.push(i);
      const x = i % w, y = (i - x) / w;
      if ((x === 0 && !pageEdge.left) || (y === 0 && !pageEdge.top) || (x === w - 1 && !pageEdge.right) || (y === h - 1 && !pageEdge.bottom))
        touchesEdge = true;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const j = ny * w + nx;
          if (!seen[j] && gray[j] <= thr) {
            seen[j] = 1;
            stack.push(j);
          }
        }
    }
    if (!touchesEdge) for (const i of shape) keep[i] = 1;
  }
  return keep;
}

/**
 * Turns one word region of the page into the model's input: a little extra room around the box
 * (Tesseract boxes often clip dots and vowel signs), then crop to the ink, pad, resize, flip.
 */
export function wordTensorData(page: HTMLCanvasElement, box: Box, expand = true): { data: Float32Array; width: number } | null {
  const bh = box.y1 - box.y0, bw = box.x1 - box.x0;
  if (bh < 4 || bw < 2) return null;
  // A margin around the box: shapes reaching the margin's outer edge come from outside and are dropped.
  const ex = expand ? bw * 0.04 + 2 : Math.max(3, bh * 0.08), ey = expand ? bh * 0.25 + 2 : Math.max(3, bh * 0.1);
  const x0 = Math.max(0, Math.floor(box.x0 - ex)), x1 = Math.min(page.width, Math.ceil(box.x1 + ex));
  const y0 = Math.max(0, Math.floor(box.y0 - ey)), y1 = Math.min(page.height, Math.ceil(box.y1 + ey));
  const w = x1 - x0, h = y1 - y0;
  const px = page.getContext("2d", { willReadFrequently: true })!.getImageData(x0, y0, w, h).data;
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  const thr = otsu(gray);
  const ink = dropBorderShapes(gray, w, h, thr, {
    left: x0 === 0,
    top: y0 === 0,
    right: x1 === page.width,
    bottom: y1 === page.height,
  });
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (ink[y * w + x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
  if (maxX < 0) return null;
  const pad = Math.max(2, Math.round(0.12 * (maxY - minY + 1)));
  const cx0 = Math.max(0, minX - pad), cy0 = Math.max(0, minY - pad);
  const cw = Math.min(w, maxX + pad + 1) - cx0, ch = Math.min(h, maxY + pad + 1) - cy0;

  const src = document.createElement("canvas");
  src.width = cw;
  src.height = ch;
  const sctx = src.getContext("2d")!;
  const crop = sctx.createImageData(cw, ch);
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      const i = (cy0 + y) * w + (cx0 + x);
      const g = ink[i] || gray[i] > thr ? gray[i] : 255; // erased foreign ink becomes paper
      const o = 4 * (y * cw + x);
      crop.data[o] = crop.data[o + 1] = crop.data[o + 2] = g;
      crop.data[o + 3] = 255;
    }
  sctx.putImageData(crop, 0, 0);
  const width = Math.max(8, Math.min(MAX_W, Math.round((cw * H) / ch)));
  const dst = document.createElement("canvas");
  dst.width = width;
  dst.height = H;
  const dctx = dst.getContext("2d", { willReadFrequently: true })!;
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, width, H);
  const out = dctx.getImageData(0, 0, width, H).data;
  const data = new Float32Array(width * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < width; x++) data[y * width + (width - 1 - x)] = 1 - out[4 * (y * width + x)] / 255; // flip
  return { data, width };
}

export interface WordReading {
  text: string;
  conf: number; // 0..100: lowest per-character probability, the model's doubt about its weakest letter
}

/** Greedy CTC decode with a per-word confidence. */
function decode(logits: Float32Array, steps: number, classes: number, units: string[]): WordReading {
  let text = "", prev = 0, minP = 1;
  for (let t = 0; t < steps; t++) {
    let best = 0, max = -Infinity;
    for (let c = 0; c < classes; c++) {
      const v = logits[t * classes + c];
      if (v > max) {
        max = v;
        best = c;
      }
    }
    if (best !== 0 && best !== prev) {
      let sum = 0;
      for (let c = 0; c < classes; c++) sum += Math.exp(logits[t * classes + c] - max);
      minP = Math.min(minP, 1 / sum);
      text += units[best] ?? "";
    }
    prev = best;
  }
  return { text: text.normalize("NFC"), conf: text ? Math.round(minP * 100) : 0 };
}

export async function readWord(page: HTMLCanvasElement, box: Box, expand = true): Promise<WordReading | null> {
  const { session, units, TensorCtor } = await loadKashmiriModel();
  const input = wordTensorData(page, box, expand);
  if (!input) return null;
  const feeds = { [session.inputNames[0]]: new TensorCtor("float32", input.data, [1, 1, H, input.width]) };
  const result = await session.run(feeds);
  const out = result[session.outputNames[0]];
  const [, steps, classes] = out.dims as number[];
  return decode(out.data as Float32Array, steps, classes, units);
}

// --------------------------------------------------------------------------- page layout

function runs(values: Int32Array | number[], minValue = 1): [number, number][] {
  const out: [number, number][] = [];
  let start = -1;
  for (let i = 0; i < values.length; i++) {
    if (values[i] >= minValue && start < 0) start = i;
    if (values[i] < minValue && start >= 0) {
      out.push([start, i]);
      start = -1;
    }
  }
  if (start >= 0) out.push([start, values.length]);
  return out;
}

export interface PageLine {
  box: Box;
  words: Box[]; // right to left
}

/**
 * Finds text lines (bands of ink rows; thin bands holding only dots or vowel signs join the nearest
 * line) and words (gaps wider than `gapFactor` x the line's ink height). Word gaps are reliable in
 * Naskh-style print; in Nastaliq, words overlap and some get merged.
 */
export function segmentPage(page: HTMLCanvasElement, gapFactor = 0.15): PageLine[] {
  const { width: w, height: h } = page;
  const px = page.getContext("2d", { willReadFrequently: true })!.getImageData(0, 0, w, h).data;
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) gray[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  const thr = otsu(gray);
  const ink = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) ink[i] = gray[i] <= thr ? 1 : 0;

  const rowInk = new Int32Array(h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rowInk[y] += ink[y * w + x];
  const minRow = Math.max(1, Math.round(w * 0.0015)); // ignore rows with a few specks of noise
  const bands = runs(rowInk, minRow);
  if (!bands.length) return [];
  const tallest = Math.max(...bands.map(([a, b]) => b - a));
  const isMain = bands.map(([a, b]) => b - a >= 0.35 * tallest);
  const main = bands.filter((_, i) => isMain[i]).map(([a, b]) => [a, b] as [number, number]);
  const original = main.map(([a, b]) => [a, b] as [number, number]); // distances use the unmerged lines
  bands.forEach(([a, b], i) => {
    if (isMain[i] || !main.length) return;
    let k = 0, best = Infinity;
    original.forEach(([m0, m1], j) => {
      const d = Math.min(Math.abs(a - m1), Math.abs(b - m0));
      if (d < best) {
        best = d;
        k = j;
      }
    });
    main[k] = [Math.min(main[k][0], a), Math.max(main[k][1], b)];
  });
  main.sort((p, q) => p[0] - q[0]);

  const lines: PageLine[] = [];
  for (const [y0, y1] of main) {
    const colInk = new Int32Array(w);
    for (let y = y0; y < y1; y++) for (let x = 0; x < w; x++) colInk[x] += ink[y * w + x];
    const spans = runs(colInk, 1);
    if (!spans.length) continue;
    const gap = gapFactor * (y1 - y0);
    const words: [number, number][] = [[spans[0][0], spans[0][1]]];
    for (let i = 1; i < spans.length; i++) {
      if (spans[i][0] - words[words.length - 1][1] > gap) words.push([spans[i][0], spans[i][1]]);
      else words[words.length - 1][1] = spans[i][1];
    }
    const boxes = words
      .filter(([a, b]) => b - a >= 2)
      .map(([a, b]) => ({ x0: a, y0, x1: b, y1 }))
      .sort((p, q) => q.x1 - p.x1); // right to left
    if (boxes.length) lines.push({ box: { x0: spans[0][0], y0, x1: spans[spans.length - 1][1], y1 }, words: boxes });
  }
  return lines;
}
