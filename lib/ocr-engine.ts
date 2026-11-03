/**
 * OCR engines for KoshurOCR Studio.
 *
 * - "browser": Tesseract.js running in a Web Worker with the Urdu (Perso-Arabic)
 *   LSTM model. Free, private, works on Vercel with no server. Accuracy is good on
 *   Naskh-style print and limited on Nastaliq.
 * - "cloud": POSTs the image to /api/ocr, which forwards it to the GPU worker set in
 *   the GPU_WORKER_URL environment variable (see python_engine/modal_app.py).
 */
import type { Worker as TesseractWorker } from "tesseract.js";
import { normalizeKashmiriUnicode } from "./kashmiri-normalizer";
import { transliterateKashmiriToRoman } from "./transliteration";

export type OcrEngine = "browser" | "cloud";

export interface OcrLine {
  id: number;
  text: string;
  roman?: string;
  poly: number[][];
  conf: number;
  page?: number;
}

export interface OcrResult {
  lines: OcrLine[];
  confidence: number; // 0..1, mean over lines
  engine: string;
}

export type ProgressFn = (status: string, progress: number) => void;

const LANG_PATH = "https://cdn.jsdelivr.net/npm/@tesseract.js-data/urd/4.0.0_best_int";

let workerPromise: Promise<TesseractWorker> | null = null;
let progressListener: ProgressFn | null = null;

function getWorker(): Promise<TesseractWorker> {
  if (!workerPromise) {
    workerPromise = (async () => {
      const { createWorker } = await import("tesseract.js");
      return createWorker("urd", 1, {
        langPath: LANG_PATH,
        logger: (m: { status: string; progress: number }) => progressListener?.(m.status, m.progress),
      });
    })();
    workerPromise.catch(() => {
      workerPromise = null; // allow a retry after a network failure
    });
  }
  return workerPromise;
}

function finishLine(text: string, poly: number[][], conf: number, id: number, page?: number): OcrLine {
  const norm = normalizeKashmiriUnicode(text.replace(/\s+/g, " ").trim());
  return { id, text: norm, roman: transliterateKashmiriToRoman(norm), poly, conf, page };
}

function meanConf(lines: OcrLine[]): number {
  return lines.length ? lines.reduce((a, l) => a + l.conf, 0) / lines.length : 0;
}

async function recognizeInBrowser(image: HTMLCanvasElement, onProgress?: ProgressFn, page?: number): Promise<OcrResult> {
  progressListener = onProgress ?? null;
  try {
    const worker = await getWorker();
    const { data } = await worker.recognize(image, {}, { blocks: true, text: true });
    const lines: OcrLine[] = [];
    for (const block of data.blocks ?? []) {
      for (const para of block.paragraphs) {
        for (const line of para.lines) {
          if (!line.text.trim()) continue;
          const { x0, y0, x1, y1 } = line.bbox;
          // Tesseract occasionally reports 0 for a whole line; fall back to its words' mean.
          const words = line.words ?? [];
          const conf =
            line.confidence > 0 || words.length === 0
              ? line.confidence
              : words.reduce((a, w) => a + w.confidence, 0) / words.length;
          lines.push(finishLine(line.text, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], conf / 100, lines.length + 1, page));
        }
      }
    }
    return { lines, confidence: meanConf(lines), engine: "Tesseract.js (Urdu model, in browser)" };
  } finally {
    progressListener = null;
  }
}

async function recognizeInCloud(image: HTMLCanvasElement, onProgress?: ProgressFn, page?: number): Promise<OcrResult> {
  onProgress?.("uploading to GPU worker", 0.2);
  const dataUrl = image.toDataURL("image/jpeg", 0.9);
  const res = await fetch("/api/ocr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl.split(",")[1] }),
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok || json.status === "error") {
    throw new Error(json.message || `Cloud OCR failed (HTTP ${res.status})`);
  }
  onProgress?.("done", 1);
  const raw: { text?: string; poly?: number[][]; conf?: number }[] = Array.isArray(json.lines) ? json.lines : [];
  const lines = raw
    .filter((l) => typeof l.text === "string" && l.text.trim())
    .map((l, i) =>
      finishLine(
        l.text as string,
        Array.isArray(l.poly) && l.poly.length === 4 ? l.poly : [[0, 0], [image.width, 0], [image.width, 40], [0, 40]],
        typeof l.conf === "number" ? l.conf : 0,
        i + 1,
        page
      )
    );
  return { lines, confidence: meanConf(lines), engine: json.metadata?.engine || "Cloud GPU worker" };
}

export function recognize(engine: OcrEngine, image: HTMLCanvasElement, onProgress?: ProgressFn, page?: number) {
  return engine === "cloud" ? recognizeInCloud(image, onProgress, page) : recognizeInBrowser(image, onProgress, page);
}
