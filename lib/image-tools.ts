/**
 * Browser image helpers: loading files into canvases, real pre-processing filters,
 * and rendering sample Kashmiri text to an image so samples go through real OCR.
 */

const MAX_SIDE = 2400; // keep OCR fast and memory use sane on phones

export function canvasFromSource(src: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.round(w * scale);
  c.height = Math.round(h * scale);
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.drawImage(src, 0, 0, c.width, c.height);
  return c;
}

export async function canvasFromFile(file: File): Promise<HTMLCanvasElement> {
  if (!file.type.startsWith("image/")) throw new Error(`"${file.name}" is not an image file.`);
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return canvasFromSource(img, img.naturalWidth, img.naturalHeight);
  } catch {
    throw new Error(`Could not read "${file.name}". Try a PNG or JPG image.`);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export interface PreprocessOptions {
  evenLighting: boolean; // divide by a blurred background to remove shadows / glare
  reduceMoire: boolean; // light blur to suppress screen-pixel patterns
}

/**
 * Returns a new grayscale canvas with the selected filters applied.
 * Even lighting: I_norm = clip(I / (blur(I) + eps) * 240, 0, 255).
 */
export function preprocess(src: HTMLCanvasElement, opts: PreprocessOptions): HTMLCanvasElement {
  if (!opts.evenLighting && !opts.reduceMoire) return src;
  const { width: w, height: h } = src;
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d", { willReadFrequently: true })!;
  ctx.filter = opts.reduceMoire ? "blur(0.7px)" : "none";
  ctx.drawImage(src, 0, 0);
  ctx.filter = "none";
  if (!opts.evenLighting) return out;

  const bg = document.createElement("canvas");
  bg.width = w;
  bg.height = h;
  const bctx = bg.getContext("2d", { willReadFrequently: true })!;
  bctx.filter = `blur(${Math.max(12, Math.round(Math.max(w, h) / 40))}px)`;
  bctx.drawImage(out, 0, 0);

  const img = ctx.getImageData(0, 0, w, h);
  const back = bctx.getImageData(0, 0, w, h).data;
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const g = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    const b = 0.299 * back[i] + 0.587 * back[i + 1] + 0.114 * back[i + 2];
    const v = Math.max(0, Math.min(255, (g / (b + 1)) * 240));
    d[i] = d[i + 1] = d[i + 2] = v;
  }
  ctx.putImageData(img, 0, 0);
  return out;
}

export const SAMPLE_TEXT = {
  dictionary: {
    font: '"Noto Naskh Arabic", serif',
    label: "Naskh",
    lines: [
      "کٲشُر لُغَت — جِلد گۄڈٕنِیُک",
      "آب — اسم مذکر: رۄپہِ ہِو شفاف پانہِ",
      "آزٲدی — اسم مؤنث: خۄشحٲلی تہٕ چُھٹکارٕ",
      "بَرگ — پَن: کُلِک زرد یا سبز حِصہٕ",
      "دۄپھ — آواز: زمینَس پؠٹھ کِہین پؠنہٕ چِھ چھَنٛڈ",
    ],
  },
  poetry: {
    font: '"Noto Nastaliq Urdu", serif',
    label: "Nastaliq",
    lines: [
      "مےٚ چُھ کٲشُر زَبانہِ پؠٹھ غُرور",
      "صدیَن پؠٹھ چھےٚ یۄس بَہان پُر نُور",
      "لَل دؠد تہٕ نُند رؠشؠن کٔر شَنٛگار",
      "مَحجوٗر تہٕ حَبہٕ خاتوٗنَن دِیُت پیار",
    ],
  },
} as const;

export type SampleKind = keyof typeof SAMPLE_TEXT;

/** Draws sample text onto a white page so the sample goes through the real OCR engine. */
export async function renderSamplePage(kind: SampleKind): Promise<HTMLCanvasElement> {
  const sample = SAMPLE_TEXT[kind];
  const size = 40;
  const lineHeight = kind === "poetry" ? size * 2.6 : size * 1.9;
  const font = `${size}px ${sample.font}`;
  await document.fonts.load(font, sample.lines.join(" "));

  const c = document.createElement("canvas");
  c.width = 1200;
  c.height = Math.round(80 + lineHeight * sample.lines.length);
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#111";
  ctx.font = font;
  ctx.direction = "rtl";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  sample.lines.forEach((line, i) => ctx.fillText(line, c.width - 60, 40 + lineHeight * (i + 0.5)));
  return c;
}
