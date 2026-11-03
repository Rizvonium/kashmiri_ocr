/**
 * Client-side PDF rendering with pdf.js. Pages are rendered one at a time, so large
 * books never leave the browser and never hit Vercel's upload limits.
 */
import type { PDFDocumentProxy } from "pdfjs-dist";

async function getPdfjs() {
  const pdfjs = await import("pdfjs-dist");
  if (!pdfjs.GlobalWorkerOptions.workerSrc) {
    pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
  }
  return pdfjs;
}

export async function openPdf(file: File): Promise<PDFDocumentProxy> {
  const pdfjs = await getPdfjs();
  const data = new Uint8Array(await file.arrayBuffer());
  try {
    return await pdfjs.getDocument({ data }).promise;
  } catch (e) {
    const msg = (e as Error).message || "";
    if (/password/i.test(msg)) throw new Error("This PDF is password-protected.");
    throw new Error(`Could not open "${file.name}" as a PDF.`);
  }
}

/** Renders a page at roughly 200 DPI (capped for phones). */
export async function renderPdfPage(pdf: PDFDocumentProxy, pageNumber: number): Promise<HTMLCanvasElement> {
  const page = await pdf.getPage(pageNumber);
  const base = page.getViewport({ scale: 1 });
  const scale = Math.min(200 / 72, 2400 / Math.max(base.width, base.height));
  const viewport = page.getViewport({ scale });
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(viewport.width);
  canvas.height = Math.round(viewport.height);
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvasContext: ctx, viewport, canvas }).promise;
  page.cleanup();
  return canvas;
}
