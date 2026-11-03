"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ScanText, Sliders, BookOpen, Image as ImageIcon, Camera, AlertTriangle, X, Loader2 } from "lucide-react";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { VerificationStudio } from "../components/VerificationStudio";
import { CameraViewfinder } from "../components/CameraViewfinder";
import { PdfBookReader } from "../components/PdfBookReader";
import { ImageUploader } from "../components/ImageUploader";
import { SynthCameraSimulator } from "../components/SynthCameraSimulator";
import { recognize, type OcrEngine, type OcrLine } from "../lib/ocr-engine";
import { canvasFromFile, preprocess, renderSamplePage, type PreprocessOptions, type SampleKind } from "../lib/image-tools";
import { openPdf, renderPdfPage } from "../lib/pdf-loader";
import { checkLine, loadLexicon, type CheckSettings } from "../lib/dictionary";
import { transliterateKashmiriToRoman } from "../lib/transliteration";

type InputMode = "pdf" | "image" | "camera";

interface PdfState {
  doc: PDFDocumentProxy;
  name: string;
  numPages: number;
}

interface Busy {
  label: string;
  progress: number;
}

// Non-PDF documents (images, camera, samples) are stored under page key 0.
const SINGLE = 0;

const PROGRESS_LABELS: Record<string, string> = {
  "loading tesseract core": "Loading OCR engine",
  "initializing tesseract": "Starting OCR engine",
  "loading language traineddata": "Downloading Urdu/Kashmiri model (first time only)",
  "initializing api": "Preparing model",
  "recognizing text": "Reading text",
};

export default function Home() {
  const [activeTab, setActiveTab] = useState<"studio" | "simulator">("studio");
  const [engine, setEngine] = useState<OcrEngine>("browser");
  const [cloudAvailable, setCloudAvailable] = useState(false);
  const [inputMode, setInputMode] = useState<InputMode>("image");
  const [prep, setPrep] = useState<PreprocessOptions>({ evenLighting: false, reduceMoire: false });
  const [dict, setDict] = useState<CheckSettings>({ enabled: true, restoreSigns: true });

  const [pdf, setPdf] = useState<PdfState | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const [pageImages, setPageImages] = useState<Record<number, HTMLCanvasElement>>({});
  const [pageLines, setPageLines] = useState<Record<number, OcrLine[]>>({});
  const [docName, setDocName] = useState<string>("");
  const [engineUsed, setEngineUsed] = useState<string>("");

  const [busy, setBusy] = useState<Busy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = useRef(false);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((j) => setCloudAvailable(Boolean(j.cloud_gpu_configured)))
      .catch(() => setCloudAvailable(false));
  }, []);

  const currentKey = inputMode === "pdf" && pdf ? pdfPage : SINGLE;
  const currentImage = pageImages[currentKey] ?? null;
  const currentLines = pageLines[currentKey] ?? [];

  const allLines = useMemo(() => {
    const keys = Object.keys(pageLines).map(Number).sort((a, b) => a - b);
    const relevant = inputMode === "pdf" && pdf ? keys.filter((k) => k !== SINGLE) : [SINGLE];
    return relevant.flatMap((k) => pageLines[k] ?? []);
  }, [pageLines, inputMode, pdf]);

  const runOcr = useCallback(
    async (image: HTMLCanvasElement, key: number, labelPrefix = "") => {
      setPageImages((m) => ({ ...m, [key]: image }));
      setBusy({ label: `${labelPrefix}Preparing image`, progress: 0 });
      try {
        const input = preprocess(image, prep);
        const result = await recognize(
          engine,
          input,
          (status, progress) =>
            setBusy({ label: labelPrefix + (PROGRESS_LABELS[status] ?? status), progress }),
          key === SINGLE ? undefined : key
        );
        let lines = result.lines;
        try {
          setBusy({ label: `${labelPrefix}Checking words against the Kashmiri dictionary`, progress: 0.95 });
          const lex = await loadLexicon();
          lines = lines.map((l) => {
            const check = checkLine(lex, l.ocrText ?? l.text, dict, l.wordConfs);
            return { ...l, text: check.text, roman: transliterateKashmiriToRoman(check.text), check };
          });
        } catch (e) {
          setError(`Dictionary check unavailable: ${(e as Error).message}. Showing the raw OCR text.`);
        }
        setPageLines((m) => ({ ...m, [key]: lines }));
        setEngineUsed(result.engine);
        if (result.lines.length === 0) {
          setError("No text was found. Try a sharper, well-lit image with the text the right way up.");
        }
        return true;
      } catch (e) {
        setError(`OCR failed: ${(e as Error).message}`);
        return false;
      } finally {
        setBusy(null);
      }
    },
    [engine, prep, dict]
  );

  const startSingle = (name: string) => {
    setError(null);
    setDocName(name);
    setPageLines((m) => ({ ...m, [SINGLE]: [] }));
  };

  const handleImageFile = async (file: File) => {
    startSingle(file.name);
    try {
      const canvas = await canvasFromFile(file);
      await runOcr(canvas, SINGLE);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleSample = async (kind: SampleKind) => {
    setInputMode("image");
    startSingle(kind === "dictionary" ? "Sample dictionary page (Naskh)" : "Sample poetry page (Nastaliq)");
    setBusy({ label: "Loading sample font", progress: 0 });
    let page: HTMLCanvasElement;
    try {
      page = await renderSamplePage(kind);
    } catch (e) {
      setBusy(null);
      setError(`Could not draw the sample: ${(e as Error).message}`);
      return;
    }
    await runOcr(page, SINGLE);
  };

  const handleCapture = async (canvas: HTMLCanvasElement) => {
    startSingle(`Camera capture ${new Date().toLocaleTimeString()}`);
    await runOcr(canvas, SINGLE);
  };

  const handleSimulatorSend = async (canvas: HTMLCanvasElement) => {
    setActiveTab("studio");
    setInputMode("image");
    startSingle("SynthCamera simulated frame");
    await runOcr(canvas, SINGLE);
  };

  // ---- PDF ----
  const pageImagesRef = useRef(pageImages);
  pageImagesRef.current = pageImages;

  const showPdfPage = async (doc: PDFDocumentProxy, n: number) => {
    setPdfPage(n);
    if (pageImagesRef.current[n]) return;
    try {
      const c = await renderPdfPage(doc, n);
      setPageImages((m) => ({ ...m, [n]: c }));
    } catch (e) {
      setError(`Could not render page ${n}: ${(e as Error).message}`);
    }
  };

  const handlePdfFile = async (file: File) => {
    setError(null);
    setBusy({ label: "Opening PDF", progress: 0 });
    try {
      const doc = await openPdf(file);
      pdf?.doc.destroy();
      setPdf({ doc, name: file.name, numPages: doc.numPages });
      const first = await renderPdfPage(doc, 1);
      setPageImages((m) => ({ [SINGLE]: m[SINGLE], 1: first }));
      setPageLines((m) => ({ [SINGLE]: m[SINGLE] ?? [] }));
      setPdfPage(1);
      setBusy(null);
    } catch (e) {
      setBusy(null);
      setError((e as Error).message);
    }
  };

  const ocrPdfPage = async (n: number, prefix = "") => {
    if (!pdf) return false;
    setError(null);
    const img = pageImagesRef.current[n] ?? (await renderPdfPage(pdf.doc, n));
    return runOcr(img, n, prefix);
  };

  const ocrAllPages = async () => {
    if (!pdf) return;
    cancelRef.current = false;
    for (let n = 1; n <= pdf.numPages; n++) {
      if (cancelRef.current) break;
      if (pageLines[n]?.length) continue;
      setPdfPage(n);
      const ok = await ocrPdfPage(n, `Page ${n}/${pdf.numPages}: `);
      if (!ok) break;
    }
  };

  const updateCurrentLines = (lines: OcrLine[]) => setPageLines((m) => ({ ...m, [currentKey]: lines }));

  const doneCount = pdf ? Object.keys(pageLines).filter((k) => Number(k) !== SINGLE && pageLines[Number(k)]?.length).length : 0;

  const modeCard = (mode: InputMode, title: string, desc: string, tag: string, Icon: typeof BookOpen, color: string) => (
    <button
      type="button"
      onClick={() => {
        setInputMode(mode);
        setError(null);
      }}
      className={`text-left cursor-pointer group p-4 rounded-xl border-2 transition ${
        inputMode === mode ? `${color} bg-slate-900 shadow-lg` : "border-slate-800 bg-slate-900/40 hover:border-slate-700"
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="p-2.5 rounded-lg bg-slate-950 text-cyan-400 border border-slate-800">
          <Icon className="w-5 h-5" />
        </div>
        <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-slate-950 text-slate-300 border border-slate-700">
          {tag}
        </span>
      </div>
      <h3 className="font-semibold text-white text-sm">{title}</h3>
      <p className="text-xs text-slate-400 mt-1">{desc}</p>
    </button>
  );

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-4 lg:px-8 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-500 flex items-center justify-center font-nastaliq text-xl font-bold text-white">
              ک
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                KoshurOCR Studio
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 font-mono">beta</span>
              </h1>
              <p className="text-xs text-slate-400">Kashmiri Perso-Arabic camera, PDF & image to Unicode</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <nav className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
              {(
                [
                  ["studio", "OCR Studio", ScanText],
                  ["simulator", "SynthCamera Simulator", Sliders],
                ] as const
              ).map(([tab, label, Icon]) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
                    activeTab === tab ? "bg-cyan-600 text-white shadow-sm" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" /> {label}
                </button>
              ))}
            </nav>

            <div className="flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs">
              <span className="text-slate-400">Engine:</span>
              <button
                onClick={() => setEngine("browser")}
                className={`px-2 py-0.5 rounded font-semibold transition ${
                  engine === "browser" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30" : "text-slate-400 hover:text-white"
                }`}
              >
                In-browser (free)
              </button>
              <button
                onClick={() => cloudAvailable && setEngine("cloud")}
                disabled={!cloudAvailable}
                title={cloudAvailable ? "Send images to your GPU worker" : "Set GPU_WORKER_URL in Vercel to enable"}
                className={`px-2 py-0.5 rounded font-semibold transition disabled:opacity-40 disabled:cursor-not-allowed ${
                  engine === "cloud" ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" : "text-slate-400 hover:text-white"
                }`}
              >
                Cloud GPU
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-6 flex flex-col gap-6">
        {activeTab === "studio" ? (
          <div className="flex flex-col gap-6">
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {modeCard("pdf", "1. PDF Book", "Open a PDF and OCR one page or the whole book. Pages are rendered in your browser.", "PDF.js", BookOpen, "border-cyan-500/60")}
              {modeCard("image", "2. Image Scan", "Upload a scan, photo or screenshot of printed Kashmiri text.", "PNG/JPG/WEBP", ImageIcon, "border-emerald-500/60")}
              {modeCard("camera", "3. Camera", "Use your phone or laptop camera with a live focus meter.", "Focus meter", Camera, "border-amber-500/60")}
            </section>

            <section className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-300 px-1">
              <span className="text-slate-500 uppercase tracking-wider font-semibold">Clean-up before OCR:</span>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={prep.evenLighting} onChange={(e) => setPrep((p) => ({ ...p, evenLighting: e.target.checked }))} />
                Even out lighting & shadows (good for phone photos)
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={prep.reduceMoire} onChange={(e) => setPrep((p) => ({ ...p, reduceMoire: e.target.checked }))} />
                Reduce screen moiré (photos of a monitor)
              </label>
            </section>
            <section className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-300 px-1 -mt-3">
              <span className="text-slate-500 uppercase tracking-wider font-semibold">Kashmiri dictionary:</span>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={dict.enabled} onChange={(e) => setDict((d) => ({ ...d, enabled: e.target.checked }))} />
                Fix OCR misreads using the dictionary (56,000 attested words)
              </label>
              <label className={`flex items-center gap-2 ${dict.enabled ? "cursor-pointer" : "opacity-40"}`}>
                <input
                  type="checkbox"
                  disabled={!dict.enabled}
                  checked={dict.restoreSigns}
                  onChange={(e) => setDict((d) => ({ ...d, restoreSigns: e.target.checked }))}
                />
                Add vowel signs the OCR missed
              </label>
            </section>

            {inputMode === "camera" && <CameraViewfinder onCapture={handleCapture} onFallbackFile={handleImageFile} disabled={!!busy} />}
            {inputMode === "image" && <ImageUploader onImageSelected={handleImageFile} onSampleSelected={handleSample} disabled={!!busy} />}
            {inputMode === "pdf" && (
              <PdfBookReader
                fileName={pdf?.name ?? null}
                totalPages={pdf?.numPages ?? 0}
                currentPage={pdfPage}
                doneCount={doneCount}
                busy={!!busy}
                onOpenFile={handlePdfFile}
                onPageChange={(n) => pdf && showPdfPage(pdf.doc, n)}
                onRunCurrentPage={() => ocrPdfPage(pdfPage)}
                onRunBatch={ocrAllPages}
                onCancel={() => (cancelRef.current = true)}
              />
            )}

            {busy && (
              <div className="flex items-center gap-3 p-3 rounded-xl bg-cyan-950/40 border border-cyan-800/50 text-sm text-cyan-200" role="status">
                <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                <span className="flex-1">{busy.label}…</span>
                <div className="w-40 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-400 transition-all" style={{ width: `${Math.round(busy.progress * 100)}%` }} />
                </div>
              </div>
            )}
            {error && (
              <div className="flex items-start gap-3 p-3 rounded-xl bg-amber-950/40 border border-amber-700/50 text-sm text-amber-200" role="alert">
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                <span className="flex-1">{error}</span>
                <button onClick={() => setError(null)} aria-label="Dismiss" className="text-amber-300 hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            <VerificationStudio
              image={currentImage}
              lines={currentLines}
              exportLines={allLines}
              onLinesChange={updateCurrentLines}
              dictSettings={dict}
              docName={inputMode === "pdf" && pdf ? `${pdf.name} — page ${pdfPage}` : docName}
              engineUsed={engineUsed}
              emptyHint={
                inputMode === "pdf"
                  ? pdf
                    ? "Click “OCR this page” to read the text on this page."
                    : "Open a PDF to get started."
                  : inputMode === "camera"
                  ? "Point the camera at printed text and press Capture."
                  : "Upload an image, or try one of the samples."
              }
            />
          </div>
        ) : (
          <SynthCameraSimulator onSendToStudio={handleSimulatorSend} />
        )}
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 px-4 lg:px-8 text-xs text-slate-500 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>KoshurOCR Studio • In-browser OCR uses Tesseract&apos;s Urdu model; accuracy on Nastaliq is limited.</p>
          <a href="https://github.com/Rizvonium/kashmiri_ocr" className="text-cyan-400 hover:underline">
            GitHub Repository
          </a>
        </div>
      </footer>
    </div>
  );
}
