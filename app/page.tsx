"use client";

import React, { useState } from "react";
import {
  ScanText,
  Sliders,
  BookOpen,
  Image as ImageIcon,
  Camera,
  Sparkles,
  Zap,
} from "lucide-react";
import { VerificationStudio, LineItem } from "../components/VerificationStudio";
import { CameraViewfinder } from "../components/CameraViewfinder";
import { PdfBookReader } from "../components/PdfBookReader";
import { ImageUploader } from "../components/ImageUploader";
import { SynthCameraSimulator } from "../components/SynthCameraSimulator";
import { normalizeKashmiriUnicode } from "../lib/kashmiri-normalizer";
import { transliterateKashmiriToRoman } from "../lib/transliteration";

const SAMPLE_DOCS = {
  dictionary: [
    { id: 1, text: "کٲشُر لُغَت — جِلد گۄڈٕنِیُک", poly: [[40, 30], [360, 30], [360, 65], [40, 65]], conf: 0.98 },
    { id: 2, text: "آب — اسم مذکر: رۄپہِ ہِو شفاف پانہِ", poly: [[40, 80], [360, 80], [360, 115], [40, 115]], conf: 0.96 },
    { id: 3, text: "آزٲدی — اسم مؤنث: خۄشحٲلی تہٕ چُھٹکارٕ", poly: [[40, 130], [360, 130], [360, 165], [40, 165]], conf: 0.97 },
    { id: 4, text: "اٟٹھ — عدد: سَتھ پتہٕ یِنہٕ وۄل عدَد", poly: [[40, 180], [360, 180], [360, 215], [40, 215]], conf: 0.95 },
    { id: 5, text: "بَرگ — پَن: کُلِک زرد یا سبز حِصہٕ", poly: [[40, 230], [360, 230], [360, 265], [40, 265]], conf: 0.94 },
    { id: 6, text: "ترٛومُت — صفت: آبَس مَنٛز کٔڈِتھ بَرٛوٛنٛہہ ترٛاوُن", poly: [[40, 280], [360, 280], [360, 315], [40, 315]], conf: 0.96 },
    { id: 7, text: "دۄپھ — آواز: زمینَس پؠٹھ کِہین پؠنہٕ چِھ چھَنٛڈ", poly: [[40, 330], [360, 330], [360, 365], [40, 365]], conf: 0.93 },
  ],
  poetry: [
    { id: 1, text: "مےٚ چُھ کٲشُر زَبانہِ پؠٹھ غُرور", poly: [[50, 40], [350, 40], [350, 80], [50, 80]], conf: 0.99 },
    { id: 2, text: "صدیَن پؠٹھ چھےٚ یۄس بَہان پُر نُور", poly: [[50, 95], [350, 95], [350, 135], [50, 135]], conf: 0.97 },
    { id: 3, text: "لَل دؠد تہٕ نُند رؠشؠن کٔر شَنٛگار", poly: [[50, 150], [350, 150], [350, 190], [50, 190]], conf: 0.96 },
    { id: 4, text: "مَحجوٗر تہٕ حَبہٕ خاتوٗنَن دِیُت پیار", poly: [[50, 205], [350, 205], [350, 245], [50, 245]], conf: 0.98 },
    { id: 5, text: "ہر کٲنٛسہِ شٲعِرَس مَنٛز چُھ اَکھ راز", poly: [[50, 260], [350, 260], [350, 300], [50, 300]], conf: 0.95 },
  ],
};

function formatInitialLines(rawList: typeof SAMPLE_DOCOCS.dictionary): LineItem[] {
  return rawList.map((item) => {
    const norm = normalizeKashmiriUnicode(item.text);
    return {
      ...item,
      text: norm,
      roman: transliterateKashmiriToRoman(norm),
    };
  });
}

// Temporary alias
const SAMPLE_DOCOCS = SAMPLE_DOCS;

export default function Home() {
  const [activeTab, setActiveTab] = useState<"studio" | "simulator">("studio");
  const [inferenceMode, setInferenceMode] = useState<"cloud" | "wasm">("cloud");
  const [activeInputMode, setActiveInputMode] = useState<"pdf" | "image" | "camera">("pdf");

  const [showCamera, setShowCamera] = useState<boolean>(false);
  const [showPdfToolbar, setShowPdfToolbar] = useState<boolean>(true);
  const [showImageUploader, setShowImageUploader] = useState<boolean>(false);

  const [pdfPage, setPdfPage] = useState<number>(1);
  const [pdfTotalPages, setPdfTotalPages] = useState<number>(14);

  const [lines, setLines] = useState<LineItem[]>(() => formatInitialLines(SAMPLE_DOCS.dictionary));

  const handleSampleLoad = (type: "dictionary" | "poetry") => {
    setLines(formatInitialLines(SAMPLE_DOCS[type]));
  };

  const handleCameraCapture = (imgData: string) => {
    setShowCamera(false);
    handleSampleLoad("poetry");
  };

  const handleSendFromSimulator = (simLines: string[]) => {
    const formatted: LineItem[] = simLines.map((str, idx) => {
      const norm = normalizeKashmiriUnicode(str);
      const y = 40 + idx * 65;
      return {
        id: idx + 1,
        text: norm,
        roman: transliterateKashmiriToRoman(norm),
        poly: [[40, y], [360, y], [360, y + 45], [40, y + 45]],
        conf: 0.96,
      };
    });
    setLines(formatted);
    setActiveTab("studio");
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* HEADER */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50 px-4 lg:px-8 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-cyan-500/20 font-nastaliq text-xl font-bold text-white">
              ک
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                  KoshurOCR Studio
                  <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 font-mono">
                    v2.0 Production
                  </span>
                </h1>
              </div>
              <p className="text-xs text-slate-400">Printed Kashmiri Nastaliq Camera, PDF & Image-to-Unicode App</p>
            </div>
          </div>

          {/* Navigation & Inference Mode */}
          <div className="flex items-center space-x-3">
            <nav className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
              <button
                onClick={() => setActiveTab("studio")}
                className={`px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
                  activeTab === "studio" ? "bg-cyan-600 text-white shadow-sm" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <ScanText className="w-3.5 h-3.5" /> OCR Studio
              </button>
              <button
                onClick={() => setActiveTab("simulator")}
                className={`px-3 py-1.5 rounded-md font-medium transition flex items-center gap-1.5 ${
                  activeTab === "simulator" ? "bg-cyan-600 text-white shadow-sm" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Sliders className="w-3.5 h-3.5" /> SynthCamera Simulator
              </button>
            </nav>

            <div className="hidden sm:flex items-center space-x-2 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs">
              <span className="text-slate-400">Engine:</span>
              <button
                onClick={() => setInferenceMode("cloud")}
                className={`px-2 py-0.5 rounded font-semibold transition ${
                  inferenceMode === "cloud"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                Cloud Turbo (Modal GPU)
              </button>
              <button
                onClick={() => setInferenceMode("wasm")}
                className={`px-2 py-0.5 rounded font-semibold transition ${
                  inferenceMode === "wasm"
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                Browser WebGPU (INT8)
              </button>
            </div>

            <a
              href="https://github.com/Rizvonium/kashmiri_ocr"
              target="_blank"
              rel="noreferrer"
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition"
            >
              <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
            </a>
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-6 flex flex-col gap-6">
        {activeTab === "studio" ? (
          <div className="flex flex-col gap-6">
            {/* 3 Input Mode Cards */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div
                onClick={() => {
                  setActiveInputMode("pdf");
                  setShowPdfToolbar(true);
                  setShowCamera(false);
                  setShowImageUploader(false);
                }}
                className={`cursor-pointer group p-4 rounded-xl border-2 transition relative overflow-hidden ${
                  activeInputMode === "pdf"
                    ? "border-cyan-500/60 bg-gradient-to-br from-slate-900 to-slate-900/80 shadow-lg shadow-cyan-950/30"
                    : "border-slate-800 bg-slate-900/40 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="p-2.5 rounded-lg bg-cyan-950 text-cyan-400 border border-cyan-800/50">
                    <BookOpen className="w-5 h-5" />
                  </div>
                  <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-300 border border-cyan-700/50">
                    pdfjs-dist 300 DPI
                  </span>
                </div>
                <h3 className="font-semibold text-white group-hover:text-cyan-300 transition text-sm">
                  1. Multi-Page PDF Book
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Render 500+ page dictionaries client-side without crashing Vercel payload limits.
                </p>
              </div>

              <div
                onClick={() => {
                  setActiveInputMode("image");
                  setShowImageUploader(true);
                  setShowCamera(false);
                }}
                className={`cursor-pointer group p-4 rounded-xl border-2 transition relative overflow-hidden ${
                  activeInputMode === "image"
                    ? "border-emerald-500/60 bg-gradient-to-br from-slate-900 to-slate-900/80 shadow-lg shadow-emerald-950/30"
                    : "border-slate-800 bg-slate-900/40 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="p-2.5 rounded-lg bg-emerald-950 text-emerald-400 border border-emerald-800/50">
                    <ImageIcon className="w-5 h-5" />
                  </div>
                  <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-700/50">
                    PNG/JPG/WEBP
                  </span>
                </div>
                <h3 className="font-semibold text-white group-hover:text-emerald-300 transition text-sm">
                  2. Image Scan & Crop
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Upload book scans, poetry snippets, or screen photos for immediate recognition.
                </p>
              </div>

              <div
                onClick={() => {
                  setActiveInputMode("camera");
                  setShowCamera(true);
                  setShowImageUploader(false);
                }}
                className={`cursor-pointer group p-4 rounded-xl border-2 transition relative overflow-hidden ${
                  activeInputMode === "camera"
                    ? "border-amber-500/60 bg-gradient-to-br from-slate-900 to-slate-900/80 shadow-lg shadow-amber-950/30"
                    : "border-slate-800 bg-slate-900/40 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="p-2.5 rounded-lg bg-amber-950 text-amber-400 border border-amber-800/50">
                    <Camera className="w-5 h-5" />
                  </div>
                  <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-700/50">
                    Laplacian Meter
                  </span>
                </div>
                <h3 className="font-semibold text-white group-hover:text-amber-300 transition text-sm">
                  3. Smartphone Camera
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Live phone camera with real-time blur gate & hand-shadow division normalization.
                </p>
              </div>
            </section>

            {/* Camera Drawer */}
            {showCamera && (
              <CameraViewfinder onCapture={handleCameraCapture} onClose={() => setShowCamera(false)} />
            )}

            {/* Image Uploader Drawer */}
            {showImageUploader && (
              <ImageUploader
                onImageSelected={() => {
                  setShowImageUploader(false);
                  handleSampleLoad("dictionary");
                }}
                onSampleSelected={(t) => {
                  setShowImageUploader(false);
                  handleSampleLoad(t);
                }}
              />
            )}

            {/* PDF Virtual Scroller Toolbar */}
            {showPdfToolbar && (
              <PdfBookReader
                fileName="kasir_lughat_vol1.pdf"
                totalPages={pdfTotalPages}
                currentPage={pdfPage}
                onPageChange={(p) => setPdfPage(p)}
                onRunCurrentPage={() => handleSampleLoad("dictionary")}
                onRunBatch={() => alert("Batch OCR queue: streaming 5 pages to Modal GPU microservice...")}
              />
            )}

            {/* Synchronized Side-by-Side Studio */}
            <VerificationStudio lines={lines} onLinesChange={setLines} onSampleLoad={handleSampleLoad} />
          </div>
        ) : (
          <SynthCameraSimulator onSendToStudio={handleSendFromSimulator} />
        )}
      </main>

      {/* FOOTER */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 px-4 lg:px-8 text-center text-xs text-slate-500 mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>KoshurOCR Studio v2.0 • Persian-Arabic Nastaliq OCR & Canonical Unicode Pipeline</p>
          <div className="flex items-center space-x-4">
            <span>Trained with KS-PRET-5M & Kaeshir Dictionary</span>
            <a href="https://github.com/Rizvonium/kashmiri_ocr" className="text-cyan-400 hover:underline">
              GitHub Repository
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
