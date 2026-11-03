"use client";

import React, { useEffect, useRef, useState } from "react";
import { Copy, Trash2, ZoomIn, ZoomOut, Plus, FileSearch } from "lucide-react";
import { DiacriticToolbar } from "./DiacriticToolbar";
import { normalizeKashmiriUnicode } from "../lib/kashmiri-normalizer";
import { transliterateKashmiriToRoman } from "../lib/transliteration";
import { exportToTxt, exportToJson, exportToDocx, exportToPdf } from "../lib/export-utils";
import type { OcrLine } from "../lib/ocr-engine";
import { WordCheckPanel } from "./WordCheckPanel";
import { checkLine, lexiconIfLoaded, type CheckSettings } from "../lib/dictionary";

export type LineItem = OcrLine;

interface VerificationStudioProps {
  image: HTMLCanvasElement | null;
  lines: LineItem[];
  exportLines: LineItem[];
  onLinesChange: (updated: LineItem[]) => void;
  dictSettings: CheckSettings;
  docName: string;
  engineUsed: string;
  emptyHint: string;
}

export const VerificationStudio: React.FC<VerificationStudioProps> = ({
  image,
  lines,
  exportLines,
  onLinesChange,
  dictSettings,
  docName,
  engineUsed,
  emptyHint,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputRefs = useRef<Record<number, HTMLInputElement | null>>({});
  const [activeLineId, setActiveLineId] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);
  const [copied, setCopied] = useState(false);

  // Keep the active line valid when the document changes.
  useEffect(() => {
    if (!lines.some((l) => l.id === activeLineId)) setActiveLineId(lines[0]?.id ?? null);
  }, [lines, activeLineId]);

  useEffect(() => setZoom(1), [image]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);
    const lw = Math.max(2, image.width / 500);
    lines.forEach((line) => {
      const isActive = line.id === activeLineId;
      ctx.beginPath();
      line.poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = isActive ? "rgba(6, 182, 212, 0.22)" : "rgba(16, 185, 129, 0.08)";
      ctx.fill();
      ctx.strokeStyle = isActive ? "#06b6d4" : "rgba(16, 185, 129, 0.8)";
      ctx.lineWidth = isActive ? lw * 1.5 : lw;
      ctx.stroke();
    });
  }, [image, lines, activeLineId]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const hit = lines.find((l) => {
      const xs = l.poly.map((p) => p[0]);
      const ys = l.poly.map((p) => p[1]);
      return x >= Math.min(...xs) && x <= Math.max(...xs) && y >= Math.min(...ys) && y <= Math.max(...ys);
    });
    if (hit) {
      setActiveLineId(hit.id);
      inputRefs.current[hit.id]?.focus();
    }
  };

  const updateLineText = (id: number, text: string) => {
    // The old word check no longer matches; it is redone when editing of the line ends.
    onLinesChange(lines.map((l) => (l.id === id ? { ...l, text, roman: transliterateKashmiriToRoman(text), check: undefined } : l)));
  };

  // Normalize once editing of a line is finished, so the cursor does not jump while typing.
  const normalizeLine = (id: number) => {
    const line = lines.find((l) => l.id === id);
    if (!line) return;
    const norm = normalizeKashmiriUnicode(line.text);
    if (line.check && line.check.text === norm) return; // unchanged: keep the check and its OCR history
    const lex = lexiconIfLoaded();
    // Re-check what the person typed, but never auto-change it.
    const check = lex ? checkLine(lex, norm, { ...dictSettings, enabled: false }) : line.check;
    if (norm !== line.text || check !== line.check) {
      onLinesChange(lines.map((l) => (l.id === id ? { ...l, text: norm, roman: transliterateKashmiriToRoman(norm), check } : l)));
    }
  };

  const insertDiacriticToActive = (char: string) => {
    if (activeLineId == null) return;
    const input = inputRefs.current[activeLineId];
    if (!input) return;
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    const newVal = input.value.slice(0, start) + char + input.value.slice(end);
    updateLineText(activeLineId, newVal);
    requestAnimationFrame(() => {
      input.focus();
      input.selectionStart = input.selectionEnd = start + char.length;
    });
  };

  const deleteLine = (id: number) => onLinesChange(lines.filter((l) => l.id !== id));

  const addLine = () => {
    const id = Math.max(0, ...lines.map((l) => l.id)) + 1;
    onLinesChange([...lines, { id, text: "", roman: "", poly: [[0, 0], [0, 0], [0, 0], [0, 0]], conf: 1, page: lines[0]?.page }]);
    setActiveLineId(id);
    requestAnimationFrame(() => inputRefs.current[id]?.focus());
  };

  const handleCopy = () => {
    const full = exportLines.map((l) => l.text).join("\n");
    navigator.clipboard
      .writeText(full)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => alert("Your browser blocked clipboard access. Use the .TXT export instead."));
  };

  const activeLine = lines.find((l) => l.id === activeLineId);
  const totalWords = lines.reduce((acc, l) => acc + l.text.trim().split(/\s+/).filter(Boolean).length, 0);
  const totalChars = lines.reduce((acc, l) => acc + l.text.length, 0);
  const checked = lines.filter((l) => l.check);
  const grounded = checked.length ? checked.reduce((a, l) => a + (l.check?.grounded ?? 0), 0) / checked.length : null;
  const meanConf = lines.length ? lines.reduce((a, l) => a + l.conf, 0) / lines.length : 0;
  const multiPage = new Set(exportLines.map((l) => l.page)).size > 1;
  const nothingToExport = exportLines.length === 0;

  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
      {/* LEFT: document image with detected line boxes */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center space-x-2 min-w-0">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shrink-0" />
            <h3 className="text-sm font-semibold text-white truncate">{docName || "Input document"}</h3>
          </div>
          <div className="flex items-center space-x-2 text-xs shrink-0">
            <button
              onClick={() => setZoom((z) => Math.min(4, z * 1.25))}
              disabled={!image}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
              title="Zoom in"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => setZoom((z) => Math.max(1, z / 1.25))}
              disabled={!image || zoom <= 1}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
              title="Zoom out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="relative w-full h-[60vh] max-h-[640px] bg-slate-950 rounded-xl overflow-auto border border-slate-800">
          {image ? (
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              style={{ width: `${zoom * 100}%` }}
              className="block mx-auto cursor-crosshair"
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 text-slate-500">
              <FileSearch className="w-10 h-10 mb-3" />
              <p className="text-sm">{emptyHint}</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 text-[11px] font-mono text-center">
          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">WORDS IN DICTIONARY</span>
            <span className={grounded == null ? "text-slate-200" : grounded >= 0.7 ? "text-emerald-400" : grounded >= 0.4 ? "text-amber-400" : "text-red-400"}>
              {grounded == null ? "—" : `${Math.round(grounded * 100)}%`}
            </span>
          </div>
          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">LINES FOUND</span>
            <span className="text-cyan-400">{lines.length}</span>
          </div>
          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
            <span className="text-slate-500 block text-[9px]">AVG CONFIDENCE</span>
            <span className={meanConf >= 0.7 ? "text-emerald-400" : meanConf >= 0.4 ? "text-amber-400" : "text-red-400"}>
              {lines.length ? `${Math.round(meanConf * 100)}%` : "—"}
            </span>
          </div>
        </div>
        {engineUsed && <p className="text-[11px] text-slate-500 px-1">Engine: {engineUsed}</p>}
      </div>

      {/* RIGHT: editable text */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3 min-w-0">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            <h3 className="text-sm font-semibold text-white">Editable Kashmiri Text</h3>
          </div>
          <div className="flex items-center gap-1.5 text-xs">
            <button
              onClick={handleCopy}
              disabled={nothingToExport}
              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1 disabled:opacity-40"
            >
              <Copy className="w-3.5 h-3.5" />
              {copied ? "Copied!" : "Copy"}
            </button>
            {(
              [
                [".TXT", () => exportToTxt(exportLines)],
                [".DOC", () => exportToDocx(exportLines)],
                [
                  ".JSON",
                  () =>
                    exportToJson(
                      exportLines.map(({ id, page, text, ocrText, roman, conf, poly, check }) => ({
                        id,
                        page,
                        text,
                        ocr_text: ocrText,
                        roman,
                        confidence: conf,
                        poly,
                        words_in_dictionary: check ? Math.round(check.grounded * 100) / 100 : undefined,
                        dictionary_fixes: check?.words
                          .filter((w) => w.text !== w.original)
                          .map((w) => ({ from: w.original, to: w.text, reason: w.status })),
                      })),
                      { engine: engineUsed, source: docName, dictionary: "github.com/Rizvonium/Kashmiri_dictionary" }
                    ),
                ],
              ] as const
            ).map(([label, fn]) => (
              <button
                key={label}
                onClick={fn}
                disabled={nothingToExport}
                className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 disabled:opacity-40"
              >
                {label}
              </button>
            ))}
            <button
              onClick={() => exportToPdf(exportLines)}
              disabled={nothingToExport}
              className="px-2 py-1 rounded bg-cyan-700 hover:bg-cyan-600 text-white font-semibold disabled:opacity-40"
            >
              PDF
            </button>
          </div>
        </div>
        {multiPage && <p className="text-[11px] text-slate-400 -mt-1">Copy and exports include all OCR&apos;d pages ({exportLines.length} lines).</p>}

        <DiacriticToolbar onInsert={insertDiacriticToActive} disabled={activeLineId == null} />

        <div className="flex items-center justify-between text-xs text-slate-400 px-1 font-mono text-[11px]">
          <span>
            Lines: {lines.length} | Words: {totalWords} | Chars: {totalChars}
          </span>
          <button onClick={addLine} disabled={!image} className="flex items-center gap-1 text-cyan-400 hover:text-cyan-300 disabled:opacity-40">
            <Plus className="w-3 h-3" /> Add line
          </button>
        </div>

        <div className="min-h-[300px] max-h-[500px] overflow-y-auto space-y-2.5 p-3 bg-slate-950 border border-slate-800 rounded-xl">
          {lines.length === 0 && <p className="text-center text-sm text-slate-500 pt-24">Recognized text will appear here.</p>}
          {lines.map((line, index) => {
            const isActive = line.id === activeLineId;
            return (
              <div
                key={line.id}
                onClick={() => setActiveLineId(line.id)}
                className={`p-2.5 rounded-xl border transition flex items-center justify-between gap-3 cursor-pointer ${
                  isActive ? "bg-slate-900 border-cyan-500" : "bg-slate-950/60 border-slate-800/80 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center space-x-2 text-xs font-mono text-slate-500 shrink-0">
                  <span
                    className={`w-5 h-5 rounded-md flex items-center justify-center ${
                      isActive ? "bg-cyan-500 text-slate-950 font-bold" : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {index + 1}
                  </span>
                  <span className={`text-[10px] font-bold ${line.conf >= 0.7 ? "text-emerald-400" : line.conf >= 0.4 ? "text-amber-400" : "text-red-400"}`}>
                    {Math.round(line.conf * 100)}%
                  </span>
                  {line.check && line.ocrText !== undefined && line.text !== line.ocrText && (
                    <span title={`OCR read: ${line.ocrText}`} className="text-[10px] text-cyan-400">fixed</span>
                  )}
                </div>
                <input
                  type="text"
                  ref={(el) => {
                    inputRefs.current[line.id] = el;
                  }}
                  dir="rtl"
                  lang="ks"
                  value={line.text}
                  onFocus={() => setActiveLineId(line.id)}
                  onChange={(e) => updateLineText(line.id, e.target.value)}
                  onBlur={() => normalizeLine(line.id)}
                  className="flex-1 min-w-0 bg-transparent text-white font-nastaliq text-base leading-[2.2] focus:outline-none focus:text-cyan-300"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteLine(line.id);
                  }}
                  aria-label={`Delete line ${index + 1}`}
                  className="text-slate-600 hover:text-red-400 p-1 shrink-0"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })}
        </div>

        <WordCheckPanel line={activeLine} onChange={(l) => onLinesChange(lines.map((x) => (x.id === l.id ? { ...l, roman: transliterateKashmiriToRoman(l.text) } : x)))} />

        <div className="p-3 bg-slate-950/80 border border-slate-800/80 rounded-xl">
          <span className="text-[11px] font-semibold text-cyan-400 uppercase tracking-wider block mb-1">
            Roman transliteration (selected line, approximate)
          </span>
          <p className="text-xs text-slate-300 font-serif leading-relaxed italic min-h-[1.25rem]">
            {activeLine ? activeLine.roman || transliterateKashmiriToRoman(activeLine.text) : "—"}
          </p>
        </div>
      </div>
    </section>
  );
};
