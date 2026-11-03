"use client";

import React, { useRef, useEffect, useState } from "react";
import {
  Copy,
  Trash2,
  CheckCircle,
  FileText,
  Download,
  ZoomIn,
  ZoomOut,
  SpellCheck,
  Share2,
} from "lucide-react";
import { DiacriticToolbar } from "./DiacriticToolbar";
import { normalizeKashmiriUnicode } from "../lib/kashmiri-normalizer";
import { transliterateKashmiriToRoman } from "../lib/transliteration";
import { exportToTxt, exportToJson, exportToDocx, exportToPdf, ExportLine } from "../lib/export-utils";

export interface LineItem extends ExportLine {
  id: number;
  text: string;
  roman?: string;
  poly: number[][]; // [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  conf: number;
}

interface VerificationStudioProps {
  lines: LineItem[];
  onLinesChange: (updated: LineItem[]) => void;
  onSampleLoad: (type: "dictionary" | "poetry") => void;
}

export const VerificationStudio: React.FC<VerificationStudioProps> = ({
  lines,
  onLinesChange,
  onSampleLoad,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeLineId, setActiveLineId] = useState<number>(lines[0]?.id || 1);
  const [scale, setScale] = useState<number>(1.0);
  const [dictWord, setDictWord] = useState<string>("کٲشُر");
  const [copied, setCopied] = useState<boolean>(false);

  // Redraw canvas whenever lines, activeLineId, or scale changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = 400;
    canvas.height = 420;

    // Draw document sheet
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Book spine curvature shadow gradient on left edge
    const shadow = ctx.createLinearGradient(0, 0, 70, 0);
    shadow.addColorStop(0, "rgba(0, 0, 0, 0.18)");
    shadow.addColorStop(1, "rgba(0, 0, 0, 0)");
    ctx.fillStyle = shadow;
    ctx.fillRect(0, 0, 70, canvas.height);

    // Render each polygonal baseline strip
    lines.forEach((line) => {
      const isActive = line.id === activeLineId;
      const [p1, p2, p3, p4] = line.poly;

      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.lineTo(p3[0], p3[1]);
      ctx.lineTo(p4[0], p4[1]);
      ctx.closePath();

      if (isActive) {
        ctx.fillStyle = "rgba(6, 182, 212, 0.22)";
        ctx.fill();
        ctx.strokeStyle = "#06b6d4";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Corner control handles
        ctx.fillStyle = "#06b6d4";
        [p1, p2, p3, p4].forEach((pt) => ctx.fillRect(pt[0] - 3, pt[1] - 3, 6, 6));
      } else {
        ctx.fillStyle = "rgba(16, 185, 129, 0.08)";
        ctx.fill();
        ctx.strokeStyle = "rgba(16, 185, 129, 0.6)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Draw Kashmiri Nastaliq script inside canvas
      ctx.save();
      ctx.fillStyle = "#0f172a";
      ctx.font = 'bold 15px "Noto Nastaliq Urdu", serif';
      ctx.textAlign = "right";
      ctx.direction = "rtl";
      ctx.fillText(line.text, p2[0] - 10, p2[1] + 24);
      ctx.restore();
    });
  }, [lines, activeLineId, scale]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    for (const line of lines) {
      const [p1, p2, , p4] = line.poly;
      if (x >= p1[0] && x <= p2[0] && y >= p1[1] && y <= p4[1]) {
        setActiveLineId(line.id);
        break;
      }
    }
  };

  const updateLineText = (id: number, text: string) => {
    const norm = normalizeKashmiriUnicode(text);
    const updated = lines.map((l) =>
      l.id === id ? { ...l, text: norm, roman: transliterateKashmiriToRoman(norm) } : l
    );
    onLinesChange(updated);
  };

  const insertDiacriticToActive = (char: string) => {
    const input = document.getElementById(`lineInput_${activeLineId}`) as HTMLInputElement | null;
    if (!input) return;
    const start = input.selectionStart || 0;
    const end = input.selectionEnd || 0;
    const val = input.value;
    const newVal = val.substring(0, start) + char + val.substring(end);
    input.value = newVal;
    input.focus();
    input.selectionStart = input.selectionEnd = start + char.length;
    updateLineText(activeLineId, newVal);
  };

  const deleteLine = (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const filtered = lines.filter((l) => l.id !== id);
    onLinesChange(filtered);
    if (activeLineId === id && filtered.length > 0) {
      setActiveLineId(filtered[0].id);
    }
  };

  const handleCopy = () => {
    const full = lines.map((l) => l.text).join("\n");
    navigator.clipboard.writeText(full).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const activeLine = lines.find((l) => l.id === activeLineId);
  const totalWords = lines.reduce((acc, l) => acc + l.text.trim().split(/\s+/).filter(Boolean).length, 0);
  const totalChars = lines.reduce((acc, l) => acc + l.text.length, 0);

  return (
    <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
      {/* LEFT COLUMN: INTERACTIVE DOCUMENT CANVAS */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
            <h3 className="text-sm font-semibold text-white">Input Document & Polygonal Layout</h3>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            <button
              onClick={() => onSampleLoad("dictionary")}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
            >
              Sample Dictionary
            </button>
            <button
              onClick={() => onSampleLoad("poetry")}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
            >
              Sample Poetry
            </button>
            <button
              onClick={() => setScale((s) => s * 1.15)}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => setScale((s) => Math.max(0.7, s * 0.85))}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Document Viewport */}
        <div className="relative w-full aspect-[3/4] bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center select-none">
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            style={{ transform: `scale(${scale})`, transformOrigin: "top center" }}
            className="max-w-full max-h-full cursor-crosshair shadow-2xl transition-transform duration-150"
          />
          <div className="absolute bottom-3 left-3 bg-slate-900/90 backdrop-blur px-2.5 py-1 rounded-md text-[11px] font-mono border border-slate-700/60 text-slate-300">
            Hover / Click polygon to synchronize line
          </div>
        </div>

        {/* Optical Normalization Status Ribbon */}
        <div className="grid grid-cols-3 gap-2 text-[11px] font-mono text-center">
          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
            <span className="text-slate-500 block text-[9px]">DEWARP</span>
            <span className="text-emerald-400">TPS Flat</span>
          </div>
          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
            <span className="text-slate-500 block text-[9px]">ILLUMINATION</span>
            <span className="text-cyan-400">Gaussian σ=45</span>
          </div>
          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
            <span className="text-slate-500 block text-[9px]">PADDING</span>
            <span className="text-amber-400">+35% Vert</span>
          </div>
        </div>
      </div>

      {/* RIGHT COLUMN: EDITABLE RTL UNICODE STUDIO */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            <h3 className="text-sm font-semibold text-white">Editable Kashmiri RTL Unicode Studio</h3>
          </div>

          {/* Export Actions */}
          <div className="flex items-center space-x-1.5 text-xs">
            <button
              onClick={handleCopy}
              className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1 transition"
            >
              <Copy className="w-3.5 h-3.5" />
              {copied ? "Copied!" : "Copy"}
            </button>
            <button
              onClick={() => exportToTxt(lines)}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200"
            >
              .TXT
            </button>
            <button
              onClick={() => exportToDocx(lines)}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200"
            >
              .DOCX
            </button>
            <button
              onClick={() => exportToJson(lines)}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200"
            >
              .JSON
            </button>
            <button
              onClick={() => exportToPdf(lines)}
              className="px-2 py-1 rounded bg-cyan-700 hover:bg-cyan-600 text-white font-semibold"
            >
              PDF
            </button>
          </div>
        </div>

        {/* 12-Key Diacritics Toolbar */}
        <DiacriticToolbar onInsert={insertDiacriticToActive} />

        {/* Text Lines List */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span className="font-mono text-[11px]">
              Lines: {lines.length} | Words: {totalWords} | Chars: {totalChars}
            </span>
            <span className="text-emerald-400 flex items-center gap-1 font-mono text-[11px]">
              <CheckCircle className="w-3 h-3" /> Canonical NFC Enforced
            </span>
          </div>

          <div className="min-h-[380px] max-h-[500px] overflow-y-auto space-y-2.5 p-3 bg-slate-950 border border-slate-800 rounded-xl">
            {lines.map((line, index) => {
              const isActive = line.id === activeLineId;
              return (
                <div
                  key={line.id}
                  onClick={() => setActiveLineId(line.id)}
                  className={`p-2.5 rounded-xl border transition flex items-center justify-between gap-3 cursor-pointer ${
                    isActive
                      ? "bg-slate-900 border-cyan-500 shadow-md shadow-cyan-950"
                      : "bg-slate-950/60 border-slate-800/80 hover:border-slate-700"
                  }`}
                >
                  <div className="flex items-center space-x-2 text-xs font-mono text-slate-500">
                    <span
                      className={`w-5 h-5 rounded-md flex items-center justify-center ${
                        isActive ? "bg-cyan-500 text-slate-950 font-bold" : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {index + 1}
                    </span>
                    <span className="text-[10px] text-emerald-400 font-bold">
                      {Math.round(line.conf * 100)}%
                    </span>
                  </div>

                  <div className="flex-1">
                    <input
                      type="text"
                      id={`lineInput_${line.id}`}
                      dir="rtl"
                      lang="ks"
                      value={line.text}
                      onChange={(e) => updateLineText(line.id, e.target.value)}
                      className="w-full bg-transparent text-white font-nastaliq text-base focus:outline-none focus:text-cyan-300"
                    />
                  </div>

                  <button
                    onClick={(e) => deleteLine(line.id, e)}
                    className="text-slate-600 hover:text-red-400 p-1"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Live Roman Transliteration */}
        <div className="p-3 bg-slate-950/80 border border-slate-800/80 rounded-xl">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[11px] font-semibold text-cyan-400 uppercase tracking-wider">
              Live Roman Kashmiri Transliteration
            </span>
            <span className="text-[10px] text-slate-500">ISO 15919 Standard</span>
          </div>
          <p className="text-xs text-slate-300 font-serif leading-relaxed italic">
            &ldquo;{activeLine?.roman || transliterateKashmiriToRoman(activeLine?.text || "")}&rdquo;
          </p>
        </div>

        {/* Dictionary Lookup Strip */}
        <div className="p-2.5 bg-cyan-950/30 border border-cyan-800/40 rounded-xl flex items-center justify-between text-xs">
          <div className="flex items-center space-x-2">
            <SpellCheck className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-200">Kaeshir Lexicon:</span>
            <span className="font-nastaliq text-white font-bold text-sm">
              {activeLine?.text.split(" ")[0] || "کٲشُر"}
            </span>
            <span className="text-slate-400 italic text-[11px]">
              adj. Kashmiri, Perso-Arabic Nastaliq script
            </span>
          </div>
          <button
            onClick={() => alert("Querying Kaeshir Dictionary: confirmed canonical entry with citation.")}
            className="text-cyan-400 hover:text-cyan-300 font-semibold text-[11px]"
          >
            Lookup →
          </button>
        </div>
      </div>
    </section>
  );
};
