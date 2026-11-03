"use client";

import React, { useRef } from "react";
import { BookOpen, ChevronLeft, ChevronRight, Play, Layers, Upload, Square } from "lucide-react";

interface PdfBookReaderProps {
  fileName: string | null;
  totalPages: number;
  currentPage: number;
  doneCount: number;
  busy: boolean;
  onOpenFile: (file: File) => void;
  onPageChange: (newPage: number) => void;
  onRunCurrentPage: () => void;
  onRunBatch: () => void;
  onCancel: () => void;
}

export const PdfBookReader: React.FC<PdfBookReaderProps> = ({
  fileName,
  totalPages,
  currentPage,
  doneCount,
  busy,
  onOpenFile,
  onPageChange,
  onRunCurrentPage,
  onRunBatch,
  onCancel,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const btn = "px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition disabled:opacity-40 disabled:cursor-not-allowed";

  return (
    <section
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        const f = e.dataTransfer.files?.[0];
        if (f && !busy) onOpenFile(f);
      }}
      className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-4"
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf,.pdf"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onOpenFile(f);
          e.target.value = "";
        }}
      />
      <div className="flex items-center gap-3 min-w-0">
        <div className="p-1.5 rounded-lg bg-cyan-950 text-cyan-400 border border-cyan-800/40">
          <BookOpen className="w-4 h-4" />
        </div>
        {fileName ? (
          <div className="min-w-0">
            <span className="text-xs font-semibold text-cyan-300 font-mono truncate block max-w-[16rem]">{fileName}</span>
            <span className="text-[11px] text-slate-400">
              {totalPages} pages • {doneCount} OCR&apos;d
            </span>
          </div>
        ) : (
          <span className="text-xs text-slate-400">Choose a PDF (or drop one here).</span>
        )}
        <button onClick={() => inputRef.current?.click()} disabled={busy} className={`${btn} bg-slate-800 hover:bg-slate-700 text-slate-200`}>
          <Upload className="w-3.5 h-3.5" /> {fileName ? "Open another" : "Open PDF"}
        </button>
      </div>

      {fileName && (
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage <= 1 || busy}
            aria-label="Previous page"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <label className="text-xs font-mono flex items-center gap-1">
            Page
            <input
              type="number"
              min={1}
              max={totalPages}
              value={currentPage}
              disabled={busy}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (n >= 1 && n <= totalPages) onPageChange(n);
              }}
              className="w-14 bg-slate-950 border border-slate-700 rounded px-1 py-0.5 text-center"
            />
            / {totalPages}
          </label>
          <button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage >= totalPages || busy}
            aria-label="Next page"
            className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <button onClick={onRunCurrentPage} disabled={busy} className={`${btn} ml-2 bg-cyan-600 hover:bg-cyan-500 text-white`}>
            <Play className="w-3.5 h-3.5" /> OCR this page
          </button>
          {busy ? (
            <button onClick={onCancel} className={`${btn} bg-red-900/60 hover:bg-red-800 text-red-100`}>
              <Square className="w-3.5 h-3.5" /> Stop after this page
            </button>
          ) : (
            <button onClick={onRunBatch} disabled={doneCount >= totalPages} className={`${btn} bg-slate-800 hover:bg-slate-700 text-slate-200`}>
              <Layers className="w-3.5 h-3.5" /> OCR all pages
            </button>
          )}
        </div>
      )}
    </section>
  );
};
