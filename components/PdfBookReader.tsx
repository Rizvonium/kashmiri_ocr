"use client";

import React, { useState } from "react";
import { BookOpen, ChevronLeft, ChevronRight, Play, Layers } from "lucide-react";

interface PdfBookReaderProps {
  fileName: string;
  totalPages: number;
  currentPage: number;
  onPageChange: (newPage: number) => void;
  onRunCurrentPage: () => void;
  onRunBatch: () => void;
}

export const PdfBookReader: React.FC<PdfBookReaderProps> = ({
  fileName,
  totalPages,
  currentPage,
  onPageChange,
  onRunCurrentPage,
  onRunBatch,
}) => {
  return (
    <section className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center space-x-3">
        <div className="p-1.5 rounded-lg bg-cyan-950 text-cyan-400 border border-cyan-800/40">
          <BookOpen className="w-4 h-4" />
        </div>
        <div>
          <span className="text-xs text-slate-400 block sm:inline mr-2">PDF Book:</span>
          <span className="text-xs font-semibold text-cyan-300 font-mono">{fileName}</span>
          <span className="ml-2 text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300">
            Page {currentPage} / {totalPages}
          </span>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <button
          onClick={() => onPageChange(Math.max(1, currentPage - 1))}
          disabled={currentPage <= 1}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
        <span className="text-xs font-mono font-bold px-2">Page {currentPage}</span>
        <button
          onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
          disabled={currentPage >= totalPages}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 disabled:opacity-40"
        >
          <ChevronRight className="w-4 h-4" />
        </button>

        <button
          onClick={onRunCurrentPage}
          className="ml-3 px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold flex items-center space-x-1.5 shadow-sm transition"
        >
          <Play className="w-3.5 h-3.5" />
          <span>OCR Current Page</span>
        </button>

        <button
          onClick={onRunBatch}
          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold flex items-center space-x-1.5 transition"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Batch Stream (1-5)</span>
        </button>
      </div>
    </section>
  );
};
