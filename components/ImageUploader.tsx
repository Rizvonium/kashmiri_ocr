"use client";

import React, { useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import type { SampleKind } from "../lib/image-tools";

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  onSampleSelected: (type: SampleKind) => void;
  disabled?: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, onSampleSelected, disabled }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && !disabled) onImageSelected(f);
      }}
      onClick={() => !disabled && fileInputRef.current?.click()}
      className={`border-2 border-dashed rounded-2xl p-6 bg-slate-900/40 text-center transition group ${
        disabled ? "opacity-60 cursor-wait" : "cursor-pointer"
      } ${dragging ? "border-cyan-400" : "border-slate-800 hover:border-cyan-500/60"}`}
    >
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onImageSelected(f);
          e.target.value = "";
        }}
      />
      <div className="w-12 h-12 mx-auto rounded-full bg-slate-800/80 flex items-center justify-center text-slate-400 group-hover:text-cyan-400 transition mb-3">
        <UploadCloud className="w-6 h-6" />
      </div>
      <h4 className="text-sm font-semibold text-white group-hover:text-cyan-300 transition">
        Drag & drop a Kashmiri scan or photo, or click to browse
      </h4>
      <p className="text-xs text-slate-400 mt-1">PNG, JPG or WEBP. Straight, well-lit, high-resolution images work best.</p>

      <div className="mt-4 pt-3 border-t border-slate-800/60 flex flex-wrap items-center justify-center gap-3 text-xs" onClick={(e) => e.stopPropagation()}>
        <span className="text-slate-500">Or try a sample page:</span>
        <button
          disabled={disabled}
          onClick={() => onSampleSelected("dictionary")}
          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 font-medium disabled:opacity-40"
        >
          Dictionary (Naskh font)
        </button>
        <button
          disabled={disabled}
          onClick={() => onSampleSelected("poetry")}
          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-emerald-300 font-medium disabled:opacity-40"
        >
          Poetry (Nastaliq font)
        </button>
      </div>
    </div>
  );
};
