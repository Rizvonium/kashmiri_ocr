"use client";

import React, { useRef } from "react";
import { UploadCloud, Image as ImageIcon } from "lucide-react";

interface ImageUploaderProps {
  onImageSelected: (file: File) => void;
  onSampleSelected: (type: "dictionary" | "poetry") => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelected, onSampleSelected }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageSelected(e.dataTransfer.files[0]);
    }
  };

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className="border-2 border-dashed border-slate-800 hover:border-cyan-500/60 rounded-2xl p-6 bg-slate-900/40 text-center cursor-pointer transition group"
    >
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onImageSelected(e.target.files[0])}
      />
      <div className="w-12 h-12 mx-auto rounded-full bg-slate-800/80 flex items-center justify-center text-slate-400 group-hover:text-cyan-400 group-hover:bg-cyan-950/60 transition mb-3">
        <UploadCloud className="w-6 h-6" />
      </div>
      <h4 className="text-sm font-semibold text-white group-hover:text-cyan-300 transition">
        Drag & Drop Kashmiri Scan or Click to Browse
      </h4>
      <p className="text-xs text-slate-400 mt-1">Supports high-res PNG, JPG, WEBP, or multi-line book crops</p>

      <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-center gap-3 text-xs" onClick={(e) => e.stopPropagation()}>
        <span className="text-slate-500">Or test samples:</span>
        <button
          onClick={() => onSampleSelected("dictionary")}
          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 font-medium"
        >
          Kāśir Lug̲h̲at (Dictionary)
        </button>
        <button
          onClick={() => onSampleSelected("poetry")}
          className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-emerald-300 font-medium"
        >
          Lalla Vakh (Poetry)
        </button>
      </div>
    </div>
  );
};
