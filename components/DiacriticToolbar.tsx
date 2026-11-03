"use client";

import React from "react";

export interface DiacriticKey {
  char: string;
  name: string;
  insertChar: string;
}

export const KASHMIRI_KEYS: DiacriticKey[] = [
  { char: "ٲ", name: "Alef Maddah Below (əː)", insertChar: "ٲ" },
  { char: "أ", name: "Alef Hamza Above", insertChar: "أ" },
  { char: "◌ٕ", name: "Hamza Below (ɨ)", insertChar: "\u0655" },
  { char: "◌ٟ", name: "Kashmiri Wavy Dagger (ɨː)", insertChar: "\u065F" },
  { char: "ۄ", name: "Waw with Ring (ɔ)", insertChar: "ۄ" },
  { char: "ؠ", name: "Kashmiri Yeh (ʲ)", insertChar: "ؠ" },
  { char: "ژ", name: "Zhe", insertChar: "ژ" },
  { char: "ژھ", name: "Zhe Aspirated", insertChar: "ژھ" },
  { char: "◌ٚ", name: "Small V Above (e/o)", insertChar: "\u065A" },
  { char: "◌ٖ", name: "Subscript Alef", insertChar: "\u0656" },
  { char: "◌ٗ", name: "Inverted Damma", insertChar: "\u0657" },
  { char: "◌ْ", name: "Sukun / Jazm", insertChar: "\u0652" },
];

interface DiacriticToolbarProps {
  onInsert: (char: string) => void;
  disabled?: boolean;
}

export const DiacriticToolbar: React.FC<DiacriticToolbarProps> = ({ onInsert, disabled }) => {
  return (
    <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-xl">
      <div className="flex items-center justify-between mb-1.5 px-1">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Kashmiri Diacritic & Character Pad
        </span>
        <span className="text-[10px] text-slate-500 font-mono">Select a line, then click to insert</span>
      </div>
      <div className="grid grid-cols-6 sm:grid-cols-12 gap-1.5 text-center font-nastaliq text-base">
        {KASHMIRI_KEYS.map((key, i) => (
          <button
            key={i}
            type="button"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => onInsert(key.insertChar)}
            disabled={disabled}
            title={key.name}
            className="p-1.5 rounded-lg bg-slate-900 hover:bg-cyan-950 text-slate-200 hover:text-cyan-300 border border-slate-800 hover:border-cyan-600 transition flex flex-col items-center justify-center active:scale-95 disabled:opacity-40"
          >
            <span className="text-base select-none leading-none">{key.char.replace("◌", "")}</span>
            <span className="text-[8px] font-mono text-slate-500 mt-1">{key.char}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
