"use client";

import React, { useEffect, useState } from "react";
import { BookOpen, Search, Undo2 } from "lucide-react";
import { joinWords, type KashmiriLexicon, type Match, type WordResult, type WordStatus } from "../lib/ks-lexicon";
import { loadLexicon, lexiconIfLoaded } from "../lib/dictionary";
import type { OcrLine } from "../lib/ocr-engine";

const STATUS: Record<WordStatus, { label: string; chip: string; help: string }> = {
  ok: { label: "in dictionary", chip: "border-emerald-600/60 text-emerald-200 bg-emerald-950/40", help: "Found exactly in the dictionary." },
  restored: {
    label: "restored",
    chip: "border-cyan-500/70 text-cyan-200 bg-cyan-950/50",
    help: "Same letters as the OCR read; the dictionary spelling added the Kashmiri letters or vowel signs the OCR missed.",
  },
  corrected: {
    label: "corrected",
    chip: "border-violet-500/70 text-violet-200 bg-violet-950/50",
    help: "The OCR was unsure and one dictionary word was a clearly better fit, so it was used.",
  },
  suggest: {
    label: "check",
    chip: "border-amber-500/70 text-amber-200 bg-amber-950/40",
    help: "Not in the dictionary as read. Similar dictionary words are listed; nothing was changed.",
  },
  unknown: {
    label: "unknown",
    chip: "border-red-500/60 text-red-200 bg-red-950/40",
    help: "Nothing similar in the dictionary. It may be a name, a rare word, or a bad misread.",
  },
  skip: { label: "", chip: "border-slate-700 text-slate-400", help: "Number or punctuation; not checked." },
};

function Meaning({ m }: { m: Match }) {
  const e = m.entry;
  return (
    <span className="text-slate-300">
      {e.gloss || <em className="text-slate-500">no English meaning recorded (attested spelling)</em>}
      {(e.pos || e.valency) && (
        <span className="text-slate-500">
          {" "}
          · {[e.pos, e.valency].filter(Boolean).join(", ")}
        </span>
      )}
    </span>
  );
}

interface Props {
  line: OcrLine | undefined;
  onChange: (line: OcrLine) => void;
}

export const WordCheckPanel: React.FC<Props> = ({ line, onChange }) => {
  const [selected, setSelected] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [lex, setLex] = useState<KashmiriLexicon | null>(lexiconIfLoaded());
  const [lookup, setLookup] = useState<Match[] | null>(null);

  useEffect(() => setSelected(null), [line?.id]);

  const check = line?.check;
  const words = check?.words ?? [];
  const word = selected != null ? words[selected] : undefined;

  const replaceWord = (i: number, patch: Partial<WordResult>) => {
    if (!line || !check) return;
    const next = words.map((w, k) => (k === i ? { ...w, ...patch } : w));
    const text = joinWords(next, check.separators);
    const grounded = next.filter((w) => w.status !== "skip");
    onChange({
      ...line,
      text,
      check: {
        ...check,
        words: next,
        text,
        grounded: grounded.length ? grounded.filter((w) => ["ok", "restored", "corrected"].includes(w.status)).length / grounded.length : 1,
      },
    });
  };

  const runLookup = async () => {
    const q = query.trim();
    if (!q) return;
    const l = lex ?? (await loadLexicon());
    setLex(l);
    setLookup(l.lookupByForm(q, 6));
  };

  const changed = words.filter((w) => w.text !== w.original).length;

  return (
    <div className="p-3 bg-slate-950/80 border border-slate-800/80 rounded-xl space-y-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-semibold text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
          <BookOpen className="w-3.5 h-3.5" /> Dictionary check (selected line)
        </span>
        {check && (
          <span className="text-[11px] text-slate-400 font-mono">
            {Math.round(check.grounded * 100)}% of words in dictionary{changed ? ` · ${changed} fixed` : ""}
          </span>
        )}
      </div>

      {!line ? (
        <p className="text-xs text-slate-500">Select a line to see each word checked against the dictionary.</p>
      ) : !check ? (
        <p className="text-xs text-slate-500">This line has not been checked yet.</p>
      ) : (
        <>
          <div dir="rtl" className="flex flex-wrap gap-1.5">
            {words.map((w, i) =>
              w.status === "skip" ? null : (
                <button
                  key={i}
                  onClick={() => setSelected(i === selected ? null : i)}
                  title={STATUS[w.status].help}
                  className={`px-2 py-0.5 rounded-md border font-nastaliq text-sm leading-[2] ${STATUS[w.status].chip} ${
                    i === selected ? "ring-2 ring-white/60" : ""
                  }`}
                >
                  {w.text}
                </button>
              )
            )}
          </div>
          <div className="flex flex-wrap gap-3 text-[10px] text-slate-500">
            {(["ok", "restored", "corrected", "suggest", "unknown"] as WordStatus[]).map((s) => (
              <span key={s} className="flex items-center gap-1">
                <span className={`w-2.5 h-2.5 rounded-sm border ${STATUS[s].chip}`} /> {STATUS[s].label}
              </span>
            ))}
          </div>

          {word && (
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-xs space-y-2">
              <p className="text-slate-400">{STATUS[word.status].help}</p>
              {word.text !== word.original && (
                <p className="flex items-center gap-2 flex-wrap">
                  <span className="text-slate-500">OCR read:</span>
                  <span className="font-nastaliq text-sm text-slate-200" dir="rtl">
                    {word.original}
                  </span>
                  <button
                    onClick={() => replaceWord(selected!, { text: word.original, status: "suggest" })}
                    className="flex items-center gap-1 text-amber-300 hover:text-amber-200"
                  >
                    <Undo2 className="w-3 h-3" /> keep what OCR read
                  </button>
                </p>
              )}
              {(word.status === "ok" || word.status === "restored" || word.status === "corrected") && word.candidates[0] && (
                <p>
                  <span className="text-slate-500">Meaning: </span>
                  <Meaning m={word.candidates.find((c) => c.entry.form === word.text) ?? word.candidates[0]} />
                </p>
              )}
              {word.candidates.filter((c) => c.entry.form !== word.text).length > 0 && (
                <div className="space-y-1">
                  <span className="text-slate-500">Other dictionary words:</span>
                  {word.candidates
                    .filter((c) => c.entry.form !== word.text)
                    .slice(0, 5)
                    .map((c) => (
                      <button
                        key={c.entry.form}
                        onClick={() => replaceWord(selected!, { text: c.entry.form, status: "ok" })}
                        className="w-full text-left flex items-baseline gap-2 px-2 py-1 rounded hover:bg-slate-800"
                      >
                        <span className="font-nastaliq text-sm text-white shrink-0" dir="rtl">
                          {c.entry.form}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono shrink-0">{Math.round(c.score * 100)}%</span>
                        <span className="truncate">
                          <Meaning m={c} />
                        </span>
                      </button>
                    ))}
                </div>
              )}
            </div>
          )}
        </>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          runLookup();
        }}
        className="flex items-center gap-2 pt-1 border-t border-slate-800/80"
      >
        <Search className="w-3.5 h-3.5 text-slate-500 shrink-0" />
        <input
          dir="rtl"
          lang="ks"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Look up a Kashmiri word…"
          className="flex-1 min-w-0 bg-transparent text-sm font-nastaliq text-white placeholder:text-slate-600 focus:outline-none py-1"
        />
        <button type="submit" className="text-[11px] text-cyan-400 hover:text-cyan-300 font-semibold">
          Look up
        </button>
      </form>
      {lookup && (
        <div className="space-y-1 text-xs">
          {lookup.length === 0 && <p className="text-slate-500">Not found in the dictionary.</p>}
          {lookup.map((m) => (
            <p key={m.entry.form} className="flex items-baseline gap-2">
              <span className="font-nastaliq text-sm text-white shrink-0" dir="rtl">
                {m.entry.form}
              </span>
              <span className="text-[10px] text-slate-500 font-mono shrink-0">{m.kind === "exact" ? "exact" : `${Math.round(m.score * 100)}%`}</span>
              <Meaning m={m} />
            </p>
          ))}
        </div>
      )}
    </div>
  );
};
