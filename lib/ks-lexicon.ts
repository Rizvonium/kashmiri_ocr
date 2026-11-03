/**
 * Dictionary-backed OCR post-correction for Kashmiri.
 *
 * Implements the "form lookup" of the dual-lookup verification guide
 * (Sheikh, 2026, sections 6.3 and 7.3) on top of the Kashmiri dictionary dataset
 * (github.com/Rizvonium/Kashmiri_dictionary):
 *
 *   1. exact key   NFC + letter fixes (ك→ک, ي→ی, ه→ہ …)       precise match first
 *   2. bare key    vowel signs removed (the guide's "skeleton")  restores dropped signs
 *   3. loose key   Kashmiri letters folded to Urdu look-alikes   restores ٲ ۄ ؠ ۍ ے …
 *                  (ٲ→ا, ۄ→و, ؠ/ۍ/ے→ی …, same rules as the dictionary website)
 *   4. fuzzy       character-trigram index narrows candidates, then an edit distance
 *                  that charges less for letters OCR confuses (ڑ/ژ/ز/ر, ب/پ/ت/ٹ/ث …)
 *
 * Code decides, the dictionary supplies: a word is only ever replaced by an attested
 * form from the lexicon, and only when the evidence is unambiguous. Everything else is
 * offered as a suggestion or flagged as unknown, never silently changed.
 *
 * This file has no imports so it can also run under Node (scripts/eval-dictionary-correction.mjs).
 */

// ---------------------------------------------------------------- normalisation keys

const KS_MAP: Record<string, string> = {
  "ك": "ک", // ك -> ک
  "ي": "ی", // ي -> ی
  "ى": "ی", // ى -> ی
  "ه": "ہ", // ه -> ہ
  "ۀ": "ۂ", // ۀ -> ۂ
  "ۭ": "ٟ", // legacy small low meem -> wavy hamza below
  "ۆ": "وٚ", // ۆ -> وٚ
  "ٳ": "اٟ", // ٳ -> اٟ
  "ـ": "", // tatweel
  "‍": "", // ZWJ
  "‎": "",
  "‏": "",
  "﻿": "",
};
const KS_MAP_RE = /[كيىهۀۭۆٳـ‍‎‏﻿]/g;
// Vowel signs and other combining marks (madda U+0653 is kept so آ stays آ).
const MARK_RE = /[ً-ْٔ-ٰ۪]/g;
const LOOSE: Record<string, string> = {
  "ٲ": "ا", "أ": "ا", "إ": "ا", "آ": "ا", // ٲ أ إ آ -> ا
  "ۄ": "و", "ؤ": "و", // ۄ ؤ -> و
  "ؠ": "ی", "ێ": "ی", "ۍ": "ی", "ئ": "ی", "ے": "ی", // ؠ ێ ۍ ئ ے -> ی
  "ۂ": "ہ", "ۃ": "ہ", "ة": "ہ", // ۂ ۃ ة -> ہ
  "ں": "ن", // ں -> ن
  "ٓ": "", "‌": "",
};
const LOOSE_RE = /[ٲأإآۄؤؠێۍئےۂۃةںٓ‌]/g;
const ARABIC_LETTER_RE = /[ؠ-يٮ-ۓۺ-ۿ]/;
// Letters and vowel signs (not Arabic punctuation such as ۔ ، ؛ ؟) make up the core of a word.
const WORD_CHARS = "\\u0620-\\u065F\\u0670-\\u06D3\\u06D5-\\u06ED\\u06FA-\\u06FF\\u200C";
const EDGE_PUNCT_RE = new RegExp(`^([^${WORD_CHARS}]*)(.*?)([^${WORD_CHARS}]*)$`, "s");

export function normalizeKs(s: string): string {
  return s
    .replace(/[ﭐ-﷿ﹰ-﻾]/g, (c) => c.normalize("NFKC"))
    .replace(/ٮ۪/g, "ؠ")
    .replace(KS_MAP_RE, (c) => KS_MAP[c])
    .replace(/(?<=[\u0620-\u06D3])[ \t]+([\u064B-\u065F\u0670])/g, "$1 ") 
    .replace(/[ \t]{2,}/g, " ") // ...re-attach a detached vowel sign; collapse the spaces it leaves
    .replace(/([ً-ٰٟ])\1+/g, "$1")
    .normalize("NFC")
    .trim();
}
export const bareKey = (s: string) => normalizeKs(s).normalize("NFD").replace(MARK_RE, "").normalize("NFC");
export const looseKey = (s: string) =>
  bareKey(s).normalize("NFD").replace(LOOSE_RE, (c) => LOOSE[c] ?? c).normalize("NFC");

// ---------------------------------------------------------------- OCR-aware edit distance

// Letters Tesseract's Urdu model commonly swaps on Kashmiri print (dots and small strokes).
const CONFUSION_GROUPS = ["ڑژزرذ", "بپتٹثنی", "جچحخ", "دڈذ", "سش", "صض", "طظ", "عغ", "فق", "کگ", "ہھ", "وۄ", "یے"];
const confusable = new Map<string, Set<string>>();
for (const g of CONFUSION_GROUPS) for (const c of g) {
  const set = confusable.get(c) ?? new Set<string>();
  for (const d of g) if (d !== c) set.add(d);
  confusable.set(c, set);
}
const CHEAP_INDEL = new Set(["ا", "و", "ی", "ہ", "ھ"]); // letters OCR drops or invents most

export function ocrDistance(a: string, b: string): number {
  const A = Array.from(a), B = Array.from(b);
  let prev = new Array(B.length + 1);
  for (let j = 0; j <= B.length; j++) prev[j] = j === 0 ? 0 : prev[j - 1] + (CHEAP_INDEL.has(B[j - 1]) ? 0.6 : 1);
  for (let i = 1; i <= A.length; i++) {
    const cur = new Array(B.length + 1);
    cur[0] = prev[0] + (CHEAP_INDEL.has(A[i - 1]) ? 0.6 : 1);
    for (let j = 1; j <= B.length; j++) {
      const x = A[i - 1], y = B[j - 1];
      const sub = x === y ? 0 : confusable.get(x)?.has(y) ? 0.4 : 1;
      cur[j] = Math.min(
        prev[j - 1] + sub,
        prev[j] + (CHEAP_INDEL.has(x) ? 0.6 : 1),
        cur[j - 1] + (CHEAP_INDEL.has(y) ? 0.6 : 1)
      );
    }
    prev = cur;
  }
  return prev[B.length];
}

export function similarity(a: string, b: string): number {
  const n = Math.max(Array.from(a).length, Array.from(b).length);
  return n ? Math.max(0, 1 - ocrDistance(a, b) / n) : 0;
}

/** Letters merged by confusion group: one wrong dot or stroke no longer changes the key. */
const GROUP_REP = new Map<string, string>();
for (const g of CONFUSION_GROUPS) for (const c of g) if (!GROUP_REP.has(c)) GROUP_REP.set(c, g[0]);
export const classKey = (s: string) => Array.from(looseKey(s), (c) => GROUP_REP.get(c) ?? c).join("");

// Kashmiri letters that the Urdu OCR model writes as a near look-alike.
const LOOKALIKE_PAIRS = new Set(
  [["أ", "ٲ"], ["ا", "ٲ"], ["ی", "ؠ"], ["ی", "ۍ"], ["ے", "ۍ"], ["و", "ۄ"], ["ۓ", "ے"]].flatMap(([a, b]) => [a + b, b + a])
);
const isMark = (c: string) => /[\u064B-\u0652\u0654-\u0670\u06EA]/.test(c);
const lookalike = (a: string, b: string) => LOOKALIKE_PAIRS.has(a + b);

/**
 * How well a dictionary spelling fits what OCR read, on the full form (letters + signs).
 * Adding a sign OCR missed is cheap; deleting a sign OCR saw is expensive.
 */
export function fitDistance(read: string, cand: string): { dist: number; deletesMarks: boolean } {
  const A = Array.from(read.normalize("NFC")), B = Array.from(cand.normalize("NFC"));
  const del = (c: string) => (isMark(c) ? 0.9 : 1); // in OCR, not in the dictionary form
  const ins = (c: string) => (isMark(c) ? 0.25 : 1); // in the dictionary form, OCR missed it
  const sub = (x: string, y: string) =>
    x === y ? 0 : isMark(x) && isMark(y) ? 0.6 : lookalike(x, y) ? 0.3 : confusable.get(x)?.has(y) ? 0.5 : 1;
  const D: number[][] = [[0]];
  for (let j = 1; j <= B.length; j++) D[0][j] = D[0][j - 1] + ins(B[j - 1]);
  for (let i = 1; i <= A.length; i++) {
    D[i] = [D[i - 1][0] + del(A[i - 1])];
    for (let j = 1; j <= B.length; j++) {
      D[i][j] = Math.min(D[i - 1][j - 1] + sub(A[i - 1], B[j - 1]), D[i - 1][j] + del(A[i - 1]), D[i][j - 1] + ins(B[j - 1]));
    }
  }
  const readMarks = A.filter(isMark), candMarks = B.filter(isMark);
  const remaining = [...candMarks];
  const deletesMarks = readMarks.some((m) => {
    const k = remaining.indexOf(m);
    if (k < 0) return true;
    remaining.splice(k, 1);
    return false;
  });
  return { dist: D[A.length][B.length], deletesMarks };
}

function trigrams(s: string): string[] {
  const p = Array.from(`#${s}#`);
  const out: string[] = [];
  for (let i = 0; i + 3 <= p.length; i++) out.push(p.slice(i, i + 3).join(""));
  return out.length ? out : [p.join("")];
}

// ---------------------------------------------------------------- lexicon

export interface LexEntry {
  form: string;
  weight: number; // 3 native headword, 2 corpus-attested, 1 phrase word / 1932, 0.5 unverified
  gloss: string;
  pos: string;
  valency: string;
  bare: string;
  loose: string;
  cls: string;
}

export type MatchKind = "exact" | "bare" | "loose" | "fuzzy";

export interface Match {
  entry: LexEntry;
  kind: MatchKind;
  score: number; // 0..1
  deletesMarks?: boolean; // using this spelling would remove a vowel sign OCR saw
}

export type LexRow = [string, number, string, string, string];

export class KashmiriLexicon {
  readonly entries: LexEntry[] = [];
  private exact = new Map<string, number>();
  private byBare = new Map<string, number[]>();
  private byLoose = new Map<string, number[]>();
  private byClass = new Map<string, number[]>();
  private grams = new Map<string, number[]>();

  constructor(rows: LexRow[], exclude?: Set<string>) {
    for (const [form, weight, gloss, pos, valency] of rows) {
      const f = normalizeKs(form);
      if (exclude?.has(f) || this.exact.has(f)) continue;
      const e: LexEntry = { form: f, weight, gloss, pos, valency, bare: bareKey(f), loose: looseKey(f), cls: classKey(f) };
      const i = this.entries.push(e) - 1;
      this.exact.set(f, i);
      push(this.byBare, e.bare, i);
      push(this.byLoose, e.loose, i);
      push(this.byClass, e.cls, i);
      for (const g of new Set(trigrams(e.cls))) push(this.grams, g, i);
    }
  }

  get size() {
    return this.entries.length;
  }

  get(form: string): LexEntry | undefined {
    const i = this.exact.get(normalizeKs(form));
    return i === undefined ? undefined : this.entries[i];
  }

  /** Lookup 1 of the guide: what does the written token correspond to in the dictionary? */
  lookupByForm(token: string, topK = 5): Match[] {
    const t = normalizeKs(token);
    const hit = this.exact.get(t);
    const bare = bareKey(t);
    const loose = looseKey(t);
    // An exact hit on a trusted spelling wins. A hit on a low-trust spelling (auto-transliterated
    // or unverified, weight < 1) only wins if no trusted spelling shares its letters.
    if (hit !== undefined && this.entries[hit].weight >= 1) return [{ entry: this.entries[hit], kind: "exact", score: 1 }];
    // Same skeleton: rank the attested spellings by how well they fit what was read.
    const rank = (ids: number[] | undefined, kind: MatchKind) =>
      (ids ?? [])
        .map((i) => {
          const e = this.entries[i];
          const fit = fitDistance(t, e.form);
          const n = Math.max(Array.from(t).length, 1);
          return { entry: e, kind, score: Math.max(0.5, 1 - fit.dist / n), deletesMarks: fit.deletesMarks };
        })
        .sort((a, b) => rankScore(b) - rankScore(a))
        .slice(0, topK);

    // Same letters (bare key) or same letters up to Kashmiri look-alikes (loose key), together.
    const same = [...new Set([...(this.byBare.get(bare) ?? []), ...(this.byLoose.get(loose) ?? [])])];
    if (hit !== undefined) {
      const trusted = same.filter((i) => this.entries[i].weight >= 2);
      if (!trusted.length) return [{ entry: this.entries[hit], kind: "exact", score: 1 }];
    }
    const sameHits = rank(same, "loose").map((m) => (m.entry.bare === bare ? { ...m, kind: "bare" as MatchKind } : m));
    if (sameHits.length) return sameHits;

    // Fuzzy: same confusion-class key, or sharing a third of the class trigrams;
    // then the OCR-aware distance on the loose keys.
    const cls = classKey(t);
    const tg = trigrams(cls);
    const counts = new Map<number, number>();
    for (const i of this.byClass.get(cls) ?? []) counts.set(i, 99);
    for (const g of new Set(tg)) for (const i of this.grams.get(g) ?? []) counts.set(i, (counts.get(i) ?? 0) + 1);
    const minShared = Math.max(1, Math.floor(tg.length / 3));
    const len = Array.from(loose).length;
    const scored: Match[] = [];
    for (const [i, c] of counts) {
      if (c < minShared) continue;
      const e = this.entries[i];
      if (Math.abs(Array.from(e.loose).length - len) > 2) continue;
      scored.push({ entry: e, kind: "fuzzy", score: similarity(loose, e.loose) });
    }
    scored.sort((a, b) => b.score + 0.015 * b.entry.weight - (a.score + 0.015 * a.entry.weight));
    return scored.slice(0, topK);
  }
}

/** Fit to what OCR read, plus trust in the spelling's source; low-trust spellings are penalised. */
export const rankScore = (m: Match) => m.score + 0.04 * m.entry.weight - (m.entry.weight < 1 ? 0.08 : 0);

function push(m: Map<string, number[]>, k: string, i: number) {
  const a = m.get(k);
  if (a) a.push(i);
  else m.set(k, [i]);
}

// ---------------------------------------------------------------- line correction

export type WordStatus =
  | "ok" // found exactly in the dictionary
  | "restored" // same letters, dictionary restored vowel signs / Kashmiri letters
  | "corrected" // OCR misread fixed from a confident fuzzy match
  | "suggest" // plausible dictionary words offered, word left as read
  | "unknown" // nothing similar in the dictionary
  | "skip"; // numbers, punctuation, Latin

export interface WordResult {
  original: string; // as OCR read it (core, without punctuation)
  text: string; // after correction
  prefix: string;
  suffix: string;
  status: WordStatus;
  candidates: Match[];
  ocrConf?: number; // 0..100 when known
}

export interface CorrectOptions {
  autoRestore: boolean; // apply same-letter restorations automatically (off = only annotate)
  minRestoreScore: number; // how well the dictionary spelling must fit what was read to restore it
  restoreSigns: boolean; // add vowel signs the dictionary spelling has but OCR did not read
  autoFuzzy: boolean; // allow replacing misreads, not just restoring signs
  minFuzzy: number; // similarity needed to suggest
  autoFuzzyScore: number; // similarity needed to auto-correct
  margin: number; // lead over the runner-up needed to auto-correct
  maxOcrConfForFuzzy: number; // only auto-correct words the OCR itself was unsure of
}

export const DEFAULT_CORRECT_OPTIONS: CorrectOptions = {
  autoRestore: true,
  minRestoreScore: 0.85,
  restoreSigns: true,
  autoFuzzy: true,
  minFuzzy: 0.6,
  autoFuzzyScore: 0.9,
  margin: 0.06,
  maxOcrConfForFuzzy: 80,
};

export function correctWord(
  lex: KashmiriLexicon,
  raw: string,
  ocrConf?: number,
  opts: CorrectOptions = DEFAULT_CORRECT_OPTIONS
): WordResult {
  const [, prefix, core, suffix] = EDGE_PUNCT_RE.exec(raw) ?? ["", "", raw, ""];
  const base: WordResult = { original: core, text: core, prefix, suffix, status: "skip", candidates: [], ocrConf };
  if (!core || !ARABIC_LETTER_RE.test(core)) return base;
  const matches = lex.lookupByForm(core);
  const top = matches[0];
  if (!top) return { ...base, status: "unknown" };
  if (top.kind === "exact") return { ...base, text: top.entry.form, status: "ok", candidates: matches };

  if (top.kind === "bare" || top.kind === "loose") {
    // Same letters. Restore only when the best-fitting attested spelling is clear and
    // only adds what OCR missed (never deletes a vowel sign that was read).
    if (top.entry.form === core) return { ...base, status: "ok", candidates: matches };
    const second = matches[1];
    const clear = !second || rankScore(top) - rankScore(second) >= 0.04;
    const ok = opts.autoRestore && clear && !top.deletesMarks && top.score >= opts.minRestoreScore && (opts.restoreSigns || bareKey(core) !== top.entry.bare);
    return ok
      ? { ...base, text: top.entry.form, status: "restored", candidates: matches }
      : { ...base, status: "suggest", candidates: matches };
  }

  const good = matches.filter((m) => m.score >= opts.minFuzzy);
  if (!good.length) return { ...base, status: "unknown", candidates: matches.slice(0, 3) };
  // Spellings that differ only in vowel signs are one candidate for the margin test.
  const rival = good.find((m) => m.entry.loose !== good[0].entry.loose);
  const sameLetters = good.filter((m) => m.entry.loose === good[0].entry.loose);
  const lead = good[0].score - (rival?.score ?? 0);
  if (sameLetters.length > 1 && sameLetters[1].entry.weight === good[0].entry.weight && opts.autoFuzzy) {
    // Letters are clear but the sign placement is not: fix the letters, keep OCR's signs.
    const best = sameLetters.find((m) => m.entry.form === m.entry.bare) ?? good[0];
    good.unshift(...good.splice(good.indexOf(best), 1));
  }
  const longEnough = Array.from(looseKey(core)).length >= 3;
  const ocrUnsure = ocrConf === undefined || ocrConf <= opts.maxOcrConfForFuzzy;
  if (opts.autoFuzzy && longEnough && ocrUnsure && good[0].score >= opts.autoFuzzyScore && lead >= opts.margin) {
    return { ...base, text: good[0].entry.form, status: "corrected", candidates: good };
  }
  return { ...base, status: "suggest", candidates: good };
}

export interface LineCorrection {
  words: WordResult[];
  separators: string[]; // whitespace between words (words.length + 1 entries)
  text: string;
  grounded: number; // share of checkable words now attested in the dictionary (guide's LGR)
}

export function correctLine(
  lex: KashmiriLexicon,
  line: string,
  wordConfs?: number[],
  opts: CorrectOptions = DEFAULT_CORRECT_OPTIONS
): LineCorrection {
  const parts = line.split(/(\s+)/);
  const words: WordResult[] = [];
  const separators: string[] = [parts[0] === "" ? "" : ""];
  let w = 0;
  for (let i = 0; i < parts.length; i++) {
    const p = parts[i];
    if (i % 2 === 1) {
      separators[separators.length - 1] += p;
      continue;
    }
    if (p === "") continue;
    words.push(correctWord(lex, p, wordConfs?.[w++], opts));
    separators.push("");
  }
  return { words, separators, text: joinWords(words, separators), grounded: groundedShare(words) };
}

export function joinWords(words: WordResult[], separators: string[]): string {
  let s = separators[0] ?? "";
  words.forEach((w, i) => {
    s += w.prefix + w.text + w.suffix + (i < words.length - 1 ? separators[i + 1] || " " : separators[i + 1] ?? "");
  });
  return s.trim();
}

export function groundedShare(words: WordResult[]): number {
  const checkable = words.filter((w) => w.status !== "skip");
  if (!checkable.length) return 1;
  return checkable.filter((w) => w.status === "ok" || w.status === "restored" || w.status === "corrected").length / checkable.length;
}
