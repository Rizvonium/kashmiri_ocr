/**
 * Loads the Kashmiri lexicon (public/lexicon/ks-lexicon.json, built from the
 * Kashmiri dictionary dataset by scripts/build_lexicon.py) once, on first use.
 */
import { KashmiriLexicon, correctLine, DEFAULT_CORRECT_OPTIONS, type LexRow, type LineCorrection } from "./ks-lexicon";

let lexPromise: Promise<KashmiriLexicon> | null = null;
let loaded: KashmiriLexicon | null = null;

export function loadLexicon(): Promise<KashmiriLexicon> {
  if (!lexPromise) {
    lexPromise = fetch("/lexicon/ks-lexicon.json")
      .then((r) => {
        if (!r.ok) throw new Error(`dictionary download failed (HTTP ${r.status})`);
        return r.json();
      })
      .then((j: { forms: LexRow[] }) => (loaded = new KashmiriLexicon(j.forms)));
    lexPromise.catch(() => (lexPromise = null));
  }
  return lexPromise;
}

export const lexiconIfLoaded = () => loaded;

export interface CheckSettings {
  enabled: boolean;
  restoreSigns: boolean;
}

export function checkLine(lex: KashmiriLexicon, text: string, settings: CheckSettings, wordConfs?: number[]): LineCorrection {
  const opts = { ...DEFAULT_CORRECT_OPTIONS, restoreSigns: settings.restoreSigns };
  // Switched off: annotate each word (found / similar / unknown) but change nothing.
  if (!settings.enabled) return correctLine(lex, text, wordConfs, { ...opts, autoRestore: false, autoFuzzy: false });
  return correctLine(lex, text, wordConfs, opts);
}
