/**
 * Kashmiri orthography normalizer for OCR output.
 *
 * Uses the same convention as the Kashmiri dictionary dataset (scripts/ks_normalize.py in
 * github.com/Rizvonium/Kashmiri_dictionary) and the dual-lookup guide (section 6.3):
 *   - Unicode NFC (standard canonical order of vowel signs),
 *   - Arabic-keyboard letters mapped to Kashmiri/Urdu ones (ك→ک, ي/ى→ی, ه→ہ),
 *   - legacy code points replaced (ۆ→وٚ, ٳ→اٟ, ٮ۪→ؠ, small low meem→ٟ),
 *   - Kashmiri letters and signs always kept as they are (ٲ ۄ ؠ ۍ ێ, ٕ ٔ ٚ ٛ ٟ …).
 *
 * Earlier versions turned ۍ into ؠ and ٛ into a sukun and re-ordered signs by hand; that
 * produced spellings the dictionary never contains, so lookups failed.
 */
import { normalizeKs } from "./ks-lexicon";

export function normalizeKashmiriUnicode(text: string): string {
  if (!text) return "";
  return text
    .replace(/[­]/g, "")
    .split("\n")
    .map((l) => normalizeKs(l).replace(/[ \t]+/g, " ").trim())
    .join("\n");
}
