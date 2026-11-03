/**
 * KashmiriOrthographyNormalizer (TypeScript)
 * Enforces canonical NFC Unicode sequences, Kashmiri-specific glyph remapping,
 * and deterministic syllable combining diacritic ordering.
 */

const MARK_PRIORITIES: Record<string, number> = {
  '\u0652': 10, // Sukun / Inverted-V Jazm
  '\u065B': 10, // African Inverted V -> Sukun
  '\u0651': 20, // Shaddah
  '\u0655': 30, // Arabic Hamza Below (short /ɨ/)
  '\u0654': 31, // Arabic Hamza Above (/ə/)
  '\u065A': 32, // Arabic Small V Above (/e/ and /o/)
  '\u065F': 33, // Arabic Wavy Dagger Below (long /ɨː/)
  '\u064E': 34, // Fatha
  '\u064F': 35, // Damma
  '\u0650': 36, // Kasra
  '\u0656': 37, // Subscript Alef
  '\u0657': 38, // Inverted Damma
  '\u0670': 39, // Superscript Alef
  '\u064B': 40, // Fathatan
  '\u064C': 41, // Dammatan
  '\u064D': 42, // Kasratan
};

const CANONICAL_REPLACEMENTS: Record<string, string> = {
  '\u0643': '\u06A9', // ك -> ک (Keheh)
  '\u064A': '\u06CC', // ي -> ی (Farsi/Urdu Yeh)
  '\u06C5': '\u06C4', // ۅ -> ۄ (Kashmiri Waw with Ring)
  '\u06CD': '\u0620', // ۍ -> ؠ (Kashmiri Yeh)
  '\u06D0': '\u0620', // ے with 2 dots -> ؠ
  '\u0673': '\u0627\u065F', // ٳ -> اٟ
  '\u065B': '\u0652', // Inverted V diacritic to Sukun
};

const DIACRITIC_REGEX = /[\u064B-\u065F\u0670]/;

function isPersoArabicLetter(ch: string): boolean {
  const code = ch.charCodeAt(0);
  return (
    ((code >= 0x0600 && code <= 0x06FF) ||
      (code >= 0x0750 && code <= 0x077F) ||
      (code >= 0xFB50 && code <= 0xFDFF) ||
      (code >= 0xFE70 && code <= 0xFEFF)) &&
    !(code >= 0x064B && code <= 0x065F)
  );
}

export function normalizeKashmiriUnicode(text: string): string {
  if (!text) return '';

  // 1. NFC normalization
  let res = text.normalize('NFC');

  // 2. Strip stray zero-width chars and soft hyphens
  res = res.replace(/[\u200E\u200F\u00AD]/g, '');

  // 3. Direct canonical replacements
  for (const [src, dst] of Object.entries(CANONICAL_REPLACEMENTS)) {
    res = res.split(src).join(dst);
  }

  // 4. Positional vowels
  // Replace ۆ with וٚ
  res = res.split('\u06C6').join('\u0648\u065A');

  // Replace ێ (medial -> یٚ, final -> ےٚ)
  res = res.replace(/\u06CE/g, (match, offset, str) => {
    const nextChar = str[offset + 1];
    if (!nextChar || !isPersoArabicLetter(nextChar)) {
      return '\u06D2\u065A'; // final ےٚ
    }
    return '\u06CC\u065A';     // medial یٚ
  });

  // 5. Deterministic Syllabic Combining Mark Re-ordering
  const chars = Array.from(res);
  const out: string[] = [];
  let i = 0;
  const n = chars.length;

  while (i < n) {
    const char = chars[i];
    if (DIACRITIC_REGEX.test(char)) {
      out.push(char);
      i++;
      continue;
    }

    out.push(char);
    i++;

    const marks: string[] = [];
    while (i < n && DIACRITIC_REGEX.test(chars[i])) {
      marks.push(chars[i]);
      i++;
    }

    if (marks.length > 1) {
      marks.sort((a, b) => (MARK_PRIORITIES[a] || 99) - (MARK_PRIORITIES[b] || 99));
    }
    out.push(...marks);
  }

  res = out.join('');

  // 6. Whitespace cleanup
  res = res.replace(/[ \t]+/g, ' ');
  res = res
    .split('\n')
    .map((l) => l.trim())
    .join('\n');

  return res;
}
