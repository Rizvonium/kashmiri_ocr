/**
 * High-fidelity Kashmiri Perso-Arabic to Roman Transliteration (ISO 15919 Standard)
 */

const CHAR_MAP: Record<string, string> = {
  'آ': 'Ā', 'ا': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 'ṭ',
  'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd',
  'ڈ': 'ḍ', 'ذ': 'z', 'ر': 'r', 'ڑ': 'ṛ', 'ز': 'z', 'ژ': 'zh',
  'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z',
  'ع': "'", 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g',
  'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'v', 'ہ': 'h',
  'ھ': 'h', 'ی': 'y', 'ے': 'e', 'ؠ': 'y', 'ۄ': 'o', 'ٲ': 'ə̄',
  'ٚ': 'e', 'ٕ': 'i', 'ٟ': 'ị̄', 'ْ': '', 'ِ': 'i', 'َ': 'a', 'ُ': 'u',
  '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
  '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
};

export function transliterateKashmiriToRoman(text: string): string {
  if (!text) return "";
  let out = "";
  const chars = Array.from(text);
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    out += CHAR_MAP[c] !== undefined ? CHAR_MAP[c] : c;
  }
  return out;
}
