// Unit tests for the dictionary post-corrector:  npm run test:js
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const { KashmiriLexicon, correctLine, correctWord, normalizeKs, DEFAULT_CORRECT_OPTIONS } = await import(`${root}lib/ks-lexicon.ts`);
const rows = JSON.parse(readFileSync(`${root}public/lexicon/ks-lexicon.json`, "utf8")).forms;
const lex = new KashmiriLexicon(rows);

test("normalizer keeps Kashmiri letters and signs, fixes Arabic-keyboard letters", () => {
  assert.equal(normalizeKs("كتاب"), "کتاب");
  assert.equal(normalizeKs("مَنٛز"), "مَنٛز"); // U+065B kept (not turned into sukun)
  assert.equal(normalizeKs("آسمٲنۍ"), "آسمٲنۍ"); // ۍ kept (not turned into ؠ)
  assert.equal(normalizeKs("اڈلنۆک"), "اڈلنوٚک");
  assert.equal(normalizeKs("اَسنہ ٕ دٕہان"), "اَسنہٕ دٕہان"); // detached vowel sign re-attached
  assert.equal(normalizeKs("اَمہ ِکِس"), "اَمہِ کِس"); // ...to the word it belongs to
});

test("exact dictionary words are left alone", () => {
  const r = correctWord(lex, "کٲشُر");
  assert.equal(r.status, "ok");
  assert.equal(r.text, "کٲشُر");
});

test("Kashmiri letters the Urdu OCR model writes as look-alikes are restored", () => {
  assert.equal(correctWord(lex, "آزأدی", 40).text, "آزٲدی");
  assert.equal(correctWord(lex, "خوشحالی", 40).text, "خۄشحٲلی");
});

test("a vowel sign the OCR read is never deleted", () => {
  const r = correctWord(lex, "بَرگ", 80);
  assert.ok(r.text.includes("َ"), `fatha lost: ${r.text}`);
});

test("near-ties are suggested, not applied", () => {
  // two equally trusted spellings with the same letters: OCR's ہو could be either
  const tie = new KashmiriLexicon([["ہِو", 2, "like", "", ""], ["ہۄ", 2, "he", "", ""]]);
  const r = correctWord(tie, "ہو", 80);
  assert.equal(r.status, "suggest");
  assert.equal(r.text, "ہو");
  assert.ok(r.candidates.length >= 2);
});

test("words absent from the dictionary are not replaced when the OCR was confident", () => {
  const r = correctWord(lex, "زمیئس", 99);
  assert.equal(r.text, "زمیئس");
});

test("switching correction off annotates but changes nothing", () => {
  const line = "کأشر لقّت جلد گوڑنٹُک";
  const r = correctLine(lex, line, undefined, { ...DEFAULT_CORRECT_OPTIONS, autoRestore: false, autoFuzzy: false });
  assert.equal(r.text, line);
  assert.ok(r.words.some((w) => w.status === "suggest"));
});

test("punctuation and spacing survive correction", () => {
  const r = correctLine(lex, "آب — اسم مذکر: پانہِ۔");
  assert.match(r.text, /—/);
  assert.match(r.text, /:/);
  assert.match(r.text, /۔$/);
});

test("Nastaliq text read without spaces is split back into dictionary words", async () => {
  const { splitRunTogether } = await import(`${root}lib/ks-lexicon.ts`);
  const pieces = splitRunTogether(lex, "مےٚچُھکٲشُرزَبانوپِؠٹھغُرور");
  assert.ok(pieces, "no split");
  for (const w of ["مےٚ", "چُھ", "کٲشُر", "غُرور"]) assert.ok(pieces.includes(w), `missing ${w}: ${pieces.join(" ")}`);
});

test("real words, short or long, are never split", async () => {
  const { splitRunTogether } = await import(`${root}lib/ks-lexicon.ts`);
  for (const w of ["چھَنٛڈ", "کٲشُر", "زمینَس", "انسٹیٹیوٹ", "آبادکٲری", "ایمبولینسہِ"]) assert.equal(splitRunTogether(lex, w), null, w);
});
