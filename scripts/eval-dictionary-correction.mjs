#!/usr/bin/env node
/**
 * Benchmark: does dictionary post-correction improve OCR?
 *
 *   node scripts/eval-dictionary-correction.mjs [--pages 6] [--holdout 0.2] [--seed 7]
 *
 * Needs Playwright with Chromium (to render ground-truth pages in real Kashmiri fonts)
 * and network access for Google Fonts and the Tesseract model. It:
 *   1. builds pages of known Kashmiri text (app samples + random attested dictionary words),
 *   2. renders each page in Naskh and in Nastaliq,
 *   3. OCRs them with the same Tesseract.js Urdu model the app uses,
 *   4. scores word accuracy and character error rate before and after correction,
 *   5. repeats with a share of the true words removed from the lexicon ("holdout"),
 *      which measures over-correction of words the dictionary does not know.
 */
import { createRequire } from "node:module";
import { readFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const { KashmiriLexicon, correctLine, normalizeKs, bareKey } = await import(join(ROOT, "lib/ks-lexicon.ts"));
const { createWorker } = require("tesseract.js");
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  console.error("This benchmark needs Playwright: npm i -D playwright && npx playwright install chromium");
  process.exit(1);
}

const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`);
  return i > 0 ? Number(process.argv[i + 1]) : d;
};
const VERBOSE = process.argv.includes("--verbose");
const PAGES = arg("pages", 6), HOLDOUT = arg("holdout", 0.2);
let seed = arg("seed", 7);
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31);

const rows = JSON.parse(readFileSync(join(ROOT, "public/lexicon/ks-lexicon.json"), "utf8")).forms;

// ---- ground truth -------------------------------------------------------------
const FIXED = [
  ["کٲشُر لُغَت — جِلد گۄڈٕنِیُک", "آب — اسم مذکر: رۄپہِ ہِو شفاف پانہِ", "آزٲدی — اسم مؤنث: خۄشحٲلی تہٕ چُھٹکارٕ", "بَرگ — پَن: کُلِک زرد یا سبز حِصہٕ"],
  ["مےٚ چُھ کٲشُر زَبانہِ پؠٹھ غُرور", "صدیَن پؠٹھ چھےٚ یۄس بَہان پُر نُور", "لَل دؠد تہٕ نُند رؠشؠن کٔر شَنٛگار", "مَحجوٗر تہٕ حَبہٕ خاتوٗنَن دِیُت پیار"],
  ["کٲشِر زَبانہِ ہٕنٛز تَوٲریخ چھےٚ واریاہ پرٛٲنؠ۔", "اَتھ مَنٛز چھےٚ شیٚنٛٹھ، شیو مَت تہٕ صُوفی شٲعِری ہٕنٛز خوٗشبوٗ۔"],
];
const pool = rows.filter((r) => r[1] >= 3 && Array.from(r[0]).length >= 3 && !/\s/.test(r[0])).map((r) => r[0]);
const pages = FIXED.slice();
while (pages.length < PAGES) {
  pages.push(Array.from({ length: 4 }, () => Array.from({ length: 5 }, () => pool[Math.floor(rand() * pool.length)]).join(" ")));
}
const gtWords = [...new Set(pages.flat().join(" ").split(/\s+/).map((w) => normalizeKs(w.replace(/[—:،۔]/g, ""))).filter(Boolean))];

// ---- metrics ------------------------------------------------------------------
function lev(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[b.length];
}
const clean = (s) => normalizeKs(s.replace(/[—:،۔\-]/g, " ")).replace(/\s+/g, " ").trim();
function score(gt, hyp) {
  const g = clean(gt), h = clean(hyp);
  const gw = g.split(" "), hw = h.split(" ");
  return {
    wordErr: lev(gw, hw) / gw.length, // word error rate, exact spelling incl. vowel signs
    charErr: lev(Array.from(g), Array.from(h)) / Array.from(g).length, // CER incl. vowel signs
    bareErr: lev(Array.from(bareKey(g)), Array.from(bareKey(h))) / Array.from(bareKey(g)).length, // CER on letters only
  };
}

// ---- render + OCR ---------------------------------------------------------------
const FONTS = { Naskh: "'Noto Naskh Arabic'", Nastaliq: "'Noto Nastaliq Urdu'" };
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
const worker = await createWorker("urd", 1, {
  langPath: "https://cdn.jsdelivr.net/npm/@tesseract.js-data/urd/4.0.0_best_int",
  cachePath: join(ROOT, "node_modules/.cache/tesseract"),
});
mkdirSync(join(ROOT, "node_modules/.cache/tesseract"), { recursive: true });

const ocrRuns = [];
for (const [fontName, family] of Object.entries(FONTS)) {
  for (const lines of pages) {
    const lh = fontName === "Nastaliq" ? 2.6 : 1.9;
    await page.setContent(`<html><head><link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&family=Noto+Naskh+Arabic&display=swap" rel="stylesheet"></head>
      <body style="margin:0;background:#fff"><div id=p dir=rtl style="font-family:${family};font-size:40px;line-height:${lh};padding:30px 50px;width:1100px">${lines.join("<br>")}</div></body></html>`);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(300);
    const png = await (await page.$("#p")).screenshot();
    const { data } = await worker.recognize(png, {}, { blocks: true });
    const ocrLines = [];
    for (const b of data.blocks ?? []) for (const p of b.paragraphs) for (const l of p.lines) {
      if (l.text.trim()) ocrLines.push({ text: l.text.trim(), confs: l.words.map((w) => w.confidence) });
    }
    ocrRuns.push({ fontName, gt: lines.join(" "), ocrLines });
  }
}
await worker.terminate();
await browser.close();

// ---- evaluate ------------------------------------------------------------------
function evaluate(lex, label) {
  if (VERBOSE) console.log(`\n--- changes: ${label}`);
  const out = {};
  for (const fontName of Object.keys(FONTS)) {
    const runs = ocrRuns.filter((r) => r.fontName === fontName);
    const acc = { before: { wordErr: 0, charErr: 0, bareErr: 0 }, after: { wordErr: 0, charErr: 0, bareErr: 0 } };
    let changed = 0, harmful = 0, helpful = 0;
    for (const r of runs) {
      const before = r.ocrLines.map((l) => l.text).join(" ");
      const after = r.ocrLines.map((l) => correctLine(lex, l.text, l.confs).text).join(" ");
      const sb = score(r.gt, before), sa = score(r.gt, after);
      for (const k of Object.keys(sb)) { acc.before[k] += sb[k] / runs.length; acc.after[k] += sa[k] / runs.length; }
      // word-level change audit: did each change move the word towards the truth?
      const gtSet = new Set(clean(r.gt).split(" "));
      for (const l of r.ocrLines) for (const w of correctLine(lex, l.text, l.confs).words) {
        if (w.text === w.original) continue;
        changed++;
        const wasRight = gtSet.has(normalizeKs(w.original)), nowRight = gtSet.has(normalizeKs(w.text));
        if (nowRight && !wasRight) helpful++;
        if (wasRight && !nowRight) harmful++;
        if (VERBOSE) console.log(`  [${fontName}] ${w.status.padEnd(9)} ${w.original} -> ${w.text}  ${wasRight ? "WAS-RIGHT" : ""}${nowRight ? "NOW-RIGHT" : ""}  conf=${Math.round(w.ocrConf ?? -1)} score=${w.candidates[0]?.score.toFixed(2)}`);
      }
    }
    out[fontName] = { ...acc, changed, helpful, harmful };
  }
  console.log(`\n=== ${label} (lexicon ${lex.size.toLocaleString()} forms) ===`);
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  for (const [f, r] of Object.entries(out)) {
    console.log(`${f.padEnd(9)} word errors ${pct(r.before.wordErr)} -> ${pct(r.after.wordErr)} | char errors ${pct(r.before.charErr)} -> ${pct(r.after.charErr)} | letter-only char errors ${pct(r.before.bareErr)} -> ${pct(r.after.bareErr)}`);
    console.log(`          words changed ${r.changed}: ${r.helpful} now exactly right, ${r.harmful} made wrong (were right before)`);
  }
  return out;
}

console.log(`Benchmark: ${pages.length} pages x ${Object.keys(FONTS).length} fonts, ${gtWords.length} distinct true words`);
evaluate(new KashmiriLexicon(rows), "Full dictionary");
const hidden = new Set(gtWords.filter(() => rand() < HOLDOUT));
evaluate(new KashmiriLexicon(rows, hidden), `Hold-out: ${hidden.size} of the true words removed from the dictionary`);
