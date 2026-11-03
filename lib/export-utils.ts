/**
 * Document & Text Export Utilities for KoshurOCR Studio
 * Supports .TXT, .JSON, .DOCX (RTL-formatted Word document), and printable Searchable PDF
 */

export interface ExportLine {
  id: number;
  text: string;
  roman?: string;
  confidence?: number;
  poly?: number[][];
}

export function exportToTxt(lines: ExportLine[], filename: string = "kashmiri_ocr_export.txt"): void {
  const content = lines.map((l) => l.text).join("\n");
  downloadBlob(new Blob([content], { type: "text/plain;charset=utf-8" }), filename);
}

export function exportToJson(
  lines: ExportLine[],
  metadata: Record<string, unknown> = {},
  filename: string = "kashmiri_ocr_export.json"
): void {
  const data = {
    metadata: {
      generator: "KoshurOCR Studio",
      language: "Kashmiri (ks)",
      script: "Nastaliq Perso-Arabic",
      timestamp: new Date().toISOString(),
      ...metadata,
    },
    lines,
  };
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), filename);
}

export function exportToDocx(lines: ExportLine[], filename: string = "kashmiri_ocr_export.doc"): void {
  // Rich HTML-based Word format with RTL Nastaliq font styling
  const linesHtml = lines
    .map(
      (l) => `
    <p dir="rtl" style="font-family: 'Noto Nastaliq Urdu', 'Gulmarg Nastaleeq', serif; font-size: 16pt; line-height: 2.4; text-align: right; margin: 8pt 0;">
      ${escapeHtml(l.text)}
    </p>
    ${
      l.roman
        ? `<p dir="ltr" style="font-family: Arial, sans-serif; font-size: 10pt; color: #64748b; font-style: italic; margin-bottom: 12pt;">${escapeHtml(l.roman)}</p>`
        : ""
    }
  `
    )
    .join("\n");

  const docHtml = `
    <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
    <head>
      <meta charset='utf-8'>
      <title>KoshurOCR Document</title>
      <style>
        body { font-family: 'Noto Nastaliq Urdu', serif; padding: 20pt; }
      </style>
    </head>
    <body dir="rtl">
      <h2 dir="rtl" style="text-align: right; font-family: 'Noto Nastaliq Urdu', serif;">کٲشُر او سی آر متن (KoshurOCR Document)</h2>
      <hr/>
      ${linesHtml}
    </body>
    </html>
  `;

  downloadBlob(new Blob([docHtml], { type: "application/msword" }), filename);
}

export function exportToPdf(lines: ExportLine[]): void {
  // Open print dialog with RTL styles
  const printWindow = window.open("", "_blank");
  if (!printWindow) {
    alert("Your browser blocked the print window. Allow pop-ups for this site to export a PDF.");
    return;
  }

  const contentHtml = lines
    .map(
      (l) => `
    <div style="margin-bottom: 16px;">
      <p dir="rtl" style="font-family: 'Noto Nastaliq Urdu', serif; font-size: 18px; line-height: 2.2; text-align: right; margin: 0;">
        ${escapeHtml(l.text)}
      </p>
      ${l.roman ? `<p style="font-family: sans-serif; font-size: 11px; color: #666; font-style: italic; margin: 4px 0 0 0;">${escapeHtml(l.roman)}</p>` : ""}
    </div>
  `
    )
    .join("");

  printWindow.document.write(`
    <!DOCTYPE html>
    <html lang="ks" dir="rtl">
    <head>
      <title>KoshurOCR Export</title>
      <link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&display=swap" rel="stylesheet">
      <style>
        body { font-family: 'Noto Nastaliq Urdu', serif; padding: 40px; color: #111; }
        @media print { body { padding: 0; } }
      </style>
    </head>
    <body>
      <h2 style="text-align: right; border-bottom: 2px solid #ccc; padding-bottom: 10px;">کٲشُر متن — KoshurOCR Export</h2>
      ${contentHtml}
      <script>
        window.onload = function() { window.print(); }
      </script>
    </body>
    </html>
  `);
  printWindow.document.close();
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
