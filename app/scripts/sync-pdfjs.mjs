// Serve a matching local worker and fonts; no third-party CDN is needed.
import { cpSync, mkdirSync, writeFileSync } from "node:fs";
const source = new URL("../node_modules/pdfjs-dist/", import.meta.url);
const target = new URL("../public/pdfjs/", import.meta.url);
mkdirSync(target, { recursive: true });
cpSync(
  new URL("build/pdf.worker.min.mjs", source),
  new URL("pdf.worker.min.mjs", target),
);
for (const folder of ["cmaps", "standard_fonts", "wasm"])
  cpSync(new URL(folder, source), new URL(folder, target), { recursive: true });
// Polyfilled worker entry (lib/pdfjs-compat.js) for browsers older than late 2025.
cpSync(new URL("../lib/pdfjs-compat.js", import.meta.url), new URL("compat.mjs", target));
writeFileSync(new URL("pdf.worker.compat.mjs", target), 'import "./compat.mjs";\nimport "./pdf.worker.min.mjs";\n');
