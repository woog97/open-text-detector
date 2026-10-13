import {readFileSync} from "node:fs";
import type {Plugin} from "vite";
/** Emits the PDF.js worker entry behind built-in polyfills (lib/pdfjs-compat.js) next to the copied pdf.worker.min.mjs. */
export function pdfjsCompatWorker():Plugin{
  return {name:"pdfjs-compat-worker",generateBundle(){
    this.emitFile({type:"asset",fileName:"pdfjs/compat.mjs",source:readFileSync(new URL("../lib/pdfjs-compat.js",import.meta.url),"utf8")});
    this.emitFile({type:"asset",fileName:"pdfjs/pdf.worker.compat.mjs",source:'import "./compat.mjs";\nimport "./pdf.worker.min.mjs";\n'});
  }};
}
