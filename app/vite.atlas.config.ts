import { defineConfig } from "vite";
import {copyFileSync, mkdirSync, readFileSync} from "node:fs";
import react from "@vitejs/plugin-react";
import {pdfjsCompatWorker} from "./build/pdfjs-compat-plugin.ts";
import { fileURLToPath } from "node:url";
export default defineConfig({
  root: fileURLToPath(new URL("./atlas-cloud", import.meta.url)),
  publicDir: fileURLToPath(new URL("./.sites-runtime/atlas/assets", import.meta.url)),
  plugins: [react(),{name:"atlas-dev-transparency",configureServer(server){
    // Dev only: the production Worker serves this file from R2. Copy it to .sites-runtime/transparency/ (git-ignored) to preview.
    server.middlewares.use("/backend/v1/transparency/ai-examples",(_req,res)=>{
      try{const body=readFileSync(new URL("./.sites-runtime/transparency/ai-examples.json",import.meta.url));res.setHeader("Content-Type","application/json");res.end(body);}
      catch{res.statusCode=404;res.end('{"detail":"Not found"}');}
    });}},{name:"atlas-security-headers",generateBundle(){this.emitFile({type:"asset",fileName:"_headers",source:readFileSync(new URL("./atlas-cloud/_headers",import.meta.url),"utf8")});}}, {name:"atlas-favicon",buildStart(){
    // The site icon lives in public/; refresh the copy in publicDir on every build so an old icon is never shipped.
    mkdirSync(new URL("./.sites-runtime/atlas/assets/",import.meta.url),{recursive:true});
    copyFileSync(new URL("./public/favicon.svg",import.meta.url),new URL("./.sites-runtime/atlas/assets/favicon.svg",import.meta.url));}}, pdfjsCompatWorker()],
  resolve: { alias: { "@": fileURLToPath(new URL("./", import.meta.url)) } },
  build: {outDir: "../dist-atlas", emptyOutDir: true, minify: true},
});
