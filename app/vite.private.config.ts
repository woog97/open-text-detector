import { defineConfig } from "vite";
import {readFileSync} from "node:fs";
import react from "@vitejs/plugin-react";
import {pdfjsCompatWorker} from "./build/pdfjs-compat-plugin.ts";
import { fileURLToPath } from "node:url";
export default defineConfig({
  root: fileURLToPath(new URL("./atlas-local", import.meta.url)),
  publicDir: fileURLToPath(new URL("./.sites-runtime/atlas/assets", import.meta.url)),
  plugins: [react(),{name:"atlas-security-headers",generateBundle(){this.emitFile({type:"asset",fileName:"_headers",source:readFileSync(new URL("./atlas-local/_headers",import.meta.url),"utf8")});}},pdfjsCompatWorker()],
  resolve: { alias: { "@": fileURLToPath(new URL("./", import.meta.url)) } },
  build: {outDir: "../dist-atlas-local", emptyOutDir: true, minify: true},
});
