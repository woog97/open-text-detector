// Preview this checkout's public website without deploying (`npm run preview:atlas`): builds dist-atlas, serves it on
// http://localhost:4789 and forwards /backend/* (catalogue, browse, results, PDFs) read-only to the live site.
// Works in a fresh worktree: it fills the build's static-asset folder (pdf.js files + favicon, as
// scripts/prepare_cloud_atlas.py does) when that folder is missing. Usage: node scripts/preview-atlas.mjs [port]
import http from "node:http";
import {readFile} from "node:fs/promises";
import {cpSync, existsSync, mkdirSync} from "node:fs";
import {execSync} from "node:child_process";

const app = new URL("../", import.meta.url).pathname, assets = `${app}.sites-runtime/atlas/assets`;
if (!existsSync(`${app}public/pdfjs/pdf.worker.min.mjs`)) execSync("node scripts/sync-pdfjs.mjs", {cwd: app, stdio: "inherit"});
if (!existsSync(`${assets}/pdfjs/pdf.worker.min.mjs`)) {
  mkdirSync(assets, {recursive: true});
  cpSync(`${app}public/pdfjs`, `${assets}/pdfjs`, {recursive: true});
  cpSync(`${app}public/favicon.svg`, `${assets}/favicon.svg`);
}
execSync("npm run build:atlas", {cwd: app, stdio: "inherit"});

const root = new URL("../dist-atlas/", import.meta.url).pathname;
const live = process.env.ATLAS_LIVE || "https://open-text-detector.woog09.workers.dev"; // ATLAS_LIVE: preview against another deployment
const port = Number(process.argv[2] || 4789);
const types = {js: "text/javascript", mjs: "text/javascript", css: "text/css", html: "text/html", svg: "image/svg+xml", json: "application/json", woff2: "font/woff2", png: "image/png", wasm: "application/wasm", bcmap: "application/octet-stream"};

http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  try {
    if (url.pathname.startsWith("/backend/")) {
      if (!["GET", "HEAD"].includes(req.method)) {res.writeHead(405); res.end(); return;}
      const headers = {};
      for (const h of ["range", "if-none-match", "if-range"]) if (req.headers[h]) headers[h] = req.headers[h];
      const upstream = await fetch(live + url.pathname + url.search, {method: req.method, headers});
      const out = {};
      for (const h of ["content-type", "content-range", "accept-ranges", "etag", "cache-control"]) {const v = upstream.headers.get(h); if (v) out[h] = v;}
      // PDF.js only uses byte ranges when it knows the file length; fetch() has already decompressed the body.
      const body = req.method === "HEAD" ? undefined : Buffer.from(await upstream.arrayBuffer());
      if (body) out["content-length"] = body.length; else if (!upstream.headers.get("content-encoding") && upstream.headers.get("content-length")) out["content-length"] = upstream.headers.get("content-length");
      res.writeHead(upstream.status, out);
      res.end(body);
      return;
    }
    // Single-page app: unknown paths without a file extension get index.html, like the Worker's asset config.
    const path = url.pathname === "/" || !/\.[a-z0-9]+$/i.test(url.pathname) ? "/index.html" : decodeURIComponent(url.pathname);
    if (path.includes("..")) {res.writeHead(400); res.end(); return;}
    const body = await readFile(root + path.slice(1));
    res.writeHead(200, {"content-type": types[path.split(".").pop()] || "application/octet-stream"});
    res.end(body);
  } catch {
    res.writeHead(404); res.end("Not found");
  }
}).listen(port, "127.0.0.1", () => console.log(`Preview: http://localhost:${port}  (data from ${live})`));
