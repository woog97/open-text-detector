// PDF.js 5 calls built-ins that only recent browsers have (Chrome 140+, Safari 18.2+, Firefox 133+):
// Uint8Array toHex/toBase64/fromBase64 and Promise.try. Without them the reader fails with
// "a.toHex is not a function". Loaded on the page and, via pdf.worker.compat.mjs, inside the PDF.js worker.
const U8 = Uint8Array;
if (!U8.prototype.toHex) {
  Object.defineProperty(U8.prototype, "toHex", {configurable: true, writable: true, value() {
    let out = ""; for (let i = 0; i < this.length; i++) out += this[i].toString(16).padStart(2, "0"); return out;
  }});
}
if (!U8.prototype.toBase64) {
  Object.defineProperty(U8.prototype, "toBase64", {configurable: true, writable: true, value() {
    let s = ""; for (let i = 0; i < this.length; i += 0x8000) s += String.fromCharCode.apply(null, this.subarray(i, i + 0x8000)); return btoa(s);
  }});
}
if (!U8.fromBase64) {
  Object.defineProperty(U8, "fromBase64", {configurable: true, writable: true, value(text) {
    const s = atob(String(text).replace(/[\s]/g, "")), out = new U8(s.length); for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i); return out;
  }});
}
if (!Promise.try) {
  Object.defineProperty(Promise, "try", {configurable: true, writable: true, value(fn, ...args) {
    return new Promise(resolve => resolve(fn(...args)));
  }});
}
if (!Promise.withResolvers) {
  Object.defineProperty(Promise, "withResolvers", {configurable: true, writable: true, value() {
    let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject};
  }});
}
