/** Shape check for the Oct 10 Calibration page JSONs: every key and value type must match the Oct 7 file of the same
 * counting mode (the `Calibration` type in app/components/model-calibration.tsx), except the dropped `candidates`;
 * sentence curves have no other_model; YearRow fields typed as in the type (over_human_p99 / document_score_median
 * number or null). Usage: node check_calibration_shape.mjs JSON_DIR */
import fs from "node:fs";
const dir = process.argv[2], modes = {all: "", nobib: "-nobib", noapp: "-noapp", body: "-body"};
const t = v => v === null ? "null" : Array.isArray(v) ? "array" : typeof v;
let bad = 0;
function cmp(a, b, path) {
  if (t(a) !== t(b) && !(path.match(/(over_human_p99|document_score_median)$/) && ["number", "null"].includes(t(a)) && ["number", "null"].includes(t(b)))) { console.log("type", path, t(a), t(b)); bad++; return; }
  if (t(a) === "object") {
    const ka = Object.keys(a).filter(k => k !== "candidates" && k !== "other_model"), kb = Object.keys(b);
    for (const k of new Set([...ka, ...kb])) { if (!(k in a) || !(k in b)) { console.log("key", path + "." + k, k in a ? "only Oct 7" : "only Oct 10"); bad++; } else cmp(a[k], b[k], path + "." + k); }
  } else if (t(a) === "array" && a.length && b.length) {
    if (t(a[0]) === "object") for (const x of b) cmp(a[0], x, path + "[]"); else for (const x of b) if (t(x) !== t(a[0])) { console.log("elem", path, t(x)); bad++; break; }
  }
}
for (const [m, s] of Object.entries(modes)) {
  const o = JSON.parse(fs.readFileSync(`${dir}/qwen36-35b-a3b-final${s}.json`)), n = JSON.parse(fs.readFileSync(`${dir}/qwen36-35b-a3b-oct10-${m}.json`));
  cmp(o, n, m);
  if (n.sentence_curves.some(c => "other_model" in c || / new final-moe/.test(c.name))) { console.log(m, "other_model or old name left"); bad++; }
  for (const c of [...n.sentence_curves, ...n.document_curves]) if (c.fpr.length !== c.tpr.length) { console.log(m, c.name, "fpr/tpr length"); bad++; }
  if (n.counting.mode !== m) { console.log(m, "counting.mode", n.counting.mode); bad++; }
  console.log(m, "checked:", n.sentence_curves.map(c => c.name).join(", "), "|", n.years.length, "year rows,", n.years_p1.length, "years_p1 rows");
}
console.log(bad ? `FAIL ${bad}` : "shape OK"); process.exit(bad ? 1 : 0);
