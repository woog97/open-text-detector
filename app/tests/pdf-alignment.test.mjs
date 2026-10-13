import { test } from "node:test";
import assert from "node:assert/strict";
import { buildIndex, alignPassages } from "../lib/pdf-alignment.ts";
const segment = (text) => [
  { start: 0, end: Array.from(text).length, score: 0.7, label: "ai_evidence" },
];
test("maps ligatures, whitespace and hyphenation across pages to original characters", () => {
  const source = "Efficient classification preserves the original meaning.";
  const index = buildIndex([
    ["Efﬁcient classi-"],
    ["fication preserves the original meaning."],
  ]);
  const [p] = alignPassages(source, segment(source), index);
  assert.equal(p.mapped, true);
  assert.deepEqual(
    p.pieces.map((x) => x.page),
    [1, 2],
  );
  assert.equal(p.pieces[0].end, 15);
});
test("refuses ambiguous and changed OCR passages", () => {
  const source = "This long passage is repeated in the paper.";
  const [p] = alignPassages(
    source,
    segment(source),
    buildIndex([[source, source]]),
  );
  assert.equal(p.mapped, false);
  assert.equal(
    alignPassages(
      source,
      segment(source),
      buildIndex([["This long passage was changed in the paper."]]),
    )[0].mapped,
    false,
  );
});
test("Python unicode offsets and browser UTF16 offsets stay distinct", () => {
  const source = "😀 This sentence contains sufficient text to match.";
  const [p] = alignPassages(source, segment(source), buildIndex([[source]]));
  assert.equal(p.text, source);
  assert.equal(p.pieces[0].start, 3);
  assert.equal(p.pieces[0].end, source.length - 1);
});
