/** Conservative bridge from old OCR offsets to the native PDF text layer.
 * Only unique, complete normalized passages are accepted. No fuzzy attribution.
 */
export type Passage = {
  start: number;
  end: number;
  score: number;
  label: string;
};
export type TextRun = {
  text: string;
  page: number;
  item: number;
  start: number;
  end: number;
  offsets: number[];
};
export function normalizeText(text: string) {
  let value = "",
    offset = 0;
  const offsets: number[] = [];
  for (const char of text) {
    for (const part of char.normalize("NFKD").toLowerCase()) {
      if (/[\p{L}\p{N}]/u.test(part)) {
        value += part;
        for (let i = 0; i < part.length; i++) offsets.push(offset);
      }
    }
    offset += char.length;
  }
  return { value, offsets };
}
export function buildIndex(pages: string[][]) {
  let text = "";
  const runs: TextRun[] = [];
  pages.forEach((items, page) =>
    items.forEach((original, item) => {
      const { value, offsets } = normalizeText(original);
      runs.push({
        text: original,
        page: page + 1,
        item,
        start: text.length,
        end: text.length + value.length,
        offsets,
      });
      text += value;
    }),
  );
  return { text, runs };
}
export function alignPassages(
  source: string,
  passages: Passage[],
  index: ReturnType<typeof buildIndex>,
) {
  const chars = Array.from(source);
  return passages.map((passage, id) => {
    const text = chars.slice(passage.start, passage.end).join("");
    const needle = normalizeText(text).value;
    const start = needle.length >= 24 ? index.text.indexOf(needle) : -1;
    const unique = start >= 0 && index.text.indexOf(needle, start + 1) === -1;
    const end = start + needle.length;
    const runs = unique
      ? index.runs.filter((run) => run.end > start && run.start < end)
      : [];
    return {
      ...passage,
      id,
      text,
      mapped: runs.length > 0,
      page: runs[0]?.page,
      pieces: runs.map((run) => {
        const a = run.offsets[Math.max(0, start - run.start)];
        const last = run.offsets[Math.min(run.end, end) - run.start - 1];
        const b = last + (run.text.codePointAt(last)! > 0xffff ? 2 : 1);
        return { page: run.page, item: run.item, start: a, end: b };
      }),
    };
  });
}
export type AlignedPassage = ReturnType<typeof alignPassages>[number];
