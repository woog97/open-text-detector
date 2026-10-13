/**
 * Bibliography and appendix sections of extracted paper text, so the reader counts and colors the same sentences as
 * the Discover flagged share. A line-for-line port of research/atlas-bib-exclusion-20261008/bib.py, which produced the
 * published per-paper counts (R2 indexes/flags/current.json); tests/paper-sections.test.mjs checks the two agree.
 *
 * Offsets are code points, like the reader's segment offsets and Python string indices. Regexes run on a copy of the
 * text with one UTF-16 unit per code point: astral characters become a BMP stand-in of the same regex class (decimal
 * digit, other letter or number, or symbol), so Python's Unicode \d, \w and \b give the same results.
 */
export type Section = 'body' | 'bib' | 'app';

const W = String.raw`[\p{L}\p{N}_]`;
const B = String.raw`(?:(?<=${W})(?!${W})|(?<!${W})(?=${W}))`; // Python's Unicode \b
const D = String.raw`\p{Nd}`;
const START = new RegExp(String.raw`(?<![^\n])[ \t]*(?:${D}{1,2}\.?[ \t]+)?(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography|Literature Cited|LITERATURE CITED|Works Cited)[ \t]*:?[ \t]*(?=\n|$)`, 'gu');
const SIGNAL = new RegExp(String.raw`${B}(?:19|20)${D}${D}[a-z]?${B}|arXiv|https?://|${B}URL${B}|${B}doi${B}|${B}pp\.|Proceedings|Conference|Journal|Transactions|Advances in|${B}In [A-Z]|et al\.|preprint|Press|${B}vol\.|${B}Vol\.`, 'u');
const GAP_LINES = 5, GAP_CHARS = 400;
const APPENDIX = new RegExp(String.raw`^[ \t]*(?:[A-H](?:\.${D}{1,2})*\.?|(?:APPENDI(?:X|CES)|Appendi(?:x|ces)|SUPPLEMENTARY MATERIALS?|Supplementary Materials?)${B}[^\n]{0,80}|[A-H](?:\.${D}{1,2})*\.?[ \t]+[A-Z][A-Z0-9 \-:,&()/]{3,90})[ \t]*$`, 'u');
const AUTHORS = /^(?:[^,]{2,40},){3,}/u;
// Python str.isspace characters.
const SPACE = new RegExp('^[' + String.raw`\t\n\v\f\r\x1c-\x1f \x85\xa0` + [0x1680, 0x2000, 0x2d, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000].map(c => String.fromCharCode(c)).join('') + ']*$', 'u');
const blank = (s: string) => SPACE.test(s);

/** The text with each code point as one UTF-16 unit (identity when the text has no astral characters). */
const ARABIC_ZERO = String.fromCharCode(0x660), ORDINAL_A = String.fromCharCode(0xaa), REPLACEMENT = String.fromCharCode(0xfffd);
function flat(text: string) {
  if (!/[\uD800-\uDFFF]/.test(text)) return text;
  let out = '';
  for (const c of text) out += c.length === 1 ? c : /\p{Nd}/u.test(c) ? ARABIC_ZERO : /[\p{L}\p{N}]/u.test(c) ? ORDINAL_A : REPLACEMENT;
  return out;
}

function looksLikeRefs(chunk: string) {
  const lines = chunk.split('\n').slice(0, 20).filter(l => !blank(l));
  return lines.length >= 3 && lines.filter(l => SIGNAL.test(l)).length >= Math.max(3, Math.floor(lines.length / 3));
}

function end(text: string, a: number) {
  let last = a, gapLines = 0, gapChars = 0, pos = a;
  for (const line of text.slice(a).split('\n')) {
    const e = pos + line.length;
    if (APPENDIX.test(line)) return gapLines === 0 || gapChars < GAP_CHARS ? Math.max(last, pos - 1) : last;
    if (SIGNAL.test(line) || AUTHORS.test(line)) { last = e; gapLines = gapChars = 0; }
    else if (!blank(line)) {
      gapLines++; gapChars += line.length;
      if (gapLines >= GAP_LINES && gapChars >= GAP_CHARS) break;
    }
    pos = e + 1;
  }
  return last;
}

/** [start, end) code-point spans of every bibliography, heading included. */
export function bibliographySpans(text: string): [number, number][] {
  const t = flat(text), spans: [number, number][] = [];
  for (const m of t.matchAll(START)) {
    if (spans.length && m.index < spans[spans.length - 1][1]) continue;
    const a = m.index + m[0].length;
    if (looksLikeRefs(t.slice(a, a + 4000))) spans.push([m.index, end(t, a)]);
  }
  return spans;
}

/** Bibliography spans and the appendix tail: everything after the first bibliography (when over 200 characters). */
export function sections(text: string) {
  const spans = bibliographySpans(text), length = flat(text).length;
  const tail: [number, number] | null = spans.length && length - spans[0][1] > 200 ? [spans[0][1], length] : null;
  return { spans, tail };
}

export function sectionOf(start: number, { spans, tail }: ReturnType<typeof sections>): Section {
  if (spans.some(([a, b]) => a <= start && start < b)) return 'bib';
  if (tail && tail[0] <= start && start < tail[1]) return 'app';
  return 'body';
}

/** Section of each segment, by its start offset. */
export function segmentSections(text: string, segments: { start: number }[]): Section[] {
  const s = sections(text);
  return segments.map(p => sectionOf(p.start, s));
}
