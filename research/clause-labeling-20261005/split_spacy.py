"""Classical clause-splitting baseline: spaCy dependency parse.

A clause starts at the leftmost token of any verb subtree attached as conj, ccomp,
advcl, relcl, csubj or parataxis that has its own subject (the report's definition:
"a group of words that contains a subject and a verb"). For conj, a directly
preceding coordinator (and/but/or) is included in the
new clause. Embedded clauses also end a clause at their last token, and verbless
pieces merge into a neighbour within the same sentence. Sentence starts are always cuts. Cuts snap back to the start of the whitespace-delimited word, so they
use the same word-start convention as the annotator.

Usage (needs spaCy + en_core_web_trf or en_core_web_sm):
  split_spacy.py IN.jsonl OUT.jsonl [--model en_core_web_sm]
IN rows need gold_id or item_id, and text. OUT rows: {id, cuts}.
"""
import argparse, json

CLAUSE_DEPS = {'conj', 'ccomp', 'advcl', 'relcl', 'csubj', 'parataxis', 'acl:relcl'}
SUBJ = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'expl'}


def next_word_start(text, i):
    while i < len(text) and not text[i].isspace():
        i += 1
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def word_start(text, i):
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return i


def cuts_for(doc, text):
    cuts = set()
    for tok in doc:
        if tok.dep_ not in CLAUSE_DEPS or tok.pos_ not in ('VERB', 'AUX'):
            continue
        if not any(c.dep_ in SUBJ for c in tok.children):
            continue
        left = tok.left_edge
        if tok.dep_ == 'conj':
            prev = doc[left.i - 1] if left.i > 0 else None
            if prev is not None and prev.dep_ == 'cc':
                left = prev
        cuts.add(word_start(text, left.idx))
        # A clause embedded before the end closes where its subtree ends.
        right = tok.right_edge
        cuts.add(next_word_start(text, right.idx + len(right.text)))
    # Sentence starts are always cuts, and verbless pieces never merge across them.
    fixed = {word_start(text, sent.start_char) for sent in doc.sents} - {0}
    cuts = sorted(c for c in cuts | fixed if 0 < c < len(text))
    # Merge pieces without any verb into a neighbour (left piece absorbs, else the next one).
    def has_verb(a, b):
        return any(t.pos_ in ('VERB', 'AUX') for t in doc if a <= t.idx < b)
    changed = True
    while changed and cuts:
        changed = False
        bounds = [0] + cuts + [len(text)]
        for j in range(len(bounds) - 1):
            if not has_verb(bounds[j], bounds[j + 1]):
                left_ok = j > 0 and bounds[j] not in fixed
                right_ok = j + 1 < len(bounds) - 1 and bounds[j + 1] not in fixed
                if left_ok or right_ok:
                    cuts.remove(bounds[j] if left_ok else bounds[j + 1])
                    changed = True
                    break
    return cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--model', default='en_core_web_sm')
    a = ap.parse_args()
    import spacy
    nlp = spacy.load(a.model)
    rows = [json.loads(l) for l in open(a.inp)]
    with open(a.out, 'w') as f:
        for r, doc in zip(rows, nlp.pipe(r['text'] for r in rows)):
            f.write(json.dumps({'id': r.get('gold_id') or r.get('item_id'), 'cuts': cuts_for(doc, r['text'])}) + '\n')
    print(len(rows), 'rows ->', a.out)


if __name__ == '__main__':
    main()
