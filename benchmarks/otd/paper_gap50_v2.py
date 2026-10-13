"""Controlled second reconstruction run: richer notes, same sources and strict judge."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import re
import shutil

import paper_gap50 as gap
import paper_gap50_fidelity as fidelity

b = gap.base
PRIOR = gap.OUT
gap.OUT = b.ROOT / 'research/data/paper-gap50-luna-v2-20260929'
fidelity.OUT = gap.OUT / 'fidelity-review'
ORIGINAL_PARSE = gap.parse
FIELDS = ['facts', 'qualifications', 'technical_terms', 'citations',
          'claim_citation_bindings', 'discourse_relations', 'rhetorical_profile', 'do_not_infer']
gap.INSTRUCTIONS = dict(gap.INSTRUCTIONS)
gap.INSTRUCTIONS['outline'] = '''Produce a loss-minimizing content specification for a writer who will NEVER see the held-out paragraph. This is exhaustive semantic extraction, not a summary. Use short, compressed notes in different wording; do not provide a paraphrased paragraph or reusable sentences. There is no 3–8 bullet limit. Split distinct claims, examples, questions, contrasts and rhetorical moves into separate facts with unique IDs [C1], [C2], etc. Preserve every clause that affects meaning, including secondary examples, explicit success claims, cross-references, limitations and evaluative framing.
For each fact, record who claims what about which subject, exact scope and conditions, and whether it is a question, assertion, expectation, proposal, suggestion, or reported result. Do not collapse an achieved result into an aim or a question into an unresolved-status assertion. A rhetorical question is NOT evidence that the field has no answer. Distinguish authors' views from attributed views. Preserve logical operators, negation, comparison direction and set membership. Keep quantifiers and force words such as most, many, all, only, often, may, expected, at least, and unlikely at their original strength. Do not replace precise technical relations with looser umbrella terms.
Attach qualifications and citation bindings to the relevant [C#]. Preserve EACH citation marker, author-year reference, footnote marker and figure/section/appendix pointer, with the exact supported claim or entity. A bag of citations alone is insufficient. If attribution is ambiguous, record that ambiguity; do not invent a binding. Record supported contrast, causal, temporal, example and concession relationships separately. Describe rhetorical purpose, critical or tentative stance, emphasis, and question-versus-assertion structure when present. Include relevant emphasis even when it carries no new experimental fact. Do not add outside knowledge or generic caveats. In do_not_infer, specify only tempting inferences that the original does NOT license (if any), without presenting them as facts.
Before returning, check every original sentence and subordinate clause against these notes. Check that no claim, scope word, question, citation binding, pointer or relevant emphasis has been lost or changed. This check must not appear as prose outside the JSON.
Return exactly {"facts":["[C1] compressed complete proposition, its force and attribution"],"qualifications":["[C1] exact scope/strength/conditions to preserve"],"technical_terms":["exact necessary term"],"citations":["exact citation/reference/footnote marker"],"claim_citation_bindings":["[C1] marker -> precise supported claim/entity or cross-reference destination"],"discourse_relations":["[C1] contrasts with [C2] in this specific respect"],"rhetorical_profile":["specific stance, emphasis or rhetorical purpose that must survive"],"do_not_infer":["unsupported conclusion to avoid"]}. Every field is a list of strings; empty lists are allowed except facts. Use 1–30 facts as needed. Avoid copying more than 10 consecutive original words in any note; names, quantifiers and citation markers can be exact. Never include the original paragraph.'''
gap.INSTRUCTIONS['writer'] = '''Write the missing middle paragraph of an AI research paper from a detailed content specification. The original paragraph is unavailable to you. The abstract and neighbors provide context and voice, but only the content notes authorize middle-paragraph claims. Return a single natural academic paragraph.
Faithfulness takes priority over compression or stylistic smoothing. Express EVERY [C#] fact and its qualifications, examples, result/success claims and references. Do not merge claims if that drops a condition, distinction, attribution or rhetorical move. Preserve exact scope and claim strength: most is not many or all; an expectation is not a result; a proposed capability is not demonstrated success; a question is not a claim that its answer is unknown. Retain the source's question-versus-assertion structure. Preserve technical relations precisely, including comparisons, negation, causality, temporal order, set membership and contrasts.
Use claim_citation_bindings to put each exact citation or footnote marker with the claim/entity it supports; never relocate it merely to the nearest fluent sentence or attach it to another claim. Preserve figure, section and appendix pointers and their referents. Preserve rhetorical_profile: criticism, tentativeness, positive result claims, emphasis and argument purpose. Match that profile rather than flattening everything to neutral polished prose. Honor do_not_infer. Do not answer rhetorical questions, add field-status conclusions, new caveats, recommendations, causal explanations or results not licensed by the notes. Do not copy or summarize the neighboring paragraphs or import extra claims from the abstract.
Draft in your own wording and organize the content coherently. Technical terms, citation markers and meaning-bearing qualifiers may remain exact. The broad word-count range is only a guide; exceed it if necessary for complete coverage. Before returning, quietly verify coverage of every [C#], qualification, citation binding, reference pointer, discourse relation and relevant tone cue. Remove any unsupported addition and restore any omitted content. Output only {"paragraph":"the complete new middle paragraph"}, with no heading, bullets, preamble, explanation or checklist.'''

def parse(stage, content, p):
    if stage != 'outline':
        return ORIGINAL_PARSE(stage, content, p)
    x = json.loads(content)
    assert set(x) == set(FIELDS), 'Use exactly the eight specified fields'
    assert all(isinstance(v, list) and all(isinstance(s, str) and s.strip() for s in v) for v in x.values()), 'Every field must be a list of nonempty strings'
    assert 1 <= len(x['facts']) <= 30, 'Use 1–30 atomic facts'
    ids = [re.match(r'^\[C(\d+)\]', s) for s in x['facts']]
    assert all(ids) and len({m[1] for m in ids}) == len(ids), 'Start each fact with a unique [C#]'
    for field in FIELDS:
        for note in x[field]:
            assert gap.longest_copy(p['held_out'], note)['words'] <= 10, 'A note copies more than 10 consecutive original words; use compressed notes in different wording'
    assert p['held_out'] not in json.dumps(x), 'Never include the original paragraph'
    return x

gap.parse = parse

async def prepare():
    assert not gap.OUT.exists(), 'Keep completed runs immutable'
    gap.OUT.mkdir(parents=True)
    for name in ['papers.jsonl', 'passages.jsonl']:
        shutil.copyfile(PRIOR / name, gap.OUT / name)
    await gap.prepare()
    mf = json.loads((gap.OUT / 'manifest.json').read_text())
    mf.update(experiment_version=2, prior_run=str(PRIOR),
              changes=['Exhaustive atomic notes instead of 3–8-bullet summary',
                       'Claim-linked citations, qualifications, discourse and rhetorical profile',
                       'Writer explicitly verifies every note and avoids unlicensed inferences'],
              comparison_design='Same frozen 50 originals and contexts; all first mechanically valid generations evaluated; unchanged strict fidelity prompt and judge settings; no semantic retries or selection')
    mf['settings']['max_tokens'] = 5000
    b.save(gap.OUT / 'manifest.json', mf)
    for module, name in [(gap, 'generation-runner.snapshot.py'), (fidelity, 'fidelity-runner.snapshot.py')]:
        shutil.copyfile(module.__file__, gap.OUT / name)
    (gap.OUT / 'runner.snapshot.py').write_bytes(Path(__file__).read_bytes())
    assert b.sha((PRIOR/'passages.jsonl').read_bytes()) == mf['passages_sha256']

def report():
    fidelity.report()
    old = {r['passage_id']: r['output'] for r in b.readl(PRIOR/'fidelity-review/responses.jsonl')}
    new = {r['passage_id']: r['output'] for r in fidelity.rows('responses.jsonl')}
    assert set(old) == set(new) and len(new) == 50
    prior_plan = json.loads((PRIOR/'fidelity-review/manifest.json').read_text())
    plan = json.loads((fidelity.OUT/'manifest.json').read_text())
    assert prior_plan['prompt'] == plan['prompt'] and prior_plan['settings'] == plan['settings']
    summary = json.loads((fidelity.OUT/'summary.json').read_text())
    # This inherited label describes the original run's 17-example screen only.
    summary['current_automated_screen_verdicts'] = summary.pop('previously_screened_17_verdicts')
    b.save(fidelity.OUT/'summary.json', summary)
    ranks = {'materially_unfaithful': 0, 'mostly_faithful_with_minor_differences': 1, 'fully_faithful': 2}
    pairs = [{'passage_id': k, 'old': old[k]['verdict'], 'new': new[k]['verdict'],
              'transition': 'uncertain' if 'uncertain' in [old[k]['verdict'],new[k]['verdict']] else
              'improved' if ranks[new[k]['verdict']] > ranks[old[k]['verdict']] else
              'worsened' if ranks[new[k]['verdict']] < ranks[old[k]['verdict']] else 'unchanged'} for k in old]
    generation_cost = json.loads((gap.OUT/'costs.json').read_text())
    total = {k: generation_cost[k] + summary['costs'][k] for k in ['input_tokens','output_tokens','reasoning_tokens','cost_usd']}
    total['calls'] = generation_cost['attempts'] + summary['costs']['calls']
    if (gap.OUT/'key-total-after.json').exists():
        total['account_usage_delta_usd'] = json.loads((gap.OUT/'key-total-after.json').read_text())['usage'] - json.loads((gap.OUT/'key-usage-before.json').read_text())['usage']
        total['account_matches_ledger'] = abs(total['account_usage_delta_usd'] - total['cost_usd']) < 1e-8
    result = {'papers':50,'old_verdicts':dict(Counter(r['verdict'] for r in old.values())),
              'new_verdicts':summary['verdicts'],'paired_changes':dict(Counter(p['transition'] for p in pairs)),
              'dimension_changed_counts':{d:{'old':sum(r['dimensions'][d]=='changed' for r in old.values()),'new':sum(r['dimensions'][d]=='changed' for r in new.values())} for d in fidelity.DIMS},
              'pairs':pairs, 'costs':{'generation_and_audit':generation_cost,'strict_fidelity_review':summary['costs'],'total':total},
              'limitations':['Same-model judgments; no independent expert gold labels', 'Same 50 development examples informed prompt changes; improvement is not evidence of held-out generalization', 'Single generation and judgment per example apart from format retries; sampling variance is unmeasured'],
              'strict_judge_prompt_unchanged':True}
    b.save(gap.OUT/'fidelity-comparison.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','outline','writer','audit','export','validate','review','report','settle','settle-total'])
    args=parser.parse_args()
    if args.action=='prepare':asyncio.run(prepare())
    elif args.action in ['outline','writer','audit']:asyncio.run(gap.run_stage(args.action))
    elif args.action=='review':asyncio.run(fidelity.run())
    elif args.action=='settle':asyncio.run(gap.snapshot('key-usage-after.json'))
    elif args.action=='settle-total':asyncio.run(gap.snapshot('key-total-after.json'))
    elif args.action=='report':report()
    else:getattr(gap,args.action)()
