"""Expand the fixed 50-paper roster to five distinct reconstruction targets each."""
import argparse
import asyncio
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import re
import shutil
import unicodedata

import paper_gap50_v2 as v2
import paper_gap250_sources as sources

gap, fidelity, b = v2.gap, v2.fidelity, v2.b
PRIOR = gap.OUT
CACHE = v2.PRIOR
OUT = b.ROOT/'research/data/paper-gap250-luna-v2-20260929'
gap.OUT = OUT
fidelity.OUT = OUT/'fidelity-review'

def detailed_parse(stage, content, p):
    try:return v2.parse(stage,content,p)
    except AssertionError as e:
        if stage!='outline' or 'copies more than 10' not in str(e):raise
        x=json.loads(content);failures=[]
        for field,notes in x.items():
            for i,note in enumerate(notes):
                overlap=gap.longest_copy(p['held_out'],note)
                if overlap['words']>10:failures.append(f'{field}[{i}] copies {overlap["words"]} words: {overlap["phrase"]}')
        raise AssertionError('; '.join(failures)+'. Split long lists into separate notes: put each author-year reference, named problem family, or mathematical tuple in its own fact or binding; do not repeat the combined list in another field. Link the separate facts by their IDs to preserve their relationship. Rephrase copied prose. Preserve all references, names, qualifiers and technical meaning.') from e

gap.parse=detailed_parse

def select():
    assert not (OUT/'manifest.json').exists(), 'Selection is frozen once calls are prepared'
    OUT.mkdir(parents=True, exist_ok=True)
    papers=b.readl(PRIOR/'papers.jsonl')
    existing={p['paper_id']:p for p in b.readl(PRIOR/'passages.jsonl')}
    selected=[];inventory=[]
    for paper in papers:
        gap.OUT=CACHE
        try: paras=sources.paragraphs(paper)
        finally: gap.OUT=OUT
        old=existing[paper['paper_id']]
        candidates=[]
        for i,target in enumerate(paras):
            if i==0 or i==len(paras)-1 or not sources.target(target):continue
            before,after=paras[i-1],paras[i+1]
            if not all(20<=len(p['text'].split())<=600 for p in [before,after]):continue
            if after['page']-before['page']>2:continue
            if target['text']==old['held_out']:continue
            candidates.append((before,target,after))
        chosen=[old]
        for ordinal in range(2,6):
            def score(triple):
                a,t,c=triple;txt=t['text']
                symbols=sum(unicodedata.category(x) in {'Sm','So'} or 'GREEK' in unicodedata.name(x,'') for x in txt)
                context_issues=sum(not re.match(r'[A-Z“]',p['text']) or not re.search(r'[.!?]$',p['text']) for p in [a,c])
                context_math=sum(not gap.clean(p) for p in [a,c])
                distance=min(abs(t['page']-p['page']) for p in chosen)
                same_section=any(t['section']==p['extracted_section_hint'] for p in chosen)
                return (symbols,context_issues,context_math,not gap.clean(t,True),same_section,-min(distance,3),abs(len(txt.split())-120),b.sha(txt))
            available=[x for x in candidates if all(x[1]['text']!=p['held_out'] for p in chosen)]
            assert available, f'Insufficient distinct targets for {paper["title"]}'
            a,t,c=min(available,key=score)
            text=a['text']+'\n\n'+t['text']+'\n\n'+c['text'];start=len(a['text'])+2
            chosen.append({'passage_id':paper['paper_id']+f'/g{ordinal:02d}',
                'paper_id':paper['paper_id'],'title':paper['title'],'split':paper['split'],
                'page':t['page'],'section':'Body paragraph','extracted_section_hint':t['section'],
                'abstract':old['abstract'],'before':a['text'],'held_out':t['text'],'after':c['text'],
                'source_paragraphs':[a,t,c],'text':text,'text_sha256':b.sha(text),
                'held_out_start':start,'held_out_end':start+len(t['text']),
                'extraction':old['extraction']+'; omit small-font footnotes, numbered captions and non-prose fragments from body context',
                'selection_note':'Additional target; 40–320 words; at most one infinity or Pi symbol; original PDF text retained',
                'raw_context_indices':[a['raw_index'],t['raw_index'],c['raw_index']]})
        assert len({p['held_out'] for p in chosen})==5
        selected.extend(chosen)
        inventory.append({'paper_id':paper['paper_id'],'additional_candidates':len(candidates),'chosen':[{'passage_id':p['passage_id'],'page':p['page'],'words':len(p['held_out'].split()),'section':p['extracted_section_hint']} for p in chosen]})
    assert len(selected)==250 and Counter(p['paper_id'] for p in selected)=={p['paper_id']:5 for p in papers}
    b.writel(OUT/'papers.jsonl',papers);b.writel(OUT/'passages.jsonl',selected)
    b.save(OUT/'selection-inventory.json',inventory)
    (OUT/'source-selector.snapshot.py').write_bytes(Path(sources.__file__).read_bytes())
    print(json.dumps({'papers':len(papers),'paragraphs':len(selected),'reused':50,'new':200}))

async def prepare():
    assert not (OUT/'manifest.json').exists()
    passages=b.readl(OUT/'passages.jsonl');assert len(passages)==250
    prior=json.loads((PRIOR/'manifest.json').read_text())
    assert prior['instructions']==gap.INSTRUCTIONS and prior['settings']['max_tokens']==5000
    mf={**prior,'created_utc':b.now(),'paragraphs':250,'paragraphs_per_paper':5,
        'prior_run':str(PRIOR),'reused_passage_ids':[p['passage_id'] for p in passages if p['passage_id'].endswith('/g01')],
        'papers_sha256':b.sha((OUT/'papers.jsonl').read_bytes()),'passages_sha256':b.sha((OUT/'passages.jsonl').read_bytes()),
        'planned_calls':{'outline':200,'writer':200,'audit':200,'strict_fidelity':200},
        'comparison_design':'200 additional distinct targets plus the 50 existing v2 targets, with all existing responses and judgments reused; generation and review prompts/settings unchanged',
        'selection_policy':'Body prose, 40–320 words, short footnotes and numbered captions excluded. Prefer original strict filter, clean adjacent body prose, different sections/pages; limited inline infinity/Pi fallback for equation-heavy papers. Sources frozen before generation.'}
    for key in ['changes','initial_passages_sha256','source_corrections']:mf.pop(key,None)
    mf['limitations']=list(prior['limitations'])+['Some new targets use wider length and limited inline-notation selection criteria; sources and generated results still require quality review','Paragraphs can appear as human context in other examples from the same paper; paper-level split assignments are unchanged']
    b.save(OUT/'manifest.json',mf)
    for name in ['outline-responses.jsonl','writer-responses.jsonl','audit-responses.jsonl','attempts.jsonl']:
        b.writel(OUT/name,[{**r,'reused_from':str(PRIOR)} for r in b.readl(PRIOR/name)])
    fidelity.OUT.mkdir(exist_ok=True)
    for name in ['responses.jsonl','attempts.jsonl']:
        b.writel(fidelity.OUT/name,[{**r,'reused_from':str(PRIOR/'fidelity-review')} for r in b.readl(PRIOR/'fidelity-review'/name)])
    status,catalog=await b.transport.fetch('models',auth=False);assert status==200
    model=next(m for m in catalog['data'] if m['id']==gap.MODEL)
    assert model['canonical_slug']==mf['canonical_slug'], 'Model version changed; do not silently change protocol'
    b.save(OUT/'model-catalog-record.json',model)
    for module,name in [(gap,'generation-runner.snapshot.py'),(v2,'v2-prompts.snapshot.py'),(fidelity,'fidelity-runner.snapshot.py')]:
        shutil.copyfile(module.__file__,OUT/name)
    (OUT/'runner.snapshot.py').write_bytes(Path(__file__).read_bytes())
    await gap.snapshot('key-usage-before.json')
    print(json.dumps({'new_planned_calls':800,'model':gap.MODEL,'reused_existing_paragraphs':50}))

async def review():
    assert len(gap.read_stage('writer'))==250
    if not (fidelity.OUT/'manifest.json').exists():
        prior=json.loads((PRIOR/'fidelity-review/manifest.json').read_text())
        plan={**prior,'created_utc':b.now(),'passages_sha256':b.sha((OUT/'passages.jsonl').read_bytes()),
              'writers_sha256':b.sha((OUT/'writer-responses.jsonl').read_bytes()),
              'scope':'250 total paragraphs; 50 prior v2 judgments reused, 200 new independent Luna reviews; no outlines or previous scores in new judge inputs'}
        b.save(fidelity.OUT/'manifest.json',plan)
        (fidelity.OUT/'runner.snapshot.py').write_bytes(Path(__file__).read_bytes())
    await fidelity.run()

def costs(attempts):
    us=[a['response']['usage'] for a in attempts];assert all('cost' in u for u in us)
    return {'calls':len(us),'input_tokens':sum(u['prompt_tokens'] for u in us),
            'output_tokens':sum(u['completion_tokens'] for u in us),
            'reasoning_tokens':sum(u.get('completion_tokens_details',{}).get('reasoning_tokens',0) for u in us),
            'cost_usd':sum(u['cost'] for u in us)}

def correct_source():
    """Record the one visually discovered selection error before any new writers run."""
    mf=json.loads((OUT/'manifest.json').read_text())
    assert not mf.get('source_corrections'), 'Correction already recorded'
    assert all(r.get('reused_from') for r in b.readl(OUT/'writer-responses.jsonl'))
    status=json.loads((OUT/'outline-status.json').read_text())
    assert 'completed' in status, 'Wait for the active outline batch to finish'
    pid='300891a62162b960cf02ce3827bb363c/g04'
    paper=next(p for p in b.readl(OUT/'papers.jsonl') if p['paper_id']==pid.split('/')[0])
    gap.OUT=CACHE
    try:paras=sources.paragraphs(paper)
    finally:gap.OUT=OUT
    i=next(i for i,p in enumerate(paras) if p['raw_index']==12)
    a,t,c=paras[i-1:i+2]
    ps=b.readl(OUT/'passages.jsonl');p=next(p for p in ps if p['passage_id']==pid)
    assert t['text'].startswith('Finally, we use the 3rd and the 4th contributions')
    shutil.copyfile(OUT/'passages.jsonl',OUT/'passages.before-source-correction.jsonl')
    text=a['text']+'\n\n'+t['text']+'\n\n'+c['text'];start=len(a['text'])+2
    p.update(page=t['page'],extracted_section_hint=t['section'],before=a['text'],held_out=t['text'],after=c['text'],
             text=text,text_sha256=b.sha(text),held_out_start=start,held_out_end=start+len(t['text']),
             source_paragraphs=[a,t,c],raw_context_indices=[a['raw_index'],t['raw_index'],c['raw_index']])
    b.writel(OUT/'passages.jsonl',ps)
    rs=b.readl(OUT/'outline-responses.jsonl')
    b.writel(OUT/'superseded-outline-responses.jsonl',[r for r in rs if r['passage_id']==pid])
    b.writel(OUT/'outline-responses.jsonl',[r for r in rs if r['passage_id']!=pid])
    mf['initial_passages_sha256']=mf['passages_sha256'];mf['passages_sha256']=b.sha((OUT/'passages.jsonl').read_bytes())
    mf['source_corrections']=[{'passage_id':pid,'at_utc':b.now(),'reason':'Visual PDF check showed the initial target spanned a paragraph introduction and the first numbered list item across a page break. Replaced with a complete paragraph on page 2; no writer had seen the initial target. Initial source and outline retained; all outline attempts remain billed.'}]
    b.save(OUT/'manifest.json',mf)
    inventory=json.loads((OUT/'selection-inventory.json').read_text())
    for row in inventory:
        for item in row['chosen']:
            if item['passage_id']==pid:item.update(page=p['page'],words=len(p['held_out'].split()),section=p['extracted_section_hint'])
    b.save(OUT/'selection-inventory.json',inventory)
    (OUT/'source-selector.corrected.snapshot.py').write_bytes(Path(sources.__file__).read_bytes())
    (OUT/'runner.snapshot.py').write_bytes(Path(__file__).read_bytes())
    print('Corrected one target; retained initial source, superseded outline and all costs.')

def export():
    with contextlib.redirect_stdout(io.StringIO()):gap.export()
    attempts=b.readl(OUT/'attempts.jsonl')
    cs=json.loads((OUT/'costs.json').read_text())
    cs['new']=costs([a for a in attempts if not a.get('reused_from')]);cs['reused']=costs([a for a in attempts if a.get('reused_from')])
    if 'account_usage_delta_usd' in cs:
        cs['account_comparison_scope']='New generation and audit calls only; reused calls were billed in the prior v2 run'
        cs['account_matches_ledger']=abs(cs['account_usage_delta_usd']-cs['new']['cost_usd'])<1e-8
    b.save(OUT/'costs.json',cs)
    summary=json.loads((OUT/'summary.json').read_text());summary['costs']=cs
    b.save(OUT/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['papers','source_passages','rows','generated_paragraphs','ai_region_tokens']}))

def report():
    with contextlib.redirect_stdout(io.StringIO()):fidelity.report()
    results=fidelity.rows('responses.jsonl');assert len(results)==250
    fs=json.loads((fidelity.OUT/'summary.json').read_text())
    fs['current_automated_screen_verdicts']=fs.pop('previously_screened_17_verdicts')
    attempts=fidelity.rows('attempts.jsonl')
    fs['costs']['new']=costs([a for a in attempts if not a.get('reused_from')])
    fs['costs']['reused']=costs([a for a in attempts if a.get('reused_from')])
    fs['costs']['account_comparison_scope']='New strict fidelity calls only'
    fs['costs']['account_matches_ledger']=abs(fs['costs']['account_usage_delta_usd']-fs['costs']['new']['cost_usd'])<1e-8
    b.save(fidelity.OUT/'summary.json',fs)
    all_attempts=b.readl(OUT/'attempts.jsonl')+attempts
    new=costs([a for a in all_attempts if not a.get('reused_from')]);reused=costs([a for a in all_attempts if a.get('reused_from')])
    if (OUT/'key-total-after.json').exists():
        new['account_usage_delta_usd']=json.loads((OUT/'key-total-after.json').read_text())['usage']-json.loads((OUT/'key-usage-before.json').read_text())['usage']
        new['account_matches_ledger']=abs(new['account_usage_delta_usd']-new['cost_usd'])<1e-8
    summary=json.loads((OUT/'summary.json').read_text())
    dataset=b.readl(OUT/'dataset.jsonl')
    new_dataset=[r for r in dataset if not r['passage_id'].endswith('/g01')]
    report={'papers':50,'paragraphs_per_paper':5,'paragraphs':250,'new_paragraphs':200,'reused_paragraphs':50,
            'verdicts':fs['verdicts'],
            'new_verdicts':dict(Counter(r['output']['verdict'] for r in results if not r.get('reused_from'))),
            'reused_verdicts':dict(Counter(r['output']['verdict'] for r in results if r.get('reused_from'))),
            'ai_region_tokens':summary['ai_region_tokens'],
            'new_ai_region_tokens':sum(sum(t['label']=='ai_rewritten' for t in r['tokens']) for r in new_dataset),
            'ai_sentences':summary['ai_sentences'],
            'new_ai_sentences':sum(sum(s['label']=='ai_rewritten' for s in r['sentences']) for r in new_dataset),
            'costs':{'new':new,'reused':reused,'total':costs(all_attempts)},
            'unchanged_protocol':True,'limitations':json.loads((OUT/'manifest.json').read_text())['limitations']}
    b.save(OUT/'expansion-report.json',report)
    print(json.dumps(report,indent=2))

def validate():
    gap.validate()
    ps=b.readl(OUT/'passages.jsonl');assert len({p['passage_id'] for p in ps})==250
    assert set(Counter(p['paper_id'] for p in ps).values())=={5}
    assert len({b.sha(p['held_out']) for p in ps})==250
    old={p['passage_id']:p for p in b.readl(PRIOR/'passages.jsonl')}
    assert all(p==old[p['passage_id']] for p in ps if p['passage_id'] in old)
    for stage in ['outline','writer','audit']:
        old_rows={r['passage_id']:r for r in b.readl(PRIOR/(stage+'-responses.jsonl'))}
        for r in b.readl(OUT/(stage+'-responses.jsonl')):
            if r.get('reused_from'):assert {k:v for k,v in r.items() if k!='reused_from'}==old_rows[r['passage_id']]
    original=json.loads((PRIOR/'manifest.json').read_text());current=json.loads((OUT/'manifest.json').read_text())
    assert original['instructions']==current['instructions'] and original['settings']==current['settings']
    assert json.loads((PRIOR/'fidelity-review/manifest.json').read_text())['prompt']==json.loads((fidelity.OUT/'manifest.json').read_text())['prompt']
    old_dataset={r['id']:r for r in b.readl(PRIOR/'dataset.jsonl')}
    assert all(r==old_dataset[r['id']] for r in b.readl(OUT/'dataset.jsonl') if r['id'] in old_dataset)
    old_judgments={r['passage_id']:r for r in b.readl(PRIOR/'fidelity-review/responses.jsonl')}
    for r in fidelity.rows('responses.jsonl'):
        if r.get('reused_from'):assert {k:v for k,v in r.items() if k!='reused_from'}==old_judgments[r['passage_id']]
    print('Validated five distinct paragraphs per paper; 50 prior examples reused unchanged; generation and fidelity prompts unchanged.')

async def settle_total():
    await gap.snapshot('key-total-after.json')
    shutil.copyfile(OUT/'key-total-after.json',fidelity.OUT/'key-after.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','prepare','outline','writer','audit','export','review','report','validate','settle','settle-total','correct-source']);a=p.parse_args()
    if a.action in ['outline','writer','audit']:asyncio.run(gap.run_stage(a.action))
    elif a.action in ['prepare','review']:asyncio.run(globals()[a.action]())
    elif a.action=='settle':asyncio.run(gap.snapshot('key-usage-after.json'))
    elif a.action=='settle-total':asyncio.run(settle_total())
    elif a.action=='correct-source':correct_source()
    else:globals()[a.action]()
