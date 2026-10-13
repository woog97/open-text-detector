"""Streaming provenance exports for larger paper collections; same annotation rules."""
import json,re,hashlib
from collections import Counter
from itertools import zip_longest
import paper_gap50 as g
base=g.base
MODEL=g.MODEL
longest_copy=g.longest_copy
read_stage=g.read_stage
OUT=None
SOURCE=None

def file_sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def writel(path,rows):
 temp=path.with_suffix(path.suffix+'.tmp')
 with temp.open('w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 temp.replace(path)

class DiskRows:
 def __init__(self,path,reset=False):
  self.path=path
  if reset:path.write_text('')
 def __iter__(self):
  with self.path.open() as f:
   for line in f:
    if line.strip():yield json.loads(line)
 def __len__(self):
  with self.path.open() as f:return sum(bool(line.strip()) for line in f)
 def append(self,row):
  with self.path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def export():
    ps=base.readl(OUT/'passages.jsonl');mf=json.loads((OUT/'manifest.json').read_text());writers=read_stage('writer');outlines=read_stage('outline');audits=read_stage('audit')
    assert len(writers)==len(outlines)==len(audits)==len(ps)
    attempts=base.readl(OUT/'attempts.jsonl');rows=DiskRows(OUT/'dataset.jsonl',reset=True);quality=[]
    attempt_by_id={a['response'].get('id'):a for a in attempts}
    manual=json.loads((OUT/'manual-review.json').read_text()) if (OUT/'manual-review.json').exists() else {'reviewed_example_ids':[],'flags':{}}
    for p in ps:
        rid=p['passage_id'];w=writers[rid];a=audits[rid]['output'];flags=[]
        for k in ['outline_faithful','content_covered','meaning_preserved','citations_preserved']:
            if not a[k]:flags.append('audit_'+k+'_failed')
        if a['clarity']=='worse':flags.append('audit_lower_clarity')
        if a['context_fit']=='poor':flags.append('audit_poor_context_fit')
        if a['issues']:flags.append('audit_issue_requires_review')
        if re.search(r'\\(?:ell|frac|math\w*|theta|alpha|beta)\b|`|\$[^$]{1,100}\$',w['output']['paragraph']):
            flags.append('generated_math_or_code_markup_requires_normalization')
        source_flags=(['audit_source_extraction_issue'] if a['source_issues'] else [])+manual['flags'].get(rid+'/source',[])
        flags+=source_flags+manual['flags'].get(rid+'/paragraph_generate',[])
        original=base.annotated_row(p,'human_original',[])
        edit={'source_start':p['held_out_start'],'source_end':p['held_out_end'],'original':p['held_out'],'replacement':w['output']['paragraph'],'operation':'paragraph_generate'}
        generated=base.annotated_row(p,'paragraph_generate',[edit],w)
        original['quality_flags']=source_flags;generated['quality_flags']=sorted(set(generated['quality_flags']+flags))
        for row in [original,generated]:
            row.update(dedup_group=row['text_sha256'],sample_weight=1.0,eligible_for_pilot_training=not bool(row['quality_flags']),
                quality_status='needs_researcher_review' if row['quality_flags'] else 'automated_checks_passed_pending_expert_review',
                provenance_scope='Full generated middle paragraph; surrounding human-source paragraphs unchanged; provenance is not independent authorship proof')
            row['development_exposed']=p.get('development_exposed',False)
            rows.append(row)
        writer_attempt=attempt_by_id[w['generation_id']]
        writer_json=writer_attempt['request']['messages'][1]['content'].split('\n\nSource JSON:\n',1)[1]
        quality.append({'passage_id':rid,'audit':a,'flags':generated['quality_flags'],'original_absent_from_writer_request':p['held_out'] not in writer_json,
            'longest_original_overlap':longest_copy(p['held_out'],w['output']['paragraph']),
            'original_words':len(p['held_out'].split()),'generated_words':len(w['output']['paragraph'].split()),
            'outline_longest_original_overlap':max(longest_copy(p['held_out'],s)['words'] for s in outlines[rid]['output']['facts'])})
    for split in ['train','validation','test']:
        base.writel(OUT/(split+'.jsonl'),(r for r in rows if r['split']==split and r['eligible_for_pilot_training']))
    def cost(xs):
        us=[x['response'].get('usage',{}) for x in xs]
        return {'attempts':len(xs),'input_tokens':sum(u.get('prompt_tokens',0) for u in us),'output_tokens':sum(u.get('completion_tokens',0) for u in us),
            'reasoning_tokens':sum(u.get('completion_tokens_details',{}).get('reasoning_tokens',0) for u in us),'cost_usd':sum(u.get('cost',0) for u in us),'cost_coverage':sum('cost' in u for u in us)}
    costs=cost(attempts);costs['provider_reported_cost_usd']=costs['cost_usd'];costs['by_operation']={s:cost([a for a in attempts if a['stage']==s]) for s in ['outline','writer','audit']}
    if (OUT/'key-usage-after.json').exists():
        costs['account_usage_delta_usd']=json.loads((OUT/'key-usage-after.json').read_text())['usage']-json.loads((OUT/'key-usage-before.json').read_text())['usage']
        costs['account_matches_ledger']=abs(costs['account_usage_delta_usd']-costs['cost_usd'])<1e-8
    base.save(OUT/'costs.json',costs)
    summary={'papers':len({p['paper_id'] for p in ps}),'source_passages':len(ps),'rows':len(rows),'unique_texts':len({r['text_sha256'] for r in rows}),'eligible_rows':sum(r['eligible_for_pilot_training'] for r in rows),
        'generated_paragraphs':len(ps),'eligible_generated_paragraphs':sum(r['eligible_for_pilot_training'] for r in rows if r['operation']=='paragraph_generate'),
        'ai_region_tokens':sum(sum(t['label']=='ai_rewritten' for t in r['tokens']) for r in rows),'eligible_ai_region_tokens':sum(sum(t['label']=='ai_rewritten' for t in r['tokens']) for r in rows if r['eligible_for_pilot_training']),
        'ai_sentences':sum(sum(s['label']=='ai_rewritten' for s in r['sentences']) for r in rows),
        'audit':{k:dict(Counter(q['audit'][k] for q in quality)) for k in ['outline_faithful','content_covered','meaning_preserved','citations_preserved','clarity','context_fit']},
        'all_originals_absent_from_writer_inputs':all(q['original_absent_from_writer_request'] for q in quality),
        'costs':costs,'limitations':mf['limitations']}
    base.save(OUT/'summary.json',summary);base.writel(OUT/'quality-reviews.jsonl',quality)
    base.save(OUT/'content-quality-audit.json',{'reviewed_final_generations':len(manual['reviewed_example_ids']),'reviewed_example_ids':manual['reviewed_example_ids'],'automated_reviewed':len(ps),'reviewer_model':MODEL})
    print(json.dumps(summary,indent=2))


def validate():
    ps=base.readl(OUT/'passages.jsonl');rows=DiskRows(OUT/'dataset.jsonl');mf=json.loads((OUT/'manifest.json').read_text());papers=base.readl(OUT/'papers.jsonl')
    assert len(ps)==mf.get('paragraphs',50) and len(rows)==2*len(ps)
    assert base.sha((OUT/'papers.jsonl').read_bytes())==mf['papers_sha256']
    assert base.sha((OUT/'passages.jsonl').read_bytes())==mf['passages_sha256']
    old={p['paper_id']:p for p in base.readl(SOURCE/'papers.jsonl')}
    assert all(p['split']==old[p['paper_id']]['split'] for p in papers)
    byid={p['passage_id']:p for p in ps};writers=read_stage('writer')
    for r in rows:
        p=byid[r['passage_id']]
        expected=p['text'] if r['operation']=='human_original' else p['before']+'\n\n'+writers[r['passage_id']]['output']['paragraph']+'\n\n'+p['after']
        assert r['text']==expected and base.sha(expected)==r['text_sha256']
        assert sum(x['end']-x['start'] for x in r['regions'])==len(expected)
        assert r['regions'][0]['start']==0 and r['regions'][-1]['end']==len(expected)
        assert all(a['end']==b['start'] for a,b in zip(r['regions'],r['regions'][1:]))
        assert all(t['text']==expected[t['start']:t['end']] for t in r['tokens'])
        if r['operation']=='paragraph_generate':
            ai=[x for x in r['regions'] if x['label']=='ai_rewritten'];assert len(ai)==1 and expected[ai[0]['start']:ai[0]['end']]==writers[r['passage_id']]['output']['paragraph']
    for stage in ['outline','writer','audit']:
        assert len(read_stage(stage))==len(ps)
    for a in base.readl(OUT/'attempts.jsonl'):
        if a['stage']=='writer':
            p=byid[a['passage_id']];user=a['request']['messages'][1]['content'];payload=json.loads(user.split('\n\nSource JSON:\n')[1])
            assert set(payload)==set(mf['writer_input_fields']) and p['held_out'] not in user
            assert len(a['request']['messages'])==2
    for f in OUT.glob('*.json*'):
        with f.open() as handle:
            for t in handle:assert 'sk-or-' not in t and 'Authorization: Bearer' not in t
    for split in ['train','validation','test']:
        assert all(a==b for a,b in zip_longest(DiskRows(OUT/(split+'.jsonl')),(r for r in rows if r['split']==split and r['eligible_for_pilot_training'])))
    result={'passed':True,'at_utc':base.now(),'checks':['Same 50 paper identities and split assignments','Frozen source hashes',f'{len(ps)} isolated outline/writer/audit outputs','Held-out original absent from every writer request, including retries','Exact three-paragraph reconstruction and unchanged neighbors','Complete character provenance and tokenizer offsets','Split exports match eligibility','No credentials in JSON artifacts'],'dataset_sha256':file_sha(OUT/'dataset.jsonl')}
    base.save(OUT/'validation.json',result);print(json.dumps(result,indent=2))

