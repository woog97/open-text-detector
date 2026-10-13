import collections,hashlib,json,math,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent;rows=json.loads((P/'dataset.json').read_text());protocol=json.loads((P/'protocol.json').read_text());out={}
for name,prompt in protocol['prompts'].items():
 d=P/name;r=json.loads((d/'results.json').read_text());assert r['state']=='complete' and not r['errors'] and r['completed']==len(rows);assert r['reported_tiers']=={'flex':len(rows)}
 counts=collections.Counter();cost=0;response_ids=set();generator=collections.defaultdict(lambda:[0,0])
 for row in rows:
  assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
  c=json.loads((d/'calls'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text());response=c['response'];request=c['request']
  assert request['messages']==[{'role':'system','content':prompt},{'role':'user','content':row['text']}]
  assert response['model']=='openai/gpt-6-luna' and response['service_tier']=='flex'
  pred=json.loads(response['choices'][0]['message']['content'])['label'];assert pred==c['prediction']
  counts[row['label'],pred]+=1;cost+=response['usage']['cost'];response_ids.add(response['id'])
  if row['label']=='AI':generator[row['generator']][0]+=int(pred=='AI');generator[row['generator']][1]+=1
 assert len(response_ids)==len(rows) and math.isclose(cost,r['cost_usd'])
 for k,labels in [('tn',('HUMAN','HUMAN')),('fp',('HUMAN','AI')),('tp',('AI','AI')),('fn',('AI','HUMAN'))]:assert r['section_counts'][k]==counts[labels]
 out[name]=r|{'generator_section_recall':dict(generator),'verified_raw_responses':True}
(P/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['Independent manuscript-section evaluation — October 3, 2026','',
'12 human historical proceedings papers (4 each ACL/ICML/NeurIPS), 12 generated manuscripts (4 each Luna/Sol/Sol6.1). Human papers exclude known prompt-tuning, demonstration, and prior mini-suite families. Generated manuscripts reuse the previous mini-suite papers; their sections were not used for prompt tuning.',
'236 section-sized chunks: 130 human, 106 AI. Target400 words, usual maximum600; short final tails can merge up to700. Paragraph boundaries preserved except oversized blocks split at500 words. Every retained body word covered once, with no overlapping windows. These are paragraph-aligned chunks, not verified semantic sections.',
'Human text is the existing extracted body-prose representation, not a complete PDF transcription. First abstract block and reference-tagged blocks excluded. AI title/abstract/reference tail and Markdown heading lines excluded; human-supplied abstract text is not labeled as AI. Both classes use identical whitespace normalization. PDF-versus-Markdown extraction differences and differing topic/year distributions remain possible confounds.',
'Every chunk classified independently in a fresh stateless request. Payload verified to contain only the frozen system prompt (including its fixed calibration demonstrations) and this chunk; no manuscript ID, heading, neighboring chunks or previous outputs. No prompt tuning on this evaluation. All calls low-reasoning Luna and explicitly confirmed Flex.', '']
for name,r in out.items():
 c=r['section_counts'];lines += [name,
 f"Section balanced accuracy: {r['section_balanced_accuracy']:.2%}.",
 f"Human-section FPR: {c['fp']}/130 ({r['human_section_fpr']:.2%}). AI-section recall: {c['tp']}/106 ({r['ai_section_recall']:.2%}).",
 f"Equal-paper-weighted human-section FPR: {r['paper_macro_human_section_fpr']:.2%}; AI-section recall: {r['paper_macro_ai_section_recall']:.2%}.",
 f"Strict-majority paper classification: human {r['paper_majority_human_correct']}/12 correct; AI {r['paper_majority_ai_correct']}/12 correct. Ties are HUMAN.",
 f"Human papers with at least one false-positive section: {r['human_papers_with_any_false_positive']}/12.",
 'AI-section recall by final generator: '+json.dumps(r['generator_section_recall']),
 f"API cost ${r['cost_usd']:.8f}.", '']
lines += [f"Total API cost ${sum(r['cost_usd'] for r in out.values()):.8f}.",
'Within-paper sections are correlated; 236 sections do not represent236 independent papers. This is a small24-paper diagnostic. Hosted inference precision is provider-managed/unreported. Files preserve dataset, manuscript provenance, frozen protocol, raw calls, section predictions, and per-paper results.']
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
with zipfile.ZipFile(P/'section-evaluation.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['dataset.json','manuscripts.json','sampling.json','protocol.json','comparison.json','REPORT.txt']:z.write(P/name,name)
 for name in out:
  for file in ['predictions.json','results.json']:z.write(P/name/file,name+'/'+file)
print('\n'.join(lines))
