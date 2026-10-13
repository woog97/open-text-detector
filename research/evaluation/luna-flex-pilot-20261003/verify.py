"""Independently recompute metrics and billing from saved API responses."""
import collections, hashlib, json, math
from pathlib import Path

p=Path(__file__).resolve().parent
manifest=json.loads((p/'manifest.json').read_text())
results=json.loads((p/'results.json').read_text())
conf=collections.Counter(); cost=0; tiers=collections.Counter(); calls=[]
for row in manifest['rows']:
    assert hashlib.sha256(row['text'].encode()).hexdigest()==row['text_sha256']
    call=json.loads((p/'calls'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text())
    assert call['state']=='complete'
    assert call['request']['messages']==[{'role':'system','content':manifest['system_prompt']},{'role':'user','content':row['text']}]
    response=call['response']
    predicted=json.loads(response['choices'][0]['message']['content'])['label']
    assert predicted==call['prediction']
    conf[row['label'],predicted]+=1
    cost+=response['usage']['cost']; tiers[str(response.get('service_tier'))]+=1;calls.append(call)
for k,pair in [('tn',('HUMAN','HUMAN')),('fp',('HUMAN','AI')),('fn',('AI','HUMAN')),('tp',('AI','AI'))]:
    assert results[k]==conf[pair]
assert len(calls)==216 and results['completed']==216
assert math.isclose(cost,results['cost_usd'])
assert tiers=={'flex':216}
assert len({c['response']['id'] for c in calls})==216
assert math.isclose(results['balanced_accuracy'],(results['tn']+results['tp'])/216)
elapsed=max(c['started']+c['elapsed_seconds'] for c in calls)-min(c['started'] for c in calls)
receipt={'verified':True,'calls':216,'unique_papers':108,'reported_tiers':dict(tiers),'cost_usd':cost,'elapsed_seconds':elapsed,
         'checks':['input hashes','text-only requests','raw-response confusion matrix','billing sum','unique generation IDs','Flex reported on all responses']}
(p/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
lo,hi=results['balanced_accuracy_paper_bootstrap_95pct']
report=f'''Luna Flex classification pilot — October 3, 2026

216 independent classification requests: 108 human paragraphs and 108 matched substantial rewrites from 108 test papers.
Model: openai/gpt-6-luna. Reasoning: low. OpenRouter OpenAI Flex-only route, no fallback. All 216 responses explicitly report service_tier=flex.

Balanced accuracy: {results['balanced_accuracy']:.2%}
Paper-paired bootstrap 95% interval: {lo:.2%}–{hi:.2%} (10,000 resamples).
Human accuracy: {results['human_accuracy']:.2%} ({results['tn']}/108).
Human false positives: {results['human_fpr']:.2%} ({results['fp']}/108).
AI rewrite recall: {results['ai_recall']:.2%} ({results['tp']}/108).
Missed rewrites: {results['fn']}/108.
Recorded API cost: ${cost:.8f}. Elapsed: {elapsed:.1f} seconds.
Tokens: {results['tokens']}.

This is a passage-level AI-involvement test, not token localization. All substantial rewrites were retained; no >50%-change filter was applied, so this is not an exact Vals replication. Both the rewrites and classifier use Luna. No contamination exclusion, prompt search, or alternate reasoning run was performed. The prompt was fixed before inference. A small observed human false-positive count does not establish a very low population FPR. Hosted inference precision is not exposed by the provider.

Original passage and rewritten passage were sent in separate stateless calls. The classifier received no paired original, source labels, generator identity, paper identifiers, or metadata. Original dataset labels and splits were not changed.

The dataset and prompt are saved in manifest.json; raw requests/responses in calls/; individual results in predictions.json; metrics in results.json. Independent verification passed: input hashes, text-only payloads, raw-response confusion matrix, billing, generation IDs, and reported Flex tier.
'''
(p/'REPORT.txt').write_text(report)
print(json.dumps(receipt))
print(report)
