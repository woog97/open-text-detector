"""Verify completed hillclimb candidates against raw calls and compare paired errors."""
import collections, hashlib, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'luna-flex-pilot-20261003'
prompts=json.loads((HERE/'prompts.json').read_text())
baseline=json.loads((BASE/'manifest.json').read_text())
rows=baseline['rows']; output=[]; errors={}; predictions={}
for name,directory in [('baseline',BASE)]+[(n,HERE/n) for n in prompts]:
    if not (directory/'results.json').exists():continue
    results=json.loads((directory/'results.json').read_text())
    if results['state']!='complete':continue
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['rows']==rows
    expected_prompt=baseline['system_prompt'] if name=='baseline' else prompts[name]
    assert manifest['system_prompt']==expected_prompt
    counts=collections.Counter();tiers=collections.Counter();models=collections.Counter();cost=0;ids=set();err=[];pred={}
    for row in rows:
        call=json.loads((directory/'calls'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')).read_text())
        assert call['state']=='complete' and call['http_status']==200
        request=call['request']; response=call['response']
        assert request['messages']==[{'role':'system','content':expected_prompt},{'role':'user','content':row['text']}]
        assert request['provider']['only']==['openai/flex'] and request['provider']['allow_fallbacks']==False
        assert request['reasoning']=={'effort':'low','exclude':True}
        predicted=json.loads(response['choices'][0]['message']['content'])['label']
        assert predicted==call['prediction'] and predicted in ['HUMAN','AI']
        pred[row['id']]=predicted
        counts[row['label'],predicted]+=1
        tiers[str(response.get('service_tier'))]+=1;models[response['model']]+=1
        cost+=response['usage']['cost'];ids.add(response['id'])
        if predicted!=row['label']:err.append({'id':row['id'],'label':row['label'],'prediction':predicted,'text':row['text']})
    assert len(ids)==216 and sum(counts.values())==216
    assert math.isclose(cost,results['cost_usd'])
    for k,pair in [('tn',('HUMAN','HUMAN')),('fp',('HUMAN','AI')),('fn',('AI','HUMAN')),('tp',('AI','AI'))]:assert results[k]==counts[pair]
    accuracy=(counts['HUMAN','HUMAN']+counts['AI','AI'])/216
    assert math.isclose(accuracy,results['balanced_accuracy'])
    output.append({'variant':name,**{k:results[k] for k in ['balanced_accuracy','human_accuracy','human_fpr','ai_recall','tn','fp','fn','tp']},
        'cost_usd':cost,'reported_tiers':dict(tiers),'models':dict(models),'verified':True})
    errors[name]=err;predictions[name]=pred
assert output and output[0]['variant']=='baseline'
best=max(output,key=lambda r:(r['balanced_accuracy'],-r['human_fpr']))
for entry in output:
    p=predictions[entry['variant']];b=predictions['baseline']
    entry['fixed_baseline_errors']=sum(p[r['id']]==r['label'] and b[r['id']]!=r['label'] for r in rows)
    entry['introduced_errors']=sum(p[r['id']]!=r['label'] and b[r['id']]==r['label'] for r in rows)
summary={'development_exposed':True,'selection_metric':'balanced_accuracy','rows':216,'papers':108,
    'completed_variants':output,'winner':best['variant'],'new_api_cost_usd':sum(r['cost_usd'] for r in output if r['variant']!='baseline'),
    'round2_api_cost_usd':sum(r['cost_usd'] for r in output if r['variant'].startswith(('v5_', 'v6_', 'v7_', 'v8_')))}
(HERE/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
(HERE/'errors.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False)+'\n')
if len(output)==len(prompts)+1:
    winning_prompt=baseline['system_prompt'] if best['variant']=='baseline' else prompts[best['variant']]
    (HERE/'winning-prompt.txt').write_text(winning_prompt+'\n')
    lines=['Luna Flex prompt hillclimb — October 3, 2026','',
        'Dataset: 108 source papers, 36 each from ACL, ICML and NeurIPS, 2013–2021. Each contributes one original paragraph and one matched existing Luna substantial rewrite (216 passages total). These are the original workflow test assignments, paragraph view only. All rewrites are included; no >50% change filter. No new texts generated.',
        '', 'This set is now development-exposed: prompts were selected and errors inspected on these same 216 passages. Results below are tuning performance, not untouched test performance. Original dataset files and split assignments were preserved; development-exposure.json records the new usage.',
        '', 'All variants: openai/gpt-6-luna, low reasoning, OpenRouter OpenAI Flex-only, 2048 output-token limit, stateless per-passage requests. No evaluated passage is supplied with its matched original or its label. Some variants add labeled demonstrations from separate calibration papers, with source identities recorded in fewshot-examples.json and any additional fewshot manifests. Hosted inference precision is unreported.',
        '', 'Variant | Balanced accuracy | AI recall | Human FPR | Correct human | Correct AI | API cost']
    for r in output:
        lines.append(f"{r['variant']} | {r['balanced_accuracy']:.2%} | {r['ai_recall']:.2%} | {r['human_fpr']:.2%} | {r['tn']}/108 | {r['tp']}/108 | ${r['cost_usd']:.8f}")
    lines += ['',f"Selected winner by balanced accuracy: {best['variant']}.",
        f"Gain over baseline: {(best['balanced_accuracy']-output[0]['balanced_accuracy'])*100:.2f} percentage points.",
        f"Cumulative API cost for prompt variants: ${summary['new_api_cost_usd']:.8f}.",
        'Reported service tiers: '+json.dumps({r['variant']:r['reported_tiers'] for r in output}),
        '', 'Iteration: v1 corrected the task definition and equal prior odds after the baseline missed 80 rewrites. v2 added hypotheses about editorial smoothing. After examining v1/v2 errors, v3 added explicit guards against false positives on conventional academic writing; v4 added independently selected calibration demonstrations to v2. No test text was inserted as a labeled demonstration. The user then requested continued iteration: v5 emphasized local linguistic repairs, v6 combined the compact task definition with the original calibration demonstrations, v7 combined grammar and discourse clues, and v8 used three different calibration pairs spanning ACL/ICML/NeurIPS. All variants, including regressions, are retained.',
        '', 'Verification: recomputed all confusion counts and costs from raw API responses; checked text-only evaluated inputs, model settings, exact dataset identity, unique response IDs within each run, and recorded service tiers. One run per prompt; no seed repeats, threshold fitting, or independent generalization evaluation. Differences can include sampling variation.',
        '', 'Files: prompts.json (every prompt); winning-prompt.txt; comparison.json; errors.json; development-exposure.json; per-variant manifest.json, calls/, predictions.json and results.json.']
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
