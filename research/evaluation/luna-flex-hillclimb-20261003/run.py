"""Bounded prompt hillclimb on a now development-exposed paired test set."""
import asyncio, fcntl, importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'luna-flex-pilot-20261003'
spec=importlib.util.spec_from_file_location('pilot',BASE/'run.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

async def main():
    prompts=json.loads((HERE/'prompts.json').read_text())
    names=[x for x in sys.argv[1:] if not x.startswith('--')]
    assert names and len(prompts)<=8 and all(x in prompts for x in names)
    rows=json.loads((BASE/'manifest.json').read_text())['rows']
    assert len(rows)==216
    pilot.dataset=lambda:rows
    reserve=0
    for prompt in prompts.values():
        pilot.SYSTEM=prompt
        reserve+=sum(pilot.bound(pilot.body(r)) for r in rows)
    assert reserve<2
    pilot.save(HERE/'development-exposure.json',{
        'date':'2026-10-03','user_requested':'iterate on the prompt to hillclimb perf on this; Keep iterating',
        'usage':'Prompt selection and error inspection; scores are development results, not untouched test estimates.',
        'original_split_unchanged':'test','source_manifest':str(BASE/'manifest.json'),
        'rows':[{'id':r['id'],'paper_id':r['paper_id'],'text_sha256':r['text_sha256']} for r in rows],
        'objective':'balanced_accuracy','maximum_new_variants':8,'maximum_new_paid_calls':1728,
        'total_budget_usd':2,'current_worst_case_usd':reserve,'model':'openai/gpt-6-luna',
        'reasoning':'low','route':'openai/flex','fallbacks':False})
    if '--check' in sys.argv:
        print(json.dumps({'variants':names,'rows_each':len(rows),'reserve_all_configured_usd':reserve,'paid_calls':0}));return
    for name in names:
        pilot.HERE=HERE/name;pilot.HERE.mkdir(exist_ok=True)
        pilot.SYSTEM=prompts[name]
        with (pilot.HERE/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            await pilot.run()
        result=json.loads((pilot.HERE/'results.json').read_text())
        assert result['state']=='complete','Stop after incomplete variant'
        pilot.save(pilot.HERE/'development-status.json',{'development_exposed':True,'prompt_variant':name,'selected_on_same_216_rows':True})

if __name__=='__main__':
    with (HERE/'hillclimb.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(main())
