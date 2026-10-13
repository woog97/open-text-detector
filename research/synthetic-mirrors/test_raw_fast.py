import asyncio,json,tempfile,unittest
from pathlib import Path
from run_raw_fast import Ledger,Capacity,Production,save,UncertainCall,BudgetStop
from run_openrouter import request,GENERATOR
from mirror_core import topic_messages,sha

def row(i):
 text=' '.join(['originalword']*80)
 return {'record_id':str(i),'text':text,'passage_sha256':sha(text),'source_id':'imdb','category':'reviews','parent_document_id':str(i)}
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_exact_target_unfiltered_and_resume(self):
  with tempfile.TemporaryDirectory() as folder:
   n=0
   async def fetch(body):
    nonlocal n;n+=1;await asyncio.sleep(.002)
    # Deliberately fails old length/repetition checks: must still be retained.
    content=json.dumps({'topic':'A consumer purchase'}) if 'response_format' in body else 'originalword'
    return 200,{'model':GENERATOR['model'],'service_tier':'flex','usage':{'cost':.00001},'choices':[{'message':{'content':content},'finish_reason':'length'}]},{}
   l=Ledger(folder,1,GENERATOR['model'],Capacity(8,8),fetch);p=Production(folder,[row(i) for i in range(80)],l,17,4,12)
   s=await p.run();self.assertEqual(s['saved_raw_documents'],17);self.assertFalse(s['filtering_started']);self.assertEqual(len(list((Path(folder)/'raw-documents').glob('*.json'))),17)
   before=n;l2=Ledger(folder,1,GENERATOR['model'],Capacity(8,8),fetch);p2=Production(folder,[row(i) for i in range(80)],l2,17,4,12);s=await p2.run();self.assertEqual(n,before);self.assertEqual(s['saved_raw_documents'],17)
 async def test_unknown_call_never_replayed(self):
  with tempfile.TemporaryDirectory() as folder:
   count=0
   async def fetch(body):
    nonlocal count;count+=1;return 0,{'error':{'type':'ReadTimeout'}},{}
   l=Ledger(folder,1,GENERATOR['model'],Capacity(),fetch);r=row(1);body=request(topic_messages(r),'topic')
   with self.assertRaises(UncertainCall):await l.call(r,'topic',body)
   before=l.uncertain
   with self.assertRaises(UncertainCall):await l.call(r,'topic',body)
   self.assertEqual(count,1);self.assertGreater(before,0);self.assertEqual(l.uncertain,before)
 async def test_shared_budget_before_dispatch(self):
  with tempfile.TemporaryDirectory() as folder:
   calls=0
   async def fetch(body):
    nonlocal calls;calls+=1;raise AssertionError('Must not dispatch')
   l=Ledger(folder,.0000001,GENERATOR['model'],Capacity(),fetch);r=row(1)
   with self.assertRaises(BudgetStop):await l.call(r,'topic',request(topic_messages(r),'topic'))
   self.assertEqual(calls,0)
 async def test_missing_tier_is_preserved_unverified(self):
  with tempfile.TemporaryDirectory() as folder:
   async def fetch(body):return 200,{'model':GENERATOR['model'],'usage':{'cost':.00001},'choices':[{'message':{'content':'{"topic":"A review"}'},'finish_reason':'stop'}]},{}
   l=Ledger(folder,1,GENERATOR['model'],Capacity(),fetch);r=row(1);c=await l.call(r,'topic',request(topic_messages(r),'topic'))
   self.assertEqual(c['tier_verification'],'unreported');self.assertNotIn('service_tier',c['response'])
if __name__=='__main__':unittest.main()
