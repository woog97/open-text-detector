import json, unittest, tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock
import paper_workflow_eval as w

class ProtocolTests(unittest.TestCase):
 def setUp(self):
  self.p={'passage_id':'paper/g01','paper_id':'paper','forum_id':None,'split':'pilot','abstract':'Abstract only.', 'before':'Earlier paragraph.', 'after':'Later paragraph.', 'held_out':'The initial experiment measures performance under several distinct settings with a fixed architecture and carefully matched computational resources. We compare the resulting predictions against three conventional estimators using the same observations and report the average prediction error. The proposed estimator reduces this error on every evaluated dataset while preserving the original uncertainty estimates under the stated assumptions. These findings suggest a practical improvement within this experimental setting but do not establish behavior outside the observed data distribution.'}
 def test_all_conditions_and_exact_unmodified_context(self):
  jobs=w.build_jobs(self.p);self.assertEqual(len(jobs),7)
  for j in jobs:
   generated='We generated this replacement text for this test.'
   for row in w.exported(j,generated):
    self.assertEqual(row['text'][row['target_start']:row['target_end']],generated)
    self.assertEqual(sum(r['end']-r['start'] for r in row['regions']),len(row['text']))
    expected='assisted_unknown' if j['condition'] in w.EDIT else 'ai'
    self.assertEqual(next(r['label'] for r in row['regions'] if r['start']==row['target_start']),expected)
 def test_held_out_is_hidden(self):
  for j in w.build_jobs(self.p)[:4]:
   x=w.writer_source(j,{'facts':['A short note']})
   self.assertNotIn(j['original'],json.dumps(x))
   with self.assertRaises(AssertionError):w.writer_source(j,{'facts':[j['original']]})
 def test_assistance_can_be_unchanged_without_false_ai_gold(self):
  j=w.build_jobs(self.p)[4]
  rows=w.exported(j,j['original'])
  self.assertTrue(all(r['label_policy']=='human_origin_ai_assisted_no_binary_gold' for r in rows))
  self.assertTrue(all(op['tag']=='equal' for op in rows[0]['edit_operations']))
 def test_judge_rejects_invented_evidence(self):
  j=w.build_jobs(self.p)[0]
  x={'fidelity':'material_difference','clarity':'same','naturalness':'same','context_fit':'same','source_uncertain':False,'differences':[{'kind':'addition','severity':'material','original_quote':'not in source','candidate_quote':'','explanation':'unsupported'}]}
  with self.assertRaises(AssertionError):w.parse_judge(json.dumps(x),j,'Candidate text.')
 def test_parentheses_citations_and_abbreviations(self):
  text='A finding is established here. (The second finding is distinct). [31] reports another result. Smith et al. (2008) present the fourth result.'
  spans=w.source.sentence_spans(text)
  self.assertEqual(len(spans),4)
  self.assertTrue(spans[-1]['text'].startswith('Smith et al. (2008)'))
  self.assertTrue(all(text[s['start']:s['end']]==s['text'] for s in spans))
 def test_source_corruption_guards(self):
  for text in ['trained on the re- red. seq. 546 645 1808 712 spective data.', 'maximiz- P ing t v t A t v t .', 'We list measurements 546 645 1808 712 here.']:
   self.assertTrue(w.source.source_flags(text))
  self.assertFalse(w.source.source_flags('We evaluate the model on three standard benchmark datasets.'))
 def test_edit_prompt_has_explicit_scope(self):
  for c in w.EDIT:
   self.assertIn('Rewrite ONLY the original_paragraph field',w.writer_prompt(c))
 def test_empty_or_multiline_generation_rejected(self):
  for s in ['', 'First line.\nSecond line.']:
   with self.assertRaises(AssertionError):w.parse_text(json.dumps({'paragraph':s}))

class ReuseTests(unittest.IsolatedAsyncioTestCase):
 async def test_reuse_requires_full_request_match(self):
  with tempfile.TemporaryDirectory() as d, patch.object(w,'OUT',Path(d)):
   (w.OUT/'pilot-v2').mkdir()
   payload={'source':'original source'};prompt='Write a paragraph.';rid='fixture/writer'
   body={'model':w.MODEL,'messages':[{'role':'system','content':w.SYSTEM},{'role':'user','content':prompt+'\n\nSource JSON:\n'+json.dumps(payload,ensure_ascii=False)}],**w.SETTINGS,'service_tier':'flex','provider':w.PROVIDER}
   w.append('pilot-v2/dispatches.jsonl',{'request_id':rid,'request_sha256':w.b.sha(json.dumps(body,sort_keys=True))})
   w.append('pilot-v2/responses.jsonl',{'request_id':rid,'stage':'writer','output':{'paragraph':'This is the original valid generated paragraph.'},'generation_id':'fixture'})
   client=w.Client(w.MODEL)
   with patch.object(w.b.transport,'fetch',new_callable=AsyncMock) as fetch:
    result=await client.call(rid,'writer',prompt,payload,w.parse_text)
    self.assertEqual(result['paragraph'],'This is the original valid generated paragraph.')
    fetch.assert_not_called()
   # A changed source must not return the previous paragraph from the reuse cache.
   (w.OUT/'responses.jsonl').unlink();client=w.Client(w.MODEL)
   with patch.object(w.b.transport,'fetch',new_callable=AsyncMock,side_effect=RuntimeError('new dispatch required')) as fetch:
    with self.assertRaisesRegex(RuntimeError,'new dispatch required'):
     await client.call(rid,'writer',prompt,{'source':'different source'},w.parse_text)
    fetch.assert_awaited_once()

if __name__=='__main__':unittest.main()
