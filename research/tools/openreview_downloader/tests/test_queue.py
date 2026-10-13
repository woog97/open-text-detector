import io,tempfile,unittest,zipfile,hashlib
from pathlib import Path
from unittest.mock import patch
from openreview_downloader import queue
from pypdf import PdfWriter
class QueueTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.db=queue.connect(self.root/'q.sqlite3')
  self.plan={'batches':[{'batch':1,'group':'test/2025','papers':[{'forum_id':'a','submission_number':10},{'forum_id':'b','submission_number':20}]}]}
  queue.seed(self.db,self.plan)
 def tearDown(self):self.db.close();self.tmp.cleanup()
 def test_seed_and_reservation_do_not_repeat(self):
  queue.seed(self.db,self.plan);self.assertEqual(self.db.execute('select count(*) from papers').fetchone()[0],2)
  rows=queue.next_batch(self.db);queue.begin_attempt(self.db,1,rows)
  self.assertEqual(queue.next_batch(self.db),[])
  with self.assertRaises(ValueError):queue.begin_attempt(self.db,1,rows)
 def test_interrupted_request_is_held(self):
  queue.begin_attempt(self.db,1,queue.next_batch(self.db));queue.recover(self.db,self.root)
  self.assertEqual(self.db.execute('select status from attempts').fetchone()[0],'uncertain');self.assertEqual(queue.next_batch(self.db),[])
 def test_saved_response_recovers_without_network(self):
  aid=queue.begin_attempt(self.db,1,queue.next_batch(self.db));folder=self.root/'responses';folder.mkdir()
  w=PdfWriter();w.add_blank_page(width=72,height=72);b=io.BytesIO();w.write(b)
  with zipfile.ZipFile(folder/f'{aid:06}.bin','w') as z:z.writestr('20_B.pdf',b.getvalue());z.writestr('10_A.pdf',b.getvalue())
  queue.recover(self.db,self.root)
  self.assertEqual(self.db.execute('select status from attempts').fetchone()[0],'completed');self.assertEqual(queue.next_batch(self.db),[])
  self.assertTrue((self.root/'pdfs/test/2025/a.pdf').exists())
  self.assertFalse((folder/f'{aid:06}.bin').exists())
 def test_cleanup_preserves_unfinished_and_missing_pdf_responses(self):
  aid=queue.begin_attempt(self.db,1,queue.next_batch(self.db))
  folder=self.root/'responses';folder.mkdir();body=folder/f'{aid:06}.bin';body.write_bytes(b'keep')
  with self.db:self.db.execute('update attempts set body_file=? where id=?',(str(body),aid))
  self.assertEqual(queue.cleanup_responses(self.db,self.root)['removed_responses'],0)
  self.assertTrue(body.exists())
  with self.db:self.db.execute("update attempts set status='completed' where id=?",(aid,))
  self.assertEqual(queue.cleanup_responses(self.db,self.root)['removed_responses'],0)
  self.assertTrue(body.exists())
 def test_duplicate_track_numbers_split(self):
  with self.db:self.db.execute("update papers set number='10' where id='b'")
  first=queue.next_batch(self.db);self.assertEqual(len(first),1);queue.begin_attempt(self.db,1,first)
  self.assertEqual([p['id'] for p in queue.next_batch(self.db)],['b'])
 def test_existing_validated_pdf_skipped(self):
  f=self.root/'known.pdf';f.write_bytes(b'%PDF-existing')
  with patch.object(queue,'local_pdfs',return_value=[('a',f,{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'pages':1})]):queue.reconcile(self.db,self.root)
  self.assertEqual([p['id'] for p in queue.next_batch(self.db)],['b'])
 def test_year_first_rotation_and_wrap(self):
  for batch,group in [(2,'b/2020'),(3,'a/2021'),(4,'a/2020'),(5,'tmlr/unknown')]:
   queue.seed(self.db,{'batches':[{'batch':batch,'group':group,'papers':[{'forum_id':f'{group}-{i}','submission_number':i} for i in range(55)]}]})
  seen=[]
  for _ in range(5):
   papers=queue.next_batch(self.db);group=papers[0]['group_key'];seen.append((group,len(papers)))
   with self.db:
    self.db.executemany("UPDATE papers SET status='downloaded' WHERE id=?",[(p['id'],) for p in papers])
    self.db.execute("INSERT OR REPLACE INTO settings VALUES('last_group',?)",(group,))
  self.assertEqual(seen,[('a/2020',50),('b/2020',50),('a/2021',50),('test/2025',2),('a/2020',5)])
 def test_zero_remaining_stops_without_second_request(self):
  self._run_mock(200,{'ratelimit-remaining':'0','ratelimit-reset':'3600'},'completed')
 def test_legacy_venues_are_held_and_skipped(self):
  queue.seed(self.db,{'batches':[{'batch':2,'group':'old/2020','papers':[{'forum_id':'legacy','submission_number':1,'api_version':1}]}]})
  self.assertEqual(self.db.execute("select status from papers where id='legacy'").fetchone()[0],'held_legacy')
  self.assertEqual(queue.group_order(self.db),['test/2025'])
  self.assertEqual([r['id'] for r in queue.next_batch(self.db)],['a','b'])
 def test_unverified_real_venue_is_rejected_before_request(self):
  self.plan['batches'][0]['papers'][0]['venue_id']='unverified/2020'
  with self.assertRaisesRegex(ValueError,'not been verified'):queue.seed(self.db,self.plan)
 def test_404_is_rejected_not_uncertain(self):
  self._run_mock(404,{},'rejected')
 def test_rejected_request_stops_without_retry(self):
  self._run_mock(429,{'retry-after':'30'},'rate_limited')
 def _run_mock(self,status,headers,expected):
  import json
  from types import SimpleNamespace
  from unittest.mock import Mock
  (self.root/'plan.json').write_text(json.dumps(self.plan))
  w=PdfWriter();w.add_blank_page(width=72,height=72);pdf=io.BytesIO();w.write(pdf);data=io.BytesIO()
  with zipfile.ZipFile(data,'w') as z:z.writestr('10_A.pdf',pdf.getvalue());z.writestr('20_B.pdf',pdf.getvalue())
  response=Mock(status_code=status,headers=headers,text='RateLimitError')
  response.iter_content.return_value=[data.getvalue()]
  context=Mock();context.__enter__=Mock(return_value=response);context.__exit__=Mock(return_value=False)
  session=Mock();session.get.return_value=context;client=SimpleNamespace(session=session,headers={})
  with patch('openreview.api.OpenReviewClient',return_value=client),patch.object(queue,'credentials',return_value=('local-test','not-a-real-password')):
   if status==404:
    with self.assertRaises(SystemExit):queue.main(['--root',str(self.root),'--max-batches','138','--stop-at-limit'])
   else:queue.main(['--root',str(self.root),'--max-batches','138','--stop-at-limit'])
  self.assertEqual(session.get.call_count,1)
  other=queue.connect(self.root/'queue.sqlite3')
  self.assertEqual(other.execute('select status from attempts').fetchone()[0],expected);other.close()
