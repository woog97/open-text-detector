import io,json,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from pypdf import PdfWriter
from requests.exceptions import ChunkedEncodingError
from openreview_downloader import queue

class RetryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.plan={'batches':[{'batch':1,'group':'test/2025','papers':[{'forum_id':'a','submission_number':1}]}]}
  (self.root/'plan.json').write_text(json.dumps(self.plan))
 def tearDown(self):self.tmp.cleanup()
 def response(self,body=None,error=None,remaining='100'):
  r=Mock(status_code=200,headers={'ratelimit-remaining':remaining,'ratelimit-reset':'3600'})
  r.iter_content.side_effect=error
  if error is None:r.iter_content.return_value=[body]
  ctx=Mock();ctx.__enter__=Mock(return_value=r);ctx.__exit__=Mock(return_value=False);return ctx
 def run_sequence(self,responses):
  session=Mock();session.get.side_effect=responses
  with patch('openreview.api.OpenReviewClient',return_value=SimpleNamespace(session=session,headers={})),patch.object(queue,'credentials',return_value=('test','test')),patch.object(queue.time,'sleep'):
   queue.main(['--root',str(self.root),'--retry-failures','--stop-at-limit','--max-batches','1'])
  return session
 def test_truncated_transfer_then_corrupt_pdf_then_success(self):
  w=PdfWriter();w.add_blank_page(width=72,height=72);b=io.BytesIO();w.write(b)
  s=self.run_sequence([self.response(error=ChunkedEncodingError('truncated')),self.response(b'%PDF-1.5\nbroken'),self.response(b.getvalue())])
  self.assertEqual(s.get.call_count,3)
  db=queue.connect(self.root/'queue.sqlite3')
  self.assertEqual([x[0] for x in db.execute('select status from attempts order by id')],['retry_scheduled','retry_scheduled','completed'])
  self.assertEqual(db.execute('select status from papers').fetchone()[0],'downloaded')
  self.assertTrue((self.root/'responses/000002.bin').exists());db.close()
 def test_zero_allowance_on_broken_body_stops_before_retry(self):
  s=self.run_sequence([self.response(error=ChunkedEncodingError('truncated'),remaining='0')])
  self.assertEqual(s.get.call_count,1)
  db=queue.connect(self.root/'queue.sqlite3');self.assertIsNotNone(db.execute("select value from settings where key='not_before'").fetchone());db.close()
 def test_retry_after_http_date_is_enforced(self):
  db=queue.connect(self.root/'queue.sqlite3')
  with patch.object(queue.time,'time',return_value=0):queue.quota_deadline(db,{'retry-after':'Thu, 01 Jan 1970 00:02:00 GMT'},503)
  self.assertEqual(float(db.execute("select value from settings where key='not_before'").fetchone()[0]),122);db.close()
 def test_auth_failure_is_not_retried(self):
  db=queue.connect(self.root/'queue.sqlite3');queue.seed(db,self.plan)
  a=queue.begin_attempt(db,1,queue.next_batch(db))
  with db:db.execute('update attempts set http_status=401 where id=?',(a,))
  self.assertFalse(queue.schedule_retry(db,a,'unauthorized'));db.close()
 def test_retry_cap_persists_and_remote_verified_is_never_reset(self):
  db=queue.connect(self.root/'queue.sqlite3');queue.seed(db,self.plan)
  for i in range(6):
   a=queue.begin_attempt(db,1,queue.next_batch(db))
   with patch.object(queue.time,'time',return_value=100):queue.schedule_retry(db,a,'broken')
   if i<5:self.assertEqual(float(db.execute("select value from settings where key='retry_not_before'").fetchone()[0]),100+5*2**i)
   self.assertFalse(queue.schedule_retry(db,a,'again'))
  self.assertEqual(db.execute('select status from papers').fetchone()[0],'retry_exhausted')
  with db:db.execute("update papers set status='remote_verified'")
  with patch.object(queue,'local_pdfs',return_value=[]):queue.reconcile(db,self.root)
  self.assertEqual(db.execute('select status from papers').fetchone()[0],'remote_verified');db.close()

 def test_partial_batch_saves_good_and_quarantines_identical_bad(self):
  db=queue.connect(self.root/'queue.sqlite3')
  plan={'batches':[{'batch':1,'group':'test/2025','papers':[{'forum_id':'a','submission_number':1},{'forum_id':'b','submission_number':2}]}]}
  queue.seed(db,plan);aid=queue.begin_attempt(db,1,queue.next_batch(db))
  w=PdfWriter();w.add_blank_page(width=72,height=72);b=io.BytesIO();w.write(b)
  folder=self.root/'responses';folder.mkdir();body=folder/f'{aid:06}.bin'
  with zipfile.ZipFile(body,'w') as z:z.writestr('1_good.pdf',b.getvalue());z.writestr('2_bad.pdf',b'%PDF-1.5\nbroken')
  with db:db.execute('update attempts set body_file=?,http_status=200 where id=?',(str(body),aid))
  with self.assertRaisesRegex(ValueError,'isolated retry'):queue.finish_body(db,self.root,db.execute('select * from attempts where id=?',(aid,)).fetchone())
  queue.schedule_retry(db,aid,'bad')
  self.assertEqual([p['id'] for p in queue.next_batch(db)],['b'])
  aid=queue.begin_attempt(db,1,queue.next_batch(db));body=folder/f'{aid:06}.bin';body.write_bytes(b'%PDF-1.5\nbroken')
  with db:db.execute('update attempts set body_file=?,http_status=200 where id=?',(str(body),aid))
  queue.finish_body(db,self.root,db.execute('select * from attempts where id=?',(aid,)).fetchone())
  self.assertEqual(dict(db.execute('select id,status from papers')),{'a':'downloaded','b':'unavailable_invalid_pdf'})
  self.assertEqual(db.execute('select count(*) from retry_failures').fetchone()[0],0)
  self.assertEqual(len(list((self.root/'invalid_pdfs').glob('*.pdf'))),1);db.close()
