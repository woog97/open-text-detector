import io, tempfile, unittest, zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from openreview_downloader.batching import download_batch, unpack_batch

def archive(entries):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:
  for name,data in entries:z.writestr(name,data)
 return b.getvalue()

class BatchTests(unittest.TestCase):
 def setUp(self):self.notes=[SimpleNamespace(id='a',number=10),SimpleNamespace(id='b',number=20)]
 def test_reordered_zip_matches_numbers(self):
  with tempfile.TemporaryDirectory() as d:
   items=[(n,'accepted',Path(d)/(n.id+'.pdf'),None) for n in self.notes]
   c=Mock();c.get_attachment.return_value=archive([('20_B.pdf',b'%PDF-b'),('10_A.pdf',b'%PDF-a')])
   self.assertEqual(download_batch(c,items),2)
   c.get_attachment.assert_called_once_with(field_name='pdf',ids=['a','b'])
   self.assertEqual((Path(d)/'a.pdf').read_bytes(),b'%PDF-a')
 def test_incomplete_zip_saves_nothing(self):
  with tempfile.TemporaryDirectory() as d:
   c=Mock();c.get_attachment.return_value=archive([('10_A.pdf',b'%PDF-a')])
   with self.assertRaises(ValueError):download_batch(c,[(n,None,Path(d)/(n.id+'.pdf'),None) for n in self.notes])
   self.assertEqual(list(Path(d).iterdir()),[])
 def test_rejects_bad_pdf_and_duplicate(self):
  for entries in [[('10_A.pdf',b'error'),('20_B.pdf',b'%PDF-b')],[('10_A.pdf',b'%PDF-a'),('10_other.pdf',b'%PDF-b')]]:
   with self.assertRaises(ValueError):unpack_batch(archive(entries),self.notes)
 def test_singleton_and_bounds(self):
  self.assertEqual(unpack_batch(b'%PDF-a',self.notes[:1]),{'a':b'%PDF-a'})
  for items in [[],[(self.notes[0],None,None,None)]*51]:
   with self.assertRaises(ValueError):download_batch(Mock(),items)
 def test_rate_limit_not_retried(self):
  c=Mock();c.get_attachment.side_effect=RuntimeError('RateLimitError 429')
  with self.assertRaises(RuntimeError):download_batch(c,[(n,None,Path(n.id),None) for n in self.notes])
  self.assertEqual(c.get_attachment.call_count,1)
