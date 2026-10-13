import base64,gzip,hashlib,json,unittest
from ingest_eff import Excluded,unpack_warc,extract,pairs

class EffTests(unittest.TestCase):
 def html(self):
  text='The people and the public need to understand the rules and the choices before them. '*35
  return ('<html lang="en"><head><meta property="article:published_time" content="2021-01-01T00:00:00Z"><link rel="canonical" href="https://www.eff.org/deeplinks/2021/01/test"></head><body class="long-read-share-links"><h1 class="page-title">Title</h1><div class="pane-eff-author"><div class="byline">By <a href="/about/staff/person">Author</a> and Plain Author</div></div><article class="node--blog--full"><div class="field--name-body"><p>'+text+'</p><blockquote><p>QUOTED EXTERNAL TEXT</p></blockquote><aside><p>DONATE WIDGET</p></aside></div></article></body></html>').encode()
 def index(self,body):return {'url':'https://www.eff.org/deeplinks/2021/01/test?ref=other','digest':base64.b32encode(hashlib.sha1(body).digest()).decode(),'filename':'crawl-data/archive.warc.gz','offset':'100','length':'999'}
 def warc(self,body,date='2021-11-01T00:00:00Z'):
  block=b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n'+body
  head=('WARC/1.0\r\nWARC-Type: response\r\nWARC-Date: '+date+'\r\nWARC-Target-URI: https://www.eff.org/deeplinks/2021/01/test?ref=other\r\nContent-Length: '+str(len(block))+'\r\n\r\n').encode()
  return gzip.compress(head+block+b'\r\n\r\n')
 def test_digest_and_capture_date_verified(self):
  body=self.html();idx=self.index(body);data=self.warc(body);decoded,h=unpack_warc(data,idx);self.assertEqual(decoded,body)
  with self.assertRaises(Excluded):unpack_warc(data,{**idx,'digest':'wrong'})
  with self.assertRaises(Excluded):unpack_warc(self.warc(body,'2022-01-01T00:00:00Z'),idx)
 def test_only_main_original_prose_and_authors(self):
  body=self.html();idx=self.index(body);_,h=unpack_warc(self.warc(body),idx);doc=extract(body,idx,h)
  self.assertNotIn('QUOTED',doc['text']);self.assertNotIn('DONATE',doc['text']);self.assertEqual([a['name'] for a in doc['authors']],['Author','Plain Author'])
  for change in [body.replace(b'lang="en"',b'lang="fr"'),body.replace(b'2021-01-01T',b'2022-01-01T'),body.replace(b'pane-eff-author',b'absent-author'),body.replace(b'The people',b'Originally published elsewhere The people')]:
   with self.assertRaises(Excluded):extract(change,idx,h)
 def test_unchanged_extracted_offsets_html_retention_source_tags(self):
  body=self.html();idx=self.index(body);data=self.warc(body);_,h=unpack_warc(data,idx);doc=extract(body,idx,h)
  results=list(pairs(doc,idx,data,body,h,[0,0,1,0]));self.assertTrue(results)
  for row,raw in results:
   self.assertEqual(doc['text'][row['raw_start']:row['raw_end']],row['text']);self.assertEqual(raw['record']['metadata']['source_html'],body.decode())
   self.assertEqual(row['source_id'],'eff');self.assertEqual(row['category'],'general_web');self.assertFalse(row['training_eligible'])
   self.assertEqual(row['archive_capture_date'],'2021-11-01T00:00:00Z')
if __name__=='__main__':unittest.main()
