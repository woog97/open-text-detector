import copy,gzip,hashlib,json,tempfile,unittest
from pathlib import Path
from ingest_cccc import route,make_pair,host,load_evidence,REVISION,download,collect
from unittest.mock import patch
from expand_pool import connect

class CcccTests(unittest.TestCase):
 def setUp(self):
  root=Path(__file__).parent;self.map=json.loads((root/'cccc-domain-map.json').read_text());self.audited=set()
  for line in (root/'evidence/cccc-domain-audit/urls_to_keep.txt').read_text().splitlines():self.audited.update([line,line.removeprefix('www.') if line.startswith('www.') else 'www.'+line])
 def row(self):
  return {'id':'doc1','created':'2019-04-15T00:00:00.000Z','source':'cccc_CC-MAIN-2019-13','metadata':{'warc_url':'https://www.windley.com/archives/2018/04/example.shtml','warc_date':'2019-04-15T00:00:00.000Z','content_type':'text/html'},'text':('The people in the garden have a useful way to think about the tools they use every day. '*7)}
 def test_capture_is_required_and_later_versions_excluded(self):
  r=self.row();self.assertIsNone(route(r,self.map,self.audited)[0]);r['metadata']['warc_date']='2024-01-01T00:00:00Z';self.assertEqual(route(r,self.map,self.audited)[0],'post2021_capture')
  r=self.row();r['created']='2022-01-01T00:00:00Z';self.assertEqual(route(r,self.map,self.audited)[0],'post2021_capture')
  r=self.row();r['metadata']['warc_url']=r['metadata']['warc_url'].replace('2018/04','2021/04');self.assertEqual(route(r,self.map,self.audited)[0],'postdate_after_capture')
 def test_exact_audit_host_and_listing_genre_boundaries(self):
  for url in ['https://www.windley.com.evil.test/2018/04/a','https://www.windley.com/archives/2018/04/','https://www.windley.com/2018/04/a?search=x','https://www.windley.com/news/2018/04/a','https://www.windley.com:bad/2018/04/a']:
   r=self.row();r['metadata']['warc_url']=url;self.assertIsNotNone(route(r,self.map,self.audited)[0],url)
  r=self.row();r['text']='Book review: '+r['text'];self.assertEqual(route(r,self.map,self.audited)[0],'review_or_scientific_genre')
 def test_original_bytes_offsets_and_missing_license_are_honest(self):
  r=self.row();line=(json.dumps(r,ensure_ascii=False)+'\n').encode();reason,pair=make_pair(r,line,'source.json.gz',2,'archivehash',self.map,self.audited,[1,1,1,1]);self.assertIsNone(reason)
  row,raw=pair;self.assertEqual(r['text'][row['raw_start']:row['raw_end']],row['text']);self.assertEqual(raw['upstream_json_line_utf8'].encode(),line);self.assertEqual(raw['upstream_json_line_sha256'],hashlib.sha256(line).hexdigest());self.assertNotIn('license',raw['record']['metadata']);self.assertIsNone(raw['license_evidence']['specific_license_version']);self.assertFalse(row['training_eligible']);self.assertEqual(row['category'],'general_web')
 def test_explicit_comments_do_not_enter_passage(self):
  r=self.row();r['text']+='\n\nComments\n\n'+('The reader made a response to this article and this is all comment text. '*50);line=json.dumps(r).encode();_,pair=make_pair(r,line,'x.gz',0,'sha',self.map,self.audited,[1,1,1,1]);self.assertIsNotNone(pair);self.assertLessEqual(pair[0]['raw_end'],r['text'].index('Comments'));self.assertNotIn('reader made',pair[0]['text'])
 def test_resuming_download_rejects_changed_preserved_archive(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);folder=base/'source-downloads/cccc';folder.mkdir(parents=True);p=folder/'a.json.gz';p.write_bytes(b'original');m={'revision':REVISION,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};p.with_suffix('.gz.manifest.json').write_text(json.dumps(m));self.assertEqual(download(base,p.name)[0],p);p.write_bytes(b'changed')
   with self.assertRaises(AssertionError):download(base,p.name)

 def test_refinement_preserves_old_cursors_and_scans_new_domains_only(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);(base/'evidence').mkdir();mapping=copy.deepcopy(self.map);mapping['canonical_hosts'].append('kauaimark.blogspot.com');(base/'evidence/cccc-domain-map.json').write_text(json.dumps(mapping));folder=base/'source-downloads/cccc';folder.mkdir(parents=True)
   old=self.row();new=self.row();new['id']='new';new['metadata']['warc_url']='https://kauaimark.blogspot.com/2018/04/personal-story.html';new['text']='This is a story about the people in our class and how they learn from their experiences. '*7
   archive=folder/'test.json.gz'
   with gzip.open(archive,'wb') as f:
    for row in [old,new]:f.write((json.dumps(row)+'\n').encode())
   archive.with_suffix('.gz.manifest.json').write_text(json.dumps({'revision':REVISION,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}));(base/'filtered-repo-info.json').write_text(json.dumps({'sha':REVISION,'siblings':[{'rfilename':archive.name}]}))
   d=connect(base/'stage.sqlite3')
   with d:d.execute('INSERT INTO cursors VALUES (?,?,?,?)',('cccc',archive.name,99,1))
   d.close()
   with patch('ingest_cccc.load_evidence',return_value=(mapping,self.audited,{})):
    collect(base,quota=5,new_domains=['kauaimark.blogspot.com'],cache_only=True,cursor_namespace='cccc-refinement-test')
   d=connect(base/'stage.sqlite3');self.assertEqual(d.execute("SELECT position,done FROM cursors WHERE source='cccc'").fetchone(),(99,1));self.assertEqual(d.execute("SELECT position,done FROM cursors WHERE source='cccc-refinement-test'").fetchone(),(2,1));self.assertEqual(d.execute('SELECT count(*) FROM passages').fetchone()[0],1);self.assertIn('kauaimark',d.execute('SELECT row FROM passages').fetchone()[0]);d.close()
 def test_added_domain_route_constraint_excludes_corporate_news(self):
  mapping=copy.deepcopy(self.map);mapping['canonical_hosts'].append('blog.mozilla.org');mapping['required_path_prefixes']={'blog.mozilla.org':['/nnethercote/']};r=self.row();r['metadata']['warc_url']='https://blog.mozilla.org/blog/2018/04/product-news';self.assertEqual(route(r,mapping,self.audited)[0],'outside_reviewed_blog_route');r['metadata']['warc_url']='https://blog.mozilla.org/nnethercote/2018/04/technical-explanation';self.assertIsNone(route(r,mapping,self.audited)[0])

 def test_oversized_paragraph_recovery_preserves_offsets_and_comments_gate(self):
  mapping=copy.deepcopy(self.map);mapping['oversized_paragraph_sentence_spans']=True;r=self.row();r['text']=('The people in the garden have a useful way to think about the tools they use every day. '*110)+'\n\nComments\n'+('The reader comment should never be selected. '*200);line=json.dumps(r).encode();reason,pair=make_pair(r,line,'source.gz',1,'hash',mapping,self.audited,[0,0,0,1]);self.assertIsNone(reason);row,raw=pair;self.assertEqual(row['text'],r['text'][row['raw_start']:row['raw_end']]);self.assertEqual(row['length_bin'],3);self.assertGreaterEqual(row['word_count'],1000);self.assertLessEqual(row['word_count'],1500);self.assertLess(row['raw_end'],r['text'].index('Comments'));self.assertIn('sentence_span',row['extraction_method'])

if __name__=='__main__':unittest.main()
