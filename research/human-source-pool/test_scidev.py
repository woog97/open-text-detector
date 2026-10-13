import json,unittest,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from html import escape
from ingest_scidev import source_url,current_evidence,matching_regions,pairs,PublisherPacer
TEXT=' '.join('The science report describes how the researchers worked with communities to understand the evidence and its implications for public policy in region '+str(i)+'.' for i in range(55))
URL='https://www.scidev.net/global/health/news/example-article.html'
def wrapper(text=TEXT):return {'source_dataset':'common-pile/cccc_filtered','source_revision':'frozen','source_file':'shard.gz','source_row':17,'archive_sha256':'archivehash','upstream_json_line_sha256':'linehash','record':{'id':'original','text':text,'metadata':{'warc_url':URL,'warc_date':'2020-05-01T00:00:00Z'}}}
def html(author='Jane Writer',notice='We encourage you to republish this article under our creative commons attribution license.'):
 embedded='<h1>A report</h1><h4>By: '+author+'</h4><div id="article-body"><p>'+TEXT+'</p></div>'
 return '<script type="application/ld+json">'+json.dumps({'@type':'WebPage','datePublished':'2020-04-02'})+'</script><div class="republish-popup"><p>'+notice+'</p><textarea class="form-control">'+escape(embedded)+'</textarea></div>'
class SciDevTests(unittest.TestCase):
 def test_shared_rate_limit_and_retry_after_preserved(self):
  with tempfile.TemporaryDirectory() as folder:
   base=Path(folder);(base/'progress').mkdir();p=PublisherPacer(base)
   p.limited(SimpleNamespace(headers={'Retry-After':'600'},status_code=429));self.assertGreater(p.blocked_until,time.time()+599)
   self.assertEqual(PublisherPacer(base).blocked_until,p.blocked_until)
   p.consecutive_limits=3
   with self.assertRaises(RuntimeError):p.before()
 def test_official_historical_news_only(self):
  self.assertEqual(source_url(wrapper()),URL)
  for kind in ('features','editorials','opinions','analysis'):
   w=wrapper();w['record']['metadata']['warc_url']=URL.replace('/news/','/'+kind+'/');self.assertIsNotNone(source_url(w))
  for url in [URL.replace('www.scidev.net','example.org'),URL.replace('/global/','/america-latina/'),URL.replace('/news/','/spotlight/')]:
   w=wrapper();w['record']['metadata']['warc_url']=url;self.assertIsNone(source_url(w))
  w=wrapper();w['record']['metadata']['warc_date']='2023-01-01';self.assertIsNone(source_url(w))
 def test_explicit_notice_byline_and_unspecified_version(self):
  e=current_evidence(html(),URL);self.assertEqual(e['author'],'Jane Writer');self.assertIn('unspecified',e['license_label']);self.assertFalse(e['current_page_is_historical_capture'])
  for page in [html(author='Editor'),html(notice='All site content is licensed under CC BY.'),html(author='Reuters')]:
   with self.assertRaises(ValueError):current_evidence(page,URL)
 def test_overlap_rejects_unrelated_body(self):
  regions,evidence=matching_regions(TEXT,'Unrelated external article about matters with no similarity. '*30);self.assertFalse(regions)
  source='NAVIGATION MENU\n'+TEXT+'\nCOPYRIGHT FOOTER';regions,evidence=matching_regions(source,TEXT);self.assertEqual(len(regions),1)
  a,b=regions[0];self.assertNotIn('NAVIGATION',source[a:b]);self.assertNotIn('COPYRIGHT',source[a:b]);self.assertGreater(evidence['current_body_match_fraction'],.99)
 def test_unchanged_offsets_capped_and_separate_evidence(self):
  w=wrapper('NAV MENU\n'+TEXT+'\nFOOTER');e=current_evidence(html(),URL);rows=list(pairs(w,e,{'url':URL,'retrieved_at':'today','sha256':'currenthash'},[9,9,9,9]));self.assertTrue(rows);self.assertLessEqual(len(rows),3);occupied=[]
  for row,raw in rows:
   a,b=row['raw_start'],row['raw_end'];self.assertEqual(w['record']['text'][a:b],row['text']);self.assertFalse(any(a<y and b>x for x,y in occupied));occupied.append((a,b))
   self.assertEqual(raw['record']['metadata']['historical_capture_wrapper'],w);self.assertEqual(row['source_revision'],'frozen');self.assertFalse(row['training_eligible']);self.assertEqual(row['category'],'news')
if __name__=='__main__':unittest.main()
