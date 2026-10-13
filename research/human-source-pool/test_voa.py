import copy,unittest
from ingest_voa import extract,pairs,Excluded
class TestVOA(unittest.TestCase):
 def sample(self):
  return dict(content_type='article',site_language='eng',predicted_language='eng',url='https://www.voanews.com/a/example/123.html',authors=['VOA News'],time_retrieved='2021-07-01T00:00:00',time_published=None,time_modified=None,title='Example',paragraphs=['The people in the city said that the changes would help them to work with their neighbors. '*20])
 def test_strict_dates_and_attribution(self):
  for field,value in [('time_retrieved',None),('time_retrieved','2022-01-01'),('time_modified','2022-01-01'),('authors',[]),('authors',['VOA News','Reuters']),('authors',['Unverified Author']),('site_language','fra')]:
   d=self.sample();d[field]=value
   with self.assertRaises(Excluded):extract(d)
 def test_verified_staff(self):
  d=self.sample();d['authors']=['Named Reporter'];e={'name':'Named Reporter','url':'https://www.voanews.com/author/named-reporter/x'}
  with self.assertRaises(Excluded):extract(d)
  doc=extract(d,{'named reporter':e});self.assertEqual(doc['byline_evidence'],[e])
  d['authors'].append('Unknown Reporter')
  with self.assertRaises(Excluded):extract(d,{'named reporter':e})
 def test_wires_and_republication(self):
  for text in ['Reuters contributed.','Reporting from the Associated Press.','AFP supplied this report.','AP contributed.','Originally published elsewhere.']:
   d=self.sample();d['paragraphs'].append(text)
   with self.assertRaises(Excluded):extract(d)
 def test_faithful_offsets_and_tags(self):
  d=self.sample();doc=extract(d);self.assertEqual(doc['text'],'\n\n'.join(d['paragraphs']));rows=list(pairs(d,doc,'article/x.json',[0,10,0,0]));self.assertTrue(rows)
  for row,raw in rows:
   self.assertEqual(row['text'],raw['record']['text'][row['raw_start']:row['raw_end']]);self.assertIsInstance(row['source_row'],int);self.assertEqual(row['source_id'],'voa');self.assertEqual(row['category'],'news');self.assertFalse(row['training_eligible']);self.assertEqual(raw['record']['metadata']['original_mot_record'],d)
if __name__=='__main__':unittest.main()
