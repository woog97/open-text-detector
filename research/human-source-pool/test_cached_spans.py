import unittest,tempfile,json,gzip,sys,contextlib,io
from pathlib import Path
from unittest.mock import patch
from recover_cached_spans import prose_regions,select_windows,eligible,main
from expand_pool import connect

P=lambda i:'The guide describes the area and explains how people can travel safely with their family while learning about the local history and the practical choices available to visitors. Section '+str(i)+'.'

class CachedSpanTests(unittest.TestCase):
 def test_duplicate_original_rows_keep_one_provenance(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);audit=root/'audit/pressbooks';audit.mkdir(parents=True);prod=root/'production.sqlite3';connect(prod).close()
   meta={'license':'https://creativecommons.org/licenses/by/4.0/','url':'https://example.test/book/'};text='\n'.join(P(i) for i in range(10))
   with gzip.open(audit/'eligible-originals.jsonl.gz','wt') as stream:
    for i in range(2):stream.write(json.dumps({'record':{'id':'doc'+str(i),'created':'2020-01-01','text':text,'metadata':meta},'source_file':'original.gz','source_row':i})+'\n')
   (audit/'status.json').write_text(json.dumps({'state':'audit_complete','source_dataset':'common-pile/pressbooks','revision':'pinned','files':[{'filename':'original.gz','sha256':'a'*64,'bytes':10}]}))
   with patch.object(sys,'argv',['recover','--source','pressbooks','--base',str(root/'stage'),'--audit-base',str(root/'audit'),'--production-db',str(prod)]),contextlib.redirect_stdout(io.StringIO()):main()
   db=connect(root/'stage/stage.sqlite3');rows=[json.loads(r[0]) for r in db.execute('SELECT row FROM passages')]
   self.assertTrue(rows);self.assertEqual({r['source_row'] for r in rows},{0});self.assertEqual(db.execute('SELECT count(*) FROM documents').fetchone()[0],1);db.close()
 def test_single_newline_paragraphs_and_short_headers_preserved(self):
  text='Introduction\n'+'\nBackground\n'.join(P(i) for i in range(40))
  spans=select_windows(text,'book',3,[0,0,0,3],[]);self.assertTrue(spans);self.assertEqual(spans[0][3],3)
  a,b,words,_=spans[0];self.assertIn('\nBackground\n',text[a:b]);self.assertEqual(words,len(text[a:b].split()))
 def test_occupied_ranges_and_document_cap(self):
  text='\n'.join(P(i) for i in range(40));occupied=[(0,600)]
  spans=select_windows(text,'article',1,[10,10,10,10],occupied);self.assertEqual(len(spans),1)
  a,b,_,_=spans[0];self.assertGreaterEqual(a,600)
 def test_reference_and_table_boundaries_not_bridged(self):
  text=P(1)+'\nTABLE | VALUE | NEXT | VALUE\n'+P(2)+'\nReferences\n'+P(3)
  regions=prose_regions(text);self.assertEqual(len(regions),2)
  self.assertTrue(all('TABLE |' not in text[a:b] and 'References' not in text[a:b] and 'Section 3' not in text[a:b] for a,b in regions))
 def test_all_available_bins_considered(self):
  text='\n'.join(P(i) for i in range(6));spans=select_windows(text,'short',1,[0,1,0,0],[]);self.assertTrue(spans);self.assertEqual(spans[0][3],1)
 def test_original_date_rights_namespace_preserved(self):
  row={'created':'2020-01-01','text':P(1),'metadata':{'namespace':'0','license':'https://creativecommons.org/licenses/by-sa/4.0/'}}
  self.assertTrue(eligible('wikivoyage',row))
  for field,value in [('namespace','1'),('license','all rights reserved'),('language','es')]:
   changed={**row,'metadata':{**row['metadata'],field:value}};self.assertFalse(eligible('wikivoyage',changed))
  self.assertFalse(eligible('wikivoyage',{**row,'created':'2024-01-01'}))
if __name__=='__main__':unittest.main()
