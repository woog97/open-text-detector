import json,gzip,io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
from expand_pool import add, claimed_date, connect, counts, namespace_ok,scan_source

class ExpansionTests(unittest.TestCase):
    def test_batch_failure_rolls_back_rows_and_cursor_before_retry(self):
        payload=gzip.compress(('\n'.join(json.dumps({'text':'unique '+str(i)}) for i in range(405))+'\n').encode())
        fail_once=[True];retry_counts=[]
        def response(*args,**kwargs):
            if retry_counts:
                db=connect(dbpath)
                self.assertEqual(counts(db),{'source':200})
                self.assertEqual(db.execute('SELECT position,done FROM cursors').fetchone(),(200,0));db.close()
            retry_counts.append(1)
            result=MagicMock();result.__enter__.return_value=result;result.raw=io.BytesIO(payload);return result
        def ingest(db,sid,spec,category,quota,row,filename,index,exclusions):
            result=add(db,{'record_id':str(index),'source_id':sid,'text':row['text'],'parent_document_id':str(index),'length_bin':0,'raw_text_sha256':str(index)},{'record':row},quota)
            if index==250 and fail_once[0]:
                fail_once[0]=False;raise RuntimeError('simulated mid-batch failure')
            return 'accepted' if result else 'duplicate'
        with tempfile.TemporaryDirectory() as tmp,patch('expand_pool.requests.get',side_effect=response),patch('expand_pool.process_row',side_effect=ingest),patch('expand_pool.time.sleep'):
            base=Path(tmp);(base/'progress').mkdir();dbpath=base/'collection.sqlite3'
            scan_source(dbpath,base,'source',{'repo_id':'fixture','revision':'fixed','files':['source.gz']},'creative',500,{})
            db=connect(dbpath);self.assertEqual(counts(db),{'source':405});self.assertEqual(db.execute('SELECT position,done FROM cursors').fetchone(),(405,1));db.close()
            self.assertEqual(len(retry_counts),2)

    def test_quota_stop_can_resume_remaining_shard_after_scaling(self):
        rows=[{'text':'text '+str(i)} for i in range(4)]
        compressed=gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode())
        def response(*args,**kwargs):
            result=MagicMock();result.__enter__.return_value=result;result.raw=io.BytesIO(compressed);return result
        def ingest(db,sid,spec,category,quota,row,filename,index,exclusions):
            record={'record_id':str(index),'source_id':sid,'text':row['text'],'parent_document_id':str(index),'length_bin':0,'raw_text_sha256':str(index)}
            return 'accepted' if add(db,record,{'record':row},quota) else 'duplicate'
        with tempfile.TemporaryDirectory() as tmp,patch('expand_pool.requests.get',side_effect=response),patch('expand_pool.process_row',side_effect=ingest):
            base=Path(tmp);(base/'progress').mkdir();dbpath=base/'collection.sqlite3'
            spec={'repo_id':'fixture','revision':'fixed','files':['source.gz']}
            scan_source(dbpath,base,'source',spec,'creative',1,{})
            db=connect(dbpath);self.assertEqual(db.execute('SELECT position,done FROM cursors').fetchone(),(1,0));db.close()
            scan_source(dbpath,base,'source',spec,'creative',3,{})
            db=connect(dbpath);self.assertEqual(counts(db),{'source':3});self.assertEqual(db.execute('SELECT position,done FROM cursors').fetchone(),(3,0));db.close()
            scan_source(dbpath,base,'source',spec,'creative',5,{})
            db=connect(dbpath);self.assertEqual(counts(db),{'source':4});self.assertEqual(db.execute('SELECT position,done FROM cursors').fetchone(),(4,1));db.close()

    def test_exact_dedup_and_quota_survive_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'collection.db'
            db = connect(path)
            row = {'record_id': 'one', 'source_id': 'source', 'text': 'A human passage',
                   'parent_document_id': 'doc', 'length_bin': 0, 'raw_text_sha256': 'raw'}
            with db:
                self.assertTrue(add(db, row, {'record': {'text': row['text']}}, 1))
                self.assertFalse(add(db, {**row, 'record_id': 'two', 'text': 'Different'}, {}, 1))
                self.assertFalse(add(db, {**row, 'record_id': 'three', 'source_id': 'other', 'text': ' a HUMAN   passage '}, {}, 1))
            db.close()
            db = connect(path)
            self.assertEqual(counts(db), {'source': 1})
            db.close()

    def test_wikipedia_and_talk_are_separate(self):
        talk = {'metadata': {'namespace': '1'}}
        article = {'metadata': {'namespace': '0'}}
        self.assertFalse(namespace_ok('wikipedia', talk))
        self.assertTrue(namespace_ok('wiki_talk', talk))
        self.assertFalse(namespace_ok('wiki_talk', article))
        self.assertTrue(namespace_ok('wikipedia', article))
        self.assertFalse(namespace_ok('wikipedia', {}))

    def test_url_date_is_claimed_not_verified(self):
        row = {'metadata': {'url': 'https://globalvoices.org/2010/04/12/story/'}}
        self.assertEqual(claimed_date('globalvoices', row), ('2010-04-12', 'dated_original_url_unverified_version'))
        row['created'] = '2024-03-01'
        self.assertEqual(claimed_date('globalvoices', row)[0], '2024-03-01')
        self.assertEqual(claimed_date('wikipedia', {'metadata': {'url': '/2010/04/12/'}})[0], '')

if __name__ == '__main__':
    unittest.main()
