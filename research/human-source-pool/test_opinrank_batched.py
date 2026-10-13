import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from collect_pool import BINS
from expand_pool import connect,add
from ingest_opinrank import pair
from ingest_opinrank_batched import prepare,flush

M={'retrieved_at':'2026-10-02','publisher':'https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset','license_evidence':'UCI CC BY4; upstream rights unknown'}

def fixture(index,bin_id):
    count=[90,210,510,1100][bin_id]
    words=('The hotel had a comfortable room and the staff were helpful to us during the visit '.split()*100)[:count]
    words[0]='Review'+str(index)
    row={'text':' '.join(words),'member':'hotels/london/hotel'+str(index),'domain':'hotels','group':'london','entity':'hotel'+str(index),
         'encoding':'utf-8','index':0,'date':'2009','title':'Visit','author':'','original_record':'original record'}
    return pair(row,M)

class BatchedTests(unittest.TestCase):
    def test_same_records_and_originals_as_sequential_across_batches(self):
        quota=17;targets={'0':5,'1':6,'2':4,'3':2}
        rows=[fixture(i,i%4) for i in range(80)]
        rows.insert(10,rows[0])
        with tempfile.TemporaryDirectory() as folder:
            baseline=connect(Path(folder)/'baseline.db');batched=connect(Path(folder)/'batched.db')
            for record,raw in rows:
                with baseline:
                    have=baseline.execute("SELECT count(*) FROM passages WHERE source='opinrank' AND bin=?",(record['length_bin'],)).fetchone()[0]
                    if have<targets[str(record['length_bin'])]:add(baseline,record,raw,quota)
            for start in range(0,len(rows),7):
                flush(batched,[prepare(*r) for r in rows[start:start+7]],quota,targets,'test',start+7)
            self.assertEqual(list(baseline.execute('SELECT id,norm,source,doc,bin,row FROM passages ORDER BY id')),list(batched.execute('SELECT id,norm,source,doc,bin,row FROM passages ORDER BY id')))
            self.assertEqual(batched.execute('SELECT count(*) FROM passages').fetchone()[0],quota)
            def docs(db):return [(key,json.loads(gzip.decompress(blob))) for key,blob in db.execute('SELECT hash,raw FROM documents ORDER BY hash')]
            self.assertEqual(docs(baseline),docs(batched));baseline.close();batched.close()
    def test_atomic_failure_rolls_back_records_and_cursor_then_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            db=connect(Path(folder)/'test.db');rows=[prepare(*fixture(i,i%4)) for i in range(3)]
            # Fail on second passage; first passage+original must not be committed alone.
            failid=rows[1][0]['record_id']
            db.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON passages WHEN NEW.id='"+failid+"' BEGIN SELECT RAISE(ABORT,'injected failure'); END")
            with self.assertRaises(sqlite3.IntegrityError):flush(db,rows,3,{str(i):3 for i in range(4)},'test',3)
            self.assertEqual(db.execute('SELECT count(*) FROM passages').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT count(*) FROM documents').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT count(*) FROM cursors').fetchone()[0],0)
            db.execute('DROP TRIGGER injected_failure');db.commit()
            n,_,_=flush(db,rows,3,{str(i):3 for i in range(4)},'test',3);self.assertEqual(n,3)
            # Replaying the same durable input is harmless and cannot exceed source quota.
            n,_,_=flush(db,rows,3,{str(i):3 for i in range(4)},'test',3);self.assertEqual(n,3)
            self.assertEqual(db.execute('SELECT position FROM cursors').fetchone()[0],3);db.close()

if __name__=='__main__':unittest.main()
