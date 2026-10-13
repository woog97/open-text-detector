import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from ingest_amazon2018 import CapturingReader,PrefixLimit,make_pair,Gate,CATEGORIES
from expand_pool import connect

ARCHIVE={'etag':'revision','last_modified':'','url':'https://example.test/original.gz','retrieved_at':'2026-10-02','prefix_file':'prefix.gz'}

def pair(user='user',product='product',category='Books',rating=3,extra=''):
    row={'reviewText':('I used the product with my family and it was useful for the work we do. '*4)+extra,
         'reviewerID':user,'asin':product,'overall':rating,'unixReviewTime':1514764800,'verified':True}
    line=(json.dumps(row)+'\n').encode()
    return make_pair(row,category,12,line,987,ARCHIVE)

class AmazonTests(unittest.TestCase):
    def test_exact_original_line_text_and_offsets(self):
        record,raw=pair(extra='Unique detail.')
        meta=raw['record']['metadata'];line=meta['original_json_line'].encode()
        self.assertEqual(hashlib.sha256(line).hexdigest(),meta['original_line_sha256'])
        self.assertEqual(meta['uncompressed_line_byte_offset'],987)
        self.assertEqual(raw['record']['text'][record['raw_start']:record['raw_end']],record['text'])
        self.assertEqual(record['category'],'reviews');self.assertFalse(record['training_eligible'])
        self.assertEqual(record['claimed_original_date'],'2018-01-01')

    def test_date_rating_and_short_text_excluded(self):
        row=pair()[1]['record']['metadata']['original_review']
        for edits in [{'overall':6},{'unixReviewTime':1735689600},{'reviewText':'Too short'}, {'reviewerID':''}]:
            candidate={**row,**edits}
            self.assertIsNone(make_pair(candidate,'Books',0,json.dumps(candidate).encode(),0,ARCHIVE))

    def test_captured_prefix_is_exact_compressed_bytes(self):
        encoded=gzip.compress(b'one\ntwo\nthree\n');out=io.BytesIO()
        capture=CapturingReader(io.BytesIO(encoded),out,len(encoded)+1)
        self.assertEqual(gzip.GzipFile(fileobj=capture).read(),b'one\ntwo\nthree\n')
        self.assertEqual(out.getvalue(),encoded)
        self.assertEqual(capture.sha.hexdigest(),hashlib.sha256(encoded).hexdigest())
        bounded=CapturingReader(io.BytesIO(encoded),io.BytesIO(),2)
        self.assertEqual(bounded.read(9),encoded[:2])
        with self.assertRaises(PrefixLimit):bounded.read(1)

    def test_gate_reviewer_product_dedup_caps(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=connect(Path(tmp)/'stage.sqlite3');gate=Gate(db,5306)
            for i in range(3):self.assertTrue(gate.accept(db,pair('user'+str(i),'sameproduct',extra='unique'+str(i)),1))
            self.assertFalse(gate.accept(db,pair('fourth','sameproduct',extra='fourth'),1))
            self.assertFalse(gate.accept(db,pair('user0','anotherproduct',extra='different'),1))
            self.assertEqual(Gate(db,5306).count,3);db.close()

    def test_category_and_rating_slots_reserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=connect(Path(tmp)/'stage.sqlite3');gate=Gate(db,5306)
            gate.categories['Books']=170;gate.ratings[('Books',5)]=170;gate.count=170
            self.assertFalse(gate.accept(db,pair('one','one',rating=5),1))
            self.assertTrue(gate.accept(db,pair('two','two',rating=1),1))
            gate.categories['Books']=gate.maximum
            self.assertFalse(gate.accept(db,pair('three','three',rating=2),2));db.close()

if __name__=='__main__':unittest.main()
