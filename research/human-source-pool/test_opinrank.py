import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from ingest_opinrank import parse_member, diversified_rows, pair
from expand_pool import connect, add, counts

TEXT = 'The hotel room was comfortable and the staff were helpful during our visit. ' * 7
M = {'retrieved_at': '2026-10-02', 'publisher': 'https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset', 'license_evidence': 'UCI CC BY 4.0; upstream rights unknown'}
class OpinrankTests(unittest.TestCase):
    def test_hotel_exact_text_and_encoding(self):
        text = TEXT + 'The café was lovely.'
        line = ('Oct 12 2009\tReview title\t' + text + '\t\n').encode('cp1252')
        row = next(parse_member('hotels/london/entity', line)); r, raw = pair(row, M)
        self.assertEqual(r['text'], text); self.assertEqual(raw['record']['text'][r['raw_start']:r['raw_end']], text)
        self.assertEqual(r['category'], 'reviews'); self.assertEqual(r['source_id'], 'opinrank')
        self.assertEqual(row['encoding'], 'cp1252'); self.assertFalse(r['training_eligible'])
    def test_car_only_review_field_and_no_other_fields(self):
        content = '<DOC><DATE>07/31/2009</DATE><AUTHOR>Name</AUTHOR><TEXT>' + TEXT + '</TEXT><FAVORITE>NEVER INCLUDE FAVORITE</FAVORITE></DOC>'
        row = next(parse_member('cars/2009/entity', content.encode())); r, raw = pair(row, M)
        self.assertEqual(r['text'], TEXT); self.assertNotIn('NEVER', r['text'])
        self.assertIn('NEVER INCLUDE FAVORITE', raw['record']['metadata']['original_record'])
    def test_round_robin_and_dedup(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            for city in ['london', 'beijing']:
                z.writestr('hotels/' + city + '/entity', ('2009\tTitle\t' + TEXT + '\t\n') * 2)
        with zipfile.ZipFile(buffer) as z:
            rows = list(diversified_rows(z))
        self.assertNotEqual(rows[0]['group'], rows[1]['group'])
        with tempfile.TemporaryDirectory() as d:
            db = connect(Path(d) / 'test.sqlite3')
            with db:
                self.assertTrue(add(db, *pair(rows[0], M), 4244))
                self.assertFalse(add(db, *pair(rows[1], M), 4244))
            self.assertEqual(counts(db), {'opinrank': 1}); db.close()
    def test_short_post_cutoff_and_malformed_excluded(self):
        row = next(parse_member('hotels/london/entity', ('2024\tTitle\t' + TEXT + '\t\n').encode()))
        self.assertIsNone(pair(row, M))
        row['date'] = '2009'; row['text'] = 'too short'; self.assertIsNone(pair(row, M))
        self.assertEqual(list(parse_member('hotels/london/entity', b'date\ttitle\ttext\textra unexpected')), [])

if __name__ == '__main__': unittest.main()
