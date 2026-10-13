import tempfile
import unittest
from pathlib import Path
from ingest_govreport import pairs, check_manifest, REPO, REVISION
from expand_pool import connect, add, counts

class GovReportTests(unittest.TestCase):
    def manifest(self):
        return dict(repo_id=REPO, revision=REVISION, split='train', source_file='document/train-00000-of-00002.parquet', sha256='hash', publisher='https://gov-report-data.github.io/', license_evidence='publisher CC BY 4.0')
    def test_excludes_test_and_validation(self):
        for split in ['test', 'validation']:
            m = self.manifest(); m['source_file'] = 'document/' + split + '-00000-of-00001.parquet'
            with self.assertRaises(ValueError): check_manifest(m)
    def test_long_report_exact_offsets_no_summary_and_source_tags(self):
        text = 'The government report describes the operation of this public program. ' * 500
        results = list(pairs({'report': text, 'summary': 'NEVER USE THIS TARGET'}, 42, self.manifest(), [4,4,4,4]))
        self.assertTrue(results)
        self.assertLessEqual(len(results), 3)
        for r, raw in results:
            self.assertEqual(text[r['raw_start']:r['raw_end']], r['text'])
            self.assertEqual(r['source_id'], 'govreport'); self.assertEqual(r['category'], 'professional')
            self.assertEqual(r['word_count'], len(r['text'].split()))
            self.assertFalse(r['training_eligible']); self.assertNotIn('NEVER USE', r['text'])
            self.assertEqual(raw['record']['metadata']['original_columns']['summary'], 'NEVER USE THIS TARGET')
    def test_dedup_and_quota_survive_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            db = connect(Path(d) / 'test.sqlite3')
            result = next(pairs({'report': 'The report describes a public policy and the people who use it. ' * 50}, 1, self.manifest(), [3,3,3,3]))
            with db:
                self.assertTrue(add(db, *result, 1)); self.assertFalse(add(db, *result, 1))
            self.assertEqual(counts(db), {'govreport': 1}); db.close()

if __name__ == '__main__': unittest.main()
