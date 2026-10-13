import importlib.util
from pathlib import Path
import tempfile
import unittest
from pypdf import PdfWriter

spec = importlib.util.spec_from_file_location('legacy', Path(__file__).resolve().parents[1] / 'run_legacy.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SkipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = runner.q.connect(self.root / 'queue.sqlite3')
        self.db.execute("INSERT INTO papers(id,group_key,batch_no,status) VALUES('paper','iclr/2013',1,'pending')")
        self.db.commit()
        self.paper = self.db.execute('SELECT * FROM papers').fetchone()
        self.pdf = self.root / 'pdfs/iclr/2013/paper.pdf'
        self.pdf.parent.mkdir(parents=True)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_adopts_complete_pdf_without_request(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(self.pdf)
        self.assertTrue(runner.skip_existing(self.db, self.paper, self.root))
        row = self.db.execute('SELECT * FROM papers').fetchone()
        self.assertEqual(row['status'], 'downloaded')
        self.assertEqual(row['pages'], 1)
        self.assertEqual(self.db.execute('SELECT count(*) FROM attempts').fetchone()[0], 0)

    def test_refreshes_stale_pending_row(self):
        self.db.execute("UPDATE papers SET status='downloaded'")
        self.db.commit()
        self.assertTrue(runner.skip_existing(self.db, self.paper, self.root))

    def test_partial_file_does_not_count_as_completed(self):
        self.pdf.write_bytes(b'%PDF-incomplete')
        self.assertFalse(runner.skip_existing(self.db, self.paper, self.root))

    def test_missing_file_remains_pending(self):
        self.assertFalse(runner.skip_existing(self.db, self.paper, self.root))


if __name__ == '__main__':
    unittest.main()
