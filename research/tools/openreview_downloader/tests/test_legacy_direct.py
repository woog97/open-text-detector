import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

spec = importlib.util.spec_from_file_location('legacy_direct', Path(__file__).resolve().parents[1] / 'run_legacy_direct.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RecoveryTests(unittest.TestCase):
    def test_legacy_mime_pdf_envelope(self):
        pdf = b'%PDF-1.4\nexample\n%%EOF'
        body = b'--abc123\r\nContent-Disposition: form-data; name="data"; filename="paper.pdf"\r\n\r\n' + pdf + b'\r\n--abc123--\r\n'
        self.assertEqual(runner.pdf_payload(body), pdf)
        self.assertEqual(runner.pdf_payload(pdf), pdf)
        with self.assertRaises(ValueError):
            runner.pdf_payload(body[:-12])
        with self.assertRaises(ValueError):
            runner.pdf_payload(b'<html>Sign in</html>')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = patch.object(runner, 'ROOT', self.root)
        self.patch.start()
        self.db = runner.q.connect(self.root / 'queue.sqlite3')
        self.db.execute("INSERT INTO papers(id,group_key,batch_no,status) VALUES('paper','iclr/2013',1,'pending')")
        self.db.commit()
        self.paper = self.db.execute('SELECT * FROM papers').fetchone()

    def tearDown(self):
        self.db.close()
        self.patch.stop()
        self.tmp.cleanup()

    def fetch(self, session):
        return runner.fetch_with_recovery(self.db, self.paper, session, 'https://example.org/pdf', None, {})

    def test_incomplete_body_and_timeout_retry_with_accounting(self):
        ok = Mock(status_code=200)
        session = Mock()
        session.get.side_effect = [requests.exceptions.ChunkedEncodingError('partial'), requests.exceptions.Timeout(), ok]
        with patch.object(runner, 'wait_or_stop', return_value=True) as wait:
            response, aid = self.fetch(session)
        self.assertIs(response, ok)
        self.assertEqual(aid, 3)
        self.assertEqual([c.args[0] for c in wait.call_args_list], [7, 14])
        self.assertEqual([r[0] for r in self.db.execute('SELECT status FROM attempts ORDER BY id')], ['retryable_error', 'retryable_error', 'inflight'])

    def test_server_error_recovers(self):
        bad = Mock(status_code=503, headers={})
        ok = Mock(status_code=200)
        session = Mock()
        session.get.side_effect = [bad, ok]
        with patch.object(runner, 'wait_or_stop', return_value=True):
            self.assertIs(self.fetch(session)[0], ok)
        bad.close.assert_called_once()

    def test_challenges_and_rate_limits_are_not_retried(self):
        for status in (403, 429):
            with self.subTest(status=status):
                self.db.execute("UPDATE papers SET status='pending'")
                self.db.commit()
                response = Mock(status_code=status)
                session = Mock()
                session.get.return_value = response
                with patch.object(runner, 'wait_or_stop') as wait:
                    self.assertIs(self.fetch(session)[0], response)
                wait.assert_not_called()
                session.get.assert_called_once()

    def test_stop_during_backoff_leaves_paper_pending(self):
        session = Mock()
        session.get.side_effect = requests.exceptions.ConnectionError()
        with patch.object(runner, 'wait_or_stop', return_value=False):
            self.assertIsNone(self.fetch(session))
        self.assertEqual(self.db.execute('SELECT status FROM papers').fetchone()[0], 'pending')


if __name__ == '__main__':
    unittest.main()
