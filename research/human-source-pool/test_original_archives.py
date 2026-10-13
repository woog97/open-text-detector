import gzip
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingest_original_archives import SOURCES, check_manifest, imdb_documents, writingprompts_documents, make_pair, collect, file_hash
from expand_pool import connect


def archive(path, files):
    with tarfile.open(path, 'w:gz') as tar:
        for name, text in files.items():
            data = text.encode('utf-8'); info = tarfile.TarInfo(name); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def prose(prefix='first', words=220):
    return prefix + ' ' + ' '.join(('the story and the person in a room with a light'.split() * (words // 12 + 1))[:words])


class OriginalArchiveTests(unittest.TestCase):
    def test_imdb_train_only_balanced_with_url_movie_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'a.tar.gz'
            archive(p, {'aclImdb/train/neg/0_1.txt': prose('negative'),
                        'aclImdb/train/pos/0_9.txt': prose('positive'),
                        'aclImdb/test/neg/0_1.txt': 'NEVER TEST',
                        'aclImdb/train/unsup/0_0.txt': 'NOT SELECTED',
                        'aclImdb/train/urls_neg.txt': 'https://www.imdb.com/title/tt001/usercomments\n',
                        'aclImdb/train/urls_pos.txt': 'https://www.imdb.com/title/tt002/usercomments\n'})
            rows = list(imdb_documents(p))
            self.assertEqual([r['sentiment'] for r in rows], ['neg', 'pos'])
            self.assertEqual([r['family_key'] for r in rows], ['tt001', 'tt002'])
            self.assertTrue(all('train/' in r['source_file'] for r in rows))

    def test_writingprompts_pairs_only_original_train_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp); p = d / 'a.tar.gz'
            archive(p, {'writingPrompts/train.wp_source': 'PROMPT ONE\nPROMPT TWO\n',
                        'writingPrompts/train.wp_target': 'ORIGINAL ONE\nORIGINAL TWO\n',
                        'writingPrompts/test.wp_target': 'NEVER TEST\n',
                        'writingPrompts/generated.wp_target': 'NEVER GENERATED\n'})
            rows = list(writingprompts_documents(p, d / 'cache'))
            self.assertEqual([r['text'] for r in rows], ['ORIGINAL ONE\n', 'ORIGINAL TWO\n'])
            self.assertEqual(rows[1]['prompt'], 'PROMPT TWO\n')
            self.assertEqual(list(writingprompts_documents(p, d / 'cache')), rows)
            archive(p, {'writingPrompts/train.wp_source': 'PROMPT\n', 'writingPrompts/train.wp_target': 'ONE\nTWO\n'})
            with self.assertRaisesRegex(ValueError, 'Mismatched'):
                list(writingprompts_documents(p, d / 'bad-cache'))

    def test_exact_offsets_metadata_and_no_prompt_in_output(self):
        for sid in SOURCES:
            m = {**SOURCES[sid], 'source_id': sid, 'retrieved_at': '2026-10-02'}
            doc = {'text': prose(words=2100) + '<br /><br /> Original ending.', 'source_file': 'train',
                   'source_row': 3, 'original_id': 'id3', 'source_url': 'url',
                   'family_key': 'family', 'family_basis': 'test', 'prompt': 'NEVER INCLUDE THIS PROMPT'}
            record, raw = make_pair(sid, doc, m, [1, 0, 0, 0])
            self.assertEqual(record['text'], doc['text'][record['raw_start']:record['raw_end']])
            self.assertEqual(record['word_count'], len(record['text'].split()))
            self.assertEqual(record['category'], SOURCES[sid]['category'])
            self.assertFalse(record['training_eligible'])
            self.assertEqual(raw['record']['text'], doc['text'])
            self.assertNotIn('NEVER INCLUDE', record['text'])
            self.assertEqual(raw['record']['metadata']['prompt'], doc['prompt'])

    def test_manifest_does_not_allow_wrong_origin_or_revision(self):
        m = {**SOURCES['imdb'], 'source_id': 'imdb'}
        check_manifest('imdb', m)
        for key in ('sha256', 'url', 'source_id'):
            with self.assertRaises(ValueError): check_manifest('imdb', {**m, key: 'wrong'})

    def test_resume_preserves_raw_dedup_quota_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); d = base / 'source-downloads/imdb'; d.mkdir(parents=True)
            (base / 'pipeline').mkdir()
            (base / 'pipeline/sampling-plan.json').write_text(json.dumps({'source_quotas': [{'source_id': 'imdb', 'planned_passages': 5}]}))
            archive(d / 'original.tar.gz', {'aclImdb/train/neg/%s_1.txt' % i: prose('document' + str(i), 220 + i * 100) for i in range(20)})
            sha = file_hash(d / 'original.tar.gz'); spec = {**SOURCES['imdb'], 'sha256': sha}
            (d / 'manifest.json').write_text(json.dumps({**spec, 'source_id': 'imdb', 'retrieved_at': '2026-10-02'}))
            with patch.dict(SOURCES, {'imdb': spec}):
                first = collect(base, 'imdb', max_new=2)
                self.assertEqual(first['count'], 2)
                second = collect(base, 'imdb')
                self.assertEqual(second['count'], 5)
                third = collect(base, 'imdb')
                self.assertEqual(third['new_candidates'], 0)
            db = connect(base / 'collection.sqlite3')
            rows = [json.loads(r[0]) for r in db.execute('SELECT row FROM passages')]
            self.assertEqual(len({r['parent_document_id'] for r in rows}), 5)
            for r in rows:
                blob = db.execute('SELECT raw FROM documents WHERE hash=?', (r['raw_text_sha256'],)).fetchone()[0]
                raw = json.loads(gzip.decompress(blob))
                self.assertEqual(raw['record']['text'][r['raw_start']:r['raw_end']], r['text'])
            db.close()

if __name__ == '__main__': unittest.main()
