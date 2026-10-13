import tempfile
import unittest
from pathlib import Path
from ingest_oanc import matches, metadata, pairs, oanc_passages, logical_paragraphs
from expand_pool import connect, add, counts


def header(name='letter.txt', date='1999-03-09'):
    return ('<cesHeader xmlns="http://www.xces.org/ns/GrAF/1.0/" date.created="2010-01-01">'
            '<fileDesc><sourceDesc><title>Source title</title><author>Original Author</author>'
            '<pubDate value="' + date + '">display date</pubDate></sourceDesc></fileDesc>'
            '<profileDesc><primaryData loc="' + name + '" medium="text"/></profileDesc></cesHeader>')


class OANCTests(unittest.TestCase):
    def test_original_annotation_offsets_without_blank_lines(self):
        paragraph = 'The letter describes the support of people in this community. ' * 6
        text = paragraph + '\n   ' + paragraph
        second = len(paragraph) + 4
        logical = '<graph xmlns="http://www.xces.org/ns/GrAF/1.0/">'
        for i,(a,b) in enumerate([(0,len(paragraph)),(second,len(text))]):
            logical += '<region xml:id="r%d" anchors="%d %d"/><node xml:id="n%d"><link targets="r%d"/></node><a label="p" ref="n%d"/>' % (i,a,b,i,i,i)
        logical += '</graph>'
        self.assertEqual(logical_paragraphs(text,logical),[(0,len(paragraph)),(second,len(text))])
        results = oanc_passages(text,'annotations-test',3,[9,0,0,0],logical)
        self.assertEqual(len(results),2)
        with self.assertRaises(ValueError): logical_paragraphs(text,logical.replace('0 '+str(len(paragraph)), '0 999999'))

    def test_whitespace_separator_joining_is_exact(self):
        paragraph = 'The letter describes the support of people in this community. ' * 20
        text = paragraph + '\n          \n       ' + paragraph
        results = oanc_passages(text, 'whitespace-test', 1, [0,0,1,0])
        self.assertEqual(len(results), 1)
        a,b,wc,bin_id = results[0]
        self.assertEqual(wc,400); self.assertEqual(bin_id,2)
        self.assertIn('\n          \n',text[a:b])

    def test_exact_component_guard(self):
        self.assertTrue(matches('oanc_icic', 'OANC-GrAF/data/written_1/letters/icic/letter.txt'))
        for name in ['OANC-GrAF/data/spoken/letters/icic/letter.txt',
                     'OANC-GrAF/data/written_1/letters/icic/letter.anc',
                     'OANC-GrAF/data/written_1/letters/icic/../other.txt',
                     'OANC-GrAF/data/written_1/journal/slate/letter.txt']:
            self.assertFalse(matches('oanc_icic', name))

    def test_metadata_uses_publication_not_conversion_date(self):
        m = metadata(header(), 'letter.txt')
        self.assertEqual(m['publication_date'], '1999-03-09')
        self.assertEqual(m['authors'], ['Original Author'])
        with self.assertRaises(ValueError): metadata(header('other.txt'), 'letter.txt')

    def test_nonoverlap_exact_offsets_quarantine_and_tags(self):
        text = '\n\n'.join('The letter discusses the importance of helping people in this community. ' * 6 + str(i) for i in range(8))
        result = list(pairs('oanc_icic', 'OANC-GrAF/data/written_1/letters/icic/letter.txt', text, header(), [20,0,0,0]))
        self.assertEqual(len(result), 3)
        spans = []
        for row, original in result:
            a, b = row['raw_start'], row['raw_end']
            self.assertEqual(row['text'], text[a:b]); self.assertEqual(row['word_count'], len(text[a:b].split()))
            self.assertFalse(any(a < end and b > start for start, end in spans)); spans.append((a,b))
            self.assertEqual(row['source_id'], 'oanc_icic'); self.assertEqual(row['category'], 'professional')
            self.assertFalse(row['training_eligible']); self.assertEqual(original['record']['text'], text)
        self.assertEqual(len({r['parent_document_id'] for r, _ in result}), 1)

    def test_post_cutoff_and_wrong_component_excluded(self):
        name = 'OANC-GrAF/data/written_1/letters/icic/letter.txt'
        text = 'The letter describes this community and its work. ' * 20
        self.assertEqual(list(pairs('oanc_icic', name, text, header(date='2024-01-01'), [9]*4)), [])
        with self.assertRaises(ValueError): list(pairs('oanc_slate', name, text, header(), [9]*4))

    def test_dedup_source_quota_and_resume(self):
        text = 'The letter discusses the importance of helping people in this community. ' * 6
        pair = next(pairs('oanc_icic', 'OANC-GrAF/data/written_1/letters/icic/letter.txt', text, header(), [9,0,0,0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'db.sqlite3'
            db = connect(path)
            with db:
                self.assertTrue(add(db, *pair, 1)); self.assertFalse(add(db, *pair, 1))
            db.close(); db = connect(path)
            with db: self.assertFalse(add(db, *pair, 1))
            self.assertEqual(counts(db), {'oanc_icic': 1}); db.close()


if __name__ == '__main__': unittest.main()
