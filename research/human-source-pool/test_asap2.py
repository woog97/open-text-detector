import unittest
from ingest_asap2 import balanced_rows, make_record

class AsapTests(unittest.TestCase):
    def fixture(self, **changes):
        return {'full_text': 'original wording,  unchanged.\n' * 40, 'essay_id': 'essay-1', 'set': 'train',
                'prompt_name': 'prompt', 'score': '3', 'assignment': 'instructions', '_source_row': '2', **changes}

    def test_original_essay_preserved_without_instruction_or_demographic_injection(self):
        row = self.fixture(gender='example', race_ethnicity='example')
        record, raw = make_record(row, 'revision', 'archive-hash')
        self.assertEqual(record['text'], row['full_text'])
        self.assertEqual(raw['record']['text'][record['raw_start']:record['raw_end']], record['text'])
        self.assertNotIn('instructions', record['text'])
        self.assertFalse(record['training_eligible'])
        self.assertIn('collection_date_unverified', record['reason_codes'])

    def test_official_test_rows_never_selected(self):
        rows = list(balanced_rows([self.fixture(set='test'), self.fixture(essay_id='train')]))
        self.assertEqual([r['essay_id'] for r in rows], ['train'])

    def test_prompt_score_cells_are_interleaved(self):
        rows = [self.fixture(essay_id=str(i)) for i in range(5)] + [self.fixture(essay_id='other', score='5')]
        first = list(balanced_rows(rows))[:2]
        self.assertEqual({r['score'] for r in first}, {'3', '5'})

    def test_protected_evaluation_prompt_families_never_enter_new_intake(self):
        rows = [self.fixture(prompt_name='  Driverless   CARS ', essay_id='protected-1'),
                self.fixture(prompt_name='Facial action coding system', essay_id='protected-2'),
                self.fixture(prompt_name='Allowed prompt', essay_id='allowed')]
        self.assertEqual([r['essay_id'] for r in balanced_rows(rows)], ['allowed'])
        for row in rows[:2]:
            self.assertIsNone(make_record(row, 'revision', 'archive-hash'))

if __name__ == '__main__':
    unittest.main()
