import unittest
from labels import record, uniform, summarize, project_offsets


class LabelContract(unittest.TestCase):
    def test_three_way_fraction(self):
        text = 'abcdefghij'
        spans = [{'start':0,'end':4,'label':'human'},
                 {'start':4,'end':8,'label':'ai_assisted'},
                 {'start':8,'end':10,'label':'ai_generated'}]
        self.assertEqual(summarize(text, spans)['weighted_ai_fraction'], .4)

    def test_unicode_offsets_preserved(self):
        text = 'A 🧬 café'
        r = uniform('x','test',text,'human')
        self.assertEqual(r['spans'][0]['end'], len(text))
        self.assertNotEqual(len(text), len(text.encode()))

    def test_cross_boundary_token_masked(self):
        text = 'abcdefgh'
        spans = [{'start':0,'end':4,'label':'human'}, {'start':4,'end':8,'label':'ai_assisted'}]
        self.assertEqual(project_offsets(text, spans, [(0,0),(0,3),(3,5),(5,8)]),
                         ([-100,0,-100,1],[False,True,False,True]))

    def test_gap_and_overlap_rejected(self):
        for start in [3,5]:
            with self.assertRaises(AssertionError):
                summarize('abcdefgh', [{'start':0,'end':4,'label':'human'},
                                        {'start':start,'end':8,'label':'ai_generated'}])

    def test_human_and_generated_extremes(self):
        self.assertEqual(uniform('x','test','abc','human')['weighted_ai_fraction'],0)
        self.assertEqual(uniform('x','test','abc','ai_generated')['weighted_ai_fraction'],1)


if __name__ == '__main__':
    unittest.main()
