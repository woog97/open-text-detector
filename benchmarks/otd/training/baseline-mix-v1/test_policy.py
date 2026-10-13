import unittest
from policy import cap_groups,blocked_groups,validate,TOTALS
class PolicyTests(unittest.TestCase):
 def test_existing_split_unchanged(self):
  pools={'human':[{'id':str(i),'group':str(i),'paper_id':str(i)} for i in range(80)]}
  self.assertEqual(cap_groups(pools,set()),set())
 def test_connected_family_exclusion(self):
  pools={'fullpapers':[{'group':'connected','paper_id':'reserved'}],'papers':[{'group':'connected','paper_id':'other'}]}
  self.assertEqual(blocked_groups(pools,{'reserved'}),{'connected'})
 def test_hard_cap_whole_families(self):
  old=TOTALS['human'];TOTALS['human']=10
  try:
   pools={'human':[{'group':str(i//3)} for i in range(10)]};blocked=cap_groups(pools,set());self.assertLessEqual(sum(r['group'] not in blocked for r in pools['human']),8)
  finally:TOTALS['human']=old
 def test_training_schedule_rejects_heldout(self):
  pools={s:[{'id':s,'group':s}] for s in TOTALS}
  with self.assertRaises(AssertionError):validate(pools,{'stage2-epoch0':[{'id':'heldout','group':'heldout','dataset':'human'}]},set())
if __name__=='__main__':unittest.main()
