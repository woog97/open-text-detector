import sys,copy,tempfile,hashlib,unittest
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from iclr2027_cleanup_gates import validate_coverage,authorize_local_path
class Gates(unittest.TestCase):
 def setUp(self):
  self.m={'papers':[{'id':'a','sha256':'abc','bytes':3}]};self.ex={'verified':True,'papers':[{'forum_id':'a'}]};self.hf={'private':True,'readback_verified':True,'ids':['a'],'records':1};self.cf={'readback_verified':True,'website_published':True,'objects':[{'id':'a','pdf_sha256':'abc','pdf_bytes':3}]}
 def test_exact(self):self.assertEqual(validate_coverage(self.m,self.ex,self.hf,self.cf),['a'])
 def test_each_missing_gate(self):
  for target,key in [(self.ex,'verified'),(self.hf,'private'),(self.hf,'readback_verified'),(self.cf,'website_published'),(self.cf,'readback_verified')]:
   target[key]=False
   with self.assertRaises(AssertionError):validate_coverage(self.m,self.ex,self.hf,self.cf)
   target[key]=True
 def test_missing_id(self):
  self.hf['ids']=[]
  with self.assertRaises(AssertionError):validate_coverage(self.m,self.ex,self.hf,self.cf)
 def test_duplicate(self):
  self.m['papers']*=2
  with self.assertRaises(AssertionError):validate_coverage(self.m,self.ex,self.hf,self.cf)
 def test_wrong_hash(self):
  self.cf['objects'][0]['pdf_sha256']='bad'
  with self.assertRaises(AssertionError):validate_coverage(self.m,self.ex,self.hf,self.cf)
 def test_local(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'pdfs';root.mkdir();p=root/'a.pdf';p.write_bytes(b'abc');r={'file':str(p),'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
   self.assertEqual(authorize_local_path(r,root),p)
   p.write_bytes(b'abd')
   with self.assertRaises(AssertionError):authorize_local_path(r,root)
   p.unlink();p.symlink_to(Path(d)/'other')
   with self.assertRaises(AssertionError):authorize_local_path(r,root)
 def test_verify_only_before_mutations(self):
  source=Path('scripts/verify_cleanup_iclr2027_chunk.py').read_text()
  self.assertLess(source.index('if args.verify_only:'),source.index('sqlite3.connect'))
  self.assertLess(source.index('if args.verify_only:'),source.index('path.unlink'))
unittest.main()
