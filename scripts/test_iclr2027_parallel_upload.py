import ast,concurrent.futures,gzip,hashlib,json,tempfile,unittest
from pathlib import Path
source=ast.parse(Path('scripts/publish_iclr2027_chunk.py').read_text());node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='publish_object');code=compile(ast.Module(body=[node],type_ignores=[]),'<publisher>','exec')
class Upload(unittest.TestCase):
 def run_objects(self,corrupt=False):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);receipts=root/'receipts';receipts.mkdir();store={};meta={};papers=[]
   sha=lambda b:hashlib.sha256(b).hexdigest()
   for n in range(8):
    id=str(n);pdf=root/(id+'.pdf');pdf.write_bytes(b'%PDF-'+id.encode());blob=root/(id+'.json');d={'text':'text'+id,'pages':[],'rectangles':[],'method':'native','mapping':{},'geometry_version':1};blob.write_text(json.dumps(d));meta[id]={'title':id};papers.append({'forum_id':id,'pdf':str(pdf),'pdf_sha256':sha(pdf.read_bytes()),'text_file':str(blob),'text_sha256':sha(d['text'].encode())})
   def r2(k,data=None,ctype=None):
    if data is not None:store[k]=data;return b''
    return b'bad' if corrupt else store[k]
   env={'ROOT':root,'object_receipts':receipts,'json':json,'gzip':gzip,'sha':sha,'decode':lambda b:json.loads(b),'meta':meta,'r2':r2,'write':lambda p,d:p.write_text(json.dumps(d))}
   exec(code,env)
   with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    if corrupt:
     with self.assertRaises(AssertionError):list(pool.map(env['publish_object'],papers))
     self.assertFalse(list(receipts.glob('*.json')))
    else:
     result=list(pool.map(env['publish_object'],papers));self.assertEqual(len(result),8);self.assertEqual(len(list(receipts.glob('*.json'))),8);self.assertTrue(all(json.loads(p.read_text())['readback_verified'] for p in receipts.glob('*.json')))
 def test_parallel_readbacks(self):self.run_objects()
 def test_corrupt_readback_no_receipts(self):self.run_objects(True)
unittest.main()
