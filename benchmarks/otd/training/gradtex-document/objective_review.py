from pathlib import Path
import sys,json,gzip,collections,hashlib
R=Path('/data/workspace/paper-diversity-v1');A=R/'objective-document-control-v1'
sys.path.insert(0,str(A))
import torch
from modeling import loss,training_loss
# Synthetic output-only algebra; not model inference.
torch.manual_seed(39)
out={k:torch.randn(*sh,dtype=torch.bfloat16,requires_grad=True) for k,sh in {'tokens':(3,4,2),'document':(3,2),'segment':(3,15),'mixed':(3,2)}.items()}
b={'labels':torch.tensor([[0,0,1,1],[0,0,0,0],[-100]*4]),'target':torch.tensor([[False,False,True,True],[True]*4,[False]*4]),'segment_labels':torch.tensor([7,0,-100]),'mixed_labels':torch.tensor([1,0,-100]),'sentence_groups':[[([0,1],0),([2,3],1)],[([0,1,2,3],0)],[]],'document_only':torch.tensor([False,False,True]),'document_labels':torch.tensor([1,0,1])}
sub={k:(v[:2] if isinstance(v,torch.Tensor) else v[:2]) for k,v in b.items()}
v,p=training_loss(out,b,2);base,_=loss({k:x[:2] for k,x in out.items()},sub,2);expected=base*(2/3)+.1*torch.nn.functional.cross_entropy(out['document'].float(),b['document_labels'])
assert torch.equal(v,expected)
v.backward();assert not out['tokens'].grad[2].any();assert not out['segment'].grad[2].any();assert not out['mixed'].grad[2].any();assert out['document'].grad[2].abs().sum()>0
# Paper-only path retains original combined-loss value and gradients plus independent doc term.
sub['document_only']=torch.zeros(2,dtype=torch.bool)
pp={k:x[:2].detach().clone().requires_grad_() for k,x in out.items()}
a,_=training_loss(pp,sub,2);base,_=loss(pp,sub,2)
assert torch.equal(a,base+.1*torch.nn.functional.cross_entropy(pp['document'].float(),sub['document_labels']))
first,_=training_loss(pp,sub,1);orig,_=loss(pp,sub,1);assert torch.equal(first,orig)
# Verify ready hashes still agree; no running file changed.
ready=json.loads((A/'READY.json').read_text());assert all(hashlib.sha256((A/n).read_bytes()).hexdigest()==h for n,h in ready['source_code_sha256'].items())
sel=(R/'gradtex-document-objective/objective_selection_followup.py').read_text();assert "sys.path.insert(0,str(R/sys.argv[1]))" in sel
result={'mixed_loss_exact_expected':True,'doc_only_token_segment_mixed_gradients_zero':True,'doc_gradient_nonzero':True,'paper_only_loss_exact_original_plus_doc':True,'stage1_exact_original_loss':True,'ready_code_hashes_unchanged':True,'selection_imports_arm_specific_code':True,'precision':'BF16 synthetic tensors, no model inference','notes':'Paper loss weighted by fraction paper rows; doc CE averaged over known labels. Effective document weight depends on count known paper document labels, which is explicit mask semantics.'}
(R/'gradtex-document-objective/objective-review-checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
