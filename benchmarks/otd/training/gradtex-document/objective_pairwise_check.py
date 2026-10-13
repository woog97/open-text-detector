"""Synthetic BF16-logit checks, no model inference."""
import json
import torch
from objective_pairwise import ranking_loss, pair_indices
z = torch.zeros(4,3,2,dtype=torch.bfloat16,requires_grad=True)
mask=torch.tensor([[True,True,False],[True,True,False],[False]*3,[False]*3])
y=ranking_loss(z,mask,[(0,1)]);y.backward()
assert torch.all(z.grad[0,:2,1]>0) and torch.all(z.grad[1,:2,1]<0)
assert not z.grad[:,2].any() and not z.grad[2:].any()
assert float(z.grad.abs().max())<=.1
better=z.detach().clone();better[1,:2,1]=2
assert ranking_loss(better,mask,[(0,1)])<y
assert ranking_loss(z,mask,[(0,1)],0).item()==0
assert ranking_loss(z,mask,[]).item()==0
assert pair_indices([{'ranking_pair_id':'p','ranking_role':r,'verified_pair_relation':True} for r in ['human','ai']])==[(0,1)]
try:pair_indices([{'ranking_pair_id':'p','ranking_role':'human','verified_pair_relation':True}])
except ValueError:pass
else:raise AssertionError('Incomplete pair accepted')
print(json.dumps({'passed':True,'BF16_synthetic_logits':True,'human_AI_gradient_directions_correct':True,'padding_and_unpaired_gradients_zero':True,'coefficient_zero_exact_zero':True,'complete_pair_required':True,'additional_model_parameters':0,'no_model_inference':True}))
