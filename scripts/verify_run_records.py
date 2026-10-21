"""Validate actual records from a fresh full run, never inferred default settings."""
import json
from pathlib import Path
import numpy as np
from itertools import combinations

def verify_profile(profile):
 assert profile['token_extraction_method']=='score_quantile' and profile['token_extraction_fraction']==.75
 assert profile['importance_min_score']==1e-9 and profile['max_elements_per_block'] is None
 assert profile['importance_output_eps']==1e-8
 assert any(b['key'].startswith('output') for b in profile['blocks'])
 for b in profile['blocks']:
  # Quantiles and the strict zero filter apply to raw scores, not normalized ratios.
  scores=np.asarray(b[b['mask_value_name']],float)
  assert scores.shape[1]==int(np.prod(b['shape']))
  assert np.array_equal(b['candidate_flat_indices'],np.arange(scores.shape[1]))
  finite=np.isfinite(scores)
  threshold=np.nanquantile(np.where(finite,scores,np.nan),.75,axis=1)
  mask=finite & (scores>1e-9) & (scores>=threshold[:,None])
  assert np.array_equal(mask,np.asarray(b['important_mask'],bool)),b['key']
  combos=[c for size in range(1,len(scores)+1) for c in combinations(range(len(scores)),size)]
  assert len(b['keep_flat_indices'])==len(combos)
  for combo,indices in zip(combos,b['keep_flat_indices']):
   assert np.array_equal(np.flatnonzero(np.any(mask[list(combo)],axis=0)),indices),(b['key'],combo)
 return len(profile['blocks'])

def verify_run_records(root,groups):
 root=Path(root);models=0;analyses=0;blocks=0
 for group in groups:
  for expected in group['runs']:
   rd=root/'simulation'/group['id']/expected['run']
   actual=json.loads((rd/'config.json').read_text())
   for key in expected:
    assert actual[key]==expected[key],(str(rd),key)
   with np.load(rd/'run_data.npz') as state:
    for key,shape in group['state_shapes'].items():
     assert list(state[key].shape)==shape,(str(rd),key)
   models+=1
 # Verify full per-matrix masks against stored scores, including output matrices.
 for path in sorted((root/'analysis').rglob('analysis.json')):
  result=json.loads(path.read_text());profile=result['weight_combo_subnetwork']
  assert result['subnet_target']=='weight'
  blocks+=verify_profile(profile)
  analyses+=1
 assert models==923 and analyses==443,(models,analyses)
 auxiliary=verify_auxiliary_records(root/'cost_training')
 return {'models_checked':models,'auxiliary_models_checked':auxiliary,'analysis_records_checked':analyses,'weight_masks_checked':blocks,'scope':'Saved training configs, tensor shapes and masks recomputed from stored scores; not an independent re-execution of importance scoring'}

def verify_auxiliary_records(root):
 auxiliary=list(Path(root).glob('*/training/*/run_*/config.json'))
 assert len(auxiliary)==18,len(auxiliary)
 rule_counts=[]
 for path in auxiliary:
  c=json.loads(path.read_text());rule_counts.append(len(c['rule_indices']))
  assert c['optimizer_name'].lower()=='sgd',(str(path),'optimizer_name')
  for key,value in dict(epochs=600,batch_size=128,lr=.05,weight_decay=0).items():
   assert c[key]==value,(str(path),key)
  for key,value in dict(l1=1,gamma=1,regularization_epsilon=1e-8).items():
   assert c['loss_config'][key]==value,(str(path),key)
  assert c['seed']==830+c['repeat_idx'] and 0<=c['repeat_idx']<3
  with np.load(path.parent/'run_data.npz') as state:
   for key,shape in {'modules_list_0_linear1_weight':(300,14),'modules_list_1_linear1_weight':(300,300),'modules_list_2_linear1_weight':(300,300),'output_block_linear1_weight':(7*len(c['rule_indices']),300)}.items():
    assert state[key].shape==shape,(str(path),key)
   assert state['x_test'].shape==(1000,14)
 assert rule_counts.count(5)==3 and rule_counts.count(1)==15
 return len(auxiliary)
