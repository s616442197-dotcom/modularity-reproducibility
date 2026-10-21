"""Training and analyses from an empty output directory; no historical cache reads."""
from pathlib import Path
import json, csv, sys, inspect
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'example')]
import analysis as ana
from main import train_one_run,save_run
from model import TorchModularMLP,TorchTinyTransformer
from dataset import BinaryPairDataset,SparsePatternDataset,MNISTPairDataset
from task import BinaryArithmeticTask,PatternLogicTask,MNISTArithmeticTask
from loss import WeightedMSELoss
import figure8 as f8,figure10 as f10,ruleweight as rw
PROTOCOL=json.loads((ROOT/'configs/analysis.json').read_text())
def js(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()))
def csvout(p,rows):
 with p.open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def build(g,c,device,smoke=False):
 n=40 if smoke else 5000;bits=7;common=dict(num_samples=n,pattern_size=bits,train_ratio=.5)
 if c['dataset']=='SparsePatternDataset':dataset=SparsePatternDataset(**common,sparsity=.7);task=PatternLogicTask(rule_indices=c['rule_indices'])
 elif c['dataset']=='MNISTPairDataset':dataset=MNISTPairDataset(**common,root=ROOT/'data',download=False,normalize=True,balanced_pairs=True,split='train');task=MNISTArithmeticTask(rule_indices=c['rule_indices'],output_bits=bits)
 else:dataset=BinaryPairDataset(**common,unique_pairs=True);task=BinaryArithmeticTask(rule_indices=c['rule_indices'])
 if c['model']=='TorchTinyTransformer':model=TorchTinyTransformer(model_dim=4 if smoke else 32,num_layers=2,ff_dim=8 if smoke else 64,train_encoder=True,device=device)
 else:
  # Dimensions are verified from stored tensors, not inferred from the experiment name.
  shape=g['state_shapes']['modules_list_0_linear1_weight']
  model=TorchModularMLP(hidden_dim=4 if smoke else shape[0],num_modules=3,device=device)
 return dataset,task,model,WeightedMSELoss(**c['loss_config'])
def train(g,c,base,device,smoke=False):
 dataset,task,model,loss=build(g,c,device,smoke);data=task.prepare(dataset.build(seed=c['seed']));d=base/g['id']/c['run'];cfg=dict(c)
 if smoke:cfg.update(epochs=2,batch_size=8,snapshot_every=1,eval_every=1)
 def snapshot(ep,m,h):save_run(base/g['id']/'snapshots'/f'epoch_{ep:06d}'/c['run'],m,data,h,{**cfg,'snapshot_epoch':ep})
 m,h=train_one_run(model,data,loss,seed=cfg['seed'],epochs=cfg['epochs'],batch_size=cfg['batch_size'],lr=cfg['lr'],optimizer_name=cfg['optimizer_name'],weight_decay=cfg['weight_decay'],snapshot_every=cfg['snapshot_every'],snapshot_callback=snapshot,eval_every=cfg.get('eval_every'),log_every=100)
 save_run(d,m,data,h,cfg);return d
def analyze(rd,output,method,device):
 keys=['importance_min_score','importance_output_eps','max_weight_elements_per_block','perturbation_batch_size','taylor_probes','token_extraction_method','token_extraction_fraction']
 return ana.analyze_run(rd,output_dir=output,mask_method=method,subnet_target='weight',analysis_device=device,correlation_sample_size=5000,**{k:PROTOCOL[k] for k in keys})
def nip(rd,g,l,method,device,epoch=5000,single=False):
 d=ana.load_run_npz(rd);s=d['state'];x=d['x_test'];y=d['y_test'];ps=int(d['pattern_size']);rules=list(map(int,d['rule_indices']));pred,_=ana.forward_from_state(s,x)
 if method=='taylor':scores=ana.taylor_importance_scores(s,x,ps,rules,batch_size=PROTOCOL['perturbation_batch_size'],device='cpu',probes=PROTOCOL['taylor_probes'],seed=PROTOCOL['taylor_seed'])['weights']
 else:
  ctx=ana.torch_ablation_context(s,x,pred,device=device);scores={}
  for spec in ana.weight_block_specs(s):
   k=spec['key'];sim=ana.torch_weight_zero_similarity(s,x,pred,ps,rules,k,np.arange(s[k].size),batch_size=64,device=device,context=ctx)
   if sim is None:raise RuntimeError('Ablation backend failed')
   scores[k]=(1-sim)*ps
 eps=PROTOCOL['fig3_nip_output_eps'] if method=='ablation' else PROTOCOL['importance_output_eps'];energy=ana.rule_output_energy(pred,ps,rules,eps=eps);rows=[];masks={}
 for i,spec in enumerate(ana.weight_block_specs(s)):
  k=spec['key'];score=np.asarray(scores[k]).reshape(len(rules),-1);q=ana.importance_score_tables(score,energy,output_eps=eps);mask=ana.importance_mask_from_tables(score,q,min_score=PROTOCOL['importance_min_score'],token_extraction_method=PROTOCOL['token_extraction_method'],token_extraction_fraction=PROTOCOL['token_extraction_fraction']);masks[k]=mask
  if i<3:rows.append({**ana.importance_weighted_specialization_summary(q,membership_mask=mask),'gamma':g,'lasso':l,'experiment_lasso':l/5,'run':rd.name,'epoch':epoch,'layer_index':i+1,'modularity_metric':'importance_weighted_NIP'})
 singles=[]
 if single:
  baseline=ana.rule_accuracy_vector(pred,y,ps,rules)
  for i in range(5):
   inds={k:np.where(v[i])[0].tolist() for k,v in masks.items()};p,_=ana.forward_from_state(ana.state_with_kept_weights(s,inds),x);acc=ana.rule_accuracy_vector(p,y,ps,rules)/baseline;ratio=sum(len(v) for v in inds.values())/sum(s[k].size for k in inds)
   for j in range(5):singles.append({'gamma':g,'lasso':l,'run':rd.name,'retained_rule_index':i+1,'evaluated_rule_index':j+1,'relative_accuracy':acc[j],'retained_weight_ratio':ratio})
 return rows,singles
def smoke(out,device):
 groups=json.loads((ROOT/'configs/training_runs.json').read_text());checks=[]
 cases=[('mlp300_binary_repeat30','ablation'),('mlp300_pattern_repeat30','ablation'),('mlp_mnist_arithmetic_repeat30','ablation'),('transformer_binary_repeat30','taylor'),('mlp300_binary_ruleweight_control_w1_l5_gamma1_repeat30','ablation'),('mlp300_binary_ruleweight_rule0_w5_l5_gamma1_repeat30','ablation')]
 for name,method in cases:
  g=next(g for g in groups if g['id']==name);rd=train(g,g['runs'][0],out/'simulation',device,True);result=analyze(rd,out/'analysis'/name/rd.name,method,device)
  acc=np.asarray(result['weight_combo_retained_accuracy']);assert acc.shape==(5,31) and np.isfinite(acc).all()
  if name=='mlp300_binary_repeat30':
   rows,singles=nip(rd,1,5,'ablation',device,single=True);assert all(np.isfinite(r['modularity']) for r in rows) and len(singles)==25
   trows,_=nip(rd,1,5,'taylor','cpu');assert len(trows)==3
   geometry,hist=f8.compute_run_rows(rd,1,5,1,2,None,np.linspace(0,1,101),True,'relu');assert len(geometry)==10 and len(hist)==300
   recomp=f10.evaluate_recomposition_run(rd,f10.condition_from_dir(rd.parent,5),result);assert len(recomp)==3 and all(np.isfinite(r['relative_accuracy']) for r in recomp)
   from example.composite import multitask_vs_separate_cost as cost
   from types import SimpleNamespace
   args=SimpleNamespace(mask_method='ablation',importance_min_score=1e-9,token_extraction_method='score_quantile',token_extraction_fraction=.75,importance_output_eps=1e-8,perturbation_batch_size=64,analysis_device=device)
   cr,_=cost.minimal_weight_extraction(rd,out/'cost_smoke',args)
   assert cr['extraction_total_scalar_weights']==228 and cr['extraction_batched_forward_calls']==6 and cr['analysis_samples']==20
   assert cr['protocol']['target']=='weight' and cr['protocol']['candidate_cap'] is None
  if 'ruleweight' in name:
   assert result['weight_combo_subnetwork']['token_extraction_method']=='score_quantile'
  checks.append({'model':name,'epochs':2,'samples':40,'analysis':method,'retained_shape':list(acc.shape),'status':'PASS'})
 js(out/'smoke.json',{'status':'PASS','full_training':False,'checks':checks});print('Smoke training and analysis passed')
def full(out,device):
 # Complete protocol; runtime depends on hardware. All runs start from fresh random initialization.
 groups=json.loads((ROOT/'configs/training_runs.json').read_text());base=out/'simulation';data=out/'results';data.mkdir();anal=out/'analysis';arrays={};selected={};threshold=[];singles=[];times=[];geometry=[];hist=[];recomp=[];weighted=[]
 for group in groups:
  folder=group['id'];first=group['runs'][0];sweep='lgamma' in folder;weight='ruleweight' in folder;g=first['loss_config'].get('gamma',1);l=5*first['loss_config']['l1']
  for config in group['runs']:
   rd=train(group,config,base,device)
   if not sweep and not weight:
    name='transformer' if 'transformer' in folder else 'mnist' if 'mnist' in folder else 'pattern' if 'pattern' in folder else 'binary'
    methods=['taylor'] if name=='transformer' else ['ablation','taylor'] if name=='binary' else ['ablation']
    for method in methods:
     key='taylor_mlp' if name=='binary' and method=='taylor' else name;res=analyze(rd,anal/key/rd.name,method,device);selected.setdefault(key,[]).append(rd.name)
     for field in ['weight_combo_exclusive_counts','weight_combo_retained_accuracy','weight_combo_keep_counts']:arrays.setdefault(key+'__'+field,[]).append(res[field])
   if weight:analyze(rd,anal/folder/rd.name,'ablation',device)
   if sweep:
    if (g==1 and l in [0,1,5,10]) or (l==5 and g in [.5,1,2]):
     rr,ss=nip(rd,g,l,'ablation',device,single=True);threshold+=rr;singles+=ss
    if g==1:
     rr,hh=f8.compute_run_rows(rd,g,l,first['loss_config']['l1'],5000,None,np.linspace(0,1,101),l in [0,1,5,10] and int(rd.name[-3:])<4,'relu');geometry+=rr;hist+=hh
     if l in [0,1,5,10]:
      for ep in [1000,2000,3000,4000,5000]:
       snap=base/folder/'snapshots'/f'epoch_{ep:06d}'/rd.name;rr,_=nip(snap,g,l,'taylor','cpu',ep);times+=rr
      res=analyze(rd,anal/'recomposition'/folder/rd.name,'ablation',device);recomp+=f10.evaluate_recomposition_run(rd,f10.condition_from_dir(base/folder,5),res)
  if weight:
   weights=first['loss_config']['rule_weights'];up=None if len(set(weights))==1 else int(np.argmax(weights));m=rw.extract_condition_metrics(anal/folder,up)
   from scipy.io import loadmat
   m['retained_runs']=np.stack([loadmat(anal/folder/c['run']/'analysis.mat')['weight_combo_retained_accuracy'][:,:5] for c in group['runs']]);weighted.append(m)
 np.savez_compressed(data/'robustness.npz',**{k:np.stack(v) for k,v in arrays.items()});js(data/'selection.json',selected);js(data/'nip.json',threshold);js(data/'ruleweight_runs.json',weighted)
 for name,rows in [('single_rule',singles),('time',times),('geometry',geometry),('correlation_hist',hist),('recomposition',recomp)]:csvout(data/(name+'.csv'),rows)
 import subprocess,shutil
 cmd=[sys.executable,str(ROOT/'example/composite/multitask_vs_separate_cost.py'),'--output-root',str(out/'cost_training'),'--repeat','3','--seed','830','--num-samples','2000','--train-ratio','0.5','--pattern-size','7','--epochs','600','--batch-size','128','--lr','0.05','--optimizer-name','sgd','--hidden-dim','300','--num-modules','3','--l1','1','--single-l1','1','--gamma','1','--device',device,'--weight-decay','0','--regularization-epsilon','1e-8','--mask-method','ablation','--token-extraction-method','score_quantile','--token-extraction-fraction','0.75','--importance-min-score','1e-9','--importance-output-eps','1e-8','--perturbation-batch-size','64','--analysis-device',device]
 subprocess.run(cmd,check=True)
 for name in ['multitask_runs.json','extraction_runs.json','separate_runs.json']:
  p=next((out/'cost_training').rglob(name));(data/'cost').mkdir(exist_ok=True);shutil.copy2(p,data/'cost'/name)
 return data
