"""Numerical checks on the exported observations, independent of rendering."""
from pathlib import Path
import sys,json,csv,collections,argparse,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.render import recompute_ruleweight, f6, rw
def main(out,data=None,mode='published',run_root=None):
 out.mkdir(parents=True,exist_ok=True);data=(data or ROOT/'results/published').resolve();checks={'data_directory':str(data),'mode':mode}
 groups=json.loads((ROOT/'configs/training_runs.json').read_text());selection=json.loads((data/'selection.json').read_text())
 assert sum(len(g['runs']) for g in groups)==923
 expected=[f'run_{i:03d}' for i in range(30) if i not in [2,11,14,17,20,22,24]]
 assert selection['transformer']==expected
 z=np.load(data/'robustness.npz')
 for name,runs in selection.items():
  a=z[name+'__weight_combo_retained_accuracy'];assert a.shape==(len(runs),5,31) and np.isfinite(a).all()
  assert len(runs)==(23 if name=='transformer' else 30)
 checks['selected_runs']={k:len(v) for k,v in selection.items()}
 for name,n in [('single_rule',4500),('time',1800),('geometry',2100),('correlation_hist',4800),('recomposition',360)]:
  with (data/f'{name}.csv').open() as h:rows=list(csv.DictReader(h))
  assert len(rows)==n,(name,len(rows));checks[name+'_rows']=len(rows)
  if name=='single_rule':assert all(0<=float(r['retained_weight_ratio'])<=1 for r in rows)
 nip=json.loads((data/'nip.json').read_text());by=collections.defaultdict(list)
 for r in nip:by[(r['gamma'],r['lasso'],r['run'],r['layer_index'])].append(r['modularity'])
 assert len(by)==540
 assert all(np.ptp(v)<1e-12 for v in by.values()),'Legacy threshold records must not change quantile-based NIP'
 checks['nip_unique_run_layers']=len(by);checks['nip_legacy_threshold_duplicates_identical']=True
 metrics=recompute_ruleweight(json.loads((data/'ruleweight_runs.json').read_text()));rw.OUTPUT_ROOT=out
 matrices=f6.figure_matrices(rw.write_summary_tables(metrics));reference=json.loads((ROOT/'docs/reference_statistics.json').read_text())['fig5_matrices']
 diffs=[float(np.max(np.abs(a-np.array(b)))) for a,b in zip(matrices,reference)]
 if mode=='published':assert max(diffs)<1e-12,'Published matrices differ from the reference'
 checks['fig5_matrix_max_abs_differences']=diffs
 assert len(f6.paired_trial_rows(metrics))==150
 checks['fig5_paired_trials']=150
 weighted=[g for g in groups if 'ruleweight' in g['id']];assert len(weighted)==6
 for g in weighted:
  assert g['state_shapes']['modules_list_0_linear1_weight']==[300,14]
  assert g['state_shapes']['modules_list_1_linear1_weight']==[300,300]
  assert g['state_shapes']['modules_list_2_linear1_weight']==[300,300]
  assert g['state_shapes']['output_block_linear1_weight']==[35,300]
  assert [r['seed'] for r in g['runs']]==list(range(300,330))
  assert all(r['epochs']==5000 and r['batch_size']==128 and r['optimizer_name']=='adam' and r['lr']==.05 for r in g['runs'])
 assert all(m['pairing']==metrics[0]['pairing'] for m in metrics)
 rows=f6.paired_trial_rows(metrics)
 r=float(np.corrcoef([x['accuracy_delta'] for x in rows],[x['weight_ratio_delta'] for x in rows])[0,1])
 assert np.isfinite(r)
 if mode=='published':assert abs(r-.783506636538771)<1e-12
 ex=json.loads((data/'cost/extraction_runs.json').read_text());assert len(ex)==3
 for row in ex:
  assert row['protocol']=={'target':'weight','method':'ablation','tau':.75,'extraction_method':'score_quantile','min_score':1e-9,'output_eps':1e-8,'candidate_cap':None}
  assert row['extraction_total_scalar_weights']==194700 and row['extraction_batched_forward_calls']==3045
  assert row['extraction_sample_forward_evaluations']==194700000 and row['extraction_sample_forward_proxy']==3045000
 accuracy=np.mean([row['token_retained_relative_accuracy_mean'] for row in ex]);assert np.isfinite(accuracy)
 if mode=='published':assert abs(accuracy-.9527236425015575)<1e-12
 for m in metrics:
  assert m['num_runs']==30 and len(m['pairing'])==30 and len({json.dumps(p,sort_keys=True) for p in m['pairing']})==30
  for key in ['baseline_acc_runs','weight_ratio_runs','retained_runs']:
   assert np.isfinite(m[key]).all(),key
  assert np.all((m['baseline_acc_runs']>=0)&(m['baseline_acc_runs']<=1))
  assert np.all((m['weight_ratio_runs']>=0)&(m['weight_ratio_runs']<=1))
 checks['fresh_provenance_status']='NOT_CHECKED'
 if run_root is not None:
  from scripts.verify_run_records import verify_run_records
  checks['fresh_provenance']=verify_run_records(run_root,groups)
  checks['fresh_provenance_status']='PASS'
 checks.update(fig5_width=300,fig5_runs=180,fig5_pooled_r=r,cost_retained_relative_accuracy=accuracy,cost_scalar_weights=194700,
  status='PASS' if mode=='published' else 'PARTIAL',data_validation_status='PASS',
  reproduction_status='PASS' if mode=='published' else 'NOT_CHECKED',
  reproduction_scope='Supplied observation statistics only; no fresh training claim' if mode=='published' else 'Fresh output structure/configuration checked; scientific conclusion agreement requires review of the recorded reference differences, not bitwise equality')
 checks['input_sha256']={str(p.relative_to(data)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.rglob('*')) if p.is_file()}
 (out/'published_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
 return checks
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path,nargs='?',default=ROOT/'outputs/published_checks');p.add_argument('--data',type=Path);p.add_argument('--mode',choices=['published','fresh'],default='published');p.add_argument('--run-root',type=Path)
 a=p.parse_args();main(a.output,a.data,a.mode,a.run_root)
