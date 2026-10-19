"""Numerical checks on the exported observations, independent of rendering."""
from pathlib import Path
import sys,json,csv,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.render import recompute_ruleweight, f6, rw
def main(out):
 out.mkdir(parents=True,exist_ok=True);data=ROOT/'results/published';checks={}
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
  rows=list(csv.DictReader((data/f'{name}.csv').open()));assert len(rows)==n,(name,len(rows));checks[name+'_rows']=len(rows)
  if name=='single_rule':assert all(0<=float(r['retained_weight_ratio'])<=1 for r in rows)
 nip=json.loads((data/'nip.json').read_text());by=collections.defaultdict(list)
 for r in nip:by[(r['gamma'],r['lasso'],r['run'],r['layer_index'])].append(r['modularity'])
 assert len(by)==540
 assert all(np.ptp(v)<1e-12 for v in by.values()),'Legacy threshold records must not change quantile-based NIP'
 checks['nip_unique_run_layers']=len(by);checks['nip_legacy_threshold_duplicates_identical']=True
 metrics=recompute_ruleweight(json.loads((data/'ruleweight_runs.json').read_text()));rw.OUTPUT_ROOT=out
 matrices=f6.figure_matrices(rw.write_summary_tables(metrics));reference=json.loads((ROOT/'docs/reference_statistics.json').read_text())['fig5_matrices']
 diffs=[float(np.max(np.abs(a-np.array(b)))) for a,b in zip(matrices,reference)]
 assert max(diffs)<1e-12;checks['fig5_matrix_max_abs_differences']=diffs
 assert len(f6.paired_trial_rows(metrics))==150
 checks['fig5_paired_trials']=150;checks['status']='PASS'
 (out/'published_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'outputs/published_checks')
