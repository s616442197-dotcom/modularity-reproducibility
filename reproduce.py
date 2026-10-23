"""Independent entry point. --all never reads the bundled published results."""
from pathlib import Path
import argparse, json, os, sys, subprocess, hashlib, datetime
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--from-results',action='store_true');g.add_argument('--all',action='store_true');g.add_argument('--smoke',action='store_true')
 p.add_argument('--output',type=Path);p.add_argument('--skip-tex',action='store_true',help='Deprecated compatibility flag; this code/data repository does not compile manuscripts.');p.add_argument('--device',default='cpu',choices=['cpu','cuda','auto'])
 a=p.parse_args();out=(a.output or ROOT/'outputs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
 if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be empty: '+str(out))
 out.mkdir(parents=True,exist_ok=True)
 if a.smoke:
  from scripts.pipeline import smoke
  smoke(out,a.device);return
 if a.all:
  from scripts.pipeline import full
  data=full(out,a.device)
 else:data=ROOT/'results/published'
 checks_out=out/'checks'
 command=[sys.executable,str(ROOT/'tests/verify_published.py'),str(checks_out),'--data',str(data),'--mode','fresh' if a.all else 'published']
 if a.all:command.extend(['--run-root',str(out)])
 with (out/'data_validation.log').open('w',encoding='utf8') as h:
  subprocess.run(command,stdout=h,stderr=subprocess.STDOUT,check=True)
 checks=json.loads((checks_out/'published_checks.json').read_text())
 from scripts.render import render
 stats=render(data,out/'figures')
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()} if a.from_results else {}
 report={'mode':'from-results' if a.from_results else 'all','status':checks['status'],'execution_status':'PASS','data_validation_status':checks['data_validation_status'],'reproduction_status':checks['reproduction_status'],'reproduction_scope':checks['reproduction_scope'],'checks':'checks/published_checks.json','training_repeated':bool(a.all),'tex_compiled':False,'input_sha256':manifest,'statistics':'figures/statistics.json'}
 (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}))
if __name__=='__main__':main()
