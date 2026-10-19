"""Independent entry point. --all never reads the bundled published results."""
from pathlib import Path
import argparse, json, os, sys, shutil, subprocess, hashlib, datetime
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
def compile_tex(out,figures,reference=False):
 target=out/'manuscript';shutil.copytree(ROOT/'manuscript',target,dirs_exist_ok=True)
 if not reference:
  for i in range(1,7):shutil.copy2(figures/f'main_fig{i}.pdf',target/'newfig'/f'main_fig{i}.pdf')
  shutil.copy2(figures/'supp_training_cost.pdf',target/'figs/supp_training_cost.pdf')
 exe=shutil.which('pdflatex');bib=shutil.which('bibtex')
 if not exe or not bib:raise RuntimeError('pdflatex and bibtex are required; install TeX Live or MiKTeX, or explicitly use --skip-tex.')
 # Both documents reference the other's aux file. Bootstrap both before checking.
 for cycle in range(4):
  for stem in ['modularity','sup']:
   with (out/(stem+'_build.log')).open('w' if cycle==0 else 'a',encoding='utf8') as h:
    subprocess.run([exe,'-interaction=nonstopmode','-halt-on-error',stem+'.tex'],cwd=target,stdout=h,stderr=subprocess.STDOUT,check=True)
    if cycle==0 and '\\bibdata' in (target/(stem+'.aux')).read_text(errors='replace'):
     subprocess.run([bib,stem],cwd=target,stdout=h,stderr=subprocess.STDOUT,check=True)
 for stem in ['modularity','sup']:
  content=(target/(stem+'.log')).read_text(errors='replace')
  if 'There were undefined references' in content or 'There were undefined citations' in content:raise RuntimeError('Unresolved references in '+stem)
def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--from-results',action='store_true');g.add_argument('--all',action='store_true');g.add_argument('--smoke',action='store_true');g.add_argument('--compile-reference',action='store_true')
 p.add_argument('--output',type=Path);p.add_argument('--skip-tex',action='store_true');p.add_argument('--device',default='cpu',choices=['cpu','cuda','auto'])
 a=p.parse_args();out=(a.output or ROOT/'outputs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
 if out.exists() and any(out.iterdir()):raise RuntimeError('Output must be empty: '+str(out))
 out.mkdir(parents=True,exist_ok=True)
 if a.smoke:
  from scripts.pipeline import smoke
  smoke(out,a.device);return
 if a.compile_reference:compile_tex(out,None,True);return
 if a.all:
  from scripts.pipeline import full
  data=full(out,a.device)
 else:data=ROOT/'results/published'
 from scripts.render import render
 stats=render(data,out/'figures')
 if not a.skip_tex:compile_tex(out,out/'figures')
 manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()} if a.from_results else {}
 report={'mode':'from-results' if a.from_results else 'all','status':'PASS','training_repeated':bool(a.all),'tex_compiled':not a.skip_tex,'input_sha256':manifest,'statistics':'figures/statistics.json'}
 (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}))
if __name__=='__main__':main()
