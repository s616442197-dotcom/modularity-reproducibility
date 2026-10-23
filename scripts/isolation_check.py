"""Copy the repository to a temporary directory and deny Python file access to source trees.

This is a Python audit-hook check, not a kernel/container security boundary.
"""
from pathlib import Path
import argparse,shutil,sys,tempfile,subprocess,os,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--deny',action='append',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);temp=Path(tempfile.mkdtemp(prefix='modularity-isolation-'));repo=temp/'repository'
 shutil.copytree(ROOT,repo,ignore=shutil.ignore_patterns('.git','outputs','__pycache__','.venv'))
 guard=temp/'guard';guard.mkdir()
 denied=[str(Path(x).resolve()) for x in a.deny]+[str(ROOT)]
 (guard/'sitecustomize.py').write_text('''import sys,os
DENIED='''+repr(denied)+'''
def audit(event,args):
 if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
  path=os.path.abspath(os.fsdecode(args[0]))
  if any(path==root or path.startswith(root+os.sep) for root in DENIED):
   raise PermissionError('Isolation check denied source access: '+path)
sys.addaudithook(audit)
''')
 env=dict(os.environ,PYTHONPATH=os.pathsep.join(filter(None,[str(guard),os.environ.get('PYTHONPATH','')])),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');reports=[]
 commands=[['reproduce.py','--from-results','--output',str(temp/'plots')],['reproduce.py','--smoke','--output',str(temp/'smoke')],['tests/verify_published.py',str(temp/'checks')]]
 for i,args in enumerate(commands):
  with (a.output/f'command_{i}.log').open('w') as h:subprocess.run([sys.executable,str(repo/args[0]),*args[1:]],cwd=temp,env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
  reports.append({'command':args,'status':'PASS'})
 for source,dest in [(temp/'plots/validation.json','render.json'),(temp/'smoke/smoke.json','smoke.json'),(temp/'checks/published_checks.json','published_checks.json')]:shutil.copy2(source,a.output/dest)
 shutil.copytree(temp/'plots/figures',a.output/'figures')
 report={'status':'PASS','mechanism':'Python audit hook, not an OS sandbox','temporary_repository':str(repo),'denied_roots':denied,'commands':reports,'tex_in_isolation':False}
 (a.output/'isolation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
