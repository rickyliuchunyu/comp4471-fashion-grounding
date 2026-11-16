"""One cached B diagnosis, one 1024 inference and comparison; 900s shared cap."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_min_scale_trial import run_bounded, arm_worker_deadline

ROOT=Path(__file__).resolve().parents[1]


def validate_configs(old,new):
    allowed={'short_edge','max_size','budget_seconds','run_scope'}
    changes={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    if not changes <= allowed or (old['short_edge'],old['max_size'])!=(800,1333) or (new['short_edge'],new['max_size'])!=(1024,1706):
        raise ValueError('Only the fixed single-scale resize may change scientifically')


def commands(plan,out):
    diag=[sys.executable,str(ROOT/'scripts/analyze_resolution_trial.py'),'--plan',str(ROOT/plan['plan_path']),'--output',str(out)]
    dev=[sys.executable,'-m','fashion_grounding.segmentation.evaluate','--package',plan['evaluation_package'],
         '--training-package',plan['training_package'],'--source',plan['upstream_source'],
         '--config',str(ROOT/plan['evaluation_config']),'--checkpoints',plan['checkpoint_path'],
         '--reuse-validated-inputs',str(Path(plan['baseline_directory'])/'evaluation_report.json'),
         '--output',str(out/'dev1024')]
    return diag,dev


def main():
    start=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--gpu-worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();plan=json.loads(Path(args.plan).read_text());out=Path(args.output).resolve()
    if args.gpu_worker:
        if os.environ.get('SEG_RESOLUTION_WORKER')!=plan['run_id']:
            raise RuntimeError('Run via outer deadline and shared GPU lock')
        arm_worker_deadline(float(os.environ['SEG_RESOLUTION_DEADLINE']))
        raise SystemExit(subprocess.call(commands(plan,out)[1],cwd=ROOT))
    from fashion_grounding.data.schema import sha256
    if plan['total_hard_limit_seconds']!=900 or plan['automatic_restart'] is not False:
        raise ValueError('Keep the authorized 900s limit')
    if Path(args.plan).resolve()!=(ROOT/plan['plan_path']).resolve():
        raise ValueError('Wrong plan path')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit source first; do not execute a dirty checkout')
    for path,expected in plan['input_sha256'].items():
        if sha256(ROOT/path)!=expected:
            raise ValueError('Pinned input changed: '+path)
    old=json.loads((ROOT/plan['baseline_config']).read_text())
    new=json.loads((ROOT/plan['evaluation_config']).read_text())
    validate_configs(old,new)
    baseline=Path(plan['baseline_directory'])/'evaluation_report.json'
    if sha256(baseline)!=plan['baseline_report_sha256']:
        raise ValueError('Wrong cached B report')
    report=json.loads(baseline.read_text())
    if report['checkpoint_sha256']!=plan['checkpoint_sha256'] or report['checkpoint_path']!=plan['checkpoint_path']:
        raise ValueError('Wrong B checkpoint')
    if not args.execute:
        print(json.dumps(dict(status='prepared_not_started',cap_seconds=900,inference_passes=1,training_updates=0)))
        return
    if sys.platform!='linux' or shutil.disk_usage(out.parent).free<3*1024**3:
        raise RuntimeError('Need the Linux server and at least 3 GiB free')
    out.mkdir(exist_ok=False)
    deadline=start+900
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             SEG_RESOLUTION_WORKER=plan['run_id'],SEG_RESOLUTION_DEADLINE=str(deadline-10))
    diag,_=commands(plan,out)
    phases=[]
    status=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),hard_limit_seconds=900,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                automatic_restart=False,new_inference_passes_authorized=1,training_updates=0)
    (out/'stage_status.json').write_text(json.dumps(status,indent=2)+'\n')
    def run(name,cmd,gpu=False):
        phase=dict(name=name,command=cmd,status='running',started_at=datetime.now(timezone.utc).isoformat())
        phases.append(phase);(out/'phases.json').write_text(json.dumps(phases,indent=2)+'\n')
        phase_start=time.monotonic()
        phase_env=env if gpu else dict(env,CUDA_VISIBLE_DEVICES='')
        try:
            with (out/(name+'.log')).open('x') as log:
                code=run_bounded(cmd,deadline-5,cwd=ROOT,env=phase_env,stdout=log,stderr=subprocess.STDOUT)
            phase.update(status='completed' if code==0 else 'failed',exit_code=code)
            if code:raise RuntimeError('Phase failed: '+name)
        except BaseException as error:
            phase.update(status='failed',error=type(error).__name__);raise
        finally:
            phase['elapsed_seconds']=time.monotonic()-phase_start
            (out/'phases.json').write_text(json.dumps(phases,indent=2)+'\n')
    try:
        run('diagnostic800',diag+['--phase','baseline'])
        seconds=int(deadline-time.monotonic())-15
        if seconds<=0:raise TimeoutError('No budget left for inference')
        lock=[sys.executable,str(ROOT/'scripts/with_gpu_lock.py'),'--coordination',plan['coordination'],
              '--run-id',plan['run_id'],'--output',str(out),'--seconds',str(seconds),'--',sys.executable,
              str(Path(__file__).resolve()),'--plan',str(ROOT/plan['plan_path']),'--output',str(out),'--gpu-worker']
        run('dev1024',lock,gpu=True)
        run('compare',diag+['--phase','compare'])
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',error=type(error).__name__);raise
    finally:
        status.update(elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
        (out/'stage_status.json').write_text(json.dumps(status,indent=2)+'\n')

if __name__=='__main__':main()
