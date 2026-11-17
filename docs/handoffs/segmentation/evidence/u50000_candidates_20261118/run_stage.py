"""One 50k dev250 pass, three offline exports, and selection; total cap 900s.

Reuse tested GPU lock, process-group deadline, cache, consumer and selector.
No retries, training, AP reruns, language ranking or held-out test access.
"""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO/'scripts'))
from run_min_scale_trial import run_bounded, arm_worker_deadline
from fashion_grounding.data.schema import read_json, write_json, sha256, require

ROOT = Path('/root/autodl-tmp/fashion-grounding')
RUN_ID = 'segmentation-u50000-dev-candidates-20261118-r01'
OUT = ROOT/'runs/segmentation'/RUN_ID
INPUTS = OUT.with_name(RUN_ID+'-inputs')
CONFIG = REPO/'configs/runs/segmentation/candidate_u50000_export.json'
EVIDENCE = Path(__file__).resolve().parent


def commands():
    final = read_json(REPO/'docs/handoffs/segmentation/evidence/tenk_completion_20261118/final_checkpoint.json')
    return [
        ('cache', [sys.executable, '-m', 'fashion_grounding.segmentation.export_candidates', 'cache',
            '--config', str(CONFIG), '--split', 'dev', '--package', str(ROOT/'shared/processed/fp_dev250_v1'),
            '--checkpoint', final['checkpoint_path'], '--source', str(ROOT/'vendor/segmentation/Mask2Former'),
            '--run-id', RUN_ID, '--output', str(OUT/'cache'), '--budget-seconds', '600']),
        ('offline', [sys.executable, str(EVIDENCE/'export_and_accept.py'), str(OUT), str(REPO)]),
        ('selection', [sys.executable, '-m', 'fashion_grounding.segmentation.select_threshold',
            '--config', str(OUT/'coverage_config.json'), '--freeze', str(INPUTS/'freeze.json'),
            '--output', str(OUT/'selection'), '--budget-seconds', '180', '--user-authorized-dev']),
    ]


def coverage_config():
    base = read_json(REPO/'configs/runs/segmentation/candidate_u18000_coverage.json')
    base.update(checkpoint_sha256=read_json(CONFIG)['checkpoint_sha256'],
                source_cache_sha256=sha256(OUT/'cache/cache.json'))
    base['packages'] = [dict(threshold=t, path=str(OUT/f'trial-{t}/candidates_predicted.json'),
        sha256=sha256(OUT/f'trial-{t}/candidates_predicted.json')) for t in (.3, .5, .7)]
    write_json(OUT/'coverage_config.json', base)


def publish():
    selected = read_json(OUT/'selection/selection.json')
    require(selected['status'] == 'selected_for_user_authorized_development', 'Selection incomplete')
    offline = read_json(OUT/'offline_receipt.json')
    require(offline['status'] == 'completed', 'Consumer validation incomplete')
    dest = ROOT/'shared/candidates/u50000-dev250-20261118-r01'
    require(not dest.exists(), 'Never overwrite a published candidate version')
    dest.mkdir()
    targets = {'cache': OUT/'cache', 'selection': OUT/'selection', 'language_dev_inputs': INPUTS,
               'selected_candidates': Path(selected['selected_package']['path']).parent}
    targets.update({f'trial-{t}': OUT/f'trial-{t}' for t in (.3, .5, .7)})
    for name, path in targets.items():
        (dest/name).symlink_to(path, target_is_directory=True)
    record = dict(status='available_fixed_for_development', published_at=datetime.now(timezone.utc).isoformat(),
        run_id=RUN_ID, source_run_directory=str(OUT), server_aliases=True,
        storage_policy='Absolute symlinks to immutable run/input directories; preserve source directories and copy resolved contents across machines.',
        checkpoint_sha256=selected['checkpoint_sha256'], completed_updates=50000,
        source_cache_sha256=selected['source_cache_sha256'], selected_threshold=selected['selected_threshold'],
        selection_reason=selected['selection_reason'], selection_sha256=sha256(OUT/'selection/selection.json'),
        selected_package=selected['selected_package'], selected_entry=str(dest/'selected_candidates/candidates_predicted.json'),
        selected_candidate_manifest_sha256=selected['selected_package']['sha256'],
        queries_sha256=selected['queries_sha256'], query_records=642, semantic_groups=321, query_images=246,
        evaluation_basis=selected['evaluation_basis'], upstream_review_status='642 accepted, 9 pending semantic groups excluded by upstream',
        consumer_verification='lcy executed pinned wyz 103dcc5 reader/crop contract on new artifacts; teammate personal sign-off not claimed',
        packages=[{k:p[k] for k in ('threshold','path','images','candidates','no_candidates_images','candidate_manifest_sha256')} for p in offline['packages']],
        reuse_note='Old 18k packages and selections remain immutable. New candidate_set_id/hash requires new downstream feature namespace.',
        scope='dev250 candidates and offline coverage selection only; no ranking, train/test candidates or training')
    write_json(OUT/'delivery_manifest.json', record)
    shutil.copy2(OUT/'delivery_manifest.json', dest/'delivery_manifest.json')


def worker(deadline):
    require(os.environ.get('SEG_U50000_WORKER') == RUN_ID, 'Use the bounded shared-lock launcher')
    arm_worker_deadline(deadline)
    phases = []
    for name, command in commands():
        if name == 'selection':
            coverage_config()
        phase = dict(name=name, command=command, status='running', started_at=datetime.now(timezone.utc).isoformat())
        phases.append(phase)
        write_json(OUT/'phases.json', phases)
        started = time.monotonic()
        with (OUT/(name+'.log')).open('x') as stream:
            code = subprocess.call(command, cwd=REPO,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0' if name == 'cache' else ''),
                stdout=stream, stderr=subprocess.STDOUT)
        phase.update(status='completed' if code == 0 else 'failed', exit_code=code,
                     elapsed_seconds=time.monotonic()-started)
        write_json(OUT/'phases.json', phases)
        require(code == 0, 'Phase failed: '+name)
    publish()


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.deadline)
        return
    if not args.execute:
        print(json.dumps(dict(status='prepared_not_started', total_hard_limit_seconds=900,
            new_inference_passes=1, optimizer_updates=0, commands=commands())))
        return
    require(sys.platform == 'linux', 'Run on the authorized server')
    require(not subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True).strip(), 'Commit source before execution')
    require(read_json(INPUTS/'server_preflight.json')['status'] == 'passed', 'New input preparation missing')
    OUT.mkdir(exist_ok=False)
    shutil.copy2(INPUTS/'grounding_inputs.py', OUT/'grounding_inputs.py')
    deadline = started+900
    record = dict(status='running', run_id=RUN_ID, started_at=datetime.now(timezone.utc).isoformat(),
        hard_limit_seconds=900, worker_deadline_monotonic=deadline-10,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        automatic_retries=0, automatic_extension=False, optimizer_updates=0, new_inference_passes_authorized=1)
    command = [sys.executable, str(REPO/'scripts/with_gpu_lock.py'), '--coordination', str(ROOT/'coordination'),
        '--run-id', RUN_ID, '--output', str(OUT), '--seconds', str(max(1,int(deadline-time.monotonic())-5)), '--',
        sys.executable, str(Path(__file__).resolve()), '--worker', '--deadline', str(deadline-10)]
    record['command'] = command
    write_json(OUT/'stage_receipt.json', record)
    env = dict(os.environ, PYTHONPATH=str(REPO/'src'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', SEG_U50000_WORKER=RUN_ID)
    try:
        with (OUT/'stage.log').open('x') as stream:
            code = run_bounded(command, deadline, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(exit_code=code, status='completed' if code == 0 else 'failed')
        require(code == 0, 'Candidate stage failed; no automatic retry')
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(OUT/'stage_receipt.json', record)


if __name__ == '__main__':
    main()
