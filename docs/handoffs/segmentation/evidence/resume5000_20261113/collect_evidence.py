import datetime
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import zipfile

root = Path('/root/autodl-tmp/fashion-grounding')
runs = root / 'runs/segmentation'
train_id = 'segmentation-resume5000-train-20261113-r01'
eval_id = 'segmentation-resume5000-ap-20261113-r01'
select_id = 'segmentation-resume5000-selection-20261113-r01'
train, evaluation, selection = [runs / s for s in (train_id, eval_id, select_id)]
read = lambda p: json.loads(p.read_text())
phases = [read(root / 'coordination' / (s + '.json')) for s in (train_id, eval_id)]
assert all(p['status'] == 'completed' for p in phases)
report = read(train/'train_report.json')
assert report['completed_updates'] == 5000 and report['resumed'] and report['start_update'] == 1000
reports = [read(evaluation/f'update_{u:06d}/evaluation_report.json') for u in (2000,3000,5000)]
chosen = read(selection/'selection.json')
rows = [json.loads(line) for line in (train/'metrics.jsonl').read_text().splitlines()]
assert [r['update'] for r in rows] == list(range(1001,5001))
summary = {
    'updates':len(rows), 'image_exposures':sum(len(r['image_ids']) for r in rows),
    'unique_images':len({im for r in rows for im in r['image_ids']}),
    'empty_augmented_images':sum(n == 0 for r in rows for n in r['instance_counts']),
    'skipped_updates':sum(r['optimizer_step_skipped'] for r in rows),
    'grad_norm_before_clip_range':[min(r['grad_norm_before_clip'] for r in rows),max(r['grad_norm_before_clip'] for r in rows)],
    'grad_norm_after_clip_max':max(r['grad_norm_after_clip'] for r in rows),
    'loss_scale_values':sorted({r['loss_scale'] for r in rows}),
    'main_lr_at_updates':{str(u):max(rows[u-1001]['learning_rates']) for u in (1001,2000,3000,5000)},
    'loss_windows':{}
}
for label, window in [('1001_1100',rows[:100]),('1901_2000',rows[900:1000]),('2901_3000',rows[1900:2000]),('4901_5000',rows[3900:4000])]:
    v = {'total':statistics.mean(r['total_loss'] for r in window)}
    for name in ('loss_ce','loss_mask','loss_dice'):
        v[name] = statistics.mean(sum(x for k,x in r['components'].items() if k == name or k.startswith(name+'_')) for r in window)
    summary['loss_windows'][label] = v
post = {
    'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'disk':{str(p):dict(zip(['total','used','free'],shutil.disk_usage(p))) for p in [Path('/'),root]},
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv'],text=True),
    'compute_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True),
    'running_records':[str(p) for p in (root/'coordination').glob('*.json') if read(p).get('status')=='running']
}
artifacts = {}
for base, prefix, names in [
    (train,'train',['train_report.json','resume_check.json','budget_extension.json','data_check.json','resolved_config.json','initialization.json','model_config.yaml']),
    (evaluation,'evaluation',['run_report.json','data_check.json','resolved_config.json']),
    (selection,'',['selection.json'])]:
    for name in names:
        p=base/name
        if p.exists(): artifacts[(prefix+'/' if prefix else '')+name]=p
for u in (2000,3000,5000):
    for name in ('evaluation_report.json','evaluation_report.sha256','coco_summary.txt'):
        artifacts[f'update_{u:06d}/{name}']=evaluation/f'update_{u:06d}'/name
for s in (train_id,eval_id):
    artifacts['coordination/'+s+'.json']=root/'coordination'/(s+'.json')
    artifacts['logs/'+s+'.log']=runs/(s+'.log')
artifacts['targeted_tests.log']=runs/'resume5000-targeted-tests-20261113.log'
artifacts['preflight.json']=runs/'resume5000-preflight-20261113.json'
meta = {
    'record_schema_version':1,'run_id':'segmentation-resume5000-stage-20261113-r01',
    'experiment_id':'S0','owner':'segmentation','executed_by':'lcy','decision_executor':'Codex','status':'completed',
    'git_commit':report['git_commit'],'seed':42,
    'authorization':{'source':'current user request','start_update':1000,'max_training_updates':5000,'checkpoint_updates':[2000,3000,5000],'complete_dev_images_per_checkpoint':250,'full_long_training':False},
    'scope':'Resume valid update1000 to5000 with budget-only extension; full dev at2000/3000/5000; not complete S0',
    'phases':phases,'gpu_lock_total_seconds':sum(p['elapsed_seconds'] for p in phases),
    'environment':{'path':str(root/'envs/segmentation-v1'),'upstream_record':'envs/segmentation/upstream.json','modified':False},
    'freeze_record':'docs/handoffs/segmentation/segmentation_train_dev_freeze_20261112.json',
    'reuse':'Prior input/COCO/LSJ/stability/evaluation evidence reused. Only six new CPU budget-extension tests; actual state restoration is part of authorized continuation, not an extra smoke.',
    'selection_command':[str(root/'envs/segmentation-v1/bin/python'),'-m','fashion_grounding.segmentation.select_checkpoint','--reports',*[str(evaluation/f'update_{u:06d}/evaluation_report.json') for u in (2000,3000,5000)],'--output',str(selection)],
    'server_only_artifacts':{'train/metrics.jsonl':{'source_path':str(train/'metrics.jsonl'),'sha256':hashlib.sha256((train/'metrics.jsonl').read_bytes()).hexdigest(),'bytes':(train/'metrics.jsonl').stat().st_size}},
    'artifacts':{k:{'source_path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in artifacts.items()}
}
restore=read(train/'resume_check.json')
assert restore['expected_digests']==restore['restored_digests'] and restore['resumed_from_update']==1000
summary['sampler_epoch_range']=[rows[0]['sampler_epoch'],rows[-1]['sampler_epoch']]
summary['all_losses_finite']=all(__import__('math').isfinite(r['total_loss']) and all(__import__('math').isfinite(v) for v in r['components'].values()) for r in rows)
meta['parent_run']='segmentation-postfix1000-train-20261112-r01'
meta['status_history']=[{'status':'running','at':phases[0]['started_at']},{'status':'completed','at':post['checked_at']}]
target=runs/'resume5000-evidence-20261113.zip'
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
    for k,v in artifacts.items(): z.write(v,k)
    for k,v in [('run_metadata.json',meta),('metric_summary.json',summary),('postflight.json',post)]: z.writestr(k,json.dumps(v,indent=2)+'\n')
print(json.dumps({'zip':str(target),'zip_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'summary':summary,'train':report,'selection':chosen,'reports':reports,'postflight':post},indent=2))
