"""Validate and print the 10k plan only. There is deliberately no execute mode."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/runs/segmentation/tenk_adaptation_plan.json'


def prepare(plan, root=ROOT, *, authorized=False):
    from fashion_grounding.data.schema import sha256
    from fashion_grounding.segmentation.training_state import read_config, config_sha256
    from fashion_grounding.segmentation.input_receipts import read_evidence_bundle, validate_chain
    from fashion_grounding.segmentation.evaluation import read_eval_config
    if any(plan[k] is not authorized for k in ('training_authorized', 'execution_enabled')):
        raise ValueError('Training authorization does not match requested preparation/execution mode')
    if plan['budget_proposal']['authorized'] is not authorized:
        raise ValueError('Budget authorization does not match requested mode')
    for name, digest in plan['input_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Prepared input changed: '+name)
    config = read_config(root/plan['training_config'])
    bundle, evidence, _ = read_evidence_bundle(root/plan['input_validation_bundle'])
    validate_chain(bundle, evidence)
    if (config['training_authorized'] is not authorized or config['initialization_mode'] != 'model_only'
            or not config['input_validation_bundle_required']
            or bundle['training_data'] != {k:config[k] for k in bundle['training_data']}
            or config['training_images'] != bundle['train_images'] or config['equivalent_epochs'] != 10
            or plan['evaluation_epochs'] != [2,4,6,8,10]):
        raise ValueError('Wrong training/evaluation exposure contract')
    source = json.loads((root/plan['model_only_source']).read_text())
    baseline = json.loads((root/plan['baseline_report']).read_text())
    evaluation = read_eval_config(root/plan['evaluation_config'])
    if (baseline['checkpoint_sha256'] != source['sha256'] or not baseline['complete']
            or not baseline['selection_eligible'] or baseline['inference_errors'] != 0
            or baseline['evaluation_data'] != bundle['evaluation_data']
            or baseline['evaluation_config_sha256'] != config_sha256(evaluation)
            or baseline['training_config_sha256'] != source['source_config_sha256']
            or baseline['training_git_commit'] != source['source_git_commit']
            or baseline['training_data'] != source['source_training_data']):
        raise ValueError('Baseline/source/evaluation identity differs')
    decision = json.loads((root/plan['visibility_decision']).read_text())
    if decision['selected_policy'] != config.get('target_crop') or not decision['preparation_gate_passed']:
        raise ValueError('Training crop lacks a passing diagnostic decision')
    epochs_to_updates = lambda e: e*config['training_images']//config['effective_batch_images']
    if config['training_images'] % config['effective_batch_images']:
        raise ValueError('Epoch boundary must align with successful updates')
    output, python = Path(plan['future_output']), plan['python']
    checkout = Path(plan['execution_checkout'])
    stages, previous = [], None
    for epoch in plan['evaluation_epochs']:
        update = epochs_to_updates(epoch)
        if update not in config['checkpoint_updates']:
            raise ValueError('Evaluation checkpoint is not saved')
        directory = output/f'epoch_{epoch:02d}'
        train = [python,'-m','fashion_grounding.segmentation.train','--package',plan['training_package'],
            '--source',plan['upstream_source'],'--config',str(checkout/plan['training_config']),
            '--input-validation-bundle',str(checkout/plan['input_validation_bundle']),
            '--stop-after',str(update),'--output',str(directory/'train')]
        train += ['--resume',str(previous)] if previous else ['--model-only',str(checkout/plan['model_only_source'])]
        checkpoint = directory/f'train/checkpoint_{update:06d}.pth'
        dev = [python,'-m','fashion_grounding.segmentation.evaluate','--package',plan['evaluation_package'],
            '--training-package',plan['training_package'],'--source',plan['upstream_source'],
            '--config',str(checkout/plan['evaluation_config']),
            '--input-validation-bundle',str(checkout/plan['input_validation_bundle']),
            '--checkpoints',str(checkpoint),'--output',str(directory/'dev')]
        stages.append(dict(equivalent_epoch=epoch, new_run_update=update,
            cumulative_image_presentations=epoch*config['training_images'], train=train, full_dev=dev))
        previous = checkpoint
    return dict(status='prepared_not_started', model_loads=0, training_updates=0, inference_images=0,
        training_config_sha256=config_sha256(config), evaluation_config_sha256=config_sha256(evaluation),
        fresh_start_update=0, source_checkpoint_update=source['source_completed_updates'],
        max_updates=config['max_updates'], max_image_presentations=100000,
        baseline_AP=baseline['metrics']['AP'], stages=stages,
        warning='Use run_tenk_adaptation.py only after explicit launch and budget authorization; never execute these bare commands on a shared GPU.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(PLAN))
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(Path(args.plan).read_text())),indent=2))


if __name__ == '__main__':
    main()
