"""Explicit source-fix recovery with an unchanged S0 update budget and recipe."""
import json
from pathlib import Path
import subprocess

from ..data.schema import sha256
from .amp_recovery import validate_policy
from .training_state import config_sha256, load_checkpoint


REPAIR_KEYS = {'amp_recovery', 'budget_seconds', 'checkpoint_updates', 'run_scope'}
REPAIR_FILES = {
    'src/fashion_grounding/segmentation/amp_recovery.py',
    'src/fashion_grounding/segmentation/resume_repair.py',
    'src/fashion_grounding/segmentation/train.py',
    'src/fashion_grounding/segmentation/training_state.py',
}


def validate_repair(state, config, policy, checkpoint_sha256, source_commit,
                    parent_source_tree, target_source_tree, changed_source_files):
    old = state['config']
    expected = dict(schema_version=1, transition_type='amp_recovery_same_budget_v1',
                    parent_checkpoint_sha256=checkpoint_sha256,
                    parent_git_commit=state['git_commit'],
                    parent_config_sha256=config_sha256(old),
                    parent_completed_updates=state['completed_updates'],
                    target_config_sha256=config_sha256(config),
                    parent_source_tree=parent_source_tree, target_source_tree=target_source_tree)
    if any(policy.get(k) != v for k, v in expected.items()):
        raise ValueError('Repair does not match pinned checkpoint/config/source')
    if state['config_sha256'] != expected['parent_config_sha256']:
        raise ValueError('Parent configuration fingerprint mismatch')
    if old.get('purpose') != 's0_training' or config.get('purpose') != 's0_training':
        raise ValueError('Only S0 source-fix recovery is allowed')
    if config['max_updates'] != old['max_updates'] or not 0 < state['completed_updates'] < config['max_updates']:
        raise ValueError('Repair must preserve the update limit')
    changes = {k: {'before': old.get(k), 'after': config.get(k)}
               for k in old.keys() | config.keys()
               if k not in old or k not in config or old[k] != config[k]}
    if changes.keys() - REPAIR_KEYS or policy.get('allowed_changes') != changes:
        raise ValueError('Repair cannot change the training recipe beyond the pinned AMP policy')
    validate_policy(config.get('amp_recovery'))
    if config.get('amp') is not True or not config.get('amp_recovery'):
        raise ValueError('Repair requires enabled bounded AMP recovery')
    files = sorted(changed_source_files)
    if not files or set(files) - REPAIR_FILES or policy.get('changed_source_files') != files:
        raise ValueError('Unreviewed source changes outside AMP recovery')
    return dict(**expected, target_git_commit=source_commit, changes=changes,
                changed_source_files=files, reason=policy['reason'],
                restored_state_overrides={}, update_budget_extended=False)


def load_repaired_checkpoint(path, config, data, policy_path, source_commit, repo_root):
    policy = json.loads(Path(policy_path).read_text())
    digest = sha256(path)
    if digest != policy.get('parent_checkpoint_sha256'):
        raise ValueError('Repair checkpoint hash mismatch')
    state = load_checkpoint(path, policy['parent_config_sha256'], data)
    parent = state['git_commit']
    subprocess.run(['git', 'merge-base', '--is-ancestor', parent, source_commit],
                   cwd=repo_root, check=True, capture_output=True)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo_root, text=True).strip()
    record = validate_repair(state, config, policy, digest, source_commit,
        git('rev-parse', parent+':src/fashion_grounding'),
        git('rev-parse', source_commit+':src/fashion_grounding'),
        git('diff', '--name-only', parent, source_commit, '--', 'src/fashion_grounding').splitlines())
    record['policy_sha256'] = sha256(policy_path)
    return state, record
