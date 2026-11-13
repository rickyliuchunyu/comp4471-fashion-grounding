"""Explicit, fingerprinted transitions that extend a training budget only."""
import json
from pathlib import Path
import subprocess

from ..data.schema import sha256
from .training_state import config_sha256, load_checkpoint


BUDGET_KEYS = {'max_updates', 'budget_seconds', 'checkpoint_interval',
               'checkpoint_updates', 'run_scope'}


def validate_extension(state, config, policy, checkpoint_sha256, source_commit,
                       parent_source_tree, target_source_tree):
    """Validate metadata without mutating any restored optimizer/RNG state."""
    old = state['config']
    expected = {
        'schema_version': 1,
        'parent_checkpoint_sha256': checkpoint_sha256,
        'parent_git_commit': state['git_commit'],
        'parent_config_sha256': config_sha256(old),
        'parent_completed_updates': state['completed_updates'],
        'target_config_sha256': config_sha256(config),
        'parent_source_tree': parent_source_tree,
        'target_source_tree': target_source_tree,
    }
    if any(policy.get(k) != v for k, v in expected.items()):
        raise ValueError('Budget extension does not match pinned checkpoint/config/source')
    if state['config_sha256'] != expected['parent_config_sha256']:
        raise ValueError('Parent configuration fingerprint mismatch')
    if old.get('purpose') != 's0_training' or config.get('purpose') != 's0_training':
        raise ValueError('Only S0 budgets can be extended')
    changed = {k: {'before': old.get(k), 'after': config.get(k)}
               for k in old.keys() | config.keys()
               if k not in old or k not in config or old[k] != config[k]}
    if changed.keys() - BUDGET_KEYS:
        raise ValueError('Budget extension cannot change the training recipe')
    if config['max_updates'] <= old['max_updates'] or config['max_updates'] <= state['completed_updates']:
        raise ValueError('Budget extension must increase the update limit')
    if policy.get('allowed_changes') != changed:
        raise ValueError('Budget changes differ from reviewed extension record')
    return dict(**expected, target_git_commit=source_commit, changes=changed,
                reason=policy.get('reason', 'explicit budget extension'))


def load_extended_checkpoint(path, config, data, policy_path, source_commit, repo_root):
    policy = json.loads(Path(policy_path).read_text())
    digest = sha256(path)
    if digest != policy.get('parent_checkpoint_sha256'):
        raise ValueError('Budget extension checkpoint hash mismatch')
    state = load_checkpoint(path, policy['parent_config_sha256'], data)
    parent = state['git_commit']
    # Source transitions are explicitly reviewed and pinned, never globally allowed.
    subprocess.run(['git', 'merge-base', '--is-ancestor', parent, source_commit],
                   cwd=repo_root, check=True, capture_output=True)
    def tree(commit):
        return subprocess.check_output(['git', 'rev-parse', commit+':src/fashion_grounding'],
                                       cwd=repo_root, text=True).strip()
    record = validate_extension(state, config, policy, digest, source_commit,
                                tree(parent), tree(source_commit))
    record['policy_sha256'] = sha256(policy_path)
    return state, record
