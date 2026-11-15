"""Pinned min-scale ablation; retain the six states and the existing AMP repair."""
import json
from pathlib import Path
import subprocess

from ..data.schema import sha256
from .resume_repair import validate_repair
from .training_state import config_sha256, load_checkpoint, read_config


OPERATIONAL_KEYS = {'budget_seconds', 'run_scope', 'checkpoint_updates'}
# A's later candidate/ranking modules are not imported by training or AP. All
# other source changes must be exactly the reviewed resume dispatch/new gate.
ALLOWED_SOURCE_FILES = {
    'src/fashion_grounding/segmentation/train.py',
    'src/fashion_grounding/segmentation/resume_recipe.py',
    'src/fashion_grounding/segmentation/export_candidates.py',
    'src/fashion_grounding/segmentation/pipeline.py',
    'src/fashion_grounding/segmentation/select_threshold.py',
}


def changes(before, after):
    return {k: dict(before=before.get(k), after=after.get(k))
            for k in before.keys() | after.keys()
            if k not in before or k not in after or before[k] != after[k]}


def validate_recipe(state, config, control, repair, policy, checkpoint_sha256,
                    source_commit, parent_tree, target_tree, changed_files):
    """Metadata-only gate; never edit checkpoint contents or restored states."""
    expected = dict(schema_version=1, transition_type='min_scale_ablation_v1',
        parent_checkpoint_sha256=checkpoint_sha256, parent_git_commit=state['git_commit'],
        parent_config_sha256=config_sha256(state['config']),
        parent_completed_updates=state['completed_updates'],
        control_config_sha256=config_sha256(control), target_config_sha256=config_sha256(config),
        parent_source_tree=parent_tree, target_source_tree=target_tree)
    if any(policy.get(k) != v for k, v in expected.items()):
        raise ValueError('Recipe transition differs from pinned checkpoint/config/source')
    # Reuse the audited parent -> A repair, including AMP controls, rather than
    # disguising those necessary historical changes as another experimental factor.
    validate_repair(state, control, repair, checkpoint_sha256, policy['control_git_commit'],
                    parent_tree, repair['target_source_tree'], repair['changed_source_files'])
    diff = changes(control, config)
    if (diff.keys() - OPERATIONAL_KEYS != {'min_scale'}
            or diff['min_scale'] != dict(before=0.1, after=0.5)
            or policy.get('control_to_target_changes') != diff):
        raise ValueError('Only min_scale 0.1 -> 0.5 may change the scientific recipe')
    if state['completed_updates'] != 18000 or config['max_updates'] != 20000:
        raise ValueError('This trial is restricted to update 18000 -> 20000')
    if not 0 < config['budget_seconds'] <= 2100 or config['checkpoint_updates'] != [20000]:
        raise ValueError('Unbounded trial or unexpected save points')
    files = sorted(changed_files)
    if (set(files) - ALLOWED_SOURCE_FILES or files != policy['changed_source_files']
            or 'src/fashion_grounding/segmentation/resume_recipe.py' not in files):
        raise ValueError('Unreviewed source changes relative to control A')
    return dict(**expected, target_git_commit=source_commit,
        control_git_commit=policy['control_git_commit'], control_to_target_changes=diff,
        changed_source_files=files, restored_state_overrides={}, update_budget_extended=False,
        reason=policy['reason'])


def audit_recipe(repo, source_commit, config, policy_path, state=None, checkpoint_sha256=None):
    repo, policy_path = Path(repo), Path(policy_path)
    policy = json.loads(policy_path.read_text())
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()
    control = policy['control_git_commit']
    repair_path = repo/policy['control_repair_policy']
    if sha256(repair_path) != policy['control_repair_policy_sha256']:
        raise ValueError('Historical AMP repair policy changed')
    repair = json.loads(repair_path.read_text())
    if git('rev-parse', control+':src/fashion_grounding') != repair['target_source_tree']:
        raise ValueError('Control source differs from the historical AMP repair')
    for ancestor in (policy['parent_git_commit'], control):
        subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, source_commit],
                       cwd=repo, check=True, capture_output=True)
    if state is None:
        # Preparation checks metadata only; this is NOT evidence of live restore.
        old = read_config(repo/policy['parent_config'])
        state = dict(config=old, config_sha256=config_sha256(old),
                     git_commit=policy['parent_git_commit'], completed_updates=18000)
    record = validate_recipe(state, config, read_config(repo/policy['control_config']), repair,
        policy, checkpoint_sha256 or policy['parent_checkpoint_sha256'], source_commit,
        git('rev-parse', policy['parent_git_commit']+':src/fashion_grounding'),
        git('rev-parse', source_commit+':src/fashion_grounding'),
        git('diff', '--name-only', control, source_commit, '--', 'src/fashion_grounding').splitlines())
    record['policy_sha256'] = sha256(policy_path)
    return record


def load_recipe_checkpoint(path, config, data, policy_path, source_commit, repo_root):
    policy = json.loads(Path(policy_path).read_text())
    digest = sha256(path)
    if digest != policy.get('parent_checkpoint_sha256'):
        raise ValueError('Recipe checkpoint hash mismatch')
    state = load_checkpoint(path, policy['parent_config_sha256'], data)
    record = audit_recipe(repo_root, source_commit, config, policy_path, state, digest)
    return state, record
