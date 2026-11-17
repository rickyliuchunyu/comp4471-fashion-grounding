#!/usr/bin/env python3
"""Prepare by default; explicit --execute/--resume operate the full100 job only."""
import argparse
import json
from pathlib import Path
import subprocess

from fashion_grounding.segmentation.full100_contract import read_json, require, resolve
from fashion_grounding.segmentation.full100_runtime import atomic_json, execute

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default='configs/runs/segmentation/full100_execution_plan.json')
    parser.add_argument('--binding', help='Actual freeze and consumer evidence binding; templates cannot execute')
    parser.add_argument('--output', required=True, help='New preparation directory or registered full100 run root')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Explicitly continue the same immutable run')
    args = parser.parse_args()
    require(not args.resume or args.execute, '--resume requires --execute')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    plan = Path(args.plan).resolve()
    contract = resolve(ROOT, plan, args.binding, commit)
    if not args.execute:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'preparation.json', contract)
        print(json.dumps(dict(status='prepared', execution_requested=False,
                              ready_for_training=contract['ready_for_training'], remaining=contract['remaining'])))
        return
    require(contract['ready_for_training'], 'Full dataset freeze and acceptance are missing; no worker started')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=ROOT, text=True).strip(), 'Commit runtime source before execution')
    expected = contract['upstream']['source_commit']
    actual = subprocess.check_output(['git', '-C', contract['runtime']['upstream_source'], 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == expected, 'Upstream implementation revision changed')
    require(not subprocess.check_output(['git', '-C', contract['runtime']['upstream_source'], 'status', '--porcelain',
                                        '--untracked-files=no'], text=True).strip(), 'Upstream tracked files changed')
    require(Path(contract['runtime']['python']).is_file(), 'Configured Python interpreter is missing')
    result = execute(contract, args.output, ROOT, resume=args.resume)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
