"""New receipt composition and bounded paired-launch cases; no model operations."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_expansion_control as runner
from fashion_grounding.data.schema import sha256
from fashion_grounding.segmentation.evaluation import choose_checkpoint, reuse_eval_package
from fashion_grounding.segmentation.input_receipts import reuse_input_receipts


def dump(path, value):
    path.write_text(json.dumps(value))


class ReceiptFixture:
    def __init__(self, root):
        self.root = root
        self.dev, self.old, self.new = [root/k for k in ('dev', 'old', 'new')]
        for i, directory in enumerate((self.dev, self.old, self.new)):
            directory.mkdir()
            dump(directory/'image_manifest.json', dict(schema_version=3, illustrative_only=False,
                 images=[dict(split='dev' if i == 0 else 'train', image_id=10*i+j) for j in range(2 if i == 2 else 1)]))
            dump(directory/'annotations_coco.json', dict(images=[dict(id=10*i+j) for j in range(2 if i == 2 else 1)],
                 annotations=[dict(id=10*i+j) for j in range(2 if i == 2 else 1)]))
        dump(self.new/'SHA256SUMS.json', {})
        ident = lambda p: dict(image_manifest_sha256=sha256(p/'image_manifest.json'), annotation_sha256=sha256(p/'annotations_coco.json'))
        self.report = json.loads((ROOT/'docs/handoffs/segmentation/evidence/min_scale_trial_20261115/dev/update_020000/evaluation_report.json').read_text())
        self.report.update(evaluation_data=ident(self.dev), training_data=ident(self.old), evaluated_images=1)
        self.receipt = json.loads((ROOT/'docs/handoffs/segmentation/evidence/train10000_acceptance_20261116/acceptance.json').read_text())
        self.receipt['package_pins'] = {'image_manifest.json': ident(self.new)['image_manifest_sha256'],
                                      'annotations_coco.json': ident(self.new)['annotation_sha256'],
                                      'SHA256SUMS.json': sha256(self.new/'SHA256SUMS.json')}
        self.receipt['protected_manifest_pins'].update(parent=ident(self.old)['image_manifest_sha256'], dev=ident(self.dev)['image_manifest_sha256'])
        self.receipt.update(total_images=2, total_instances=2, old_records_exact=dict(images=1, annotations=1))
        self.receipt['added_loader'].update(images=1, instances=1)
        self.plan = dict(package_pins=self.receipt['package_pins'], protected_manifest_pins=self.receipt['protected_manifest_pins'],
                         producer_build_commit=self.receipt['producer_build_commit'], parent_annotation_sha256=ident(self.old)['annotation_sha256'])
        self.bundle = dict(schema_version=1, mode='accepted_expansion', evaluation_data=ident(self.dev), training_data=ident(self.new),
                           dev_images=1, dev_instances=1, train_images=2, train_instances=2)
        self.cfg = dict(purpose='dev_checkpoint_selection', **ident(self.dev))
        self.save()

    def save(self):
        for name, value in [('dev_report', self.report), ('acceptance', self.receipt), ('acceptance_plan', self.plan)]:
            dump(self.root/(name+'.json'), value)
        self.run = dict(status='completed', exit_code=0, git_commit=self.receipt['source_commit'], artifacts={
            'acceptance.json': {'sha256': sha256(self.root/'acceptance.json')},
            'acceptance_plan.json': {'sha256': sha256(self.root/'acceptance_plan.json')}})
        dump(self.root/'acceptance_run.json', self.run)
        self.bundle['references'] = {k: dict(path=k+'.json', sha256=sha256(self.root/(k+'.json')))
                                    for k in ('dev_report', 'acceptance', 'acceptance_plan', 'acceptance_run')}
        dump(self.root/'bundle.json', self.bundle)

    def consume(self):
        return reuse_input_receipts(self.dev, self.cfg, self.new, self.root/'bundle.json')


class ReceiptTests(unittest.TestCase):
    def test_compose_without_pixels_and_keep_old_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            f = ReceiptFixture(Path(directory)); old_bytes = (f.root/'dev_report.json').read_bytes()
            with (patch('PIL.Image.open', side_effect=AssertionError('image read')),
                  patch('fashion_grounding.segmentation.evaluation.decode_mask', side_effect=AssertionError('mask read'))):
                _, _, check = f.consume()
                self.assertFalse(check['image_and_mask_reaudit'])
                self.assertFalse(check['formal_three_party_freeze'])
                self.assertFalse(check['test_files_read'])
                with self.assertRaisesRegex(ValueError, 'training identity'):
                    reuse_eval_package(f.dev, f.cfg, f.new, f.root/'dev_report.json')
                f.bundle.update(mode='unchanged_train', training_data=f.report['training_data'], train_images=1, train_instances=1)
                f.bundle['references'] = {'dev_report': f.bundle['references']['dev_report']}
                dump(f.root/'bundle.json', f.bundle)
                _, _, old = reuse_input_receipts(f.dev, f.cfg, f.old, f.root/'bundle.json')
                self.assertEqual(old['validation_mode'], 'unchanged_train')
            self.assertEqual((f.root/'dev_report.json').read_bytes(), old_bytes)

    def test_live_metadata_and_receipt_hash_drift_rejected(self):
        for name in ('dev/image_manifest.json', 'new/annotations_coco.json', 'new/SHA256SUMS.json', 'acceptance.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                f = ReceiptFixture(Path(directory)); p = f.root/name
                p.write_text(p.read_text()+' ')
                with self.assertRaises(ValueError): f.consume()

    def test_semantic_chain_rejects_incomplete_wrong_parent_and_leak(self):
        changes = [lambda f: f.report.update(complete=False),
                   lambda f: f.plan.update(parent_annotation_sha256='0'*64),
                   lambda f: f.receipt['protected_manifest_pins'].update(dev='0'*64),
                   lambda f: f.receipt['added_cross_partition_overlap_counts']['dev'].update(group_id=1),
                   lambda f: f.receipt['added_cross_partition_overlap_counts'].pop('test'),
                   lambda f: f.receipt.update(total_images=3),
                   lambda f: f.receipt['reused_audits']['near_duplicates'].update(flags=[{'conflict': True}])]
        for i, change in enumerate(changes):
            with self.subTest(case=i), tempfile.TemporaryDirectory() as directory:
                f = ReceiptFixture(Path(directory)); change(f); f.save()
                with self.assertRaises(ValueError): f.consume()

    def test_run_must_bind_exact_acceptance_and_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            f = ReceiptFixture(Path(directory)); f.run['artifacts']['acceptance_plan.json']['sha256'] = '0'*64
            dump(f.root/'acceptance_run.json', f.run)
            f.bundle['references']['acceptance_run']['sha256'] = sha256(f.root/'acceptance_run.json')
            dump(f.root/'bundle.json', f.bundle)
            with self.assertRaisesRegex(ValueError, 'does not bind'): f.consume()


def process_alive(pid):
    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith('Z')


def wait_file(path):
    until = time.monotonic()+4
    while not path.exists() and time.monotonic() < until:
        time.sleep(.01)
    if not path.exists(): raise AssertionError('Child fixture did not start')


class RunnerTests(unittest.TestCase):
    def test_prepare_does_not_execute_and_commands_keep_exact_scope(self):
        plan = runner.read(ROOT/runner.PLAN)
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, 'argv', ['run', '--output', directory+'/never-created']), \
             patch.object(runner, 'run_pair', side_effect=AssertionError('execution')), contextlib.redirect_stdout(io.StringIO()) as output:
            runner.main(); preview = json.loads(output.getvalue())
            self.assertFalse(Path(directory+'/never-created').exists())
        self.assertEqual(preview['training_updates'], 0)
        for arm in plan['arms']:
            phases = runner.commands(plan, arm, '/tmp/proposed-pair')
            self.assertEqual([x[0] for x in phases], ['train_20000', 'full_dev_10000_20000', 'select_and_compare'])
            self.assertIn('--checkpoint', phases[0][1]); self.assertNotIn('--resume', phases[0][1])
            self.assertIn('--input-validation-bundle', phases[1][1])
            self.assertEqual(sum('/checkpoint_' in x for x in phases[1][1]), 2)
            self.assertTrue(any(x.endswith('checkpoint_010000.pth') for x in phases[1][1]))
            self.assertTrue(any(x.endswith('checkpoint_020000.pth') for x in phases[1][1]))
        for changes in [dict(per_arm_hard_seconds=10801), dict(automatic_extension=True), dict(automatic_restart=True)]:
            bad = copy.deepcopy(plan); bad['budget'].update(changes)
            with self.assertRaises(ValueError): runner.validate_plan(bad)

    def test_two_phases_share_deadline_and_kill_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); pidfile = out/'descendant.pid'
            spawn = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c', {('import os,time;from pathlib import Path;Path('+repr(str(pidfile))+').write_text(str(os.getpid()));time.sleep(30)')!r}]); time.sleep(30)"
            phases = [('first', [sys.executable, '-c', 'import time;time.sleep(.35)']), ('second', [sys.executable, '-c', spawn])]
            code = f"import sys,time;sys.path.insert(0,{str(ROOT/'scripts')!r});from run_expansion_control import execute_phases;execute_phases({phases!r},{str(out)!r},time.monotonic()+1.5)"
            started = time.monotonic(); p = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
            try:
                wait_file(pidfile); p.wait(timeout=5)
                self.assertLess(time.monotonic()-started, 2.8)
                self.assertLess(p.returncode, 0)
                self.assertFalse(process_alive(int(pidfile.read_text())))
                rows = runner.read(out/'phases.json')
                self.assertEqual([r['status'] for r in rows], ['completed', 'running'])
            finally:
                if p.poll() is None: os.killpg(p.pid, signal.SIGKILL); p.wait()

    def test_failed_phase_stops_and_failed_arm_never_starts_next(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); marker = out/'should-not-exist'
            phases = [('bad', [sys.executable, '-c', 'raise SystemExit(7)']), ('after', [sys.executable, '-c', f'open({str(marker)!r},"w").close()'])]
            code = f"import sys,time;sys.path.insert(0,{str(ROOT/'scripts')!r});from run_expansion_control import execute_phases;sys.exit(execute_phases({phases!r},{str(out)!r},time.monotonic()+3))"
            p = subprocess.run([sys.executable, '-c', code], start_new_session=True, timeout=5)
            self.assertEqual(p.returncode, 7); self.assertFalse(marker.exists())
            plan = runner.read(ROOT/runner.PLAN)
            with patch.object(runner, 'git', side_effect=lambda *args: 'test-head' if args[0]=='rev-parse' else ''), \
                 patch.object(runner.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 30*1024**3})()), \
                 patch.object(runner, 'run_bounded', return_value=7) as bounded:
                with self.assertRaises(RuntimeError): runner.run_pair(plan, out/'pair', time.monotonic(), 'test-head')
                self.assertEqual(bounded.call_count, 1)
                self.assertFalse((out/'pair/D10k_targeted').exists())
                self.assertEqual(runner.read(out/'pair/stage_status.json')['status'], 'failed')

    @unittest.skipUnless(sys.platform == 'linux', 'Linux production parent-death path')
    def test_linux_wrapper_death_kills_worker_group(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory); desc = out/'desc.pid'; workerpid = out/'worker.pid'
            command = [sys.executable, '-c', f'import os,time;from pathlib import Path;Path({str(desc)!r}).write_text(str(os.getpid()));time.sleep(30)']
            worker = f"import sys,time;sys.path.insert(0,{str(ROOT/'scripts')!r});from run_expansion_control import execute_phases;execute_phases({[('wait',command)]!r},{str(out)!r},time.monotonic()+20)"
            wrapper = f"import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session=True);Path({str(workerpid)!r}).write_text(str(p.pid));time.sleep(30)"
            p = subprocess.Popen([sys.executable, '-c', wrapper], start_new_session=True)
            try:
                wait_file(desc); p.kill(); p.wait(timeout=2)
                until = time.monotonic()+2
                while process_alive(int(desc.read_text())) and time.monotonic()<until: time.sleep(.02)
                self.assertFalse(process_alive(int(desc.read_text())))
                self.assertFalse(process_alive(int(workerpid.read_text())))
            finally:
                if p.poll() is None: p.kill();p.wait()
                if workerpid.exists() and process_alive(int(workerpid.read_text())):
                    os.killpg(int(workerpid.read_text()), signal.SIGKILL)

    def test_exhausted_pair_never_launches_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'pair';plan=runner.read(ROOT/runner.PLAN)
            with patch.object(runner,'git',side_effect=lambda *a:'test-head' if a[0]=='rev-parse' else ''), \
                 patch.object(runner.shutil,'disk_usage',return_value=type('Disk',(),{'free':30*1024**3})()), \
                 patch.object(runner,'run_bounded',side_effect=AssertionError('expired launch')) as bounded:
                with self.assertRaises(TimeoutError): runner.run_pair(plan,out,time.monotonic()-21601,'test-head')
                bounded.assert_not_called()
                self.assertEqual(runner.read(out/'stage_status.json')['arms'][0]['status'],'failed')
                self.assertEqual(runner.read(out/'C5k/run_metadata.json')['status'],'failed')


class ComparisonTests(unittest.TestCase):
    def fixture(self, out):
        plan = runner.read(ROOT/runner.PLAN)
        base = runner.read(ROOT/plan['historical_report'])
        reports = {}
        for name, scores in [('C5k', [61., 61.]), ('D10k_targeted', [61.1, 61.6])]:
            folder = out/name;folder.mkdir();(folder/'train').mkdir()
            dump(folder/'train/train_report.json', {'initial_parameter_digests': {'backbone': 'same-b', 'classifier': 'same-c'}})
            rows = []
            for update, score in zip([10000, 20000], scores):
                r = copy.deepcopy(base)
                r.update(completed_updates=update, training_config_sha256=name,
                         training_data={'image_manifest_sha256': name, 'annotation_sha256': name},
                         checkpoint_path=str(folder/f'{update}.pth'), checkpoint_sha256=name+str(update))
                r['metrics'].update(AP=score, AP_small=10. if name=='C5k' else 11.2, AP75=68.)
                r['metrics']['per_category_AP'] = {str(i):50. for i in range(1,9)}
                rows.append(r)
            reports[name] = rows
        self.save(out, reports)
        return plan, reports

    def save(self, out, reports):
        for name, rows in reports.items():
            evidence = []
            for r in rows:
                p = out/name/f'{r["completed_updates"]}.json';dump(p,r)
                evidence.append({'path':str(p),'sha256':sha256(p)})
            dump(out/name/'selection.json',dict(status='completed',reports=evidence,selected=choose_checkpoint(rows)))

    def test_separate_dataset_contrast_endpoint_and_historical_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);plan,reports=self.fixture(out)
            with self.assertRaises(ValueError): choose_checkpoint([reports['C5k'][0],reports['D10k_targeted'][1]])
            result=runner.compare_pair(plan,out)
            self.assertAlmostEqual(result['D10k_minus_C5k']['AP'],.6)
            self.assertTrue(result['practical_screen_passed'])
            self.assertEqual(result['best_by_dev_AP']['group'],'historical_B20k')
            self.assertEqual(runner.read(out/'C5k/selection.json')['selected']['completed_updates'],10000)
            self.assertFalse(result['weights_or_candidates_replaced'])

    def test_missing_endpoint_and_different_initial_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);plan,reports=self.fixture(out)
            reports['D10k_targeted'].pop();self.save(out,reports)
            with self.assertRaisesRegex(ValueError,'Missing paired endpoint'): runner.compare_pair(plan,out)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);plan,reports=self.fixture(out)
            dump(out/'D10k_targeted/train/train_report.json',{'initial_parameter_digests': {'backbone':'different','classifier':'same-c'}})
            with self.assertRaisesRegex(ValueError,'Different initial parameters'): runner.compare_pair(plan,out)


if __name__ == '__main__': unittest.main()
