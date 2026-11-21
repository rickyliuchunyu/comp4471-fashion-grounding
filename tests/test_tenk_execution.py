"""New segmented execution only: synthetic receipts/processes, no GPU/model."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_tenk_adaptation as runner
from fashion_grounding.data.schema import sha256
from fashion_grounding.segmentation.tenk_execution import (
    read, write_atomic, select_progress, audit_training, audit_evaluation)
from fashion_grounding.segmentation.evaluation import choose_checkpoint

PLAN = read(runner.PLAN)
BASE = read(ROOT/PLAN['baseline_report'])
SOURCE = read(ROOT/PLAN['model_only_source'])
BUNDLE = read(ROOT/PLAN['input_validation_bundle'])
CONTEXT = dict(git_commit='new-source', training_config_sha256='new-config',
    evaluation_config_sha256=BASE['evaluation_config_sha256'],
    training_data=BUNDLE['training_data'], evaluation_data=BUNDLE['evaluation_data'], model_only_source=SOURCE)


def report(epoch, ap):
    r = copy.deepcopy(BASE)
    r.update(completed_updates=epoch*5000, checkpoint_path=f'/new/{epoch}.pth', checkpoint_sha256=str(epoch)*64,
             training_git_commit=CONTEXT['git_commit'], evaluation_git_commit=CONTEXT['git_commit'],
             training_data=CONTEXT['training_data'], training_config_sha256=CONTEXT['training_config_sha256'])
    r['metrics']['AP'] = ap
    return r


def fixture(directory, epoch=2, previous=None):
    train, dev = directory/'train', directory/'dev'
    train.mkdir(parents=True)
    update = epoch*5000
    cp = train/f'checkpoint_{update:06d}.pth'
    cp.write_bytes(b'synthetic checkpoint; never loaded')
    cp.with_suffix('.sha256').write_text(sha256(cp)+'\n')
    start = previous['completed_updates'] if previous else 0
    write_atomic(train/'train_report.json', dict(status='completed', purpose='s0_training', git_commit='new-source',
        config_sha256='new-config', start_update=start, completed_updates=update, updates_this_process=update-start,
        sample_count=(update-start)*2, resumed=bool(previous), all_losses_and_gradients_finite=True,
        skipped_optimizer_updates=0, backbone_and_classifier_updated=True,
        checkpoints=[dict(update=update, path=cp.name, sha256=sha256(cp))]))
    write_atomic(train/'data_check.json', dict(**BUNDLE['training_data'], images=10000, instances=39699))
    write_atomic(train/'initialization.json', dict(mode='model_only', source=SOURCE, new_completed_updates=0,
        loaded_components=['model'], reset_components=['optimizer','scheduler','scaler','sampler','rng','amp_recovery'],
        classifier_reinitialized=False))
    if previous:
        components={k:'digest' for k in ('model','optimizer','scheduler','scaler','sampler','rng')}
        write_atomic(train/'resume_check.json', dict(status='passed', resumed_from_update=start,
            checkpoint_sha256=previous['checkpoint_sha256'], exact_components=sorted(components),
            expected_digests=components, restored_digests=components))
    r = report(epoch, 65.)
    r.update(checkpoint_path=str(cp), checkpoint_sha256=sha256(cp),
             overlap=dict(image_id=0, sha256=0, group_id=0, source_identity=0))
    path = dev/f'update_{update:06d}'/'evaluation_report.json'
    path.parent.mkdir(parents=True)
    write_atomic(path.parent/'predictions.json', [])
    r['predictions_sha256']=sha256(path.parent/'predictions.json')
    write_atomic(path, r)
    path.with_suffix('.sha256').write_text(sha256(path))
    write_atomic(dev/'run_report.json', dict(status='completed', reports=[str(path)],
        optimizer_updates=0, purpose='dev_checkpoint_selection'))
    return train, dev, dict(new_run_update=update), path


class GateTests(unittest.TestCase):
    def test_default_entry_and_disabled_execute_have_no_side_effects(self):
        with patch.object(runner, 'launch', side_effect=AssertionError('must not launch')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('no child process')):
            with patch.object(sys, 'argv', ['run_tenk_adaptation.py']), patch('builtins.print') as printed:
                runner.main()
            result=json.loads(printed.call_args[0][0])
            self.assertEqual([s['new_run_update'] for s in result['stages']], [10000,20000,30000,40000,50000])
            self.assertFalse(result['execution_enabled'])
            with patch.object(sys, 'argv', ['run_tenk_adaptation.py','--execute']), self.assertRaises(ValueError):
                runner.main()

    def test_all_four_authorization_flags_and_hash_pin_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            shutil.copytree(ROOT/'configs',root/'configs')
            (root/'docs').symlink_to(ROOT/'docs', target_is_directory=True)
            p=copy.deepcopy(PLAN)
            p['training_authorized']=p['execution_enabled']=p['budget_proposal']['authorized']=True
            path=root/p['training_config']
            config=read(path); config['training_authorized']=True
            write_atomic(path,config); p['input_sha256'][p['training_config']]=sha256(path)
            runner.validate(p,root,authorized=True)
            for key in ('training_authorized','execution_enabled','budget_proposal'):
                bad=copy.deepcopy(p)
                if key=='budget_proposal': bad[key]['authorized']=False
                else: bad[key]=False
                with self.subTest(key=key),self.assertRaises(ValueError): runner.validate(bad,root,authorized=True)
            config['training_authorized']=False; write_atomic(path,config)
            with self.assertRaisesRegex(ValueError,'Prepared input changed'): runner.validate(p,root,authorized=True)
            p['input_sha256'][p['training_config']]=sha256(path)
            with self.assertRaisesRegex(ValueError,'exposure contract'): runner.validate(p,root,authorized=True)

    def test_changed_caps_and_stopping_rejected(self):
        for section,key,value in [('budget_proposal','total_hard_seconds',28801),
            ('budget_proposal','per_segment_hard_seconds',7201),
            ('stopping','automatic_extension',True),('stopping','automatic_restart',True),
            ('stopping','consecutive_dev_evaluations_without_new_best',4)]:
            p=copy.deepcopy(PLAN); p[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): runner.validate(p)
        p=copy.deepcopy(PLAN); p['input_sha256'].pop(p['baseline_report'])
        with self.assertRaisesRegex(ValueError,'Missing execution input pin'): runner.validate(p)

    def test_wrapper_failure_is_recorded_without_retry_and_output_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan_path=root/'plan.json'; write_atomic(plan_path,PLAN)
            p={**PLAN,'execution_checkout':tmp,'python':sys.executable,'future_output':str(root/'output'),
               'training_package':tmp,'evaluation_package':tmp,'upstream_source':tmp}
            context={**CONTEXT,'model_only_source':{'path':str(plan_path)}}
            with patch.object(runner,'ROOT',root), patch.object(sys,'platform','linux'), \
                 patch.object(runner,'source_guard'), patch.object(subprocess,'check_output',return_value='new-source\n'), \
                 patch.object(subprocess,'run'), patch.object(shutil,'disk_usage') as usage, \
                 patch.object(runner,'run_bounded',side_effect=TimeoutError('test hard timeout')) as bounded:
                usage.return_value.free=100*1024**3
                with self.assertRaises(TimeoutError):
                    runner.launch(plan_path,p,{},BASE,context,started=time.monotonic())
                self.assertEqual(bounded.call_count,1)
                self.assertEqual(read(root/'output/run_metadata.json')['status'],'failed')
                with self.assertRaisesRegex(ValueError,'new absolute run output'):
                    runner.launch(plan_path,p,{},BASE,context,started=time.monotonic())
                self.assertEqual(bounded.call_count,1)


class SelectionTests(unittest.TestCase):
    def test_baseline_kept_on_lower_or_equal_ap_and_plateau_at_epoch6(self):
        ap=BASE['metrics']['AP']
        result=select_progress(BASE,[report(2,ap-1),report(4,ap),report(6,ap-2)],[2,4,6])
        self.assertEqual(result['selected_origin'],'cached_C5k20k')
        self.assertEqual(result['stop_reason'],'plateau')

    def test_improvement_resets_plateau_ties_choose_earliest_new_update(self):
        result=select_progress(BASE,[report(2,65),report(4,65),report(6,66),report(8,66)],[2,4,6,8])
        self.assertEqual(result['consecutive_without_new_best'],1)
        self.assertEqual(result['selected_checkpoint']['completed_updates'],30000)
        self.assertEqual(result['stop_reason'],'continue')

    def test_ten_epoch_limit_and_strict_training_identity_not_weakened(self):
        rs=[report(e,65+e) for e in (2,4,6,8,10)]
        self.assertEqual(select_progress(BASE,rs,[2,4,6,8,10])['stop_reason'],'max_equivalent_epochs')
        with self.assertRaisesRegex(ValueError,'Incomparable'): choose_checkpoint([BASE,rs[0]])
        bad=copy.deepcopy(rs); bad[1]['training_data']={'wrong':'data'}
        with self.assertRaises(ValueError): select_progress(BASE,bad,[2,4,6,8,10])
        bad=copy.deepcopy(rs); bad[0]['evaluation_data']={'wrong':'dev'}
        with self.assertRaises(ValueError): select_progress(BASE,bad[:1],[2])


class ReceiptTests(unittest.TestCase):
    def test_model_only_then_exact_resume_and_full_dev_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b,s,_=fixture(Path(tmp)/'first')
            cp=audit_training(a,s,None,CONTEXT)
            audit_evaluation(b,cp,CONTEXT)
            a,b,s,_=fixture(Path(tmp)/'second',4,cp)
            next_cp=audit_training(a,s,cp,CONTEXT)
            audit_evaluation(b,next_cp,CONTEXT)
            path=a/'resume_check.json'; bad=read(path); bad['restored_digests']['rng']='wrong'
            write_atomic(path,bad)
            with self.assertRaisesRegex(ValueError,'Resume'): audit_training(a,s,cp,CONTEXT)

    def test_wrong_exposure_and_checkpoint_receipts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,_,s,_=fixture(Path(tmp))
            original=read(a/'train_report.json')
            for key,value in [('sample_count',19999),('start_update',1),('config_sha256','wrong'),
                              ('all_losses_and_gradients_finite',False),('resumed',True)]:
                write_atomic(a/'train_report.json',{**original,key:value})
                with self.subTest(key=key),self.assertRaises(ValueError): audit_training(a,s,None,CONTEXT)
            write_atomic(a/'train_report.json',original)
            (a/'checkpoint_010000.sha256').write_text('wrong')
            with self.assertRaises(ValueError): audit_training(a,s,None,CONTEXT)

    def test_incomplete_wrong_identity_and_tampered_dev_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b,s,path=fixture(Path(tmp)); cp=audit_training(a,s,None,CONTEXT); original=read(path)
            for key,value in [('complete',False),('evaluated_images',249),('checkpoint_sha256','wrong'),
                ('evaluation_git_commit','wrong'),('queries_read',True),('training_data',{})]:
                write_atomic(path,{**original,key:value}); path.with_suffix('.sha256').write_text(sha256(path))
                with self.subTest(key=key),self.assertRaises(ValueError): audit_evaluation(b,cp,CONTEXT)
            write_atomic(path,original); path.with_suffix('.sha256').write_text(sha256(path))
            (path.parent/'predictions.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'prediction digest'): audit_evaluation(b,cp,CONTEXT)


class SequencingTests(unittest.TestCase):
    def simulate(self, output, scores, *, failure=None, total=100., cost=1.):
        stages=[dict(equivalent_epoch=e,new_run_update=e*5000,train=['train',str(e)],full_dev=['full_dev',str(e)])
                for e in (2,4,6,8,10)]
        now=[0.]; calls=[]; arms=[]
        def phase(command,log):
            calls.append(tuple(command)); now[0]+=cost
            if tuple(command)==failure: raise RuntimeError('injected phase failure')
        def train(d,s,p):
            if ('receipt',str(s['equivalent_epoch']))==failure: raise ValueError('injected receipt failure')
            return dict(completed_updates=s['new_run_update'],checkpoint_sha256='cp')
        def evaluate(d,c):
            e=c['completed_updates']//5000
            r=report(e,scores[e//2-1])
            if ('select',str(e))==failure: r['training_git_commit']='mismatch'
            return r
        try:
            result=runner.execute_segments(dict(stages=stages),BASE,output,total,
                run_phase=phase,train_receipt=train,eval_receipt=evaluate,guard=lambda:None,
                arm=arms.append,clock=lambda:now[0])
            return result,calls,arms
        except Exception:
            self.failed_calls=calls; self.failed_arms=arms
            raise

    def test_all_segments_in_order_and_plateau_stops_next_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            result,calls,arms=self.simulate(tmp,[64,64,64,70,71])
            self.assertEqual(result['stop_reason'],'plateau')
            self.assertEqual(calls,[(p,str(e)) for e in (2,4,6) for p in ('train','full_dev')])
            self.assertTrue(all(d==100 for d in arms))  # Total cap never resets to now+100.

    def test_each_phase_failure_retains_last_choice_and_stops(self):
        for phase in ('train','full_dev','receipt','select'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises((ValueError,RuntimeError)):
                    self.simulate(tmp,[65,66,67,68,69],failure=(phase,'4'))
                self.assertEqual(read(Path(tmp)/'selection.json')['completed_equivalent_epochs'],2)
                self.assertFalse((Path(tmp)/'epoch_06').exists())
                self.assertEqual(read(Path(tmp)/'stages.json')[-1]['status'],'failed')

    def test_total_budget_carries_across_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError): self.simulate(tmp,[65,66,67,68,69],total=5,cost=2)
            self.assertEqual(self.failed_calls,[('train','2'),('full_dev','2'),('train','4')])
            self.assertEqual(read(Path(tmp)/'selection.json')['completed_equivalent_epochs'],2)
            self.assertTrue(all(d==5 for d in self.failed_arms))

    def test_segment_budget_covers_train_and_dev_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError): self.simulate(tmp,[65]*5,total=28800,cost=3700)
            self.assertEqual(self.failed_calls,[('train','2'),('full_dev','2')])
            self.assertEqual(self.failed_arms,[7200])
            self.assertFalse((Path(tmp)/'selection.json').exists())

    def test_real_segment_timer_kills_only_its_own_child_group(self):
        # This new wiring test uses sleeping CPU children, never the trainer.
        # Linux parent-death behavior is reused from the existing tested helper.
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'child-completed'
            code = '''import json, pathlib, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
import run_tenk_adaptation as r
from fashion_grounding.segmentation.tenk_execution import read
base=read(pathlib.Path(r.ROOT)/read(r.PLAN)['baseline_report'])
r.SEGMENT_SECONDS=0.4
def phase(command,log):
    subprocess.run([sys.executable,'-c',"import time,pathlib; time.sleep(1.1); pathlib.Path("+repr(sys.argv[3])+").touch()"],check=True)
r.execute_segments({'stages':[{'equivalent_epoch':2,'new_run_update':10000,'train':[], 'full_dev':[]}]},base,
    sys.argv[2],time.monotonic()+10,run_phase=phase,train_receipt=lambda *a:None,
    eval_receipt=lambda *a:None,guard=lambda:None)
'''
            process=subprocess.Popen([sys.executable,'-c',code,str(ROOT/'scripts'),tmp,str(marker)],
                start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            _,err=process.communicate(timeout=15)
            self.assertEqual(process.returncode,-9,err)
            time.sleep(1.2)
            self.assertFalse(marker.exists())


if __name__ == '__main__': unittest.main()
