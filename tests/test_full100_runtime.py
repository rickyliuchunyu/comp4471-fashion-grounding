"""CPU contract/orchestration fixtures, not real data acceptance or model smoke."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fashion_grounding.segmentation.full100_contract import (
    FREEZE_CHECKS, PROTOCOL, cache_identity, digest_json, read_json, resolve, sha256, validate_report,
)
from fashion_grounding.segmentation.full100_runtime import (
    atomic_json, checkpoints, gpu_slot, run_engine, worker_command,
)
from fashion_grounding.segmentation.full100_worker import LossSummary, training_records

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'configs/runs/segmentation/full100_execution_plan.json'
COMMIT = 'a' * 40


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path.resolve()), 'sha256': sha256(path)}


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.plan = read_json(PLAN)
        categories = read_json(ROOT/'configs/categories.json')['categories']
        self.identities = {}
        for split, count, offset in [('train', 256, 0), ('dev', 3, 1000000)]:
            package = self.root/split
            images = [dict(image_id=offset+i+1, file_name=f'images/{offset+i+1}.jpg', width=8, height=8,
                           split=split, source_dataset='fashionpedia', source_split='official_'+split,
                           source_image_id=str(i), group_id=f'{split}-{i}',
                           sha256=hashlib.sha256(f'{split}-{i}'.encode()).hexdigest()) for i in range(count)]
            coco = dict(images=[dict(id=im['image_id'], **{k: im[k] for k in ('file_name','width','height')})
                                for im in images], categories=categories,
                        annotations=[dict(id=i+1, image_id=im['image_id'], category_id=i%8+1,
                                          bbox=[0,0,1,1], area=1, iscrowd=0,
                                          segmentation={'size':[8,8], 'counts':'01o1'})
                                     for i,im in enumerate(images)])
            m=write(package/'image_manifest.json', dict(schema_version=3, illustrative_only=False, images=images))
            c=write(package/'annotations_coco.json', coco)
            write(package/'SHA256SUMS.json', {'synthetic_fixture_only':True})
            self.identities[split] = dict(image_manifest_sha256=m['sha256'], annotation_sha256=c['sha256'])
        evaluation = read_json(ROOT/self.plan['evaluation']['path'])
        evaluation.update(dev_images=3, dev_instances=3)
        evaluation['evaluation_config_template'].update(self.identities['dev'])
        self.plan['evaluation'] = write(self.root/'evaluation_plan.json', evaluation)
        self.plan_path = self.root/'plan.json'
        write(self.plan_path, self.plan)
        proof = write(self.root/'proof.json', {'status':'passed', 'synthetic_fixture_only':True})
        self.receipt_path = self.root/'acceptance.json'
        self.receipt = dict(status='passed', scope='full_fashionpedia_segmentation_input',
            source_commit=COMMIT, training_data=self.identities['train'], evaluation_data=self.identities['dev'],
            mapping_sha256=self.plan['mapping']['sha256'], train_images=256, train_instances=256,
            dev_images=3, dev_instances=3, catalog_sha256=sha256(self.root/'train/SHA256SUMS.json'),
            image_and_mask_acceptance_complete=True, near_duplicate_review_complete=True,
            training_id_mapping={str(i):i-1 for i in range(1,9)}, protected_test_manifest_sha256='b'*64,
            overlap={k:{v:0 for v in ('image_id','source_identity','group_id','sha256')} for k in ('dev','test')})
        self.receipt_ref = write(self.receipt_path, self.receipt)
        acknowledgements={member:write(self.root/(member+'.json'), dict(member=member,decision='approved',
            reviewer_identity='synthetic fixture only', reviewed_at='fixture', **self.identities['train']))
            for member in ('wkq','lcy','wyz')}
        self.freeze_path = self.root/'freeze.json'
        self.freeze=dict(formal_freeze=True,status='frozen',dataset='fashionpedia',source_split='train',
            mapping={**self.plan['mapping'],'path':str(ROOT/self.plan['mapping']['path'])},
            actual_train_images=256,actual_train_instances=256,**self.identities['train'],
            source_accounting=dict(coverage='all_eligible_official_train',source_images=512,retained_images=256,excluded_images=256),
            protected=dict(test_manifest_sha256='b'*64,existing_dev_manifest_sha256=self.identities['dev']['image_manifest_sha256'],
                           existing_dev_annotation_sha256=self.identities['dev']['annotation_sha256']),
            checks=[dict(id=k,status='passed',evidence=self.receipt_ref if k=='consumer_acceptance' else proof)
                    for k in sorted(FREEZE_CHECKS)],acknowledgements=acknowledgements)
        self.dev_ref=write(self.root/'dev_receipt.json', dict(purpose='dev_checkpoint_selection',protocol=PROTOCOL,
            complete=True,selection_eligible=True,inference_errors=0,evaluated_images=3,evaluation_data=self.identities['dev']))
        self.binding_path=self.root/'binding.json'
        self.refresh()

    def refresh(self):
        self.receipt_ref=write(self.receipt_path,self.receipt)
        for check in self.freeze['checks']:
            if check['id']=='consumer_acceptance': check['evidence']=self.receipt_ref
        freeze_ref=write(self.freeze_path,self.freeze)
        write(self.binding_path,dict(schema_version=1,purpose='full100_data_binding',
            train_package=str(self.root/'train'),dev_package=str(self.root/'dev'),
            references=dict(freeze=freeze_ref,acceptance=self.receipt_ref,dev_report=self.dev_ref)))

    def contract(self):
        return resolve(ROOT,self.plan_path,self.binding_path,COMMIT)


def engine_contract():
    # A four-node fixture exercises the scheduler without any GPU/model work.
    return dict(ready_for_training=True,source_commit=COMMIT,train_config_sha256='1'*64,
        eval_config_sha256='2'*64,inputs=dict(binding_sha256='3'*64,dev_images=3,
            training_data={'image_manifest_sha256':'4'*64,'annotation_sha256':'5'*64},
            evaluation_data={'image_manifest_sha256':'6'*64,'annotation_sha256':'7'*64}),
        schedule=dict(max_updates=40,checkpoint_updates=[10,20,30,40],
                      evaluation_nodes=[{'epoch':2,'update':20},{'epoch':4,'update':40}]),
        runtime={'python':'fixture-python'})


class FakeWorker:
    """Synthetic checkpoint bytes and AP records; never a numerical training loop."""
    def __init__(self,contract,fail=None):
        self.contract,self.fail=contract,fail
        self.calls=[]

    def checkpoint(self,output,update):
        path=output/f'checkpoint_{update:06d}.pth'
        path.write_bytes(f'synthetic checkpoint {update}'.encode())
        digest=sha256(path)
        path.with_suffix('.sha256').write_text(digest+'\n')
        stat=path.stat()
        row=dict(path=str(path),update=update,sha256=digest,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                 training_config_sha256=self.contract['train_config_sha256'],source_commit=COMMIT,
                 binding_sha256=self.contract['inputs']['binding_sha256'])
        atomic_json(path.with_suffix('.receipt.json'),row)
        return row

    def __call__(self,mode,output,stop,resume_checkpoint,eval_checkpoint):
        self.calls.append((mode,stop,resume_checkpoint,eval_checkpoint))
        output.mkdir(exist_ok=False)
        if mode=='train':
            start=int(Path(resume_checkpoint).stem.rsplit('_',1)[1]) if resume_checkpoint else 0
            for update in self.contract['schedule']['checkpoint_updates']:
                if start<update<=stop:
                    self.checkpoint(output,update)
                    if self.fail==('train',update): raise RuntimeError('injected training interruption')
        else:
            if self.fail==('evaluate',stop): raise RuntimeError('injected evaluation failure')
            cp=read_json(Path(eval_checkpoint).with_suffix('.receipt.json'))
            write(output/'predictions.json',{'synthetic_fixture_only':True,'update':stop})
            report=dict(**cache_identity(self.contract,cp),purpose='dev_checkpoint_selection',protocol=PROTOCOL,
                        metric_units='percent',complete=True,selection_eligible=True,inference_errors=0,
                        evaluated_images=3,metrics={'AP':80 if stop==20 else 70},checkpoint_path=eval_checkpoint,
                        predictions_sha256=sha256(output/'predictions.json'))
            write(output/'evaluation_report.json',report)
            (output/'evaluation_report.sha256').write_text(sha256(output/'evaluation_report.json')+'\n')


class Full100ContractTests(unittest.TestCase):
    def test_metadata_adapter_preserves_native_order_and_crowd_policy(self):
        manifest={'images':[dict(image_id=2,file_name='b.jpg',height=8,width=8),
                            dict(image_id=1,file_name='a.jpg',height=8,width=8)]}
        rle={'size':[8,8],'counts':'01o1'}
        coco={'annotations':[dict(id=3,image_id=1,category_id=8,bbox=[0,0,1,1],segmentation=rle),
                             dict(id=1,image_id=1,category_id=1,bbox=[0,0,1,1],segmentation=rle),
                             dict(id=2,image_id=1,category_id=2,bbox=[0,0,1,1],segmentation=rle,iscrowd=1)]}
        rows=training_records('/not_opened',manifest,coco,bbox_mode=1)
        self.assertEqual([r['image_id'] for r in rows],[1,2])
        self.assertEqual([a['category_id'] for a in rows[0]['annotations']],[0,7])
        self.assertEqual(rows[0]['annotations'][0]['segmentation'],rle)
        self.assertEqual(rows[1]['annotations'],[])

    def test_complete_binding_compiles_and_missing_images_are_not_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp); c=f.contract()
            self.assertTrue(c['ready_for_training'])
            self.assertEqual(len(c['schedule']['evaluation_nodes']),50)
            self.assertEqual(c['train_config']['max_updates'],1600)
            self.assertIsNone(c['train_config']['budget_seconds'])
            self.assertEqual(c['train_config']['accumulation_steps'],8)
            self.assertFalse((Path(tmp)/'train/images').exists())

    def test_pending_freeze_and_wrong_acknowledgement_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(tmp)
            f.freeze['formal_freeze']=False;f.refresh()
            with self.assertRaisesRegex(ValueError,'formal freeze'):f.contract()
            f.freeze['formal_freeze']=True
            f.freeze['acknowledgements']['wyz']=f.freeze['acknowledgements']['wkq'];f.refresh()
            with self.assertRaisesRegex(ValueError,'acknowledgement'):f.contract()

    def test_incomplete_acceptance_and_nonzero_overlap_fail(self):
        for mutation in [lambda f:f.receipt.update(image_and_mask_acceptance_complete=False),
                         lambda f:f.receipt['overlap']['test'].update(group_id=1),
                         lambda f:f.receipt['training_id_mapping'].update({'8':8})]:
            with tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp);mutation(f);f.refresh()
                with self.assertRaises(ValueError):f.contract()

    def test_catalog_and_metadata_drift_rejected(self):
        for name in ['SHA256SUMS.json','image_manifest.json']:
            with tempfile.TemporaryDirectory() as tmp:
                f=Fixture(tmp);p=Path(tmp)/'train'/name;p.write_text(p.read_text()+' ')
                with self.assertRaises(ValueError):f.contract()

    def test_uncompressed_masks_refused_before_training(self):
        from fashion_grounding.segmentation.full100_contract import package_metadata
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(tmp)
            path = Path(tmp)/'train/annotations_coco.json'
            coco = read_json(path)
            coco['annotations'][0]['segmentation']['counts'] = [0, 1, 63]
            ref = write(path, coco)
            identity = dict(f.identities['train'], annotation_sha256=ref['sha256'])
            categories = {c['id']:c['name'] for c in coco['categories']}
            with self.assertRaisesRegex(ValueError, 'compressed RLE'):
                package_metadata(Path(tmp)/'train', identity, 'train', categories)

    def test_no_data_execute_fails_before_output_or_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'no_run'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_full100.py'),'--execute','--output',str(output)],
                                  cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('no worker started',result.stderr)
            self.assertFalse(output.exists())

    def test_worker_requires_owning_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,'-m','fashion_grounding.segmentation.full100_worker',
                                   '--contract','not_read.json','--mode','train','--output',tmp],
                                  cwd=ROOT,capture_output=True,text=True,
                                  env={k:v for k,v in os.environ.items() if k!='FULL100_COORDINATION_RECORD'})
            self.assertNotEqual(result.returncode,0)
            self.assertIn('shared-lock full100 entry',result.stderr)


class Full100ResourceTests(unittest.TestCase):
    def fixture(self, tmp):
        root = Path(tmp).resolve()
        coordination = root/'coordination'
        coordination.mkdir()
        contract = engine_contract()
        contract['runtime'].update(runs_root=str(root), coordination=str(coordination),
                                   disk_reserve_bytes=1, estimated_checkpoint_bytes=1)
        return contract, root/'segmentation-full100-fixture', coordination

    def test_occupied_gpu_refuses_without_touching_active_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, output, coordination = self.fixture(tmp)
            record = coordination/'current_pair.json'
            write(record, dict(status='running', run_id='C5k-fixture'))
            before = record.read_bytes()
            with patch('sys.platform', 'linux'), patch('subprocess.check_output', side_effect=['123\n', '5461\n']):
                with self.assertRaisesRegex(ValueError, 'GPU occupied'):
                    with gpu_slot(c, output): self.fail('occupied GPU entered')
            self.assertEqual(record.read_bytes(), before)
            self.assertFalse(output.exists())
            self.assertEqual(len(list(coordination.glob('*.json'))), 1)

    def test_foreign_running_record_is_not_reconciled(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, output, coordination = self.fixture(tmp)
            record = coordination/'foreign.json'
            write(record, dict(status='running', entry='other', wrapper_pid=99999999))
            before = record.read_bytes()
            with patch('sys.platform', 'linux'), patch('subprocess.check_output', side_effect=['', '0\n']):
                with self.assertRaisesRegex(ValueError, 'Unresolved running'):
                    with gpu_slot(c, output): self.fail('foreign record ignored')
            self.assertEqual(record.read_bytes(), before)

    def test_explicit_own_resume_reconciles_only_dead_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, output, coordination = self.fixture(tmp)
            output.mkdir()
            record = coordination/'own.json'
            write(record, dict(status='running', entry='full100', output=str(output),
                               contract_sha256=digest_json(c), wrapper_pid=99999999))
            with patch('sys.platform', 'linux'), patch('subprocess.check_output', side_effect=['', '0\n']), \
                 patch('fashion_grounding.segmentation.full100_runtime.pid_exists', return_value=False):
                with gpu_slot(c, output, resume=True) as current:
                    self.assertEqual(read_json(record)['status'], 'interrupted')
                    self.assertEqual(read_json(current)['status'], 'running')
                self.assertEqual(read_json(current)['status'], 'completed')

    def test_held_shared_lock_stops_before_gpu_probe(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            c, output, coordination = self.fixture(tmp)
            with (coordination/'gpu.lock').open('a') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('sys.platform', 'linux'), patch('subprocess.check_output') as probe:
                    with self.assertRaises(BlockingIOError):
                        with gpu_slot(c, output): self.fail('shared lock bypassed')
                    probe.assert_not_called()


class Full100OrchestrationTests(unittest.TestCase):
    def test_complete_order_and_best_is_not_latest_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=engine_contract();worker=FakeWorker(c);out=Path(tmp)/'run'
            result=run_engine(c,out,worker)
            self.assertEqual([(x[0],x[1]) for x in worker.calls],[('train',20),('evaluate',20),('train',40),('evaluate',40)])
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['latest_recovery_update'],40)
            self.assertEqual(read_json(out/'selection.json')['completed_updates'],20)
            self.assertTrue(worker.calls[2][2].endswith('checkpoint_000020.pth'))

    def test_failed_evaluation_reuses_checkpoint_and_prior_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=engine_contract();out=Path(tmp)/'run';first=FakeWorker(c,('evaluate',40))
            with self.assertRaises(RuntimeError):run_engine(c,out,first)
            second=FakeWorker(c);run_engine(c,out,second,resume=True)
            self.assertEqual([(x[0],x[1]) for x in second.calls],[('evaluate',40)])

    def test_mid_segment_restart_uses_latest_epoch_and_absolute_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=engine_contract();out=Path(tmp)/'run';first=FakeWorker(c,('train',30))
            with self.assertRaises(RuntimeError):run_engine(c,out,first)
            second=FakeWorker(c);run_engine(c,out,second,resume=True)
            self.assertEqual([(x[0],x[1]) for x in second.calls],[('train',40),('evaluate',40)])
            self.assertTrue(second.calls[0][2].endswith('checkpoint_000030.pth'))

    def test_resume_mismatch_rejected_without_changing_existing_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=engine_contract();out=Path(tmp)/'run';run_engine(c,out,FakeWorker(c))
            before=(out/'stage_status.json').read_bytes()
            altered=copy.deepcopy(c);altered['source_commit']='b'*40
            with self.assertRaises(ValueError):run_engine(altered,out,FakeWorker(altered),resume=True)
            self.assertEqual((out/'stage_status.json').read_bytes(),before)

    def test_completed_resume_does_not_retrain_or_reinfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=engine_contract();out=Path(tmp)/'run';run_engine(c,out,FakeWorker(c))
            worker=FakeWorker(c);run_engine(c,out,worker,resume=True)
            self.assertEqual(worker.calls,[])

    def test_corrupt_cached_predictions_stop_instead_of_becoming_a_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=engine_contract();out=Path(tmp)/'run';run_engine(c,out,FakeWorker(c))
            p=next(out.glob('attempts/*/evaluate/predictions.json'));p.write_text('changed')
            worker=FakeWorker(c)
            with self.assertRaisesRegex(ValueError,'predictions changed'):run_engine(c,out,worker,resume=True)
            self.assertEqual(worker.calls,[])

    def test_incomplete_or_wrong_population_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=engine_contract();out=Path(tmp)/'run';run_engine(c,out,FakeWorker(c))
            saved=checkpoints(out,c)
            r=read_json(next(out.glob('attempts/*/evaluate/evaluation_report.json')))
            r['evaluated_images']=2
            with self.assertRaisesRegex(ValueError,'Incomplete dev'):validate_report(r,c,saved[r['completed_updates']])
            r['evaluated_images']=3;r['evaluation_config_sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'cache identity'):validate_report(r,c,saved[r['completed_updates']])

    def test_worker_command_keeps_absolute_stop_and_resume_distinct_from_eval(self):
        c=engine_contract()
        train=worker_command(c,'/run','train','/run/new',40,'/run/checkpoint_30.pth',None)
        evaluate=worker_command(c,'/run','evaluate','/run/dev',40,None,'/run/checkpoint_40.pth')
        self.assertIn('--resume',train);self.assertIn('40',train)
        self.assertNotIn('--checkpoint',train);self.assertNotIn('--resume',evaluate)

    def test_streamed_loss_aggregates_do_not_retain_update_rows(self):
        summary=LossSummary()
        for i in range(10000):summary.add(dict(total_loss=float(i+1),image_ids=[1,2],instance_counts=[0,3]))
        self.assertEqual(summary.result()['sample_count'],20000)
        self.assertEqual(summary.result()['empty_augmented_images'],10000)
        self.assertEqual(summary.result()['min_loss'],1)
        self.assertEqual(len(summary.__dict__),7)


if __name__=='__main__':unittest.main()
