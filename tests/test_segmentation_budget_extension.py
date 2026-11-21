"""Only new budget-transition behavior; no Detectron2/model smoke."""
import copy
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from fashion_grounding.segmentation.resume_extension import validate_extension, load_extended_checkpoint
from fashion_grounding.segmentation.training_state import (
    read_config, config_sha256, checkpoint_due, UpdateSchedule, StatefulSampler,
    build_optimizer, capture_rng, restore_rng, state_digest, save_checkpoint, load_checkpoint)


CONFIG = Path(__file__).resolve().parents[1]/'configs/runs/segmentation/s0_postfix_1000.json'


class BudgetExtensionTests(unittest.TestCase):
    def setUp(self):
        self.old = read_config(CONFIG)
        self.new = {**self.old, 'max_updates': 5000, 'budget_seconds': 2100,
                    'checkpoint_updates': [2000, 3000, 5000], 'run_scope': 'bounded_extension'}
        self.state = dict(config=self.old, config_sha256=config_sha256(self.old),
                          git_commit='parent', completed_updates=1000)
        self.policy = self.policy_for(self.new)

    def policy_for(self, config):
        return dict(schema_version=1, parent_checkpoint_sha256='a'*64,
                    parent_git_commit='parent', parent_config_sha256=config_sha256(self.old),
                    parent_completed_updates=1000, target_config_sha256=config_sha256(config),
                    parent_source_tree='oldtree', target_source_tree='newtree',
                    allowed_changes={k:{'before':self.old.get(k),'after':config.get(k)}
                        for k in self.old.keys() | config.keys()
                        if k not in self.old or k not in config or self.old[k]!=config[k]})

    def validate(self, state=None, config=None, policy=None, **kw):
        return validate_extension(state or self.state, config or self.new, policy or self.policy,
            kw.get('checkpoint', 'a'*64), 'current', kw.get('parent_tree','oldtree'),
            kw.get('target_tree','newtree'))

    def test_valid_extension_is_non_mutating_and_pins_parent(self):
        before=copy.deepcopy(self.state)
        result=self.validate()
        self.assertEqual(before,self.state)
        self.assertEqual(result['parent_completed_updates'],1000)
        self.assertEqual(result['changes']['max_updates'],{'before':1000,'after':5000})

    def test_recipe_changes_rejected_even_with_matching_policy(self):
        for key,value in [('learning_rate',2e-4),('warmup_updates',0),('seed',43),
                          ('image_size',640),('image_manifest_sha256','f'*64),
                          ('new_recipe_option',None)]:
            with self.subTest(key=key):
                config={**self.new,key:value}
                with self.assertRaisesRegex(ValueError,'training recipe'):
                    self.validate(config=config,policy=self.policy_for(config))

    def test_checkpoint_config_and_source_mismatch_rejected(self):
        for kw in [{'checkpoint':'b'*64},{'parent_tree':'wrong'},{'target_tree':'wrong'},
                   {'state':{**self.state,'git_commit':'wrong'}},
                   {'state':{**self.state,'completed_updates':999}},
                   {'state':{**self.state,'config_sha256':'bad'}},
                   {'policy':{**self.policy,'target_config_sha256':'bad'}},
                   {'policy':{**self.policy,'allowed_changes':{}}}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError): self.validate(**kw)

    def test_nonextension_and_invalid_save_boundaries_rejected(self):
        for limit in [1000,500]:
            config={**self.new,'max_updates':limit}
            with self.assertRaisesRegex(ValueError,'increase'):
                self.validate(config=config,policy=self.policy_for(config))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json'
            for updates in [[],[3000,2000],[2000,2000],[5001],[True]]:
                p.write_text(json.dumps({**self.new,'checkpoint_updates':updates}))
                with self.subTest(updates=updates),self.assertRaisesRegex(ValueError,'checkpoint updates'):
                    read_config(p)
        self.assertEqual([u for u in range(1001,5001) if checkpoint_due(self.new,u,5000)],
                         [2000,3000,5000])

    def test_extended_budget_keeps_lr_optimizer_rng_and_trajectory(self):
        # Exercise the NEW configuration transition across the old max_updates.
        old={**self.old,'max_updates':2,'warmup_updates':2,'lr_milestones':[4]}
        new={**old,'max_updates':6,'budget_seconds':2100,'checkpoint_updates':[4,6]}
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        model=torch.nn.Linear(2,1); opt=build_optimizer(model,old)
        sched=UpdateSchedule(opt,old); sampler=StatefulSampler(3,42)
        def step(model,opt,sched,sampler):
            index=sampler.next(); x=torch.randn(1,2)+random.random()+np.random.rand()
            lr=[g['lr'] for g in opt.param_groups]
            opt.zero_grad();loss=(model(x)-index).square().mean();loss.backward();opt.step();sched.step()
            return index,float(loss.detach()),lr
        for _ in range(2): step(model,opt,sched,sampler)
        state=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),sampler=sampler.state_dict(),rng=capture_rng(),
            config=old,config_sha256=config_sha256(old),data={'hash':'d'*64},
            git_commit='parent',completed_updates=2))
        expected=[step(model,opt,sched,sampler) for _ in range(4)]
        final=state_digest(model.state_dict())
        self.old=old
        policy=self.policy_for(new);policy['parent_completed_updates']=2
        self.validate(state=state,config=new,policy=policy)
        fresh=torch.nn.Linear(2,1);fo=build_optimizer(fresh,new)
        fs=UpdateSchedule(fo,new);sp=StatefulSampler(3,42)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pth';save_checkpoint(path,state)
            # Ordinary resume must still reject a different config hash.
            with self.assertRaisesRegex(ValueError,'configuration or dataset'):
                load_checkpoint(path,config_sha256(new),state['data'])
            saved=load_checkpoint(path,config_sha256(old),state['data'])
            fresh.load_state_dict(saved['model']);fo.load_state_dict(saved['optimizer'])
            fs.load_state_dict(saved['scheduler']);sp.load_state_dict(saved['sampler']);restore_rng(saved['rng'])
            self.assertEqual(state_digest(saved['optimizer']),state_digest(fo.state_dict()))
            actual=[step(fresh,fo,fs,sp) for _ in range(4)]
        self.assertEqual(expected,actual)
        self.assertEqual(final,state_digest(fresh.state_dict()))

    def test_loader_binds_real_checkpoint_and_reviewed_source_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
            git('init');git('config','user.email','fixture@example.invalid');git('config','user.name','Fixture')
            p=root/'src/fashion_grounding/fixture.py';p.parent.mkdir(parents=True);p.write_text('old\n')
            git('add','.');git('-c','commit.gpgsign=false','commit','-m','parent')
            parent=git('rev-parse','HEAD');parent_tree=git('rev-parse','HEAD:src/fashion_grounding')
            p.write_text('reviewed extension\n');git('add','.');git('-c','commit.gpgsign=false','commit','-m','extension')
            current=git('rev-parse','HEAD');target_tree=git('rev-parse','HEAD:src/fashion_grounding')
            state={**self.state,'git_commit':parent,'data':{'hash':'d'*64},'scheduler':{'completed_updates':1000}}
            checkpoint=root/'checkpoint.pth';digest=save_checkpoint(checkpoint,state)
            policy={**self.policy,'parent_git_commit':parent,'parent_checkpoint_sha256':digest,
                    'parent_source_tree':parent_tree,'target_source_tree':target_tree}
            policy_path=root/'transition.json';policy_path.write_text(json.dumps(policy))
            loaded,record=load_extended_checkpoint(checkpoint,self.new,state['data'],policy_path,current,root)
            self.assertEqual(loaded,state)
            self.assertEqual(record['target_git_commit'],current)
            with self.assertRaisesRegex(ValueError,'configuration or dataset'):
                load_extended_checkpoint(checkpoint,self.new,{'hash':'wrong'},policy_path,current,root)
            policy['target_source_tree']=parent_tree;policy_path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError,'pinned checkpoint/config/source'):
                load_extended_checkpoint(checkpoint,self.new,state['data'],policy_path,current,root)


if __name__=='__main__': unittest.main()
