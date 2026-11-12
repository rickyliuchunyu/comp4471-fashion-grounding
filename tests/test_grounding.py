"""Hand-checkable grounding boundaries; fake features are not model evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from fashion_grounding.data.schema import write_json, sha256
from fashion_grounding.grounding.inputs import CandidatePack, inference_query, crop_regions, Candidate
from fashion_grounding.grounding.cache import FeatureCache, visual_spec, text_spec, cache_key
from fashion_grounding.grounding.rules import category_ranks,position_scores,rule_coverage
from fashion_grounding.grounding.predict import rank_scores,prediction_record,read_queries,stage_inputs,execute

ROOT=Path(__file__).resolve().parents[1]
class FakeEncoder:
    dimension=2;logit_scale=0.;logit_bias=0.;signature={"revision":"synthetic-test-only"}
    def __init__(self): self.image_calls=0;self.text_calls=0
    def encode_images(self,images):
        self.image_calls+=1
        return np.eye(2,dtype=np.float32)
    def encode_texts(self,texts):
        self.text_calls+=1
        return np.array([[1.,0.]],dtype=np.float32)

class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        shutil.copytree(ROOT/"examples",self.root/"examples")
        self.pack=CandidatePack(self.root/"examples/candidates_gt.json",True)
        self.raw=json.loads((self.root/"examples/query_record.json").read_text())
        self.q=inference_query(self.raw)
        self.image,self.cs=self.pack.load(self.q.image_id)
        self.cache=FeatureCache(self.root/"cache")
        self.ref={"path":"inputs/candidates_gt.json","sha256":self.pack.digest}
    def tearDown(self): self.tmp.cleanup()
    def parsed(self,**kw):
        return dict(category_id=7,direction="left",category_status="unique",
                    direction_status="unique",unsupported_operator=False,**kw)
    def test_rgb_crops_match_original_pixels(self):
        crops=crop_regions(self.image,self.cs)
        self.assertEqual([x.size for x in crops],[(2,4),(2,4)])
        self.assertEqual(crops[0].mode,"RGB")
        np.testing.assert_array_equal(np.asarray(crops[0]),np.asarray(self.image)[1:5,1:3])
    def test_answers_do_not_cross_inference_boundary(self):
        changed={**self.raw,"target_instance_id":999,"target_category_id":1,"query_types":["fake"]}
        self.assertEqual(inference_query(changed),self.q)
        self.assertFalse(hasattr(self.q,"target_instance_id"))
        self.assertFalse(hasattr(self.cs[0],"source_annotation_id"))
    def test_corrupt_mask_rejected(self):
        p=self.root/"examples/fixtures/mask_1.json";p.write_text("{}")
        with self.assertRaisesRegex(ValueError,"mask hash"):self.pack.load(self.q.image_id)
    def test_duplicate_candidate_rejected(self):
        p=self.root/"examples/candidates_gt.json";obj=json.loads(p.read_text())
        obj["images"][0]["candidates"][1]["candidate_id"]=obj["images"][0]["candidates"][0]["candidate_id"]
        write_json(p,obj)
        with self.assertRaisesRegex(ValueError,"candidate ID"): CandidatePack(p,True)
    def test_unreviewed_queries_rejected_by_default(self):
        path=self.root/"q.jsonl";path.write_text(json.dumps(self.raw)+"\n")
        with self.assertRaisesRegex(ValueError,"unreviewed"):read_queries(path,self.pack)
        self.assertEqual(read_queries(path,self.pack,True),[self.q])
    def test_empty_and_duplicate_queries_rejected(self):
        path=self.root/"q.jsonl";path.write_text("")
        with self.assertRaisesRegex(ValueError,"empty"):read_queries(path,self.pack,True)
        path.write_text((json.dumps(self.raw)+"\n")*2)
        with self.assertRaisesRegex(ValueError,"duplicate"):read_queries(path,self.pack,True)
    def test_b1_hand_calculation(self):
        scores=rank_scores(np.eye(2,dtype=np.float32),np.array([[0.,1.]],np.float32),np.log(2),-3)
        np.testing.assert_allclose(scores,[-3,-1])
        record=prediction_record(self.q,self.pack,"run","B1",self.ref,self.cs,scores)
        self.assertEqual(record["selected_candidate"]["candidate_id"],self.cs[1].candidate_id)
    def test_position_rank_and_direction(self):
        np.testing.assert_array_equal(category_ranks(self.cs,0),[0,1])
        np.testing.assert_array_equal(position_scores(self.cs,self.parsed()),[1,-1])
        p=self.parsed();p["direction"]="right"
        np.testing.assert_array_equal(position_scores(self.cs,p),[-1,1])
    def test_position_ties_singletons_and_other_classes(self):
        a=self.cs[0]
        b=Candidate("tie",7,None,a.bbox_xyxy,a.mask,None)
        c=Candidate("other",1,None,self.cs[1].bbox_xyxy,self.cs[1].mask,None)
        np.testing.assert_array_equal(category_ranks([a,b,c],0),[.5,.5,.5])
        np.testing.assert_array_equal(position_scores([a,b,c],self.parsed()),[0,0,0])
    def test_shared_parser_coverage_and_singleton_no_effect(self):
        from dataclasses import replace
        lexicon=json.loads((ROOT/"configs/rule_parser.json").read_text())
        q=replace(self.q, query="the bag on the left", language="en")
        counts=rule_coverage(self.pack,[q],lexicon,1)
        self.assertEqual([counts[k] for k in ("queries","syntax_applicable","matching_category","nonzero_rule")],[1,1,1,1])
        self.assertEqual(rule_coverage(self.pack,[q],lexicon,0)["nonzero_weighted_rule"],0)
        self.pack.rows[q.image_id]["candidates"]=self.pack.rows[q.image_id]["candidates"][:1]
        counts=rule_coverage(self.pack,[q],lexicon,1)
        self.assertEqual(counts["matching_category"],1)
        self.assertEqual(counts["nonzero_rule"],0)

    def test_unsupported_parser_preserves_b1(self):
        for key,value in [("category_status","conflict"),("direction_status","missing"),("unsupported_operator",True)]:
            p=self.parsed();p[key]=value
            np.testing.assert_array_equal(position_scores(self.cs,p),[0,0])
    def test_cache_corruption_detected(self):
        spec={"test":"one"};v=np.eye(2,dtype=np.float32);self.cache.put(spec,v)
        np.testing.assert_array_equal(self.cache.get(spec,2,2),v)
        (self.root/"cache"/cache_key(spec)/"features.npy").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError,"corrupted"):self.cache.get(spec,2,2)
    def test_cache_invalidates_source_revision_and_text(self):
        spec=visual_spec(self.pack,self.q.image_id,{"revision":"a"})
        for field in ["candidate_source","candidate_ids","candidate_manifest_sha256","encoder"]:
            changed=copy.deepcopy(spec);changed[field]="different"
            self.assertNotEqual(cache_key(spec),cache_key(changed))
        self.assertNotEqual(cache_key(text_spec("left","en",{})),cache_key(text_spec("right","en",{})))
        self.assertNotEqual(cache_key(text_spec("left","en",{})),cache_key(text_spec("left","zh",{})))
    def test_warm_cache_identical_and_no_reencoding(self):
        e=FakeEncoder()
        first=execute(self.pack,[self.q],e,self.cache,"r","B1",self.ref)
        second=execute(self.pack,[self.q],e,self.cache,"r","B1",self.ref)
        self.assertEqual(first,second);self.assertEqual((e.image_calls,e.text_calls),(1,1))
    def test_inference_failure_preserves_query(self):
        e=FakeEncoder()
        def fail(_): raise RuntimeError("private failure path")
        e.encode_texts=fail
        result=execute(self.pack,[self.q],e,self.cache,"r","B1",self.ref)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0][0]["status"],"inference_error")
        self.assertNotIn("private failure path",json.dumps(result[0][0]))
    def test_no_candidates_and_upstream_error_distinct(self):
        for status in ("no_candidates","inference_error"):
            self.pack.rows[self.q.image_id]["status"]=status
            self.pack.rows[self.q.image_id]["candidates"]=[]
            result=execute(self.pack,[self.q],FakeEncoder(),self.cache,"r","B1",self.ref)
            self.assertEqual(result[0][0]["status"],status)
            self.assertIsNone(result[0][0]["selected_candidate"])
    def test_staged_inputs_resolve_and_keep_hash(self):
        out=self.root/"run";out.mkdir()
        ref=stage_inputs(self.pack,out)
        other=CandidatePack(out/ref["path"],True)
        self.assertEqual(other.digest,self.pack.digest)
        image,cs=other.load(self.q.image_id)
        self.assertEqual(image.size,self.image.size)
        self.assertEqual([c.candidate_id for c in cs],[c.candidate_id for c in self.cs])
    def test_b2_uses_injected_shared_parser_and_complete_query(self):
        calls=[]
        def parser(text,language,lexicon):
            calls.append((text,language));p=self.parsed();p["direction"]="right";return p
        result=execute(self.pack,[self.q],FakeEncoder(),self.cache,"r","B2",self.ref,2,{},parser)
        self.assertEqual(calls,[(self.q.query,self.q.language)])
        self.assertEqual(result[0][0]["selected_candidate"]["candidate_id"],self.cs[1].candidate_id)

if __name__=="__main__": unittest.main()
