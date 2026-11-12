import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import test_query_builder as builder_fixture
from publish_reviewed_queries import publish
from run_b0 import run
from evaluate_predictions import evaluate_files
from fashion_grounding.data.schema import write_json,read_json,sha256
from fashion_grounding.data.candidates import load_candidates

class FilePipelineTests(unittest.TestCase):
 def setUp(self):
  self.fixture=builder_fixture.QueryBuilderTests(methodName="test_preserves_all_candidates_and_source_metadata")
  self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.fixture.run_build();self.base=self.fixture.base;self.source=self.base/"out"
  self.review={"manifest_id":"out","review_format_version":2,"reviewer":"wkq","query_sha256":sha256(self.source/"queries_draft.jsonl"),"disclosed_image_ids":[],
   "rows":[{"semantic_query_id":"neutral-q1","decision":"human_selected","selected_annotation_id":1,"note":"","edited_queries":{"zh":"左边的包","en":"left bag"},"checks":{"unique":True,"translation":True,"mask":True}}]}
  self.rp=self.base/"review.json";self.release=self.base/"release"
 def publish(self):
  write_json(self.rp,self.review);return publish(self.source,self.rp,self.release)
 def test_promotion_requires_all_quality_checks(self):
  self.review["rows"][0]["checks"]["mask"]=False
  self.assertEqual(self.publish()["accepted_records"],0)
 def test_edits_and_notes_are_not_implicitly_accepted(self):
  self.review["rows"][0]["edited_queries"]["en"]="red bag"
  self.assertEqual(self.publish()["accepted_records"],0)
 def test_changed_query_hash_rejected(self):
  self.review["query_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"hash"):self.publish()
 def test_exposed_training_review_is_explicit(self):
  self.review["disclosed_image_ids"]=[1]
  self.assertEqual(self.publish()["previously_exposed_accepted_records"],2)
 def test_nontrain_publication_blocked(self):
  # Mutate a valid source split and its references consistently.
  m=read_json(self.source/"image_manifest.json");m["images"][0]["split"]="test";write_json(self.source/"image_manifest.json",m)
  c=read_json(self.source/"candidates_gt.json");c["image_manifest"]["sha256"]=sha256(self.source/"image_manifest.json");write_json(self.source/"candidates_gt.json",c)
  q=[json.loads(s) for s in (self.source/"queries_draft.jsonl").read_text().splitlines()]
  for x in q:x["split"]="test"
  (self.source/"queries_draft.jsonl").write_text("".join(json.dumps(x)+"\n" for x in q))
  self.review["query_sha256"]=sha256(self.source/"queries_draft.jsonl")
  with self.assertRaisesRegex(ValueError,"training"):self.publish()
 def test_complete_portable_b0_to_evaluator_and_hash_guard(self):
  self.assertEqual(self.publish()["accepted_records"],2)
  output=self.base/"run"
  run(self.release/"candidates_gt.json",self.release/"queries_accepted.jsonl",Path(__file__).resolve().parents[1]/"configs/rule_parser.json",output,"unit")
  args=[output/"queries.jsonl",output/"predictions.jsonl",output/"inputs/candidates_gt.json",self.release/"annotations_coco.json"]
  with self.assertRaisesRegex(ValueError,"train diagnostic"):evaluate_files(*args,self.base/"blocked.json")
  result=evaluate_files(*args,self.base/"metrics.json",True)
  self.assertEqual(result["summary"]["oracle_selection_accuracy"],1)
  predictions=[json.loads(s) for s in args[1].read_text().splitlines()]
  predictions[0]["candidate_manifest"]["sha256"]="0"*64
  args[1].write_text("".join(json.dumps(x)+"\n" for x in predictions))
  with self.assertRaisesRegex(ValueError,"hash"):evaluate_files(*args,self.base/"bad.json",True)
 def test_candidate_hash_and_box_tampering(self):
  c=read_json(self.source/"candidates_gt.json")
  c["images"][0]["candidates"][0]["bbox_xyxy"]=[0,0,1,1]
  write_json(self.source/"candidates_gt.json",c)
  with self.assertRaisesRegex(ValueError,"box"):load_candidates(self.source/"candidates_gt.json")
 def test_candidate_duplicate_and_unsupported_status(self):
  c=read_json(self.source/"candidates_gt.json")
  c["images"][0]["candidates"][1]["candidate_id"]=c["images"][0]["candidates"][0]["candidate_id"]
  write_json(self.source/"candidates_gt.json",c)
  with self.assertRaisesRegex(ValueError,"duplicate"):load_candidates(self.source/"candidates_gt.json")
