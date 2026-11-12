import copy,json,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.query_rules import parse_query
from fashion_grounding.baselines.category_position import select_target
from fashion_grounding.evaluation.matching import associate,mask_iou
from fashion_grounding.evaluation.metrics import evaluate

LEX=json.loads((Path(__file__).resolve().parents[1]/"configs/rule_parser.json").read_text())
class ParserTests(unittest.TestCase):
 def test_contract_examples(self):
  for text,lang,cat,direction in [("左边的包","zh",7,"left"),("the bag on the left","en",7,"left"),("连衣裙","zh",5,None),("the dress","en",5,None),("钱包","zh",7,None)]:
   p=parse_query(text,lang,LEX);self.assertEqual((p.category_id,p.direction),(cat,direction));self.assertFalse(p.unconsumed_spans)
 def test_bright_is_not_right(self):
  p=parse_query("bright bag","en",LEX);self.assertIsNone(p.direction);self.assertEqual(p.unconsumed_spans,["bright"])
 def test_appearance_preserved_for_b2(self):
  p=parse_query("the red bag on the right","en",LEX);self.assertEqual(p.direction,"right");self.assertIn("red",p.unconsumed_spans)
 def test_negation_before_filler(self):
  for text,lang in [("不是左边的包","zh"),("not the bag on the left","en")]:
   self.assertTrue(parse_query(text,lang,LEX).unsupported_operator)
 def test_conflicts(self):
  self.assertEqual(parse_query("左边和右边的包","zh",LEX).direction_status,"conflict")
  self.assertEqual(parse_query("the bag and the shoe","en",LEX).category_status,"conflict")
 def test_normalization_and_boundaries(self):
  self.assertEqual(parse_query("ＴＨＥ  ＢＡＧ！","en",LEX).category_id,7)
  self.assertEqual(parse_query("handbag","en",LEX).category_status,"missing")
 def test_repeated_alias_same_category(self):
  self.assertEqual(parse_query("bag wallet","en",LEX).category_status,"unique")
 def test_unknown_language(self):
  with self.assertRaises(ValueError):parse_query("bag","fr",LEX)

class B0Tests(unittest.TestCase):
 def setUp(self):
  self.cs=[{"candidate_id":"a","category_id":7,"bbox_xyxy":[0,0,2,2]},{"candidate_id":"b","category_id":7,"bbox_xyxy":[4,4,6,6]}]
 def select(self,q,cs=None,status="ok"):return select_target(q,"en",self.cs if cs is None else cs,LEX,status)
 def test_four_directions(self):
  for d,cid in [("left","a"),("right","b"),("upper","a"),("lower","b")]:
   self.assertEqual(self.select("the "+d+" bag")["selected_candidate"]["candidate_id"],cid)
 def test_ambiguous_class_and_unknown_modifier(self):
  for q in ["the bag","the red bag","not the left bag","the shoe"]:
   self.assertEqual(self.select(q)["status"],"rule_unsupported")
 def test_unique_class(self):
  self.assertEqual(self.select("bag",self.cs[:1])["status"],"success")
 def test_coordinate_tie_is_not_id_tiebreak(self):
  self.cs[1]["bbox_xyxy"]=[0,4,2,6]
  self.assertEqual(self.select("left bag")["status"],"rule_unsupported")
 def test_empty_and_error(self):
  self.assertEqual(self.select("bag",[])["status"],"no_candidates")
  self.assertEqual(self.select("bag",[],"inference_error")["status"],"inference_error")
 def test_poisoned_extra_labels_cannot_change_choice(self):
  self.cs[0]["target_instance_id"]=999
  self.assertEqual(self.select("right bag")["selected_candidate"]["candidate_id"],"b")

class EvaluationTests(unittest.TestCase):
 def setUp(self):
  self.a=np.array([[1,1,0,0],[1,1,0,0]],bool);self.b=~self.a
  self.anns=[{"id":1,"image_id":1,"category_id":7,"iscrowd":0,"area":4},{"id":2,"image_id":1,"category_id":7,"iscrowd":0,"area":4}]
  self.gt={1:self.a,2:self.b}
  self.cs=[{"candidate_id":"p1","category_id":7,"score":.9},{"candidate_id":"p2","category_id":7,"score":.8}]
  self.cm={"p1":self.a,"p2":self.b}
  self.pkg={"candidate_source":"predicted","candidate_set_id":"s","images":[{"image_id":1,"status":"ok","candidates":self.cs}]}
  self.q=[{"query_id":"q"+str(i),"semantic_query_id":"s"+str(i),"image_id":1,"query":"bag","language":"en","split":"dev","target_instance_id":t,"review_status":"accepted","query_types":["category"]} for i,t in enumerate([1,2])]
  self.p=[{"schema_version":3,"illustrative_only":False,**{k:q[k] for k in ("query_id","image_id","query","language","split")},"run_id":"r","method_id":"B0","candidate_source":"predicted","candidate_set_id":"s","status":"success","selected_candidate":{"candidate_id":"p"+str(i+1),"grounding_score":None},"error":None} for i,q in enumerate(self.q)]
 def result(self):return evaluate(self.q,self.p,self.pkg,self.cm,self.anns,self.gt)
 def test_perfect(self):
  s=self.result()["summary"];self.assertEqual(s["predicted_selection_accuracy"],1);self.assertEqual(s["mean_target_mask_iou"],1)
 def test_miss_failure_denominator_and_identity(self):
  self.cs.pop();self.p[1].update(status="no_output",selected_candidate=None)
  s=self.result()["summary"]
  self.assertEqual(s["queries"],2);self.assertEqual(s["predicted_selection_accuracy"],.5);self.assertEqual(s["matched_candidate_coverage"],.5);self.assertEqual(s["conditional_selection_accuracy"],1)
  self.assertEqual(s["predicted_selection_accuracy"],s["matched_candidate_coverage"]*s["conditional_selection_accuracy"])
 def test_wrong_class_geometric_success_not_selection(self):
  for c in self.cs:c["category_id"]=4
  s=self.result()["summary"]
  self.assertEqual(s["geometric_recall_at_0_5"],1);self.assertEqual(s["success_at_0_5"],1);self.assertEqual(s["predicted_selection_accuracy"],0);self.assertEqual(s["matched_candidate_coverage"],0);self.assertIsNone(s["conditional_selection_accuracy"])
 def test_duplicate_candidate_cannot_claim_same_gt(self):
  self.cm["p2"]=self.a
  self.assertEqual(associate(self.cs,self.anns,self.cm,self.gt),{"p1":1})
 def test_score_and_id_tiebreak(self):
  self.cm["p2"]=self.a;self.cs[1]["score"]=.95
  self.assertEqual(associate(self.cs,self.anns,self.cm,self.gt),{"p2":1})
  self.cs[0]["score"]=.95
  self.assertEqual(associate(list(reversed(self.cs)),self.anns,self.cm,self.gt),{"p1":1})
 def test_gt_tiebreak_and_crowd(self):
  self.gt[2]=self.a
  self.assertEqual(associate(self.cs[:1],list(reversed(self.anns)),self.cm,self.gt),{"p1":1})
  self.anns[0]["iscrowd"]=1
  self.assertEqual(associate(self.cs[:1],self.anns,self.cm,self.gt),{"p1":2})
 def test_exact_half_iou_threshold(self):
  self.cm["p1"]=np.ones((2,4),bool)
  self.assertEqual(mask_iou(self.cm["p1"],self.a),.5)
  self.assertEqual(associate(self.cs[:1],self.anns,self.cm,self.gt),{"p1":1})
 def test_wrong_instance_preserves_actual_iou(self):
  self.gt[2]=np.ones((2,4),bool);self.anns[1]["area"]=8
  self.cm["p2"]=self.gt[2];self.p[0]["selected_candidate"]["candidate_id"]="p2"
  row=self.result()["rows"][0];self.assertFalse(row["correct"]);self.assertEqual(row["target_mask_iou"],.5)
 def test_empty_candidates(self):
  self.pkg["images"][0].update(candidates=[],status="no_candidates")
  for p in self.p:p.update(status="no_candidates",selected_candidate=None)
  s=self.result()["summary"];self.assertEqual(s["mean_target_mask_iou"],0);self.assertIsNone(s["conditional_selection_accuracy"])
 def test_upstream_error_must_propagate(self):
  self.pkg["images"][0].update(candidates=[],status="inference_error")
  for p in self.p:p.update(status="no_output",selected_candidate=None)
  with self.assertRaises(ValueError):self.result()
  for p in self.p:p.update(status="inference_error",error={"code":"x","message":"failed"})
  self.assertEqual(self.result()["summary"]["queries"],2)
 def test_missing_duplicate_unknown_predictions(self):
  p=copy.deepcopy(self.p)
  for rows in [p[:1],p+[p[0]],[{**p[0],"query_id":"unknown"},p[1]]]:
   self.p=rows
   with self.assertRaises(ValueError):self.result()
 def test_mixed_run_and_bad_selected_id(self):
  self.p[1]["run_id"]="other"
  with self.assertRaises(ValueError):self.result()
  self.p[1]["run_id"]="r";self.p[1]["selected_candidate"]["candidate_id"]="unknown"
  with self.assertRaises(ValueError):self.result()
 def test_gt_oracle(self):
  self.pkg["candidate_source"]="gt"
  for i,c in enumerate(self.cs):c.update(source_annotation_id=i+1,score=None)
  for p in self.p:p["candidate_source"]="gt"
  s=self.result()["summary"];self.assertEqual(s["oracle_selection_accuracy"],1);self.assertIsNone(s["predicted_selection_accuracy"])
 def test_external_does_not_claim_internal_accuracy(self):
  self.pkg["candidate_source"]="external"
  for p in self.p:p["candidate_source"]="external"
  s=self.result()["summary"];self.assertIsNone(s["macro_category_accuracy"]);self.assertIsNone(s["predicted_selection_accuracy"]);self.assertEqual(s["success_at_0_5"],1)
 def test_paired_both_wrong_not_success(self):
  self.q[1]["semantic_query_id"]=self.q[0]["semantic_query_id"]
  self.q[1]["language"]="zh";self.q[1]["target_instance_id"]=1;self.p[1]["language"]="zh"
  self.p[0]["selected_candidate"]["candidate_id"]="p2"
  pair=self.result()["paired"]["semantic_query_id"]
  self.assertEqual(pair["selection_consistency"],1);self.assertEqual(pair["all_correct_rate"],0)
 def test_unreviewed_rejected(self):
  self.q[0]["review_status"]="needs_revision"
  with self.assertRaises(ValueError):self.result()
 def test_no_empty_evaluation_or_nan_scores(self):
  with self.assertRaises(ValueError):evaluate([],[],self.pkg,self.cm,self.anns,self.gt)
  self.p[0]["selected_candidate"]["grounding_score"]=float("nan")
  with self.assertRaises(ValueError):self.result()
