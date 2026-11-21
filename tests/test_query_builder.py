import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_query_review import build
from render_query_review import review_payload
from fashion_grounding.data.schema import write_json,encode_mask,sha256
class QueryBuilderTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.base=Path(self.tmp.name);self.source=self.base/"source";self.source.mkdir()
  (self.source/"images").mkdir();(self.source/"masks").mkdir()
  Image.new("RGB",(10,10),"white").save(self.source/"images/a.png")
  im={"image_id":1,"file_name":"images/a.png","width":10,"height":10,"sha256":sha256(self.source/"images/a.png"),"source_dataset":"fashionpedia","source_image_id":"0","group_id":"g1","split":"train"}
  common={"schema_version":3,"illustrative_only":False}
  manifest={**common,"manifest_id":"source","mapping_version":"x","images":[im]}
  write_json(self.source/"image_manifest.json",manifest)
  annotations=[];candidates=[]
  for aid,x in [(1,1),(2,6)]:
   mask=np.zeros((10,10),bool);mask[2:5,x:x+2]=True;rle=encode_mask(mask)
   a={"id":aid,"image_id":1,"source_annotation_id":aid-1,"category_id":7,"source_category_id":24,"source_category_name":"bag, wallet","iscrowd":0,"area":6,"bbox":[x,2,2,3],"segmentation":rle};annotations.append(a)
   write_json(self.source/f"masks/{aid}.json",rle)
   candidates.append({"candidate_id":f"c{aid}","category_id":7,"score":None,"bbox_xyxy":[x,2,x+2,5],"mask_reference":f"masks/{aid}.json","mask_sha256":sha256(self.source/f"masks/{aid}.json"),"decoder_query_id":None,"source_annotation_id":aid})
  ci={**im,"id":1,"license":5,"supervised_category_ids":list(range(1,9))};del ci["image_id"]
  coco={**common,"images":[ci],"annotations":annotations,"licenses":[{"id":5}],"categories":[{"id":7,"name":"bag"}]}
  write_json(self.source/"annotations_coco.json",coco)
  write_json(self.source/"source_subset.json",{"images":[{"id":0}],"annotations":[],"categories":[]})
  write_json(self.source/"mapping_config.json",{"mapping_version":"x"})
  write_json(self.source/"candidates_gt.json",{**common,"candidate_source":"gt","candidate_set_id":"s","image_manifest":{"path":"image_manifest.json","sha256":sha256(self.source/"image_manifest.json")},"provenance":{"annotation_manifest_sha256":sha256(self.source/"annotations_coco.json")},"images":[{"image_id":1,"image_size_wh":[10,10],"status":"ok","candidates":candidates}]})
  for name in ("queries_draft.jsonl","queries_accepted.jsonl"):(self.source/name).write_text("")
  self.spec={"source_packages":[str(self.source)],"drafts":[{"semantic_query_id":"neutral-q1","image_id":1,"target_instance_id":1,"zh":"左边的包","en":"left bag","query_types":["spatial"],"competing_instance_ids":[2]}]}
 def run_build(self):
  write_json(self.base/"spec.json",self.spec)
  return build(self.base/"spec.json",self.base/"out")
 def test_preserves_all_candidates_and_source_metadata(self):
  result=self.run_build();self.assertEqual(result["instances"],2);self.assertEqual(result["accepted"],0)
  out=json.loads((self.base/"out/annotations_coco.json").read_text())
  self.assertEqual(out["images"][0]["license"],5)
  self.assertEqual(out["images"][0]["supervised_category_ids"],list(range(1,9)))
  self.assertEqual((self.base/"out/masks/1.json").read_bytes(),(self.source/"masks/1.json").read_bytes())
 def test_blind_payload_does_not_include_answers(self):
  self.run_build()
  d=review_payload(self.base/"out")
  self.assertEqual(set(d["queries"][0]),{"semantic_query_id","image_id","zh","en"})
  for forbidden in ("target_instance_id","draft_target","target_category_id","query_types","competing_instance_ids"):
   self.assertNotIn(forbidden,json.dumps(d))
 def test_invalid_rival_rejected(self):
  self.spec["drafts"][0]["competing_instance_ids"]=[1]
  with self.assertRaisesRegex(ValueError,"invalid rival"):self.run_build()
 def test_unknown_target_rejected(self):
  self.spec["drafts"][0]["target_instance_id"]=999
  with self.assertRaisesRegex(ValueError,"unknown target"):self.run_build()
 def test_duplicate_semantic_rejected(self):
  self.spec["drafts"].append(copy.deepcopy(self.spec["drafts"][0]))
  with self.assertRaisesRegex(ValueError,"duplicate semantic"):self.run_build()
 def test_published_output_not_overwritten(self):
  self.run_build()
  with self.assertRaisesRegex(ValueError,"new output version"):self.run_build()
