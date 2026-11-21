"""Focused regression checks for new staged-expansion wiring; no model tests."""
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from expand_segmentation_staged import incremental_groups,hit_records,GOALS
from build_staged_segmentation import verify_membership
M=(1<<63)-1
def item(key,h,f,g=None):
 return {"key":key,"sha256":"unique-"+key,"original_url":None,"phash":h,"phash_flip":f,"group_id":g}
class StagedExpansion(unittest.TestCase):
 def test_mirror_mirror_only_conflict(self):
  old=item("val-1",0,M,"protected")
  new=item("train-1",0x5555555555555555,M^1)
  eligible,_,edges,blocked=incremental_groups([old],[new])
  self.assertFalse(eligible);self.assertTrue(any(e.get("distance")==1 for e in edges));self.assertEqual(blocked[0]["reason"],"component_reaches_frozen_or_protected")
 def test_bridge_isolates_entire_new_component(self):
  old=[item("train-10",0,0,"train"),item("val-20",4095,4095,"protected")]
  new=[item("train-30",63,63),item("train-31",63,63)]
  eligible,_,edges,blocked=incremental_groups(old,new)
  self.assertFalse(eligible);self.assertEqual(len(blocked),2)
  self.assertTrue(all(e["b"].startswith(("train-30","train-31")) for e in edges))
 def test_one_representative_per_new_group(self):
  eligible,_,_,blocked=incremental_groups([],[item("train-2",42,42),item("train-3",43,43)])
  self.assertEqual(eligible,["train-2"]);self.assertEqual(len(blocked),1)
 def test_thresholds_and_cutout_evidence(self):
  rows=[{"category_id":24,"area":1024},{"category_id":18,"area":256},{"category_id":1,"area":10000,"attribute_ids":[308]}]
  hits=hit_records(rows)
  self.assertTrue({"bag_all","small_bag","small_watch","extreme_priority","complex_boundary_source"}<=hits)
  self.assertNotIn("small_bag",hit_records([{"category_id":24,"area":1025}]))
  self.assertNotIn("complex_boundary_source",hit_records([{"category_id":23,"area":10}]))
 def test_parent_and_delta_union_and_immutable_records(self):
  with tempfile.TemporaryDirectory() as d:
   parent,delta,combined=[Path(d)/k for k in ("p","d","c")]
   fields={"image_manifest.json":["images"],"annotations_coco.json":["images","annotations"],"candidates_gt.json":["images"],"source_subset.json":["images","annotations"]}
   for folder,ids in [(parent,[1]),(delta,[2]),(combined,[1,2])]:
    folder.mkdir()
    for name,keys in fields.items():
     (folder/name).write_text(json.dumps({k:[{"id":i,"image_id":i,"group_id":"fixed-"+str(i)} for i in ids] for k in keys}))
   self.assertTrue(verify_membership(parent,delta,combined))
   path=combined/"image_manifest.json";obj=json.loads(path.read_text());obj["images"][0]["group_id"]="renumbered";path.write_text(json.dumps(obj))
   with self.assertRaises(ValueError):verify_membership(parent,delta,combined)
if __name__=="__main__":unittest.main()
