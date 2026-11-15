import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from expand_segmentation import hits,choose_targeted,isolate_new_groups,merge
from build_large_packages import build_package
from fashion_grounding.data.schema import read_json,encode_mask,sha256
class ExpansionTests(unittest.TestCase):
 def test_four_orientations_catch_mirror_mirror_pair(self):
  from build_large_packages import group_images
  rows=[{"key":"a","sha256":"a","phash":0,"phash_flip":2**63-1},{"key":"b","sha256":"b","phash":2**32-1,"phash_flip":2**63-2}]
  old,_=group_images(rows);self.assertNotEqual(old["a"],old["b"])
  robust,_=group_images(rows,all_orientations=True);self.assertEqual(robust["a"],robust["b"])
 def test_four_orientations_catch_reverse_mirror_pair(self):
  from build_large_packages import group_images
  rows=[{"key":"val-1","sha256":"a","phash":0,"phash_flip":2**32-1},{"key":"train-2","sha256":"b","phash":2**63-1,"phash_flip":1}]
  assignment,_=group_images(rows,all_orientations=True)
  available,_=isolate_new_groups([2],assignment,{"val-1":{}})
  self.assertEqual(available,[])
 def test_small_boundary_inclusive_and_source_fine_preserved(self):
  self.assertIn("small_sock",hits([{"category_id":8,"source_category_id":22,"area":1024}]))
  self.assertNotIn("small_sock",hits([{"category_id":8,"source_category_id":22,"area":1025}]))
  self.assertIn("bag_all",hits([{"category_id":24,"area":9000}]))
 def test_broad_coverage_guides_targeting(self):
  hm={1:{"small_sock"},2:{"small_bag","bag_all"}}
  self.assertEqual(choose_targeted([1,2],hm,1,{"small_sock":300}),[2])
 def test_no_empty_filler_or_duplicate_counting(self):
  self.assertEqual(choose_targeted([1,2],{1:set(),2:{"small_sock"}},5),[2])
 def test_frozen_component_excluded_and_one_new_representative(self):
  a={"val-9":"g1","train-2":"g1","train-3":"g2","train-4":"g2"}
  available,q=isolate_new_groups([2,3,4],a,{"val-9":{}})
  self.assertEqual(available,[3]);self.assertEqual(len(q),2)
  self.assertEqual(q[0]["related_frozen_keys"],["val-9"])
 def test_old_train_component_also_excluded(self):
  available,q=isolate_new_groups([2],{"train-1":"g","train-2":"g"},{"train-1":{}})
  self.assertEqual(available,[]);self.assertEqual(q[0]["reason"],"group_touches_frozen_train_or_dev_test")
 def test_size_report_counts_images_once_and_uses_inclusive_boundaries(self):
  from report_segmentation_expansion import summarize
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   (root/"image_manifest.json").write_text(json.dumps({"images":[{"image_id":1,"group_id":"g","width":100,"height":100}]}))
   anns=[{"image_id":1,"category_id":8,"source_category_id":22,"area":area,"iscrowd":0} for area in (256,1024,1025)]
   (root/"annotations_coco.json").write_text(json.dumps({"annotations":anns}))
   report=summarize(root)
   self.assertEqual(report["small_instances_le1024"],2)
   self.assertEqual(report["extreme_instances_le256"],1)
   self.assertEqual(report["target_image_hits"]["small_sock"],1)
   self.assertEqual(len(report["source_categories"]),26)
   self.assertEqual(report["source_categories"]["22"]["small_images"],1)
 def test_merge_preserves_old_bytes_ids_and_records(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);shared=root/"shared";shared.mkdir();run=root/"run";run.mkdir()
   config=read_json(Path(__file__).resolve().parents[1]/"configs/source_category_mapping.json")
   mapping={x["source_id"]:x for x in config["fashionpedia"]}
   cats=read_json(Path(__file__).resolve().parents[1]/"configs/categories.json")["categories"]
   for iid,name,out in [(7,"a.png",shared/"fp_train5000_v1"),(17,"b.png",run/"fp_added5000_targeted_v1")]:
    path=root/name;Image.new("RGB",(40,40),"red").save(path)
    mask=np.zeros((40,40),bool);mask[5:35,5:35]=True
    im={"id":iid,"width":40,"height":40,"file_name":name,"license":1}
    ann={"id":iid+2,"image_id":iid,"category_id":0,"segmentation":encode_mask(mask),"iscrowd":0}
    sources={"train":{"data":{"info":{},"licenses":[{"id":1}],"categories":[{"id":0,"name":"shirt"}],"attributes":[]},"by":{iid:[ann]},"hash":"0"*64}}
    inv={"key":{"source_image":im,"path":str(path),"sha256":sha256(path),"group_id":"g"+str(iid)}}
    build_package(out,"train",["key"],inv,sources,mapping,config,cats)
   (run/"sampling_sidecar.json").write_text("[]")
   old=shared/"fp_train5000_v1";before={str(p.relative_to(old)):p.read_bytes() for p in old.rglob("*") if p.is_file()}
   with patch("expand_segmentation.SHARED",shared):merge(run)
   out=run/"fp_train10000_targeted_v1"
   self.assertEqual(read_json(out/"validation_report.json")["images"],2)
   for name,data in before.items():self.assertEqual((old/name).read_bytes(),data)
   self.assertEqual(read_json(out/"annotations_coco.json")["annotations"][0],read_json(old/"annotations_coco.json")["annotations"][0])
   self.assertEqual((out/"masks/10.json").read_bytes(),(old/"masks/10.json").read_bytes())
