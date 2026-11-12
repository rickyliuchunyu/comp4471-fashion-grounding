import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_large_packages import group_images,phash,build_package
from build_machine_queries import dominant_color,proposed_queries,select_images
from fashion_grounding.data.schema import encode_mask

class LargeDataTests(unittest.TestCase):
 def test_exact_hash_group_even_different_phash(self):
  items=[{"key":"train-1","sha256":"same","phash":0,"phash_flip":0},{"key":"val-2","sha256":"same","phash":2**63-1,"phash_flip":2**63-1}]
  groups,_=group_images(items);self.assertEqual(groups["train-1"],groups["val-2"])
 def test_mirror_near_duplicate_group(self):
  items=[{"key":"a","sha256":"a","phash":0,"phash_flip":2**63-1},{"key":"b","sha256":"b","phash":2**63-2,"phash_flip":0}]
  groups,pairs=group_images(items);self.assertEqual(groups["a"],groups["b"]);self.assertEqual(len(pairs),1)
 def test_phash_threshold_six_not_seven(self):
  items=[{"key":"a","sha256":"a","phash":0,"phash_flip":0},{"key":"b","sha256":"b","phash":127,"phash_flip":127}]
  groups,_=group_images(items);self.assertNotEqual(groups["a"],groups["b"])
  items[1]["phash"]=63;groups,_=group_images(items);self.assertEqual(groups["a"],groups["b"])
 def test_source_url_transitive_group(self):
  items=[{"key":"a","sha256":"1","original_url":"https://example.org/photo.jpg","phash":0,"phash_flip":0},{"key":"b","sha256":"2","original_url":"https://example.org/photo.jpg","phash":2**63-1,"phash_flip":2**63-1},{"key":"c","sha256":"2","phash":1234567,"phash_flip":1234567}]
  groups,_=group_images(items);self.assertEqual(len(set(groups.values())),1)
 def test_website_homepage_not_an_image_identity(self):
  items=[{"key":"a","sha256":"a","original_url":"https://unsplash.com","phash":0,"phash_flip":0},{"key":"b","sha256":"b","original_url":"https://unsplash.com","phash":2**63-1,"phash_flip":2**63-1}]
  groups,_=group_images(items);self.assertNotEqual(groups["a"],groups["b"])
 def test_color_uses_mask_not_background(self):
  rgb=np.zeros((60,60,3),np.uint8);rgb[:]=[0,0,255];rgb[10:50,10:50]=[255,0,0]
  m=np.zeros((60,60),bool);m[10:50,10:50]=True
  color,confidence=dominant_color(rgb,m);self.assertEqual(color,3);self.assertEqual(confidence,1)
 def test_ambiguous_color_not_named(self):
  rgb=np.zeros((60,60,3),np.uint8);rgb[:30]=[255,0,0];rgb[30:]=[0,0,255]
  color,_=dominant_color(rgb,np.ones((60,60),bool));self.assertIsNone(color)
 def test_small_mask_not_named(self):
  color,_=dominant_color(np.ones((10,10,3),np.uint8)*255,np.ones((10,10),bool));self.assertIsNone(color)
 def test_clothing_fragments_not_automatic_spatial_pair(self):
  rgb=np.ones((100,200,3),np.uint8)*50;anns=[]
  for aid,x in [(1,10),(2,120)]:
   m=np.zeros((100,200),bool);m[10:90,x:x+60]=True
   anns.append({"id":aid,"source_category_id":6,"category_id":2,"iscrowd":0,"area":4800,"bbox":[x,10,60,80],"segmentation":encode_mask(m)})
  props=proposed_queries({"image_id":1,"height":100,"width":200},anns,rgb)
  self.assertFalse(any(p["kind"]=="spatial" for p in props))
 def test_unknown_rival_color_does_not_prove_uniqueness(self):
  rgb=np.zeros((100,200,3),np.uint8);rgb[:]=[255,0,0];anns=[]
  for aid,x,w in [(1,10,60),(2,120,5)]:
   m=np.zeros((100,200),bool);m[10:90,x:x+w]=True
   anns.append({"id":aid,"source_category_id":24,"category_id":7,"iscrowd":0,"area":80*w,"bbox":[x,10,w,80],"segmentation":encode_mask(m)})
  self.assertEqual(proposed_queries({"image_id":1,"height":100,"width":200},anns,rgb),[])
 def test_broad_top_checks_other_source_fine_classes(self):
  rgb=np.zeros((100,200,3),np.uint8);rgb[:]=[255,0,0];anns=[]
  for aid,x,fine in [(1,10,1),(2,120,0)]:
   m=np.zeros((100,200),bool);m[10:90,x:x+60]=True
   anns.append({"id":aid,"source_category_id":fine,"category_id":1,"iscrowd":0,"area":4800,"bbox":[x,10,60,80],"segmentation":encode_mask(m)})
  self.assertEqual(proposed_queries({"image_id":1,"height":100,"width":200},anns,rgb),[])
 def test_selection_does_not_claim_unavailable_images(self):
  props=[{"kind":"appearance","target":{"id":1,"category_id":7}}]
  self.assertEqual(len(select_images({1:props},10)),1)

class ValidationNamespaceTests(unittest.TestCase):
 def test_validation_ids_and_source_masks_preserved(self):
  from collections import defaultdict
  from fashion_grounding.data.schema import read_json,sha256
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);image=root/"a.png";Image.new("RGB",(40,40),"red").save(image)
   mask=np.zeros((40,40),bool);mask[5:35,5:35]=True
   im={"id":7,"width":40,"height":40,"file_name":"a.png","license":1}
   ann={"id":9,"image_id":7,"category_id":0,"segmentation":encode_mask(mask),"iscrowd":0}
   config=read_json(Path(__file__).resolve().parents[1]/"configs/source_category_mapping.json")
   mapping={r["source_id"]:r for r in config["fashionpedia"]}
   categories=read_json(Path(__file__).resolve().parents[1]/"configs/categories.json")["categories"]
   data={"info":{},"licenses":[{"id":1}],"categories":[{"id":0,"name":"shirt, blouse"}],"attributes":[]}
   sources={"val":{"data":data,"by":{7:[ann]},"hash":"0"*64}}
   inventory={"val-7":{"source_image":im,"path":str(image),"sha256":sha256(image),"group_id":"v7"}}
   build_package(root/"out","dev",["val-7"],inventory,sources,mapping,config,categories)
   coco=read_json(root/"out/annotations_coco.json")
   self.assertEqual(coco["images"][0]["id"],1000000008)
   self.assertEqual(coco["annotations"][0]["id"],1000000010)
   self.assertEqual(coco["annotations"][0]["source_annotation_id"],9)
   self.assertEqual(coco["annotations"][0]["segmentation"],encode_mask(mask))
