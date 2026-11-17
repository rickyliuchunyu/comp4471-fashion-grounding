"""Input boundary tests for dev visual releases; no real dataset required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_dev_visual_review import prepare

class DevVisualTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.im=1000000001; self.ann=1000000002
        self.manifest={"images":[{"image_id":self.im,"split":"dev"}]}
        self.write_manifest()
        (self.root/"annotations_coco.json").write_text(json.dumps({"annotations":[
            {"id":self.ann,"image_id":self.im,"iscrowd":0,"category_id":1,
             "source_category_id":0,"source_category_name":"shirt"}]}))
        self.specs=[{"image":1,"target":2,"zh":"红色上衣","en":"the red top","sheet":1}]
        self.old=[{"semantic_query_id":"old","query_id":"old-"+l,"language":l,
                   "image_id":self.im,"target_instance_id":self.ann,"split":"dev",
                   "query":t,"query_types":["appearance"],"review_status":"needs_revision"}
                  for l,t in [("zh","红上衣"),("en","red top")]]
        self.decisions={"old":{"action":"revise","reason":"visually checked","sheet":1,
                               "texts":{"zh":"红色上衣","en":"the red top"}}}
    def write_manifest(self):
        (self.root/"image_manifest.json").write_text(json.dumps(self.manifest))
    def call(self):
        return prepare(self.root,self.old,self.specs,self.decisions,"new-dev")
    def test_preserves_target_and_old_identity_without_accepting(self):
        _,rows,audit,q=self.call()
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["semantic_query_id"],"old")
        self.assertEqual(rows[0]["target_instance_id"],self.ann)
        self.assertTrue(all(r["reviewer"] is None and r["review_status"]=="needs_revision" for r in rows))
        self.assertEqual(audit[0]["old_records"],self.old)
        self.assertEqual(q,[])
    def test_rejects_train_partition(self):
        self.manifest["images"][0]["split"]="train";self.write_manifest()
        with self.assertRaises(ValueError):self.call()
    def test_requires_complete_visual_coverage(self):
        self.specs=[]
        with self.assertRaises(ValueError):self.call()
    def test_requires_every_old_query_audit(self):
        self.decisions={}
        with self.assertRaises(ValueError):self.call()
    def test_rejects_wrong_target(self):
        self.specs[0]["target"]=999
        with self.assertRaises(ValueError):self.call()
    def test_rejects_duplicate_visual_target(self):
        self.specs*=2
        with self.assertRaises(ValueError):self.call()
    def test_rejects_inconsistent_translation_targets(self):
        self.old[1]["target_instance_id"]=999
        with self.assertRaises(ValueError):self.call()
    def test_quarantine_keeps_original_records(self):
        self.decisions["old"]["action"]="quarantine"
        with self.assertRaises(ValueError):self.call()
        data=json.loads((self.root/"annotations_coco.json").read_text())
        alternative=dict(data["annotations"][0],id=1000000003)
        data["annotations"].append(alternative)
        (self.root/"annotations_coco.json").write_text(json.dumps(data))
        self.specs[0]["target"]=3
        _,rows,audit,q=self.call()
        self.assertEqual(q,self.old)
        self.assertEqual(audit[0]["action"],"quarantine")
        self.assertTrue(all(r["semantic_query_id"]!="old" for r in rows))
if __name__=="__main__":unittest.main()
