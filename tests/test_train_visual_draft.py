import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_train_visual_draft import prepare

class TrainVisualTests(unittest.TestCase):
    def setUp(self):
        self.m={"images":[{"image_id":1,"split":"train"}]}
        self.c={"annotations":[dict(id=10,image_id=1,category_id=1,source_category_id=0,source_category_name="shirt",iscrowd=0)]}
        self.q=[dict(query_id="q-"+l,semantic_query_id="q",image_id=1,target_instance_id=10,target_category_id=1,split="train",language=l,query=t) for l,t in (("zh","白色衬衫"),("en","the white shirt"))]
        self.s={"viewed_sheets":[1],"overrides":{},"replacements":[],"unresolved_images":[]}
        self.i=[{"image_id":1,"sheet":1,"targets":[{"id":10}]}]
    def run_prepare(self):
        return prepare(self.m,self.c,self.q,self.s,self.i,"version")
    def test_machine_review_never_becomes_human_acceptance(self):
        rows,quarantine,audit=self.run_prepare()
        self.assertEqual(len(rows),2)
        self.assertTrue(all(q["review_status"]=="needs_revision" and q["reviewer"] is None for q in rows))
        self.assertEqual(audit[0]["action"],"retained")
    def test_incomplete_visual_coverage_rejected(self):
        self.s["viewed_sheets"]=[]
        with self.assertRaisesRegex(ValueError,"incomplete visual"):self.run_prepare()
    def test_wrong_target_image_rejected(self):
        self.c["annotations"][0]["image_id"]=2
        with self.assertRaisesRegex(ValueError,"target/image mismatch"):self.run_prepare()
    def test_incomplete_translation_rejected(self):
        self.q.pop()
        with self.assertRaisesRegex(ValueError,"incomplete bilingual"):self.run_prepare()
    def test_nontrain_rejected(self):
        self.m["images"][0]["split"]="dev"
        with self.assertRaisesRegex(ValueError,"train-only"):self.run_prepare()
    def test_unknown_override_rejected(self):
        self.s["overrides"]={"99":{"zh":"白色","en":"white"}}
        with self.assertRaisesRegex(ValueError,"unknown"):self.run_prepare()
    def test_body_relation_rejected(self):
        self.s["overrides"]={"10":{"zh":"穿在身上的衬衫","en":"the shirt being worn"}}
        with self.assertRaisesRegex(ValueError,"body-relation"):self.run_prepare()
    def test_quarantine_requires_explicit_image_gap(self):
        self.s["overrides"]={"10":{"action":"quarantine","reason":"bad mask"}}
        with self.assertRaisesRegex(ValueError,"missing image"):self.run_prepare()
        self.s["unresolved_images"]=[{"image_id":1}]
        rows,qs,audit=self.run_prepare()
        self.assertEqual((len(rows),len(qs)),(0,2))
    def test_clear_replacement_preserves_old_quarantine(self):
        self.s["overrides"]={"10":{"action":"quarantine","reason":"bad mask"}}
        self.c["annotations"].append(dict(self.c["annotations"][0],id=11))
        self.s["replacements"]=[[1,11,"红色衬衫","the red shirt"]]
        rows,qs,audit=self.run_prepare()
        self.assertEqual(len(qs),2)
        self.assertEqual({q["target_instance_id"] for q in rows},{11})
        self.assertEqual(audit[-1]["action"],"replacement")
if __name__=="__main__":unittest.main()
