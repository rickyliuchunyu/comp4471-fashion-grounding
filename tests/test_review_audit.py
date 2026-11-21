import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.review import audit_review

class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.m={"manifest_id":"v1","images":[{"image_id":1,"split":"train"},{"image_id":2,"split":"train"}]}
        self.a=[{"id":1,"image_id":1,"category_id":7,"iscrowd":0},{"id":2,"image_id":1,"category_id":7,"iscrowd":0},{"id":3,"image_id":2,"category_id":7,"iscrowd":0}]
        self.q=[{"query_id":"q-"+lang,"semantic_query_id":"s","language":lang,"image_id":1,"target_instance_id":1,"split":"train","query":"left bag","query_types":["category","spatial"]} for lang in ["zh","en"]]
        self.r={"manifest_id":"v1","reviewer":"human","rows":[{"semantic_query_id":"s","decision":"human_selected","selected_annotation_id":1,"note":""}]}
    def run_audit(self, disclosed=()):
        return audit_review(self.m,self.a,self.q,self.r,disclosed)
    def test_agreement_never_automatically_accepted(self):
        before=copy.deepcopy(self.r)
        result=self.run_audit()
        self.assertEqual(result["status_counts"],{"target_agreement_only":1})
        self.assertEqual(result["accepted_promoted"],0)
        self.assertEqual(self.r,before)
    def test_disagreement_is_preserved(self):
        self.r["rows"][0]["selected_annotation_id"]=2
        self.assertEqual(self.run_audit()["rows"][0]["status"],"target_disagreement")
    def test_disclosed_answer_is_not_blind_agreement(self):
        self.assertEqual(self.run_audit({1})["rows"][0]["status"],"answer_disclosed")
    def test_missing_review_is_pending(self):
        self.r["rows"]=[]
        self.assertEqual(self.run_audit()["rows"][0]["status"],"not_reviewed")
    def test_unknown_and_duplicate_ids(self):
        self.r["rows"].append(copy.deepcopy(self.r["rows"][0]))
        with self.assertRaises(ValueError): self.run_audit()
        self.r["rows"]=self.r["rows"][:1]
        self.r["rows"][0]["semantic_query_id"]="unknown"
        with self.assertRaises(ValueError): self.run_audit()
    def test_cross_image_and_crowd_targets(self):
        self.r["rows"][0]["selected_annotation_id"]=3
        with self.assertRaises(ValueError): self.run_audit()
        self.r["rows"][0]["selected_annotation_id"]=1
        self.a[0]["iscrowd"]=1
        with self.assertRaises(ValueError): self.run_audit()
    def test_translation_and_manifest_mismatch(self):
        self.q[1]["target_instance_id"]=2
        with self.assertRaises(ValueError): self.run_audit()
        self.q[1]["target_instance_id"]=1
        self.r["manifest_id"]="v2"
        with self.assertRaises(ValueError): self.run_audit()
    def test_rejection_notes_and_malformed_decisions(self):
        row=self.r["rows"][0]
        row["note"]="mask concern"
        self.assertEqual(self.run_audit()["rows"][0]["status"],"agreement_with_note")
        row.update(decision="needs_revision",selected_annotation_id=None)
        self.assertEqual(self.run_audit()["rows"][0]["status"],"needs_revision")
        row.update(decision="human_selected",selected_annotation_id=True)
        with self.assertRaises(ValueError): self.run_audit()


    def as_v2(self):
        self.r["review_format_version"]=2
        self.r["rows"][0].update(edited_queries={"zh":"left bag","en":"left bag"},checks={"unique":True,"translation":True,"mask":True})
    def test_edited_text_preserved_and_never_accepted(self):
        self.as_v2()
        self.r["rows"][0]["edited_queries"]["zh"]="corrected wording"
        r=self.run_audit()
        self.assertEqual(r["rows"][0]["status"],"edited_query_pending_recheck")
        self.assertEqual(r["rows"][0]["edited_queries"]["zh"],"corrected wording")
        self.assertEqual(r["accepted_promoted"],0)
    def test_incomplete_checks_block_quality_claim(self):
        self.as_v2()
        self.r["rows"][0]["checks"]["mask"]=False
        self.assertEqual(self.run_audit()["rows"][0]["status"],"incomplete_quality_checks")
    def test_export_disclosure_is_preserved(self):
        self.as_v2()
        self.r["disclosed_image_ids"]=[1]
        self.assertEqual(self.run_audit()["rows"][0]["status"],"answer_disclosed")
    def test_v2_requires_boolean_checks_and_bilingual_edits(self):
        self.as_v2()
        self.r["rows"][0]["checks"]["mask"]="yes"
        with self.assertRaises(ValueError): self.run_audit()
    def test_edit_keeps_disagreement_and_exposure_evidence(self):
        self.as_v2()
        self.r["disclosed_image_ids"]=[1]
        self.r["rows"][0].update(selected_annotation_id=2,edited_queries={"zh":"changed","en":"changed"})
        r=self.run_audit()["rows"][0]
        self.assertFalse(r["target_agrees"])
        self.assertTrue(r["answer_disclosed"])

if __name__=="__main__": unittest.main()
