import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from revise_dev_queries import coverage, pairs, features, build
from fashion_grounding.data.schema import sha256

def rows(sid, image, split, zh="红色上衣", en="the red top"):
    return [{"semantic_query_id":sid,"language":l,"image_id":image,"target_instance_id":image*10,
             "target_category_id":1,"split":split,"query":t} for l,t in (("zh",zh),("en",en))]

class SimplificationTests(unittest.TestCase):
    def test_counts_semantics_once_across_languages(self):
        r=coverage(rows("t",1,"train"),rows("d",2,"dev"))
        self.assertEqual(r["features"]["red"]["train"]["semantic_groups"],1)
        self.assertEqual(r["train_records"],2)
    def test_reports_missing_explicit_supervision(self):
        r=coverage(rows("t",1,"train"),rows("d",2,"dev","粉色连帽上衣","the pink hooded top"))
        self.assertEqual(r["attention"][0]["missing_train_text_features"],["hooded","pink"])
    def test_train_dev_overlap_rejected(self):
        with self.assertRaises(ValueError):coverage(rows("t",1,"train"),rows("d",1,"dev"))
    def test_non_train_source_rejected(self):
        with self.assertRaises(ValueError):coverage(rows("t",1,"dev"),rows("d",2,"dev"))
    def test_incomplete_or_inconsistent_translation_rejected(self):
        with self.assertRaises(ValueError):pairs(rows("t",1,"train")[:1])
        rs=rows("t",1,"train");rs[1]["target_instance_id"]=90
        with self.assertRaises(ValueError):pairs(rs)
    def test_compound_colors_and_watch_not_body_relation(self):
        p=pairs(rows("x",1,"dev","红黑色手表","the red and black watch"))["x"]
        self.assertEqual(features(p),{"red","black"})
    def test_source_hash_and_ambiguous_visual_target_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); parent=p/"parent";parent.mkdir()
            old=rows("d",2,"dev")
            (parent/"queries_draft.jsonl").write_text("".join(json.dumps(q)+"\n" for q in old))
            (parent/"SHA256SUMS.json").write_text("{}")
            spec={"parent_queries_sha256":"wrong","decisions":{}}
            decision=p/"decisions.json";decision.write_text(json.dumps(spec))
            out=p/"output"
            with self.assertRaisesRegex(ValueError,"hash mismatch"):build(parent,decision,p/"train.jsonl",out)
            spec["parent_queries_sha256"]=sha256(parent/"queries_draft.jsonl")
            spec["decisions"]={"d":{"action":"revise","compatible_candidate_ids":[20,21]}}
            decision.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,"unique visual target"):build(parent,decision,p/"train.jsonl",out)
            self.assertFalse(out.exists())
if __name__=="__main__":unittest.main()
