import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from publish_reviewed_dev import publish
from fashion_grounding.data.schema import sha256

class PublishDevTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.src=self.root/"source";self.src.mkdir();self.rev=self.root/"reviews";self.rev.mkdir();self.out=self.root/"out"
        for name in ("images","masks"):(self.src/name).mkdir()
        def write(name,data):(self.src/name).write_text(json.dumps(data))
        write("image_manifest.json",dict(manifest_id="dev",images=[dict(image_id=1,split="dev")]))
        write("annotations_coco.json",dict(annotations=[dict(id=10,image_id=1,category_id=1,iscrowd=0),dict(id=11,image_id=1,category_id=1,iscrowd=0)]))
        for name in ("mapping_config.json","source_subset.json","annotation_notes.json"):write(name,{})
        write("candidates_gt.json",dict(candidate_set_id="old",image_manifest={"sha256":"old"},images=[]))
        qs=[]
        for sid in ("a","b","c"):
            for lang in ("zh","en"):qs.append(dict(query_id=sid+lang,semantic_query_id=sid,language=lang,query="white shirt",image_id=1,target_instance_id=10,split="dev",query_types=["appearance"],annotator="machine",review_status="needs_revision",reviewer=None))
        (self.src/"queries_draft.jsonl").write_text("".join(json.dumps(q)+"\n" for q in qs))
        self.review=dict(review_format_version=2,manifest_id="dev",query_sha256=sha256(self.src/"queries_draft.jsonl"),reviewer="wkq",rows=[
            dict(semantic_query_id=sid,decision="human_selected",selected_annotation_id=aid,note="",edited_queries={"zh":"white shirt","en":"white shirt"},checks={"unique":True,"translation":True,"mask":True})
            for sid,aid in (("a",10),("b",11),("c",10))])
    def tearDown(self):self.tmp.cleanup()
    def call(self):
        (self.rev/"page-human-review.json").write_text(json.dumps(self.review))
        with patch("publish_reviewed_dev.validate_package",return_value={"fixture_validation_mocked":True}):
            return publish(self.src,self.rev,self.out)
    def test_agreement_only_and_preserve_pending_original(self):
        self.review["rows"][2].update(decision="not_reviewed",selected_annotation_id=None)
        result=self.call()
        self.assertEqual((result["accepted_semantics"],result["pending_semantics"]),(1,2))
        pending=[json.loads(l) for l in (self.out/"queries_draft.jsonl").read_text().splitlines()]
        self.assertTrue(all(q["target_instance_id"]==10 and q["review_status"]=="needs_revision" for q in pending))
    def test_missing_page_rejected_before_output(self):
        self.review["rows"].pop()
        with self.assertRaisesRegex(ValueError,"page missing"):self.call()
        self.assertFalse(self.out.exists())
    def test_wrong_hash_rejected(self):
        self.review["query_sha256"]="bad"
        with self.assertRaisesRegex(ValueError,"identity"):self.call()
    def test_duplicate_reviews_rejected(self):
        self.review["rows"].append(copy.deepcopy(self.review["rows"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):self.call()
    def test_notes_edits_and_missing_checks_not_accepted(self):
        self.review["rows"][0]["note"]="uncertain"
        self.review["rows"][1]["selected_annotation_id"]=10
        self.review["rows"][1]["edited_queries"]["en"]="changed"
        self.review["rows"][2]["checks"]["mask"]=False
        with self.assertRaisesRegex(ValueError,"no fully agreed"):self.call()
    def test_exposure_not_mislabeled_blind(self):
        self.review["disclosed_image_ids"]=[1]
        with self.assertRaisesRegex(ValueError,"no fully agreed"):self.call()
if __name__=="__main__":unittest.main()
