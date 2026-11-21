import json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from apply_query_revisions import apply
class VisualRevisionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  def put(name,value):(self.root/name).write_text(json.dumps(value))
  put("image_manifest.json",{"manifest_id":"sample"})
  self.rows=[{"image_id":1,"target_instance_id":2,"semantic_query_id":"q1","language":l,"query":q,"query_types":["appearance"],"review_status":"needs_revision","reviewer":None} for l,q in (("zh","old-zh"),("en","old-en"))]
  self.before="".join(json.dumps(q)+"\n" for q in self.rows)
  for name in ("queries_draft.jsonl","queries_machine_train.jsonl"):(self.root/name).write_text(self.before)
  (self.root/"queries_accepted.jsonl").write_text("")
  put("annotation_evidence.json",[{"semantic_query_id":"q1","image_id":1,"target_instance_id":2}])
  put("annotation_report.json",{"accepted":0})
  self.change={"manifest_id":"sample","revisions":[{"image_id":1,"target_instance_id":2,"before":{"zh":"old-zh","en":"old-en"},"after":{"zh":"new-zh","en":"new-en"},"query_type":"combined","reason":"visible target ambiguity"}]}
  self.patchpath=self.root/"patch.json";self.save()
 def save(self):self.patchpath.write_text(json.dumps(self.change))
 @patch("apply_query_revisions.validate_package",return_value={"status":"passed"})
 def test_revision_keeps_targets_pending_and_backup(self,_):
  apply(self.root,self.patchpath)
  rows=[json.loads(s) for s in (self.root/"queries_draft.jsonl").read_text().splitlines()]
  self.assertEqual([q["query"] for q in rows],["new-zh","new-en"])
  self.assertTrue(all(q["target_instance_id"]==2 and q["review_status"]=="needs_revision" and q["reviewer"] is None for q in rows))
  self.assertEqual((self.root/"queries_before_visual_revision.jsonl").read_text(),self.before)
  self.assertEqual((self.root/"queries_accepted.jsonl").read_text(),"")
  self.assertEqual((self.root/"queries_machine_train.jsonl").read_text(),(self.root/"queries_draft.jsonl").read_text())
 def test_stale_patch_fails_before_writing(self):
  self.change["revisions"][0]["before"]["en"]="wrong";self.save()
  with self.assertRaises(ValueError):apply(self.root,self.patchpath)
  self.assertEqual((self.root/"queries_draft.jsonl").read_text(),self.before)
  self.assertFalse((self.root/"visual_revisions.json").exists())
 def test_cannot_overwrite_accepted_package(self):
  (self.root/"queries_accepted.jsonl").write_text("{}")
  with self.assertRaises(ValueError):apply(self.root,self.patchpath)
 @patch("apply_query_revisions.validate_package",return_value={"status":"passed"})
 def test_dev_draft_does_not_become_training_labels(self,_):
  (self.root/"queries_machine_train.jsonl").write_text("")
  apply(self.root,self.patchpath)
  self.assertEqual((self.root/"queries_machine_train.jsonl").read_text(),"")
