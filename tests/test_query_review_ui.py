"""UI behavior checks in a small JS DOM stub; not a browser rendering test."""
import json,re,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from render_query_review import PAGE
try:
 import quickjs
except ImportError:
 quickjs=None

@unittest.skipUnless(quickjs,"install requirements-data-review-test.txt to execute JS checks")
class ReviewPageTests(unittest.TestCase):
 def setUp(self):
  payload={"manifest_id":"test","query_sha256":"hash","disclosed_image_ids":[],"images":{"1":{"image":"x","candidates":[{"number":1,"annotation_id":11,"crop":"x"},{"number":2,"annotation_id":12,"crop":"y"}]}},"queries":[{"semantic_query_id":"neutral-001","image_id":1,"zh":"左边的包","en":"left bag"},{"semantic_query_id":"neutral-002","image_id":1,"zh":"右边的包","en":"right bag"}]}
  self.ctx=quickjs.Context()
  self.ctx.eval("""
const elements={},storage={},alerts=[];let exported;
function elem(){return {value:'',checked:false,textContent:'',children:[],append(...xs){this.children.push(...xs)},replaceChildren(){this.children=[]},click(){if(this.onclick)this.onclick()}}}
const document={getElementById(id){return elements[id]||(elements[id]=elem())},createElement:elem};
const localStorage={getItem:k=>storage[k]||null,setItem:(k,v)=>storage[k]=v};
const URL={createObjectURL:b=>{exported=b.data[0];return 'blob'},revokeObjectURL:u=>{}};
function Blob(data,opts){this.data=data}function alert(s){alerts.push(s)}function setTimeout(f,t){f()}
""")
  self.ctx.eval("document.getElementById('review-data').textContent="+json.dumps(json.dumps(payload))+";")
  self.ctx.eval(re.findall(r"<script>(.*?)</script>",PAGE,re.S)[0])
 def run_js(self,s):return self.ctx.eval(s)
 def test_cannot_complete_without_selection_and_checks(self):
  self.run_js("$('decision').value='human_selected';$('decision').onchange()")
  self.assertEqual(self.run_js("row().decision"),"not_reviewed")
  self.assertEqual(self.run_js("alerts.length"),1)
 def complete(self):
  self.run_js("$('gallery').children[0].onclick();for(let k of ['unique','translation','mask']){$(k).checked=true;$(k).onchange()};$('decision').value='human_selected';$('decision').onchange()")
 def test_export_roundtrip_includes_unreviewed_and_edits(self):
  self.complete()
  self.run_js("$('editZh').value='浅色包';$('editZh').oninput();$('export').onclick()")
  result=json.loads(self.run_js("exported"))
  self.assertEqual(result["review_format_version"],2)
  self.assertEqual(result["rows"][0]["selected_annotation_id"],11)
  self.assertEqual(result["rows"][0]["edited_queries"]["zh"],"浅色包")
  self.assertEqual(result["rows"][1]["decision"],"not_reviewed")
 def test_reject_keeps_proposal_but_no_final_target(self):
  self.complete()
  self.run_js("$('decision').value='needs_revision';$('decision').onchange();$('export').onclick()")
  r=json.loads(self.run_js("exported"))["rows"][0]
  self.assertIsNone(r["selected_annotation_id"])
  self.assertEqual(r["proposed_annotation_id"],11)
 def test_changing_candidate_resets_quality_checks(self):
  self.complete()
  self.run_js("$('gallery').children[1].onclick()")
  self.assertEqual(self.run_js("row().decision"),"not_reviewed")
  self.assertFalse(self.run_js("row().checks.mask"))
 def test_navigation_restores_note_and_choice(self):
  self.complete()
  self.run_js("$('note').value='check';$('note').oninput();$('next').onclick();$('prev').onclick()")
  self.assertEqual(self.run_js("$('note').value"),"check")
  self.assertEqual(self.run_js("row().selected_annotation_id"),11)

if __name__=="__main__":unittest.main()
