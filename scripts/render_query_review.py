"""Render a self-contained, answer-blind review page for bilingual training drafts."""
import argparse,base64,io,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,decode_mask,sha256,validate_package

def data_uri(im):
 b=io.BytesIO();im.save(b,format="JPEG",quality=85)
 return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()

def review_payload(root):
 validate_package(root)
 manifest=read_json(root/"image_manifest.json")
 coco=read_json(root/"annotations_coco.json")
 groups=defaultdict(dict)
 for q in (json.loads(s) for s in (root/"queries_draft.jsonl").read_text().splitlines() if s.strip()):
  groups[q["semantic_query_id"]][q["language"]]=q
 images={}
 for im in manifest["images"]:
  pic=Image.open(root/im["file_name"]).convert("RGB")
  preview=pic.copy();preview.thumbnail((1000,1000))
  candidates=[]
  for n,a in enumerate(sorted((a for a in coco["annotations"] if a["image_id"]==im["image_id"] and not a["iscrowd"]),key=lambda a:a["id"]),1):
   mask=decode_mask(a["segmentation"],im["height"],im["width"])
   rgb=np.asarray(pic).copy()
   rgb[mask]=(rgb[mask]*.55+np.array([0,215,255])*.45).astype(np.uint8)
   x,y,w,h=a["bbox"]; pad=18
   box=(max(0,x-pad),max(0,y-pad),min(im["width"],x+w+pad),min(im["height"],y+h+pad))
   crop=Image.fromarray(rgb).crop(box);crop.thumbnail((250,250))
   candidates.append({"number":n,"annotation_id":a["id"],"crop":data_uri(crop)})
  images[str(im["image_id"])]={"image":data_uri(preview),"candidates":candidates}
 rows=[{"semantic_query_id":sid,"image_id":qs["zh"]["image_id"],"zh":qs["zh"]["query"],"en":qs["en"]["query"]} for sid,qs in sorted(groups.items())]
 notes=read_json(root/"annotation_notes.json")
 return {"manifest_id":manifest["manifest_id"],"query_sha256":sha256(root/"queries_draft.jsonl"),"images":images,"queries":rows,"disclosed_image_ids":[d["image_id"] for d in notes.get("disputes",[]) if d.get("prior_answer_exposure")]}

PAGE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>服饰指代查询 · 人工审核</title><style>
body{font:16px/1.6 system-ui;margin:0;background:#f4f5f7;color:#192536}header{padding:20px 5%;background:#14283e;color:white}main{padding:20px 5%;max-width:1350px;margin:auto}
button,select,input,textarea{font:inherit;padding:9px;border:1px solid #aab6c5;border-radius:6px}button{cursor:pointer;background:white}button:focus-visible{outline:3px solid #087fbc}.layout{display:grid;grid-template-columns:1fr 1fr;gap:24px}.photo{width:100%;max-height:720px;object-fit:contain;background:white}.panel{background:white;padding:20px;border-radius:12px}.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.candidate{display:flex;flex-direction:column;align-items:center}.candidate img{width:100%;height:150px;object-fit:contain}.selected{outline:3px solid #007cac;background:#e6f7ff}textarea{display:block;box-sizing:border-box;width:100%;min-height:70px;margin:8px 0}label{display:block;margin:10px 0}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}.muted{color:#53667b}.query{font-size:19px;font-weight:600}#status{min-height:28px;color:#007059}@media(max-width:850px){.layout{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(2,1fr)}}
</style><header><h1>服饰指代查询 · 人工审核</h1><div>机器看图初标 → 你独立选目标 → 返回审核文件。<span id="total"></span> 组中英文，参考答案不在此页面。</div></header>
<main><details><summary>怎么审核（先读一次）</summary><p>先看原图与两种语言，独立选择描述指向的整件衣物。下方候选卡片中的青色区域是该候选的 mask；点击卡片即可选择。候选编号只在本图内有效。</p><p>不必逐个检查所有框：重点确认描述是否唯一、两种语言是否一致，以及你选中的 mask 是否对应整件目标。多个候选都合适、看不清、mask 分裂或缺失时，选择“需要修改”，写明原因。</p><p>中文和英文可以修改；修改会单独保留，返回后重新核对，不直接变成 accepted。旧分歧图片已看过，属于再次审核，不算独立盲审。暂存依赖浏览器，建议及时导出；换浏览器不会自动迁移。</p></details>
<div class="controls"><label>审核人 <input id="reviewer" value="wkq"></label><button id="export">导出审核 JSON</button><span id="progress"></span></div>
<div class="controls"><button id="prev">上一组</button><strong id="counter"></strong><button id="next">下一组</button><span id="exposure"></span></div>
<div class="layout"><section><img class="photo" id="photo" alt="原始图片"><h3>候选 · 青色为目标 mask</h3><div class="gallery" id="gallery"></div></section>
<section class="panel"><div class="query" id="zh"></div><p id="en"></p><label>审核结果 <select id="decision"><option value="not_reviewed">尚未审核</option><option value="human_selected">已选目标，完成核对</option><option value="needs_revision">需要修改 / 目标不唯一 / mask 有问题</option></select></label><p id="chosen"></p>
<label><input type="checkbox" id="unique"> 描述可以唯一指向我选的目标</label><label><input type="checkbox" id="translation"> 中英文指向同一目标、含义一致</label><label><input type="checkbox" id="mask"> 选中 mask 对应整件目标，没有明显分裂或误标</label>
<details><summary>修改描述（可选）</summary><label>中文<textarea id="editZh"></textarea></label><label>English<textarea id="editEn"></textarea></label></details>
<label>备注<textarea id="note" placeholder="没有问题可留空；有问题请写具体原因"></textarea></label><p class="muted">这是训练描述草稿。导出后由脚本比对，再处理修改和分歧，不自动计入正式训练/评价。</p><div id="status" role="status"></div></section></div></main>
<script id="review-data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('review-data').textContent), $=id=>document.getElementById(id), key='fg-review-'+D.manifest_id+'-'+D.query_sha256;
let idx=0, saved={reviewer:'wkq',rows:{}};
try{const raw=localStorage.getItem(key);if(raw)saved=JSON.parse(raw)}catch(e){}
$('reviewer').value=saved.reviewer||'wkq';$('total').textContent=D.queries.length;
function row(){let q=D.queries[idx];return saved.rows[q.semantic_query_id]||(saved.rows[q.semantic_query_id]={semantic_query_id:q.semantic_query_id,decision:'not_reviewed',selected_annotation_id:null,note:'',checks:{unique:false,translation:false,mask:false},edited_queries:{zh:q.zh,en:q.en}})}
function persist(){saved.reviewer=$('reviewer').value.trim();try{localStorage.setItem(key,JSON.stringify(saved));$('status').textContent='已在当前浏览器暂存'}catch(e){$('status').textContent='浏览器暂存不可用，请及时导出'}$('progress').textContent=Object.values(saved.rows).filter(r=>r.decision!=='not_reviewed').length+' / '+D.queries.length+' 组已记录'}
function render(){let q=D.queries[idx],r=row(),im=D.images[q.image_id];$('photo').src=im.image;$('counter').textContent=(idx+1)+' / '+D.queries.length;$('zh').textContent=q.zh;$('en').textContent=q.en;$('decision').value=r.decision;$('note').value=r.note;for(let k of ['unique','translation','mask'])$(k).checked=r.checks[k];$('editZh').value=r.edited_queries.zh;$('editEn').value=r.edited_queries.en;
$('exposure').textContent=D.disclosed_image_ids.includes(q.image_id)?'旧分歧图片 · 再次审核':'新图片';
$('gallery').replaceChildren();for(let c of im.candidates){let b=document.createElement('button');b.className='candidate'+(c.annotation_id===r.selected_annotation_id?' selected':'');let img=document.createElement('img');img.src=c.crop;img.alt='候选 '+c.number;let label=document.createElement('span');label.textContent='候选 '+c.number;b.append(img,label);b.onclick=()=>{r.selected_annotation_id=c.annotation_id;r.checks={unique:false,translation:false,mask:false};r.decision="not_reviewed";persist();render()};$('gallery').append(b)}
let selected=im.candidates.find(c=>c.annotation_id===r.selected_annotation_id);$('chosen').textContent=selected?'当前选择：候选 '+selected.number:'尚未选择目标';$('prev').disabled=idx===0;$('next').disabled=idx===D.queries.length-1;persist()}
$('prev').onclick=()=>{idx--;render()};$('next').onclick=()=>{idx++;render()};
$('decision').onchange=()=>{let r=row();if($('decision').value==='human_selected'&&(!r.selected_annotation_id||!Object.values(r.checks).every(Boolean))){alert('请先选择目标，并确认三个检查项；有疑问请选择需要修改。');$('decision').value=r.decision;return}r.decision=$('decision').value;persist()};
for(let k of ['unique','translation','mask'])$(k).onchange=()=>{let r=row();r.checks[k]=$(k).checked;if(!r.checks[k]&&r.decision==='human_selected'){r.decision='needs_revision';$('decision').value=r.decision}persist()};
for(let [id,field] of [['editZh','zh'],['editEn','en']])$(id).oninput=()=>{row().edited_queries[field]=$(id).value;persist()};
$('note').oninput=()=>{row().note=$('note').value;persist()};$('reviewer').oninput=persist;
$('export').onclick=()=>{if(!$('reviewer').value.trim()){alert('请填写审核人');return}let rows=D.queries.map(q=>{let r=saved.rows[q.semantic_query_id]||{semantic_query_id:q.semantic_query_id,decision:'not_reviewed',selected_annotation_id:null,note:'',checks:{unique:false,translation:false,mask:false},edited_queries:{zh:q.zh,en:q.en}};return {...r,proposed_annotation_id:r.selected_annotation_id,selected_annotation_id:r.decision==='human_selected'?r.selected_annotation_id:null}});let result={review_format_version:2,manifest_id:D.manifest_id,query_sha256:D.query_sha256,reviewer:$('reviewer').value.trim(),disclosed_image_ids:D.disclosed_image_ids,exported_at:new Date().toISOString(),rows};let url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download=D.manifest_id+'-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();
</script></html>'''

def render(root,output):
 payload=review_payload(root)
 text=PAGE.replace("__DATA__",json.dumps(payload,ensure_ascii=False).replace("<","\\u003c"))
 output.write_text(text,encoding="utf-8")
 return payload

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("package",type=Path);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 render(a.package,a.output)
