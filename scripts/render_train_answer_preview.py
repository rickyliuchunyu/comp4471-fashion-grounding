"""Create separate answer previews from already-rendered blind review pages."""
import argparse,html,json,re
from pathlib import Path

PAGE="""<!doctype html><meta charset="utf-8"><title>Codex 初标答案预览</title>
<style>body{font:18px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:16px;background:#f5f7fa}img{max-width:100%;max-height:650px}button{font:inherit;margin:12px;padding:10px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:24px}section{background:white;padding:20px}</style>
<h1>Codex 初标答案预览 · 尚未人工验收</h1>
<p>本页展示初标选择。看过答案后对该图的复核不再属于独立盲审；请在审核备注中写“看过答案预览”。正式审核请返回 <a href="index.html">审核目录</a>。</p>
<button id="prev">上一组</button><span id="counter"></span><button id="next">下一组</button>
<h2 id="zh"></h2><p id="en"></p><p id="target"></p>
<div class="grid"><section><h3>原图</h3><img id="photo"></section><section><h3>初标所选 mask（青色）</h3><img id="crop"></section></div>
<script>const D=__DATA__;let i=0;function show(){const q=D.queries[i],im=D.images[String(q.image_id)],a=im.candidates.find(x=>x.annotation_id===q.target_instance_id);document.getElementById('counter').textContent=(i+1)+' / '+D.queries.length;document.getElementById('zh').textContent=q.zh;document.getElementById('en').textContent=q.en;document.getElementById('target').textContent='图片 '+q.image_id+' · 初标候选 '+a.number+' · annotation '+a.annotation_id;document.getElementById('photo').src=im.image;document.getElementById('crop').src=a.crop;}document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show()};document.getElementById('next').onclick=()=>{i=Math.min(D.queries.length-1,i+1);show()};show();</script>"""

def build(root, folder):
    rows=[json.loads(l) for l in (root/"queries_draft.jsonl").read_text().splitlines() if l.strip()]
    targets={q["semantic_query_id"]:q["target_instance_id"] for q in rows}
    links=[];seen=set()
    for page in sorted(folder.glob("review-*.html")):
        text=page.read_text()
        marker='<script id="review-data" type="application/json">'
        start=text.index(marker)+len(marker)
        data,end=json.JSONDecoder().raw_decode(text[start:].lstrip())
        assert all("target_instance_id" not in q for q in data["queries"]), "blind page leaked answers"
        for q in data["queries"]:
            seen.add(q["semantic_query_id"])
            q["target_instance_id"]=targets[q["semantic_query_id"]]
            assert any(a["annotation_id"]==q["target_instance_id"] for a in data["images"][str(q["image_id"])]["candidates"])
        name=page.name.replace("review-","answers-")
        (folder/name).write_text(PAGE.replace("__DATA__",json.dumps(data,ensure_ascii=False).replace("<","\\u003c")),encoding="utf-8")
        links.append('<li><a href="'+name+'">'+html.escape(name)+'</a></li>')
    assert seen==set(targets)
    index=folder/"index.html"
    index.write_text(index.read_text()+'<h2>另行查看初标答案</h2><p>先做独立审核可避免答案影响判断。若看过答案，请在审核备注记录。</p><ol>'+"".join(links)+"</ol>",encoding="utf-8")
    return {"preview_pages":len(links),"semantic_groups":len(seen),"blind_pages_target_free":True}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("package",type=Path);p.add_argument("folder",type=Path);a=p.parse_args()
    print(json.dumps(build(a.package,a.folder)))
