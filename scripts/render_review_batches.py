"""Produce manageable answer-blind human-review pages without accepting machine labels."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_query_review import review_payload,PAGE
from fashion_grounding.data.schema import validate_package,require
def build(root,out,size=50,visual_dev=False):
 require(size>0,"positive page size required");require(not out.exists(),"output exists")
 validate_package(root)
 if visual_dev:
  manifest=json.loads((root/"image_manifest.json").read_text())
  require(all(im["split"]=="dev" for im in manifest["images"]),"visual-dev rendering requires a dev package")
 out.mkdir(parents=True)
 ids=sorted({json.loads(line)["semantic_query_id"] for line in (root/"queries_draft.jsonl").read_text().splitlines() if line.strip()})
 links=[]
 for n,start in enumerate(range(0,len(ids),size),1):
  page=f"review-{n:03d}.html"
  payload=review_payload(root,set(ids[start:start+size]),validate=False)
  template=PAGE if visual_dev else PAGE.replace("机器看图初标","机器规则初标（颜色估计/来源属性/几何）")
  if visual_dev:
   template=template.replace("机器看图初标","Codex 逐图初标（待人工复核）").replace("这是训练描述草稿。","这是 dev 开发集描述草稿，不用于训练。").replace("'新图片'","'未登记既往答案曝光'")
  template=template.replace("D.query_sha256;","D.query_sha256+'-page-"+str(n)+"';")
  template=template.replace("D.manifest_id+'-human-review.json'","D.manifest_id+'-page-"+str(n).zfill(3)+"-human-review.json'")
  (out/page).write_text(template.replace("__DATA__",json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")),encoding="utf-8")
  links.append(f'<li><a href="{page}">第 {n} 页：{len(payload["queries"])} 组</a></li>')
 (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>分批审核</title><h1>机器初标 · 分批人工审核</h1><p>每页单独暂存、单独导出。初标尚未人工验收；不要将导出文件覆盖为训练答案。请返回全部审核 JSON，由审计脚本核对并处理分歧。</p><ol>'+"".join(links)+"</ol>",encoding="utf-8")
 print(json.dumps({"pages":len(links),"semantics":len(ids),"answers_embedded":False}))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("package",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--page-size",type=int,default=50);p.add_argument("--visual-dev",action="store_true");a=p.parse_args();build(a.package,a.output,a.page_size,a.visual_dev)
