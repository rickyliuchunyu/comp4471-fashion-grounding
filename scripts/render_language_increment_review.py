"""Render only new queries as blind review plus separately linked answer previews."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from render_query_review import review_payload,PAGE
from render_train_answer_preview import PAGE as ANSWERS
from fashion_grounding.data.schema import validate_package,require,write_json
def build(root,out,size=25):
 require(not out.exists(),"review output exists")
 validate_package(root);out.mkdir(parents=True)
 rows=[json.loads(l) for l in (root/"queries_increment.jsonl").read_text().splitlines() if l.strip()]
 targets={q["semantic_query_id"]:q["target_instance_id"] for q in rows};ids=sorted(targets);links=[];answerlinks=[]
 for n,start in enumerate(range(0,len(ids),size),1):
  payload=review_payload(root,set(ids[start:start+size]),validate=False)
  require(all("target_instance_id" not in q for q in payload["queries"]),"blind payload leaks target")
  template=PAGE.replace("机器看图初标","Codex 逐图初标（待人工复核）").replace("'新图片'","'未登记既往答案曝光'")
  if rows[0]["split"]=="dev":template=template.replace("这是训练描述草稿。","这是独立 dev 困难集草稿，不用于训练。")
  template=template.replace("D.query_sha256;","D.query_sha256+'-page-"+str(n)+"';").replace("D.manifest_id+'-human-review.json'","D.manifest_id+'-page-"+str(n).zfill(3)+"-human-review.json'")
  name=f"review-{n:03d}.html"
  (out/name).write_text(template.replace("__DATA__",json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")),encoding="utf-8")
  links.append(f'<li><a href="{name}">第 {n} 页：{len(payload["queries"])} 组</a></li>')
  for q in payload["queries"]:q["target_instance_id"]=targets[q["semantic_query_id"]]
  name=f"answers-{n:03d}.html"
  (out/name).write_text(ANSWERS.replace("__DATA__",json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")),encoding="utf-8")
  answerlinks.append(f'<li><a href="{name}">答案预览 {n}</a></li>')
 (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>新增数据审核</title><h1>新增查询人工审核</h1><p>本目录仅包含新增查询。每页独立暂存和导出。请保留所有 JSON；看过答案请在备注注明。</p><h2>独立审核</h2><ol>'+"".join(links)+'</ol><h2>另行查看 AI 初标答案</h2><ol>'+"".join(answerlinks)+'</ol>',encoding="utf-8")
 result=dict(pages=len(links),semantic_groups=len(ids),blind_payload_target_free=True,scope="increment only",source_package=str(root))
 write_json(out/"review_manifest.json",result);return result
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("package",type=Path);p.add_argument("output",type=Path);a=p.parse_args();print(json.dumps(build(a.package,a.output)))
