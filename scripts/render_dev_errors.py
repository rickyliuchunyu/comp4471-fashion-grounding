"""Render a fixed audit sample of failures with GT and selected mask overlays."""
import argparse,html,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,decode_mask,safe_path

def render(run,package,out):
    out.mkdir(exist_ok=False)
    report=read_json(run/"error_analysis.json");coco=read_json(package/"annotations_coco.json")
    ims={i["id"]:i for i in coco["images"]};anns={a["id"]:a for a in coco["annotations"]}
    queries=[json.loads(l) for l in (package/"queries_accepted.jsonl").read_text().splitlines()]
    english={q["semantic_query_id"]:q["query"] for q in queries if q["language"]=="en"}
    candidates={}
    for cond in ("gt","predicted"):
        p=run/cond/"inputs"/("candidates_gt.json" if cond=="gt" else "candidates_predicted.json")
        candidates[cond]=(p,{c["candidate_id"]:c for im in read_json(p)["images"] for c in im["candidates"]})
    links=[]
    for n in range(0,len(report["examples"]),4):
        rows=report["examples"][n:n+4];sheet=Image.new("RGB",(1500,440*len(rows)),"white");draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows):
            im=ims[row["image_id"]];pic=Image.open(package/im["file_name"]).convert("RGB")
            a=anns[row["target_instance_id"]];target=decode_mask(a["segmentation"],im["height"],im["width"])
            cond="gt" if row["reason"]=="GT_selection_failure" else "predicted"
            pp,cs=candidates[cond];selected=row["selected_candidate_id"];pred=None
            if selected is not None:pred=decode_mask(read_json(safe_path(pp.parent,cs[selected]["mask_reference"])),im["height"],im["width"])
            for col,mask,color,title in [(0,None,None,"Original"),(1,target,[0,220,100],"Ground truth"),(2,pred,[255,70,70],"Selected by M1")]:
                rgb=np.asarray(pic).copy()
                if mask is not None:rgb[mask]=(rgb[mask]*.4+np.array(color)*.6).astype("uint8")
                preview=Image.fromarray(rgb);preview.thumbnail((480,370));sheet.paste(preview,(col*500,j*440+60))
                draw.text((col*500,j*440+40),title,fill="black")
            draw.text((0,j*440),str(row["image_id"])+" | "+row["reason"]+" | "+english[row["semantic_query_id"]],fill="black")
            draw.text((0,j*440+20),"Selected IoU="+str(round(row["target_mask_iou"],3))+" | target="+str(row["target_instance_id"])+" | selected="+str(selected),fill="black")
        name="errors-"+str(n//4+1).zfill(2)+".jpg";sheet.save(out/name,quality=92);links.append('<h2>'+name+'</h2><img style="max-width:100%" src="'+name+'">')
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>dev错误样例</title><h1>固定顺序抽取的12个错误样例</h1><p>原图、绿色GT、红色M1所选mask。按候选缺失、候选存在但选择失败、GT选择失败各取前4张不同图片；不是随机样本或错误占比估计。</p>'+"".join(links),encoding="utf-8")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("run","package","output"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();render(a.run,a.package,a.output)
