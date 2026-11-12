"""Transparent CPU weak language annotation from mask pixels, source attributes and geometry."""
import argparse,json,random,shutil,sys,concurrent.futures
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,decode_mask,sha256,require,validate_package
from fashion_grounding.data.convert import COMMON

FINE={0:("衬衫","shirt"),1:("上衣","top"),2:("毛衣","sweater"),3:("开衫","cardigan"),4:("夹克","jacket"),5:("背心","vest"),6:("长裤","trousers"),7:("短裤","shorts"),8:("半身裙","skirt"),9:("大衣","coat"),10:("连衣裙","dress"),12:("披肩","cape"),13:("眼镜","glasses"),14:("帽子","hat"),15:("头饰","head accessory"),16:("领带","tie"),17:("手套","glove"),18:("手表","watch"),19:("腰带","belt"),20:("暖腿套","leg warmer"),21:("连裤袜","tights"),22:("袜子","sock"),23:("鞋","shoe"),24:("包","bag"),25:("围巾","scarf"),26:("雨伞","umbrella")}
COLORS=[("黑色","black"),("白色","white"),("灰色","gray"),("红色","red"),("橙色","orange"),("黄色","yellow"),("绿色","green"),("蓝色","blue"),("紫色","purple"),("粉色","pink"),("棕色","brown"),("米色","beige")]
PATTERNS={322:("格纹","checkered"),323:("波点","polka-dot"),325:("花卉图案","floral"),328:("条纹","striped")}
QUARANTINE={7074,9713,42143,45411,48755,49055,13898,18413}
def dominant_color(rgb,mask):
    # Erode one pixel to reduce background boundary contamination.
    core=mask.copy()
    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:core &= np.roll(mask,(dy,dx),(0,1))
    core[[0,-1],:]=False;core[:,[0,-1]]=False
    if core.sum()<200:return None,0.0
    hsv=np.asarray(Image.fromarray(rgb).convert("HSV"),dtype=float)[core]
    h=hsv[:,0]*360/255;s=hsv[:,1]/255;v=hsv[:,2]/255
    label=np.full(len(h),-1,int)
    # Avoid treating saturated dark navy/purple as black, or pink as red.
    label[(v<.10)|((v<.25)&(s<.20))]=0
    neutral=(s<.12)&(v>=.25)
    label[neutral&(v>=.85)]=1;label[neutral&(v<.85)]=2
    chrom=(label<0)&(v>=.25)&(s>=.45)
    label[chrom&((h<10)|(h>=350))&(s>=.65)]=3
    label[chrom&(h>=48)&(h<65)&(v>=.70)&(s>=.60)]=5
    label[chrom&(h>=85)&(h<165)]=6
    label[chrom&(h>=185)&(h<250)]=7
    # Orange/brown/beige and purple/pink boundaries are not reliable enough
    # for this CPU estimator: leave them unlabeled instead of guessing.
    counts=np.bincount(label[label>=0],minlength=12);best=int(counts.argmax());fraction=float(counts[best]/len(label))
    threshold=.80 if best in (0,1,2) else .75
    return (best if fraction>=threshold else None),fraction


def box_overlap(a,b):
    ax,ay,aw,ah=a["bbox"];bx,by,bw,bh=b["bbox"]
    intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return intersection/min(aw*ah,bw*bh)

def proposed_queries(im,anns,rgb):
    valid=[a for a in anns if not a["iscrowd"]]
    fine=defaultdict(list);features={};proposals=[]
    for a in valid:
        fine[a["source_category_id"]].append(a)
        if a["area"]<1024 or min(a["bbox"][2:])<24:continue
        color,confidence=dominant_color(rgb,decode_mask(a["segmentation"],im["height"],im["width"]))
        patterns=[p for p in a.get("attribute_ids",[]) if p in PATTERNS]
        features[a["id"]]={"color":color,"color_fraction":confidence,"pattern":patterns[0] if len(patterns)==1 else None}
    def add(a,zh,en,kind,evidence,contrast=None):
        proposals.append({"target":a,"zh":zh,"en":en,"kind":kind,"evidence":evidence,"contrast":contrast,"same_fine_count":len(fine[a["source_category_id"]])})
    for a in valid:
        if a["id"] not in features:continue
        f=features[a["id"]];group=[x for x in valid if x["category_id"]==a["category_id"]];zh,en=FINE[a["source_category_id"]]
        others=[features.get(x["id"],{}) for x in group if x["id"]!=a["id"]]
        # Compare all same-project-class rivals: broad terms such as top can overlap shirt/sweater.
        # Unknown rival appearance is not evidence that a descriptor is unique.
        if f["color"] is not None and all(x.get("color") is not None and x["color"]!=f["color"] for x in others):
            cz,ce=COLORS[f["color"]]
            add(a,f"{cz}的{zh}",f"the {ce} {en}","appearance",{"measurement":"mask-interior HSV dominant-color heuristic; not VLM/human judgment",**f})
        if f["pattern"] is not None and all(x.get("pattern") is not None and x["pattern"]!=f["pattern"] for x in others):
            pz,pe=PATTERNS[f["pattern"]]
            add(a,f"有{pz}的{zh}",f"the {pe} {en}","appearance",{"measurement":"official localized textile-pattern attribute, unverified visually",**f})
    # Conservative candidate-level spatial pairs. Clothing fragments remain a risk;
    # avoid multi-region apparel/limb classes entirely for automatic spatial labels.
    for source_id,group in fine.items():
        if source_id not in (14,23,24,26) or len(group)!=2:continue
        if any(a["id"] not in features for a in group):continue
        if box_overlap(*group)>.05:continue
        ordered=sorted(group,key=lambda a:a["bbox"][0]+a["bbox"][2]/2)
        centers=[a["bbox"][0]+a["bbox"][2]/2 for a in ordered]
        if centers[1]-centers[0]<im["width"]*.12:continue
        contrast=f"auto-spatial-{im['image_id']}-{source_id}"
        zh,en=FINE[source_id]
        for a,dz,de in zip(ordered,("左边","右边"),("left","right")):
            add(a,f"画面{dz}的{zh}",f"the {en} on the {de} side of the image","spatial",{"measurement":"same-fine-class disjoint boxes; no physical-instance human verification","rival_ids":[x["id"] for x in group if x["id"]!=a["id"]]},contrast)
            f=features[a["id"]]
            if f["color"] is not None:
                cz,ce=COLORS[f["color"]]
                add(a,f"画面{dz}的{cz}{zh}",f"the {ce} {en} on the {de} side of the image","combined",{"measurement":"box geometry plus mask-interior color heuristic",**f})
    return proposals

def select_images(pool,count):
    selected=[];counts=Counter();used=set()
    # Prioritize scarce supported classes, cap easy shoe-only spatial pairs.
    while len(selected)<count and len(used)<len(pool):
        best=None;bestscore=-1
        for iid,props in pool.items():
            if iid in used:continue
            cats={p["target"]["category_id"] for p in props}
            score=sum(1/(1+counts[c]) for c in cats)
            nonshoe=any(p["kind"]=="spatial" and p["target"]["category_id"]!=6 for p in props)
            score+=.025 if nonshoe else 0
            score+=.02 if any(p["kind"]=="appearance" and p.get("same_fine_count",1)>=2 for p in props) else 0
            if score>bestscore:best=iid;bestscore=score
        if best is None:break
        used.add(best);props=pool[best]
        pairs=defaultdict(list)
        for p in props:
            if p["kind"]=="spatial":pairs[p["contrast"]].append(p)
        usable=[ps for ps in pairs.values() if len(ps)==2 and (ps[0]["target"]["category_id"]!=6 or counts[6]<max(1,len(selected)*.15))]
        if usable:
            chosen=min(usable,key=lambda ps:counts[ps[0]["target"]["category_id"]])
        else:
            ranked=sorted((p for p in props if p["kind"]!="spatial"),key=lambda p:(counts[p["target"]["category_id"]],p.get("same_fine_count",1)<2,p["kind"]!="combined",p["target"]["id"]))
            chosen=[];targets=set()
            for p in ranked:
                if p["target"]["id"] not in targets:chosen.append(p);targets.add(p["target"]["id"])
                if len(chosen)==2:break
        if not chosen:continue
        selected.append((best,chosen))
        counts.update(p["target"]["category_id"] for p in chosen)
    return selected

def build_language(source,output,count,allow_fewer=False):
    require(not output.exists(),"output exists")
    manifest=read_json(source/"image_manifest.json");coco=read_json(source/"annotations_coco.json");cs=read_json(source/"candidates_gt.json")
    images={x["image_id"]:x for x in manifest["images"]};by=defaultdict(list)
    for a in coco["annotations"]:by[a["image_id"]].append(a)
    pool={}
    def propose(im):
        if im["image_id"] in QUARANTINE:return im["image_id"],[]
        with Image.open(source/im["file_name"]) as img:rgb=np.asarray(img.convert("RGB"))
        return im["image_id"],proposed_queries(im,by[im["image_id"]],rgb)
    # map preserves manifest order, so parallel CPU execution does not change sampling.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for n,(iid,props) in enumerate(executor.map(propose,manifest["images"]),1):
            if props:pool[iid]=props
            if n%500==0:print("annotate",source.name,n,"eligible",len(pool),flush=True)
    selected=select_images(pool,count);require(bool(selected) and (allow_fewer or len(selected)==count),"insufficient images with conservative appearance/spatial labels")
    ids={iid for iid,_ in selected};output.mkdir(parents=True)
    for folder in ("images","masks"):(output/folder).mkdir()
    ims=[x for x in manifest["images"] if x["image_id"] in ids]
    for im in ims:shutil.copyfile(source/im["file_name"],output/im["file_name"])
    coco["images"]=[x for x in coco["images"] if x["id"] in ids];coco["annotations"]=[x for x in coco["annotations"] if x["image_id"] in ids]
    manifest["images"]=ims;manifest["manifest_id"]=output.name
    cs["images"]=[x for x in cs["images"] if x["image_id"] in ids];cs["candidate_set_id"]=output.name+"-gt"
    for im in cs["images"]:
        for c in im["candidates"]:shutil.copyfile(source/c["mask_reference"],output/c["mask_reference"])
    write_json(output/"image_manifest.json",manifest);write_json(output/"annotations_coco.json",coco)
    cs["image_manifest"]["sha256"]=sha256(output/"image_manifest.json");cs["provenance"]["annotation_manifest_sha256"]=sha256(output/"annotations_coco.json")
    write_json(output/"candidates_gt.json",cs);shutil.copyfile(source/"mapping_config.json",output/"mapping_config.json")
    rows=[];evidence=[]
    for iid,props in selected:
        im=images[iid]
        for p in props:
            sid=f"{output.name}-q{len(evidence)+1:05d}";a=p["target"]
            evidence.append({"semantic_query_id":sid,"image_id":iid,"target_instance_id":a["id"],"evidence":p["evidence"],"physical_instance_review":"not_performed"})
            for lang in ("zh","en"):
                rows.append({**COMMON,"query_id":sid+"-"+lang,"semantic_query_id":sid,"contrast_set_id":p["contrast"],"image_id":iid,"source_dataset":"fashionpedia","split":im["split"],"query":p[lang],"language":lang,"target_instance_id":a["id"],"target_category_id":a["category_id"],"same_category_instance_count":sum(x["category_id"]==a["category_id"] and not x["iscrowd"] for x in by[iid]),"query_types":[p["kind"]],"spatial_reference":"image_view","annotation_version":output.name,"annotator":"codex_cpu_weak_labeler_v2","reviewer":None,"review_status":"needs_revision","target_source_category_id":a["source_category_id"],"target_source_category_name":a["source_category_name"],"same_source_category_instance_count":sum(x["source_category_id"]==a["source_category_id"] and not x["iscrowd"] for x in by[iid]),"authorship":"machine_generated_from_pixels_source_attributes_and_geometry_not_individually_visually_reviewed","template_family":"weak_"+p["kind"]+"_v2","paraphrase_group_id":None,"training_policy":"experimental_machine_labels_only_not_human_accepted"})
    text="".join(json.dumps(q,ensure_ascii=False)+"\n" for q in rows)
    (output/"queries_draft.jsonl").write_text(text,encoding="utf-8")
    (output/"queries_machine_train.jsonl").write_text(text if all(im["split"]=="train" for im in ims) else "",encoding="utf-8")
    (output/"queries_accepted.jsonl").write_text("")
    raw=read_json(source/"source_subset.json")
    source_ids={int(im["source_image_id"]) for im in ims}
    raw["images"]=[im for im in raw["images"] if im["id"] in source_ids]
    raw["annotations"]=[ann for ann in raw["annotations"] if ann["image_id"] in source_ids]
    write_json(output/"source_subset.json",raw)
    exposed=set()
    shared=Path("/root/autodl-tmp/fashion-grounding/shared/processed")
    for prior in ("fp_pilot20_v2","fp_grounded9_draft_v2"):
        prior_manifest=shared/prior/"image_manifest.json"
        if prior_manifest.exists():exposed.update(x["image_id"] for x in read_json(prior_manifest)["images"])
    write_json(output/"annotation_notes.json",{"disputes":[{"image_id":iid,"prior_answer_exposure":True,"reason":"previous pilot annotation/review exposure"} for iid in sorted(ids & exposed)],"human_review":"pending","evidence_file":"annotation_evidence.json"})
    write_json(output/"annotation_evidence.json",evidence)
    semantic=[q for q in rows if q["language"]=="zh"]
    report={"color_policy":"v2 conservative black/white/gray/red/yellow/green/blue; other hues abstain","requested_images":count,"eligible_pool_images":len(pool),"images":len(ims),"semantics":len(semantic),"language_records":len(rows),"accepted":0,"types":dict(Counter(q["query_types"][0] for q in semantic)),"target_categories":dict(Counter(q["target_category_id"] for q in semantic)),"same_category_competition_candidates":sum(q["same_category_instance_count"]>=2 for q in semantic),"same_fine_category_competition_candidates":sum(q["same_source_category_instance_count"]>=2 for q in semantic),"individually_visual_reviewed":0,"limitations":["Color is a thresholded pixel estimate, not a semantic ground-truth guarantee","Official pattern attributes are reused, not individually verified","No claim that annotation count guarantees distinct physical garments","Cannot enter accepted-only evaluator or final test","Not sufficient evidence of language model quality"]}
    write_json(output/"annotation_report.json",report);write_json(output/"validation_report.json",validate_package(output))
    return report
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--count",type=int,default=1000);p.add_argument("--allow-fewer",action="store_true");a=p.parse_args()
    print(json.dumps(build_language(a.source,a.output,a.count,a.allow_fewer),indent=2))
