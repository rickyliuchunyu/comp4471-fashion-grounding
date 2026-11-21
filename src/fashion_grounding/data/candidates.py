"""Validate file-level candidates without loading query answers."""
import math
from PIL import Image
from .schema import read_json,sha256,safe_path,require,unique_index,decode_mask,mask_box

def finite(value):
    return type(value) in (int,float) and math.isfinite(value)

def load_candidates(path):
    root=path.parent;p=read_json(path)
    require(p["schema_version"]==3 and p["illustrative_only"] is False,"real candidate schema 3 required")
    require(isinstance(p["candidate_set_id"],str) and bool(p["candidate_set_id"]),"candidate set ID")
    source=p["candidate_source"];require(source in ("gt","predicted","external"),"candidate source")
    ref=p["image_manifest"];mp=safe_path(root,ref["path"])
    require(sha256(mp)==ref["sha256"],"image manifest hash")
    manifest=read_json(mp)
    require(manifest["schema_version"]==3 and manifest["illustrative_only"] is False,"image manifest schema")
    images=unique_index(manifest["images"],"image_id");rows=unique_index(p["images"],"image_id")
    require(set(images)==set(rows),"candidate image coverage")
    prov=p["provenance"]
    if source=="predicted":
        require(bool(prov["producer_run_id"]) and all(isinstance(prov[k],str) and len(prov[k])==64 for k in ("checkpoint_sha256","decoder_config_sha256")),"prediction provenance")
    if source=="gt":
        require(isinstance(prov["annotation_manifest_sha256"],str) and len(prov["annotation_manifest_sha256"])==64,"GT annotation provenance")
    seen=set();masks={}
    for iid,row in rows.items():
        im=images[iid];w,h=im["width"],im["height"]
        image_path=safe_path(mp.parent,im["file_name"])
        require(sha256(image_path)==im["sha256"],"image file hash")
        with Image.open(image_path) as image:
            image.load();require(image.size==(w,h),"image dimensions")
        require(row["image_size_wh"]==[w,h],"image size")
        require(row["status"] in ("ok","no_candidates","inference_error"),"image status")
        cs=row["candidates"]
        require((bool(cs) and row["status"]=="ok") or (not cs and row["status"]!="ok"),"empty/status mismatch")
        if row["status"]=="inference_error":require(bool(row.get("error")),"upstream error required")
        decoder=[];aids=[]
        for c in cs:
            cid=c["candidate_id"];require(isinstance(cid,str) and cid and cid not in seen,"duplicate/empty candidate ID");seen.add(cid)
            require(c["category_id"] in range(1,9) or (source=="external" and c["category_id"] is None),"candidate category")
            if source=="gt":
                require(c["score"] is None and c["decoder_query_id"] is None and type(c["source_annotation_id"]) is int,"GT fields")
                aids.append(c["source_annotation_id"])
            else:
                require(finite(c["score"]),"candidate score")
                require(c["source_annotation_id"] is None,"prediction must not carry GT association")
                if source=="predicted":
                    require(type(c["decoder_query_id"]) is int and c["decoder_query_id"]>=0,"decoder ID")
                    decoder.append(c["decoder_query_id"])
            mask_path=safe_path(root,c["mask_reference"])
            require(sha256(mask_path)==c["mask_sha256"],"mask hash")
            mask=decode_mask(read_json(mask_path),h,w)
            require(c["bbox_xyxy"]==mask_box(mask),"candidate box differs from mask")
            masks[cid]=mask
        require(len(decoder)==len(set(decoder)),"duplicate decoder query")
        if source=="gt":require(aids==sorted(set(aids)),"GT order/duplicate source annotation")
        else:require(cs==sorted(cs,key=lambda c:(-c["score"],c["candidate_id"])),"prediction ordering")
    return p,manifest,masks
