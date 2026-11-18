"""Migrate only byte-verified identical visual inputs into a new package cache key."""
import argparse,json
from pathlib import Path
from fashion_grounding.data.schema import read_json,write_json,sha256,safe_path,require
from fashion_grounding.grounding.inputs import CandidatePack
from fashion_grounding.grounding.cache import FeatureCache,visual_spec,cache_key
def run(a):
    source=CandidatePack(a.source);dest=CandidatePack(a.destination)
    require(source.package["candidate_source"]==dest.package["candidate_source"]=="gt","GT-only equivalence migration")
    signature=read_json(a.encoder_metadata)["encoder"];cache=FeatureCache(a.cache);records=[]
    source_cache=FeatureCache(a.source_cache) if getattr(a,"source_cache",None) else cache
    for iid in source.images:
        require(source.images[iid]==dest.images.get(iid),"image metadata differs")
        require(source.rows[iid]==dest.rows.get(iid),"candidate metadata/order differs")
        n=len(source.rows[iid]["candidates"])
        oldspec=visual_spec(source,iid,signature);newspec=visual_spec(dest,iid,signature)
        features=source_cache.get(oldspec,n,signature["dimension"])
        if features is None or cache.get(newspec,n,signature["dimension"]) is not None:continue
        # Verify actual bytes on both sides, not only matching hash strings.
        for pack in (source,dest):
            im=pack.images[iid]
            require(sha256(safe_path(pack.manifest_path.parent,im["file_name"]))==im["sha256"],"image bytes changed")
            for c in pack.rows[iid]["candidates"]:
                require(sha256(safe_path(pack.path.parent,c["mask_reference"]))==c["mask_sha256"],"mask bytes changed")
        cache.put(newspec,features)
        records.append({"image_id":iid,"source_key":cache_key(oldspec),"destination_key":cache_key(newspec),"candidate_count":n})
    write_json(a.output,{"scope":"identical image bytes, mask bytes, full ordered candidate and image metadata; same encoder signature",
        "source_candidate_sha256":source.digest,"destination_candidate_sha256":dest.digest,"images":len(records),"entries":records})
    print(json.dumps({"migrated_identical_images":len(records)}),flush=True)
    return len(records)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("source","destination","encoder-metadata","cache","output"):p.add_argument("--"+k,required=True)
    p.add_argument("--source-cache")
    run(p.parse_args())
