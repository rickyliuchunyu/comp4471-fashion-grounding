"""Bounded real-image encoder smoke; drafts are diagnostics, never benchmark scores."""
import argparse,json,time
from pathlib import Path
import torch
from fashion_grounding.data.schema import write_json,sha256,validate_package
from fashion_grounding.grounding.inputs import CandidatePack,crop_regions
from fashion_grounding.encoders.siglip import SiglipEncoder

def main():
    p=argparse.ArgumentParser();p.add_argument("--package",required=True);p.add_argument("--snapshot",required=True)
    p.add_argument("--revision",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    package=Path(a.package);checked=validate_package(package)
    pack=CandidatePack(package/"candidates_gt.json")
    crop_count=0
    for iid in pack.rows:
        image,cs=pack.load(iid);crop_count+=len(crop_regions(image,cs))
    rows=[json.loads(line) for line in (package/"queries_draft.jsonl").read_text().splitlines()]
    image,cs=pack.load(rows[0]["image_id"]);crops=crop_regions(image,cs)[:2]
    texts=[q["query"] for q in rows if q["image_id"]==rows[0]["image_id"]][:2]
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    encoder=SiglipEncoder(a.snapshot,a.revision,"cuda",8)
    verified=encoder.verify_logits(crops,texts)
    rejected=False
    try: encoder.encode_texts(["bag "*1000])
    except ValueError: rejected=True
    if not rejected: raise RuntimeError("Overlength text was silently accepted")
    torch.cuda.synchronize()
    report={"status":"passed","purpose":"engineering_smoke_only","formal_metrics":None,
            "source":"Fashionpedia train pilot with unreviewed draft text",
            "data_validation":checked,"valid_crops":crop_count,"logits":verified,
            "overlength_rejected":rejected,"seconds":time.monotonic()-start,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "encoder":encoder.signature,"candidate_sha256":pack.digest}
    write_json(out/"smoke_report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="encoder"},ensure_ascii=False))
if __name__=="__main__":main()
