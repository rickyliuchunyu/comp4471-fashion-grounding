"""Fetch or verify the fixed public SigLIP 2 snapshot; no credentials required."""
import argparse,hashlib,json,time
from pathlib import Path
import requests

MODEL="google/siglip2-base-patch16-224"
REVISION="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
WEIGHT_SHA256="612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--endpoint",choices=["https://huggingface.co","https://hf-mirror.com"],default="https://huggingface.co")
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--max-seconds",type=int,default=900)
    a=p.parse_args();root=Path(a.output)
    if a.verify_only:
        meta=json.loads((root/"download_manifest.json").read_text())
        assert meta["revision"]==REVISION and meta["model_id"]==MODEL
        for name,info in meta["files"].items():
            assert Path(name).name==name and digest(root/name)==info["sha256"],name
        assert meta["files"]["model.safetensors"]["sha256"]==WEIGHT_SHA256
        print(json.dumps({"status":"verified","revision":REVISION,"files":len(meta["files"])}))
        return
    if a.max_seconds<=0:p.error("positive download budget required")
    root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();s=requests.Session()
    response=s.get(a.endpoint+"/api/models/"+MODEL+"/revision/"+REVISION+"?blobs=true",timeout=30)
    response.raise_for_status();meta=response.json()
    if meta["sha"]!=REVISION:raise RuntimeError("revision mismatch")
    records={}
    for item in meta["siblings"]:
        name=item["rfilename"]
        if not name.endswith((".json",".model",".safetensors")):continue
        if Path(name).name!=name:raise RuntimeError("unexpected nested model file")
        path=root/name;partial=root/(name+".partial");h=hashlib.sha256();size=0
        print("Downloading",name,item["size"],flush=True)
        with s.get(a.endpoint+"/"+MODEL+"/resolve/"+REVISION+"/"+name,stream=True,timeout=(20,60)) as r:
            r.raise_for_status()
            with partial.open("xb") as f:
                for block in r.iter_content(1024*1024):
                    if time.monotonic()-started>a.max_seconds:raise TimeoutError("download budget exceeded")
                    f.write(block);h.update(block);size+=len(block)
        expected=WEIGHT_SHA256 if name=="model.safetensors" else item.get("lfs",{}).get("sha256")
        if size!=item["size"] or (expected and h.hexdigest()!=expected):raise RuntimeError("file integrity mismatch: "+name)
        partial.rename(path)
        records[name]={"bytes":size,"sha256":h.hexdigest(),"upstream_lfs_sha256":expected}
    (root/"download_manifest.json").write_text(json.dumps({"model_id":MODEL,"revision":REVISION,"source":a.endpoint,"files":records},indent=2)+"\n")
    print("Download verified:",root)
if __name__=="__main__":main()
