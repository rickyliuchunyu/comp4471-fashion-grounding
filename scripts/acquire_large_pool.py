"""Acquire a versioned Fashionpedia pool, using checked ZIP member ranges on CPU."""
import argparse,concurrent.futures,json,random,struct,sys,threading,time,zlib,zipfile
from collections import defaultdict,Counter
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fashion_grounding.data.schema import read_json,write_json,sha256,decode_mask,require
from acquire_pilot import RemoteZip

def download_members(url,images,dest,workers=12):
 remote=RemoteZip(url)
 with zipfile.ZipFile(remote) as z:
  infos={Path(i.filename).name:i for i in z.infolist() if not i.is_dir()}
 dest.mkdir(parents=True,exist_ok=True);local=threading.local()
 def get(im):
  name=Path(im["file_name"]).name;info=infos[name];path=dest/name
  if path.exists() and path.stat().st_size==info.file_size and zlib.crc32(path.read_bytes())==info.CRC:return
  if not hasattr(local,"session"):local.session=requests.Session()
  last=None
  for attempt in range(5):
   try:
    start=info.header_offset;end=start+30+len(info.filename.encode("utf-8"))+65536+info.compress_size-1
    end=min(end,remote.length-1)
    r=local.session.get(url,headers={"Range":f"bytes={start}-{end}","If-Match":remote.etag},timeout=90)
    r.raise_for_status();require(r.status_code==206,"range not honored")
    require(r.headers.get("Content-Range")==f"bytes {start}-{end}/{remote.length}","range mismatch")
    data=r.content;require(data[:4]==b"PK\x03\x04","local ZIP header")
    fields=struct.unpack("<4s5H3I2H",data[:30]);offset=30+fields[-2]+fields[-1]
    compressed=data[offset:offset+info.compress_size]
    require(len(compressed)==info.compress_size,"short ZIP member")
    content=compressed if info.compress_type==0 else zlib.decompress(compressed,-15) if info.compress_type==8 else None
    require(content is not None and len(content)==info.file_size and zlib.crc32(content)==info.CRC,"member CRC")
    partial=path.with_suffix(".partial");partial.write_bytes(content);partial.replace(path);return
   except Exception as exc:last=exc;time.sleep(attempt+1)
  raise RuntimeError(f"download failed {name}: {last}")
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
  futures=[ex.submit(get,im) for im in images]
  for n,f in enumerate(concurrent.futures.as_completed(futures),1):
   f.result()
   if n%100==0 or n==len(images):print("download",dest.name,n,len(images),flush=True)
 return {"url":url,"etag":remote.etag,"bytes":remote.length,"members":len(images),"crc_checked":True}

def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--train-source",type=Path,required=True);a=p.parse_args()
 a.run.mkdir(parents=True,exist_ok=True);raw=a.run/"raw";raw.mkdir(exist_ok=True)
 val=raw/"instances_attributes_val2020.json"
 if not val.exists():
  response=requests.get("https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json",timeout=90);response.raise_for_status()
  partial=val.with_suffix(".partial");partial.write_bytes(response.content);read_json(partial);partial.replace(val)
 config=read_json(Path(__file__).resolve().parents[1]/"configs/source_category_mapping.json")
 mapping={r["source_id"]:r for r in config["fashionpedia"]}
 selections={};provenance={};rejected={}
 for split,path,count in [("train",a.train_source,6000),("val",val,None)]:
  data=read_json(path);by=defaultdict(list)
  for ann in data["annotations"]:by[ann["image_id"]].append(ann)
  pool=sorted(data["images"],key=lambda x:x["id"]);random.Random(2026).shuffle(pool)
  chosen=[];bad=Counter()
  for im in pool:
   anns=by[im["id"]]
   if any(mapping[x["category_id"]]["action"]=="exclude_image" for x in anns):bad["jumpsuit_image"]+=1;continue
   whole=[x for x in anns if mapping[x["category_id"]]["action"]=="map"]
   if not whole:bad["no_whole_instances"]+=1;continue
   try:
    for x in whole:decode_mask(x["segmentation"],im["height"],im["width"])
   except (ValueError,KeyError,TypeError):bad["invalid_mask_image"]+=1;continue
   chosen.append(im)
   if count and len(chosen)==count:break
  print("selected",split,len(chosen),dict(bad),flush=True)
  selections[split]={"source_path":str(path),"source_sha256":sha256(path),"images":chosen}
  rejected[split]=dict(bad)
  write_json(raw/"selection.json",{"seed":2026,"selections":selections,"rejected":rejected,"status":"acquisition_pool_not_final_split"})
  url="https://s3.amazonaws.com/ifashionist-dataset/images/"+("train2020.zip" if split=="train" else "val_test2020.zip")
  provenance[split]=download_members(url,chosen,raw/split)
  write_json(raw/"download_provenance.json",provenance)
 print("ACQUISITION_COMPLETE",flush=True)
if __name__=="__main__":main()
