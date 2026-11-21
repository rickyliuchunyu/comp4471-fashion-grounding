"""Sample newly available shoes and bags by source mask topology for AI inspection."""
import ctypes
import random
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from full_train_inventory import STAGED,RUN,sources,read_json,write_json,require
from fashion_grounding.data.schema import decode_mask,mask_box
from pathlib import Path

def main():
    seen={r["key"] for r in read_json(STAGED/"pool_fingerprints.json")}
    inv=[r for r in read_json(RUN/"eligible.json") if r["key"] not in seen]
    random.Random(2026).shuffle(inv)
    d,by=sources()
    topo=ctypes.CDLL(str(RUN/"topology.so")).topology
    topo.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
    groups={"shoe_hole":[],"bag_sparse":[],"separated":[]}
    for r in inv:
        if all(len(a)==4 for a in groups.values()):break
        im=r["source_image"]
        for a in by[im["id"]]:
            if a["category_id"] not in (23,24):continue
            m=decode_mask(a["segmentation"],im["height"],im["width"])
            x,y,x1,y1=mask_box(m)
            c=np.ascontiguousarray(m[y:y1,x:x1],dtype=np.uint8);holes=ctypes.c_int()
            nc=topo(c.ctypes.data,c.shape[0],c.shape[1],ctypes.byref(holes))
            choices=[("shoe_hole",a["category_id"]==23 and holes.value>0),("bag_sparse",a["category_id"]==24 and c.mean()<0.5),("separated",nc>1)]
            key=next((k for k,ok in choices if ok and len(groups[k])<4),None)
            if not key:continue
            image=Image.open(r["path"]).convert("RGB")
            box=(max(0,x-10),max(0,y-10),min(im["width"],x1+10),min(im["height"],y1+10))
            pic=np.array(image.crop(box));mask=m[box[1]:box[3],box[0]:box[2]]
            faded=pic.copy();faded[mask]=(.55*pic[mask]+.45*np.array([255,30,40])).astype(np.uint8)
            groups[key].append((Image.fromarray(pic),Image.fromarray(faded),{"image_id":im["id"]+1,"source_annotation_id":a["id"],"stratum":key,"category":a["category_id"],"components":nc,"holes":holes.value}))
            break
    rows=[]
    for group,items in groups.items():
        grid=Image.new("RGB",(1152,800),"white");draw=ImageDraw.Draw(grid)
        for i,(a,b,row) in enumerate(items):
            x=i%2*576;y=i//2*400
            draw.text((x+3,y+4),str(row["source_annotation_id"])+" "+group,fill="black")
            for j,pic in enumerate((a,b)):
                pic=ImageOps.contain(pic,(280,365))
                grid.paste(pic,(x+j*288,y+25))
            rows.append(row)
        grid.save(RUN/(group+".jpg"),quality=94)
    write_json(RUN/"visual_sample.json",rows)
if __name__=="__main__":main()
