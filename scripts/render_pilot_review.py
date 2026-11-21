"""Render original/GT overlay evidence and a local, answer-hidden human review form."""
from collections import defaultdict
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_grounding.data.schema import decode_mask, read_json

root = Path(sys.argv[1])
manifest = read_json(root / "image_manifest.json")
coco = read_json(root / "annotations_coco.json")
annotations = defaultdict(list)
for a in coco["annotations"]:
    annotations[a["image_id"]].append(a)
queries = [json.loads(x) for x in (root / "queries_draft.jsonl").read_text(encoding="utf-8").splitlines()]
grouped_queries = defaultdict(list)
for q in queries:
    grouped_queries[q["semantic_query_id"]].append(q)
palette = [(220, 55, 55), (15, 130, 220), (36, 175, 80), (240, 150, 10), (150, 65, 220), (20, 170, 170), (225, 70, 155), (120, 130, 35)]
font_path = Path("C:/Windows/Fonts/arial.ttf")
font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
small = ImageFont.truetype(str(font_path), 16) if font_path.exists() else font
(root / "overlays").mkdir(exist_ok=True)
tiles, sections = [], []
for im in manifest["images"]:
    original = Image.open(root / im["file_name"]).convert("RGB")
    pixels = np.array(original).astype(float)
    rows = annotations[im["image_id"]]
    for index, a in enumerate(rows):
        mask = decode_mask(a["segmentation"], im["height"], im["width"])
        pixels[mask] = pixels[mask] * 0.62 + np.array(palette[index % len(palette)]) * 0.38
    overlay = Image.fromarray(pixels.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    for index, a in enumerate(rows):
        x, y, w, h = a["bbox"]
        color = palette[index % len(palette)]
        draw.rectangle((x, y, x+w-1, y+h-1), outline=color, width=3)
        label = str(a["id"])
        tx, ty = min(x, max(0, im["width"]-110)), min(y, max(0, im["height"]-25))
        draw.rectangle((tx, ty, tx+108, ty+24), fill=(255,255,255))
        draw.text((tx+2, ty), label, fill=color, font=font)
    overlay.save(root / "overlays" / f"{im['image_id']}.jpg", quality=92)
    tile = Image.new("RGB", (440, 300), "white")
    for j, picture in enumerate((original, overlay)):
        thumb = picture.copy(); thumb.thumbnail((216, 260))
        tile.paste(thumb, (j*220 + (216-thumb.width)//2, 28))
    ImageDraw.Draw(tile).text((6, 5), f"Image {im['image_id']} | original / GT", fill="black", font=small)
    tiles.append(tile)
    legend = " · ".join(f"{a['id']}: {html.escape(a['source_category_name'])}" for a in rows)
    query_html = []
    for semantic, qs in grouped_queries.items():
        if qs[0]["image_id"] != im["image_id"]:
            continue
        texts = " / ".join(html.escape(q["query"]) for q in qs)
        options = '<option value="">尚未复核</option><option value="reject">不唯一／描述不准确／需要修改</option>' + "".join(f'<option value="{a["id"]}">{a["id"]} — {html.escape(a["source_category_name"])}</option>' for a in rows if not a["iscrowd"])
        query_html.append(f'<div class="query"><b>{texts}</b><p>只看图片与描述，选择目标实例。确认中英含义一致后记录。</p><select data-semantic="{semantic}">{options}</select><input data-note="{semantic}" placeholder="修改建议或问题"><details><summary>选择后查看草稿目标（不计作独立盲选）</summary>目标 ID：{qs[0]["target_instance_id"]}</details></div>')
    sections.append(f'<section><h2>图片 {im["image_id"]}</h2><div class="pair"><img src="{im["file_name"]}" alt="原图"><img src="overlays/{im["image_id"]}.jpg" alt="GT mask 与实例 ID"></div><p class="legend">{legend}</p>{"".join(query_html)}</section>')
contact = Image.new("RGB", (440*4, 300*((len(tiles)+3)//4)), (225,225,225))
for i, tile in enumerate(tiles):
    contact.paste(tile, ((i%4)*440, (i//4)*300))
contact.save(root / "contact_sheet.jpg", quality=90)
data = json.dumps({"manifest_id": manifest["manifest_id"], "queries": queries}, ensure_ascii=False).replace("<", "\\u003c")
document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Fashionpedia 20 张数据复核</title><style>
body{max-width:1200px;margin:30px auto;padding:0 24px;font:16px/1.65 system-ui;background:#f4f5f7;color:#19212b}header,section{background:white;padding:24px;margin:20px 0;border-radius:12px}.pair{display:flex;gap:12px}.pair img{width:49%;object-fit:contain;max-height:650px}.legend{font-size:13px}.query{border-top:1px solid #ddd;padding:18px 0}select,input,button{font:inherit;padding:8px;margin:5px}input{max-width:90%}button{cursor:pointer}details{font-size:13px;color:#5b6471}.tools{position:sticky;top:0;background:#eff5ff;padding:8px;z-index:1}h1{font-size:26px}</style>
<header><h1>Fashionpedia 训练集 · 20 张接口小包</h1><p>图片和实例标注来自官方训练集；下列 40 组双语描述为机器生成草稿，尚无人审通过。这里的 GT 叠图是数据检查，不是模型预测。</p><p>逐项选择你认为的目标；有歧义或翻译问题选“需要修改”。导出结果不会自动修改正式数据或标记 accepted。人工复核及上下游验收完成前，P2 仍未完成。请同时检查 mask 是否对应完整实例。</p></header>
<div class="tools">复核人：<input id="reviewer" placeholder="填写你的真实成员简称"><button id="export">导出复核记录 JSON</button><span id="progress"></span></div>'''
document += "".join(sections)
document += '<script>const DATA=' + data + ''';
const selects=[...document.querySelectorAll('select')];
function progress(){document.getElementById('progress').textContent=selects.filter(x=>x.value).length+'/'+selects.length+' 组已填写'}
const key='review-'+DATA.manifest_id;
function save(){try{localStorage.setItem(key,JSON.stringify({reviewer:document.getElementById('reviewer').value,rows:selects.map(s=>({id:s.dataset.semantic,value:s.value,note:document.querySelector('[data-note="'+s.dataset.semantic+'"]').value}))}));}catch(e){}progress();}
try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved){document.getElementById('reviewer').value=saved.reviewer;for(const row of saved.rows){const s=selects.find(x=>x.dataset.semantic===row.id);if(s){s.value=row.value;document.querySelector('[data-note="'+row.id+'"]').value=row.note;}}}}catch(e){}
selects.forEach(x=>x.addEventListener('change',save));document.querySelectorAll('input').forEach(x=>x.addEventListener('input',save));progress();
document.getElementById('export').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('请填写复核人');return;}
const rows=selects.map(s=>({semantic_query_id:s.dataset.semantic,selected_annotation_id:s.value && s.value!=='reject'?Number(s.value):null,decision:s.value==='reject'?'needs_revision':s.value?'human_selected':'not_reviewed',note:document.querySelector('[data-note="'+s.dataset.semantic+'"]').value}));
const output={manifest_id:DATA.manifest_id,reviewer,reviewed_at:new Date().toISOString(),purpose:'Human review evidence; acceptance requires checked import, not automatic',rows};
const blob=new Blob([JSON.stringify(output,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='pilot20-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
</script></html>'''
(root / "review.html").write_text(document, encoding="utf-8")
print("Rendered", len(tiles), "overlays and", len(grouped_queries), "bilingual review items")
