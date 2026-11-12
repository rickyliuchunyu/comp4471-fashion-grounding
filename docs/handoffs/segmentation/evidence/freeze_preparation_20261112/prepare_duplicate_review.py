"""Prepare human review of known pairs, without decisions or data changes."""
import hashlib
import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

root = Path('/root/autodl-tmp/fashion-grounding')
run = root / 'runs/segmentation/segmentation-freeze-preparation-20261112-r01'
out = run / 'duplicate_review'
out.mkdir(exist_ok=False)
preflight = json.loads((run / 'freeze_preflight.json').read_text())
fingerprints = json.loads((root / 'shared/processed/fp_data_eval_large_v1/fingerprints.json').read_text())
by_key = {r['key']: r for r in fingerprints}
rows, cards = [], []
for i, pair in enumerate(preflight['duplicate_pair_dispositions'], 1):
    row = dict(pair, pair_id=f'pair_{i:03d}', human_decision='pending',
               reviewer=None, reviewed_at=None, rationale=None,
               conservative_group_action='retain_existing_group_and_quarantine')
    sheet = Image.new('RGB', (1200, 640), 'white')
    draw = ImageDraw.Draw(sheet)
    for k, side in enumerate(('a', 'b')):
        data = by_key[pair[side]]
        raw = Path(data['path'])
        row[f'sha256_{side}'] = data['sha256']
        row[f'group_{side}'] = data['group_id']
        # Existing image hashes were validated by production; this renders only.
        im = Image.open(raw).convert('RGB')
        sheet.paste(ImageOps.contain(im, (590, 590)), (k*600, 40))
        draw.text((k*600+5, 5), f"{pair[side]} | {pair['split_'+side]}", fill='black')
    name = row['pair_id'] + '.jpg'
    sheet.save(out / name, quality=93)
    row['sheet_sha256'] = hashlib.sha256((out/name).read_bytes()).hexdigest()
    rows.append(row)
    cards.append(f'<section><h2>{i}: {html.escape(pair["a"])} / {html.escape(pair["b"])}</h2>'
                 f'<p>pHash distance={pair["distance"]}; existing same-group={pair["same_group"]}. '
                 'Inspect originals if uncertain. Keep conservative grouping unless a separately reviewed split revision is published.</p>'
                 f'<a href="{name}"><img loading="lazy" width="1000" src="{name}"></a>'
                 f'<p>Decision <select id="d{i}"><option>pending</option><option>same_or_near_duplicate</option>'
                 '<option>different</option><option>uncertain</option></select> '
                 f'Reason <input id="r{i}" size="70"></p></section>')
review = dict(status='human_review_pending', formal_freeze=False,
              purpose='data quality only; no model/test metric inspection', pairs=rows)
(out/'review_template.json').write_text(json.dumps(review, indent=2)+'\n')
script = '''
function save(){const r=DATA; r.reviewer=document.getElementById('name').value.trim();
if(!r.reviewer){alert('Enter actual human reviewer name');return;}
r.reviewed_at=new Date().toISOString();r.pairs.forEach((x,i)=>{
x.human_decision=document.getElementById('d'+(i+1)).value;
x.rationale=document.getElementById('r'+(i+1)).value;
x.reviewer=r.reviewer;x.reviewed_at=r.reviewed_at;});
r.status=r.pairs.some(x=>['pending','uncertain'].includes(x.human_decision))?'human_review_incomplete':'human_review_recorded_pending_team_freeze';
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(r,null,2)],{type:'application/json'}));
a.download='duplicate_human_review.json';a.click();URL.revokeObjectURL(a.href);}
'''.replace('DATA', json.dumps(review))
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Known near-duplicate review</title>'
    '<style>body{font:16px sans-serif;max-width:1100px;margin:auto}section{border-top:1px solid #bbb;padding:12px}img{max-width:100%}</style>'
    '<h1>53 known near-duplicate pairs: human review pending</h1>'
    '<p>All known pairs already grouped conservatively; cross-source validation images quarantined. '
    'This material does not establish discovery of all duplicates. Test images, where present, are for quality review only.</p>'
    '<p>Actual human reviewer <input id="name"><button onclick="save()">Download review record</button></p>'
    + ''.join(cards) + '<script>'+script+'</script>')
print(json.dumps({'pairs': len(rows), 'path': str(out), 'human_decisions': 0}))
