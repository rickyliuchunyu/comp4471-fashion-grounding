"""Plot saved diagnostic JSON only. No COCO matching or image inference."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.analysis);summary=json.loads((root/'small_errors.json').read_text());analysis=json.loads((root/'analysis.json').read_text());updates=summary['updates']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(12,8),layout='constrained');grid=fig.add_gridspec(2,2)
ax=fig.add_subplot(grid[0,:]);left=np.zeros(2)
segments=[('same_iou_ge_075','Same class IoU >= .75','#308576'),('same_iou_050_075','Same class IoU .50-.75','#d0aa45'),('wrong_class_only_iou_ge_050','Only wrong class IoU >= .50','#9467bd'),('partial_or_mixed_iou_010_050','Any class IoU .10-.50','#dc8e68'),('no_saved_mask_iou_ge_010','No saved mask IoU >= .10','#b54953')]
for key,label,color in segments:
 values=np.array([summary['summaries'][u]['all_small']['geometry_partition'].get(key,0) for u in updates]);ax.barh(updates,values,left=left,label=label,color=color)
 for i,v in enumerate(values):
  if v:ax.text(left[i]+v/2,i,str(v),ha='center',va='center',fontsize=12)
 left+=values
ax.set(xlim=(0,138),xlabel='Small GT count (138 instances, 96 images)',title='Saved-mask geometry partition; wrong-class-only count = 0 at both checkpoints')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=9)
b=fig.add_subplot(grid[1,0]);keys=['3','6','7','8'];labels=['skirt\nn=1','shoes\nn=68','bag\nn=2','accessory\nn=67'];x=np.arange(4)
for j,u in enumerate(updates):
 values=[analysis['per_category'][u][k]['sizes']['small']['AP'] for k in keys];bars=b.bar(x+(j-.5)*.34,values,width=.34,label='u'+u,color=['#386cb0','#e19d48'][j]);b.bar_label(bars,fmt='%.2f',fontsize=9,padding=2)
b.set(xticks=x,xticklabels=labels,ylabel='COCO small-object AP (0-100)',ylim=(0,30),title='AP-small averages these four classes equally');b.legend();b.grid(axis='y',alpha=.15)
c=fig.add_subplot(grid[1,1]);ts=list(range(50,100,5))
for u,color in zip(updates,['#386cb0','#e19d48']):
 rows=json.loads((root/f'small_gt_{u}.json').read_text());counts=[sum(r['matched_detection_ids'][str(t)]>0 for r in rows) for t in ts];c.plot(np.array(ts)/100,counts,marker='o',label='u'+u,color=color)
c.set(xticks=np.array(ts)/100,ylim=(0,138),xlabel='Mask IoU threshold',ylabel='Matched small GT / 138',title='COCO all-area matches, filtered to small GT');c.legend();c.grid(alpha=.15)
fig.suptitle('S0 small-object diagnosis from saved dev250 predictions',fontsize=16)
fig.savefig(a.output,dpi=180)
print(a.output)
