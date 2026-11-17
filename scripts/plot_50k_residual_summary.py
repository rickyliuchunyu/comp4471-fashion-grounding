"""Scientific summary of the existing residual diagnostic JSON; no model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=json.loads(Path(a.summary).read_text());g=s['groups'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
labels=['Matched >= .75','Matched .50-.75','Correct raw mask, score filtered','Partial/mixed raw geometry','Only wrong class overlaps','No raw overlap >= .10']
small=g['small'];v=[small['matched75'],small['matched50']-small['matched75'],small['residual_causes']['score_filtered'],small['residual_causes']['partial_or_mixed_geometry'],small['residual_causes']['wrong_class_only'],small['residual_causes']['no_saved_mask_iou_ge_010']]
bars=axes[0].barh(labels[::-1],v[::-1],color=['#2f7f73','#80b6a9','#c99139','#d66c4a','#8c6bb1','#9b3030'][::-1]);axes[0].bar_label(bars,padding=3)
axes[0].set(xlabel='GT count (all 138 small instances)',xlim=(0,55),title='Small-object outcomes at fixed score 0.3')
keys=['small','category_6','category_7','category_8'];names=['Small\nn=138','Shoes\nn=327','Bags\nn=45','Accessories\nn=200'];x=np.arange(4)
for offset,key,color,label in [(-.18,'matched50','#3e7d9b','IoU >= .50'),(.18,'matched75','#e6ab54','IoU >= .75')]:
    values=[100*g[k][key]/g[k]['gt_count'] for k in keys]
    bars=axes[1].bar(x+offset,values,width=.36,color=color,label=label)
    axes[1].bar_label(bars,labels=[f"{g[k][key]}/{g[k]['gt_count']}" for k in keys],padding=3,fontsize=8)
axes[1].set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='Matched GT (%)',title='Fixed candidate matching; not AP')
axes[1].legend(loc='lower right');axes[1].grid(axis='y',alpha=.15)
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('50k / dev250: saved predictions and candidate cache only',fontsize=15)
target=Path(a.output);assert not target.exists();fig.savefig(target,dpi=170)
print(target)
