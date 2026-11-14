from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
data=json.loads((root/'reports/cpu_results.json').read_text())['datasets']
names=[x['name'].replace('_',' ') for x in data]
savings=[100*(1-x['modes']['adaptive']['container_bytes']/x['modes']['raw']['container_bytes']) for x in data]
cost=[x['modes']['adaptive']['encode_ms_median']/x['modes']['raw']['encode_ms_median'] for x in data]
fig,axes=plt.subplots(1,2,figsize=(13,6.5),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
axes[0].barh(names,savings,color='#226d85');axes[0].set_xlabel('Container size reduction vs raw + zlib (%)');axes[0].set_xlim(0,110);axes[0].invert_yaxis()
axes[1].barh(names,cost,color='#b86b32');axes[1].axvline(1,color='black',lw=1);axes[1].set_xlabel('Adaptive / baseline encoding time (ratio)')
for i,(s,c) in enumerate(zip(savings,cost)):
 axes[0].text(s+1,i,f'{s:.1f}%',va='center',fontsize=9)
 axes[1].text(c+0.03,i,f'{c:.2f}×',va='center',fontsize=9)
axes[1].set_xlim(0,max(cost)*1.22)
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('PixelPack: compression savings have an encoding cost',fontsize=16,fontweight='bold',y=.98)
fig.text(.5,.015,'Measured CPU run · 11 fixed datasets · 7 repetitions · zlib level 6 · 2 OpenMP threads\nAdaptive mode tries both representations. CUDA has not been run.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(root/'reports/cpu_tradeoffs.png',dpi=150);plt.close(fig)
