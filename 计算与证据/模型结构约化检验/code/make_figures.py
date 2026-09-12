import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_structure_test import ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})

def main():
    s=json.loads((ROOT/'results/study.json').read_text(encoding='utf-8'))
    scales=json.loads((ROOT/'results/reduction_scales.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    for q,col in [(3,'#2556b8'),(4,'#c36713')]:
        rows=[r for r in s['cases'] if r['q']==q and r['shape_power']==0]
        x=[r['eta'] for r in rows]
        ax[0].plot(x,[r['change_pct'] for r in rows],'o-',lw=2,color=col,label=f'第{q}问')
        ax[1].plot(x,[r['max_sampled_T_difference_C'] for r in rows],'o-',lw=2,color=col,label=f'第{q}问')
    ax[0].set(title='终止时间：相对全部表面汽化的变化',ylabel='时长变化 / %')
    ax[1].set(title='局部温场：最大采样温度差',ylabel='温度差 / °C')
    for a in ax:a.set_xlabel('内部潜热输运比例 η（情景参数）');a.grid(alpha=.2);a.legend()
    fig.suptitle('保持题给总扩散定律与边界：时长稳定不代表局部温度不变',fontsize=14)
    fig.tight_layout();fig.savefig(ROOT/'figures/01_相变位置敏感性.png',bbox_inches='tight');plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    for q,col in [(3,'#2556b8'),(4,'#c36713')]:
        rows=sorted([r for r in scales['bounds'] if r['q']==q],key=lambda r:r['C'])
        ax[0].loglog([r['C'] for r in rows],[r['vapor_mass_fraction_bound']*100 for r in rows],'o-',color=col,label=f'第{q}问')
    ax[0].set(title='孔隙蒸气库存的条件性上界',xlabel='总干基含水率 C',ylabel='蒸气水量 / 总水量上界（%）')
    d=scales['prior_coefficient_consistency']
    ax[1].semilogy(d['C_liquid'],np.array(d['D_vapor_equilibrium'])/d['D_given'],color='#a43948',lw=2)
    ax[1].axhline(1,color='gray',ls='--');ax[1].set(title='独立双相试算的参数并未匹配题给D',xlabel='液态干基含水率 C',ylabel='试算平衡蒸气扩散贡献 / 题给D')
    for a in ax:a.grid(alpha=.2)
    ax[0].legend();fig.suptitle('小库存可支持简化储量，但不能自动证明局部平衡或小通量',fontsize=14)
    fig.tight_layout();fig.savefig(ROOT/'figures/02_库存界限与参数一致性.png',bbox_inches='tight');plt.close(fig)
    print('FIGURES_DONE',flush=True)

if __name__=='__main__':main()
