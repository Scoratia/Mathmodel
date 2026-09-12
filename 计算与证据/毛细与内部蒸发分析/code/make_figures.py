from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from two_phase import Model,Parameters,ROOT

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors={'neither':'#8b95a5','capillary':'#d57918','internal':'#227ab5','coupled':'#26784f'}
labels={'neither':'仅表面局部汽化（双机制关闭）','capillary':'毛细迁移＋表面汽化',
        'internal':'内部汽化＋蒸气扩散','coupled':'毛细与内部汽化耦合'}

def finish(fig,name):
    fig.savefig(ROOT/'figures'/name,bbox_inches='tight');plt.close(fig)

def main():
    results=json.loads((ROOT/'results/study.json').read_text(encoding='utf-8'))
    histories={k:np.array(json.loads((ROOT/f'results/history_{k}.json').read_text(encoding='utf-8'))['rows']) for k in colors}
    fig,ax=plt.subplots(2,2,figsize=(13.5,9))
    for k,a in histories.items():
        t=a[:,0]/3600
        for axis,col in [(ax[0,0],1),(ax[0,1],2),(ax[1,0],3),(ax[1,1],4)]:
            axis.plot(t,a[:,col],color=colors[k],label=labels[k],lw=2)
    ax[0,0].set_title('总含水率的体积平均');ax[0,1].set_title('最湿位置：决定是否全部达标')
    for axis in ax[0]:axis.set_ylabel('总干基含水率 / kg/kg');axis.axhline(.15,color='#b12735',ls='--',lw=1)
    ax[1,0].set_title('中心附近温度');ax[1,1].set_title('外表面邻近单元温度')
    for axis in ax[1]:axis.set_ylabel('温度 / °C')
    for axis in ax.ravel():axis.set_xlabel('时间 / h');axis.grid(alpha=.2);axis.set_xlim(0,120)
    ax[0,0].legend(fontsize=9,loc='upper right')
    fig.suptitle('毛细输运与内部蒸发的独立作用及耦合｜固定圆柱，探索参数，N=80',fontsize=15)
    fig.tight_layout();finish(fig,'01_机制对照_含水率与温度.png')

    z=np.load(ROOT/'results/fields_coupled_fine.npz');t=z['time_s']/3600;r=z['r_m']*1000;f=z['values']
    mask=(t>=.1)&(t<=72);tm=t[mask];fm=f[mask]
    latent=Model.Lv0+(Model.cv-Model.cl)*fm[:,0]
    power=latent*fm[:,3]
    vmax=max(100,float(np.percentile(np.abs(power),99.5)))
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    im=ax[0].pcolormesh(tm,r,fm[:,1].T,cmap='YlGnBu',shading='auto',vmin=.10,vmax=2.55)
    fig.colorbar(im,ax=ax[0],label='液态水干基含水率 / kg/kg')
    im=ax[1].pcolormesh(tm,r,power.T,cmap='RdBu',shading='auto',norm=SymLogNorm(linthresh=10,vmin=-vmax,vmax=vmax))
    fig.colorbar(im,ax=ax[1],label='内部相变吸热 / W/m³（负值为放热）')
    ax[0].set_title('含水率时空分布');ax[1].set_title('蒸发吸热与凝结放热的位置')
    for a in ax:a.set_xlabel('时间 / h');a.set_ylabel('半径 / mm（0为中心，20为表面）')
    fig.suptitle('耦合模型的内部相变并非均匀分布｜探索参数，N=160',fontsize=15)
    fig.tight_layout();finish(fig,'02_内部相变时空分布.png')

    fig,ax=plt.subplots(1,2,figsize=(13,5))
    for k in ['capillary','internal','coupled']:
        zz=np.load(ROOT/f'results/fields_{k}.npz');tt=zz['time_s'];ff=zz['values'];rr=zz['r_m']*1000
        idx=np.argmin(abs(tt-72*3600))
        ax[0].plot(rr,ff[idx,1]+ff[idx,2]/Model.rho_d,color=colors[k],lw=2,label=labels[k])
        a=histories[k];mask=a[:,0]>=3600
        ax[1].plot(a[mask,0]/3600,(a[mask,6]+a[mask,7])*3600*1000,color=colors[k],label=labels[k])
    ax[0].set(title='72小时径向剖面',xlabel='半径 / mm',ylabel='总干基含水率 / kg/kg')
    ax[0].axhline(.15,color='#b12735',ls='--',lw=1)
    ax[1].set(title='实际离开药材的水',xlabel='时间 / h',ylabel='总排水速率 / g/h')
    for a in ax:a.grid(alpha=.2);a.legend(fontsize=9)
    fig.suptitle('干表层不代表中心达标｜相同探索参数下的机理对照',fontsize=15)
    fig.tight_layout();finish(fig,'03_径向梯度与失水速率.png')

    rows=[('基准耦合',results['cases']['coupled'])]+[(v['name'],v) for v in results['sensitivity']]
    names=[r[0] for r in rows];maxima=[r[1]['snapshots']['24.000000']['max_total_C'] for r in rows]
    means=[r[1]['snapshots']['24.000000']['mean_total_C'] for r in rows]
    fig,ax=plt.subplots(figsize=(10,5.5));x=np.arange(len(rows));w=.36
    ax.bar(x-w/2,maxima,w,color='#245fad',label='最湿点');ax.bar(x+w/2,means,w,color='#75b799',label='平均值')
    ax.set_xticks(x,names);ax.axhline(.15,color='#b12735',ls='--',lw=1,label='达标阈值')
    ax.set_ylabel('24小时总干基含水率 / kg/kg');ax.legend();ax.grid(axis='y',alpha=.2)
    ax.set_title('缺失参数会显著改变结果：这是情景范围，不是置信区间',pad=15)
    fig.tight_layout();finish(fig,'04_参数影响.png')

    m=Model();C=np.geomspace(.07,2.55,250);T=50.
    S=m.rho_d*C/(m.rho_l*m.eps)
    Dcap=m.rho_l*m.p.permeability/m.mu_l*S**3*m.rho_l*m.Rv*(T+273.15)*m.b/C**2/m.rho_d
    Dgiven=2.4e-3*np.exp(-.45/C-3850/(T+273.15))
    fig,ax=plt.subplots(1,2,figsize=(12,4.8))
    ax[0].plot(C,m.aw(C),lw=2,color='#26784f');ax[0].axvline(.1,color='gray',ls='--')
    ax[0].axhline(np.exp(-m.b/.1),color='gray',ls=':')
    ax[0].set(xlabel='液态水干基含水率 C',ylabel='水活度 aw',title='假设的保水关系：aw=exp(−b/C)')
    ax[1].loglog(C,Dcap,lw=2,color='#d57918',label='假设毛细系数的等温等效D')
    ax[1].loglog(C,Dgiven,lw=2,color='#245fad',label='题给有效D（参照，不相加）')
    ax[1].set(xlabel='液态水干基含水率 C',ylabel='扩散系数 / m²/s',title='新增毛细项可能与题给D重复计入')
    for a in ax:a.grid(alpha=.2)
    ax[1].legend(fontsize=9);fig.tight_layout();finish(fig,'05_闭合假设与重复计数检查.png')
    print('FIVE_FIGURES_WRITTEN',flush=True)

if __name__=='__main__':main()

