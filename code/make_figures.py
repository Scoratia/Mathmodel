"""Publication-style diagnostic figures; no generated or invented observational data."""
import os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from model import ROOT,Settings,Environment


def main():
    for name in ['Microsoft YaHei','SimHei','Noto Sans CJK SC']:
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams['font.family']=name;break
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
                        'font.size':10,'figure.dpi':160,'savefig.dpi':200})
    colors=['#184A67','#C45E37','#407B68','#92624A']
    res=ROOT/'results';out=ROOT/'figures';out.mkdir(exist_ok=True)
    v=json.loads((res/'validation.json').read_text(encoding='utf-8'))
    d=np.load(res/'profiles_full_precision.npz');env=Environment(Settings(problem=4))
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    a=env.a;r=env.rad
    axs[0].plot(a[:,0]/3600,a[:,1],color=colors[0],lw=1.1,label='实测烘房温度')
    axs[0].axhline(env.tail[0],color=colors[1],ls='--',lw=1,label='后期延拓值')
    axs[0].set(xlabel='时间 / h',ylabel='温度 / °C',title='烘房升温边界');axs[0].legend(fontsize=8)
    axs[1].plot(a[:,0]/3600,a[:,2],color=colors[0],lw=1.1)
    axs[1].set(xlabel='时间 / h',ylabel='给定浓度 / kg/kg',title='烘房水分边界')
    tt=np.linspace(0,72*3600,500);ri,tau=env.r_fit
    axs[2].plot(r[:,0]/3600,r[:,1],'.',color=colors[0],ms=3,label='实测半径')
    axs[2].plot(tt/3600,100*env.radius(tt),color=colors[2],lw=1.2,label='保形插值 主模型')
    axs[2].plot(tt/3600,ri+(2-ri)*np.exp(-tt/tau),color=colors[1],ls='--',lw=1,label='指数拟合 对照')
    axs[2].set(xlabel='时间 / h',ylabel='半径 / cm',title='收缩边界');axs[2].legend(fontsize=8)
    fig.savefig(out/'01_输入与收缩.png');plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for col,q in enumerate([3,4]):
        ts=d[f'q{q}_time_s']/3600;x=d[f'q{q}_x'];C=d[f'q{q}_C_dry_basis'];T=d[f'q{q}_T_C'];R=d[f'q{q}_radius_m']
        for row,mat,title,unit in [(0,T,'温度','°C'),(1,C,'水分浓度','kg/kg')]:
            # Material coordinate, explicitly labelled. It never labels shrinking material as a fixed radius.
            im=axs[row,col].pcolormesh(ts,x,mat,shading='auto',cmap='viridis')
            axs[row,col].set(xlabel='时间 / h',ylabel='材料位置 r/R(t)',title=f'问题 {q}  {title}')
            if row==0:
                axs[row,col].set_xlim(0,4)
                axs[row,col].set_title(f'问题 {q}  温度 前4小时')
            fig.colorbar(im,ax=axs[row,col],label=unit,pad=.02)
    fig.savefig(out/'02_热湿时空分布.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for q,color in zip([3,4],colors):
        ts=d[f'q{q}_time_s']/3600;C=d[f'q{q}_C_dry_basis'];x=d[f'q{q}_x']
        faces=(x[1:]+x[:-1])/2;w=np.diff(np.r_[0,faces,1]**2)/2
        axs[0].plot(ts,C[0],color=color,label=f'问题 {q} 中心')
        axs[0].plot(ts,2*w@C,color=color,ls='--',label=f'问题 {q} 截面平均')
        axs[1].plot(ts,C[0],color=color,label=f'问题 {q} 中心')
        mt=v['stopping_comparison'][str(q)]['mean_threshold_time_h']
        mc=v['stopping_comparison'][str(q)]['center_C_at_mean_threshold']
        axs[1].scatter([mt],[mc],color=color,s=24,zorder=5)
    for ax in axs:
        ax.axhline(.15,color='#555',ls=':',label='要求上限 0.15')
        ax.set(xlabel='时间 / h',ylabel='水分浓度 / kg/kg');ax.legend(fontsize=8)
    axs[0].set_title('中心与平均含水率');axs[1].set(title='停机附近 放大',xlim=(30,60),ylim=(.12,.26))
    fig.savefig(out/'03_达标与提前停机风险.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    labels={'D -10%':'扩散系数 −10%','D +10%':'扩散系数 +10%','hm -20%':'传质系数 −20%','hm +20%':'传质系数 +20%',
            'tail T -1 C':'后期温度 −1°C','tail T +1 C':'后期温度 +1°C','tail C -0.01':'后期浓度 −0.01','tail C +0.01':'后期浓度 +0.01'}
    for ax,q in zip(axs,[3,4]):
        rows=[r for r in v['sensitivity'] if r['problem']==q and r['case'] in labels]
        yy=np.arange(len(rows));vals=[r['change_pct'] for r in rows]
        ax.barh(yy,vals,color=[colors[1] if val>0 else colors[0] for val in vals])
        ax.set_yticks(yy,[labels[r['case']] for r in rows]);ax.axvline(0,color='#777',lw=.7)
        ax.set(xlabel='烘干时长变化 / %',title=f'问题 {q} 单因素情景');ax.invert_yaxis()
    fig.savefig(out/'04_参数敏感性.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for q,color in zip([3,4],colors):
        rows=[r for r in v['mesh'] if r['problem']==q]
        n=np.array([r['n'] for r in rows]);t=np.array([r['critical_time_h'] for r in rows])*3600
        # Compare to finest computed solution; do not present it as exact analytical truth.
        axs[0].loglog(n[:-1],abs(t[:-1]-t[-1]),'o-',color=color,label=f'问题 {q}')
    axs[0].set(xlabel='径向区间数 N',ylabel='相对 N=1280 的时长差 / s',title='网格收敛');axs[0].legend()
    for q,color in zip([3,4],colors):
        ts=d[f'q{q}_time_s']/3600;C=d[f'q{q}_C_dry_basis'];x=d[f'q{q}_x'];R=d[f'q{q}_radius_m']
        for h,style in [(6,'--'),(24,':'),(v['crossing'][str(q)]['critical_h'],'-')]:
            k=np.argmin(abs(ts-h))
            axs[1].plot(x*R[k]*100,C[:,k],style,color=color,label=f'问题 {q} {h:.1f} h')
    axs[1].set(xlabel='实际半径 / cm',ylabel='水分浓度 / kg/kg',title='实际位置上的水分剖面');axs[1].legend(fontsize=7)
    fig.savefig(out/'05_收敛与空间剖面.png');plt.close(fig)
    print('5 figures written')

if __name__=='__main__':main()

