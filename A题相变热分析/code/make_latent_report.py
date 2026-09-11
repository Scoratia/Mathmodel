from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})


def main():
    s=json.loads((ROOT/'results/study.json').read_text(encoding='utf-8'))
    q1=json.loads((ROOT/'results/question1_comparison.json').read_text(encoding='utf-8'))
    assert s['numerical_status']=='PASS'
    compare=[];energy=[];thermal=[];intervals=[];cases=[];variable=[];mesh=[]
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,q in zip(axs,[3,4]):
        rows=np.asarray(json.loads((ROOT/f'results/history_q{q}.json').read_text(encoding='utf-8'))['rows'])
        cut=rows[:,0]<=12*3600
        t=rows[cut,0]/3600
        ax.plot(t,rows[cut,3],color='#475569',ls='--',label='烘房温度')
        ax.plot(t,rows[cut,1],color='#1D4ED8',label='药材中心（计相变热）')
        ax.plot(t,rows[cut,2],color='#B45309',label='药材表面（计相变热）')
        ax.plot(t,rows[cut,9],color='#B91C1C',ls=':',label='条件性空气露点')
        ax.set(title=f'第{q}问：表面相变模型',xlabel='时间 / h',ylabel='温度 / °C')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
        c=s['crossing'][str(q)]
        compare.append(f"| 第{q}问 | {c['without_latent_h']:.4f} | {c['with_latent_h']:.4f} | {c['change_h']:.4f} | {c['change_pct']:.2f}% |")
        d=s['validation'][str(q)]['diagnostics'];e=s['validation'][str(q)]['energy'][-1]
        energy.append(f"| 第{q}问 | {e['convective_heat_input_J']/1000:.3f} | {e['latent_energy_J']/1000:.3f} | {e['outward_water_sensible_energy_J']/1000:.3f} | {e['stored_sensible_energy_change_J']/1000:.3f} | {e['energy_residual_J']:.4f} |")
        for r in d['temperature_comparison']:
            thermal.append(f"| 第{q}问 | {r['time_h']:.1f} | {r['baseline_center_T_C']:.3f} | {r['center_T_C']:.3f} | {r['baseline_surface_T_C']:.3f} | {r['surface_T_C']:.3f} |")
        interval='；'.join(f"{lo/3600:.3f}–{hi/3600:.3f} h" for lo,hi in d['conditional_evaporation_below_air_dewpoint_intervals_s']) or '未检出'
        intervals.append(f"| 第{q}问 | {d['surface_min_C']:.3f} | {d['surface_min_time_h']:.3f} | {interval} | {d['conditional_max_required_water_activity']:.3f} |")
    fig.savefig(ROOT/'figures/蒸发降温与露点检查.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.2),layout='constrained')
    for q,color in [(3,'#1D4ED8'),(4,'#B45309')]:
        a=[r for r in s['cases'] if r['problem']==q and r['law']=='constant 2.38 MJ/kg']
        ax.plot([r['fraction'] for r in a],[r['time_h'] for r in a],'-o',label=f'第{q}问',color=color)
    ax.set(xlabel='相变热耦合强度 λ（人为对照参数）',ylabel='临界烘干时间 / h',title='保持同一有效水分边界的相变热对照')
    ax.grid(alpha=.2);ax.legend();fig.savefig(ROOT/'figures/相变热对烘干时长的影响.png',bbox_inches='tight');plt.close(fig)
    for r in s['cases']:
        if r['law']=='constant 2.38 MJ/kg':
            cases.append(f"| 第{r['problem']}问 | {r['fraction']:.2f} | {r['time_h']:.6f} | {r['change_pct']:.3f}% |")
        else:variable.append(f"| 第{r['problem']}问 | {r['constant_same_grid_h']:.6f} | {r['time_h']:.6f} | {(r['time_h']-r['constant_same_grid_h'])*60:+.3f} |")
    for r in s['mesh']:
        mesh.append(f"| 第{r['problem']}问 | {r['n']} | {r['time_h']:.8f} | {r.get('delta_time_s',float('nan')):.6f} | {r.get('max_T_delta_C',float('nan')):.3e} | {r.get('max_C_delta',float('nan')):.3e} |")
    humidity=s['conditional_humidity_audit'];verification=s['thermophysical_verification']
    text=r'''# 平衡含水率解释与相变热影响分析

本轮在干物质守恒修正版上实际加入了表面汽化热，完成相变热耦合强度对照、四档网格、两种求解器复核、IAPWS温变汽化潜热对照和独立质量/能量核算。原四份基准表未被这些条件性情景替换。

本轮继承上一轮的密度取舍：初始密度来自题给经验公式，后续真实湿密度由干物质、水量和当前体积确定，不再严格保留经验ρ(C)的整个变化函数。这项取舍没有因加入相变热而消失。

## 1 “平衡湿度”具体指什么

此前的说法不够精确：这里应称“药材平衡含水率”Ceq。它是药材在给定温度和空气湿度下，经过足够长时间，不再净吸水或净失水时的干基含水率。它不是空气相对湿度，也不是每千克干空气所含的水蒸气质量。

- 药材C：kg水/kg干药材。
- 空气含湿量W：若按常用湿空气定义，是kg水蒸气/kg干空气。
- 空气相对湿度RH：水蒸气分压与该温度饱和蒸气压之比，无量纲。

两种kg/kg分母不同，不能直接当作同一变量相减。原题明确药材浓度是干基含水率，但没有进一步解释烘房“水分浓度”的换算。原模型把烘房数值当作已经折算的有效Ceq，这是一项需要说明的简化假设。

**条件性换算**：假设附件空气浓度就是W，气压为101325 Pa，并使用理想湿空气关系

$$p_v=\frac{PW}{0.621945+W},\qquad RH=\frac{p_v}{p_{sat}(T_a)}.$$

后期Ta=__TA__°C、W=__W__对应RH约__RH__%，露点约__DEW__°C。这不能推出药材Ceq=0.05；还缺少药材的解吸等温线，通常表示为Ceq=f(T,RH)，或水活度aw=g(C,T)。湿空气定义与换算依据：[ASHRAE湿空气学](https://handbook.ashrae.org/Handbooks/F25/SI/F25_Ch01/F25_Ch01_si.aspx)。

## 2 相变热如何加入

采用“所有净排出水都在表面汽化”的简化对照。在守恒修正版中，实际表面水质量流率为

$$\dot m_w=A_s\rho_d h_m(C_s-C_{eq})\quad[\mathrm{kg/s}],$$
$$P_{evap}=\lambda L_v\dot m_w\quad[\mathrm W].$$

λ=0是原显热基准，λ=1为全部表面汽化热耦合，0.25、0.5、0.75用于检查影响随耦合强度的变化；λ不是实测蒸发比例，也不是置信概率。真实干燥不能解释为其余水无须汽化热，它只是模型对照参数。

表面能量条件为

$$-k\partial_nT=h(T_s-T_a)+\lambda L_vj_w.$$

内部仍用守恒修正版的水分显热通量与干固体/水储能。相变热只加入一次，不再在整个体积中重复扣除Lv·dC/dt。主对照Lv=2.38×10^6 J/kg；另外按IAPWS饱和水物性计算Lv(Ts)，比较固定潜热的近似影响。

物理机制：汽化消耗热量，使药材低于烘房温度；题给D随温度下降而减小，后续水分迁移变慢，延长达标时间。

## 3 实际计算影响

下表两列共用干物质守恒密度、有效水分边界、半径、传热传质系数，仅改变是否加入汽化热。临界条件为全场最大C=0.15；严格低于阈值取临界时刻之后。

| 问题 | 不计显式相变热/h | 计全部表面相变热/h | 延长/h | 增幅 |
| --- | ---: | ---: | ---: | ---: |
__COMPARE__

该对照说明相变热对当前模型不可忽略，不能直接据此认定右列为真实药材的准确时长。它仍依赖有效边界和表面相变位置假设，且存在下述热力学解释冲突。

![相变热对照](figures/相变热对烘干时长的影响.png)

| 问题 | λ | 同网格N=160临界时间/h | 相对λ=0变化 |
| --- | ---: | ---: | ---: |
__CASES__

温度对照如下，均为实际计算值；全场更详细的时间序列见results/history_q3.json与history_q4.json。

| 问题 | 时间/h | 无相变热中心/°C | 有相变热中心/°C | 无相变热表面/°C | 有相变热表面/°C |
| --- | ---: | ---: | ---: | ---: | ---: |
__THERMAL__

第一问另作了30分钟对照：中心温度由__Q1BC__°C变为__Q1LC__°C，表面由__Q1BS__°C变为__Q1LS__°C。由于附录2的D只依赖C而不依赖T，两次含水率解的最大差仅__Q1CD__，符合其方程结构。这些大幅降温同样只能作为旧有效边界下的机理对照，不能解释为真实药材在当前湿空气中就会降到这些温度。

## 4 发现的物理一致性问题：不能只加潜热就定稿

在“附件浓度是真实空气含湿量”的解释下，常规界面传质的驱动方向应由aw(C_s,T_s)psat(T_s)−pv决定。对于常规稳定液态/吸附水界面，aw不超过1。若T_s低于空气露点，则psat(T_s)<pv，即使按纯水表面aw=1，也不具备净向外蒸发的正驱动力。

本次发现，现有有效浓度边界在部分时段仍规定向外排水，同时潜热把表面温度压到上述露点以下。因此它在此空气解释下有符号一致性冲突。这不是浮点误差，也不是需要更细网格才能解决的问题。

| 问题 | 采样最低表温/°C | 最低值时间/h | 向外排水且低于条件性露点的时段 | 最大pv/psat(Ts) |
| --- | ---: | ---: | --- | ---: |
__INTERVALS__

最后一列大于1意味着要维持零净驱动力都需要超出通常范围的水活度，更不用说正蒸发通量。时段先按30s/60s采样定位，再用根求解细化交点；最低温度仅是采样最低值，不声称求得连续时间的严格全局最小值。

![温度与露点](figures/蒸发降温与露点检查.png)

这项诊断始终附带标准气压、W的真实空气解释、常规表面传质等假设；如果题面本就定义了有效折算浓度，就不能把该露点条件当成对题设模型的无条件否定。它说明真正工程模型需要同时处理湿度驱动力和相变热。

建议的完整边界形式为j_w=β_p[aw(C_s,T_s)psat(T_s)−pv]，并将同一个j_w用于质量和潜热。当前没有药材aw关系，也没有明确的气膜系数β_p；题给hm不能不经单位和定义换算就直接代入压差形式。不能通过猜药材种类、随意套其他材料参数来得到“唯一精确答案”。COMSOL也用水活度与饱和蒸气条件建立相变耦合：[官方建模说明](https://www.comsol.com/blogs/how-to-model-heat-and-moisture-transport-in-porous-media-with-comsol)。

## 5 能量与质量核算

使用独立时间采样积分核对

$$\Delta H=Q_{conv,in}-Q_{latent}-H_{water,out}.$$

表中Qconv仅是对这根模型药材的对流供热积分，不能当作整间烘房或整台设备耗电；没有包括热风发生、排风、设备和围护结构损失。

| 问题 | 对流供热/kJ | 汽化热/kJ | 流出水显热/kJ | 储存显热变化/kJ | 收支残差/J |
| --- | ---: | ---: | ---: | ---: | ---: |
__ENERGY__

详细数据同时保留10s、5s积分结果；在4h环境切换点分段。干物质量也逐时按保存的密度和体积重建验证，继续保持恒定。

## 6 数值与水物性验证

| 问题 | 网格N | 临界时间/h | 相邻网格差/s | 最大温度差/°C | 最大含水率差 |
| --- | ---: | ---: | ---: | ---: | ---: |
__MESH__

时间求解器BDF与Radau在同一N=160下分别复核第三、四问。网格表中的分布差覆盖早期、3h、4h及后续多个时点，不冒称检查了所有可能的时空位置。完整数值证据见results/study.json。

水的饱和蒸气压、两相密度及温变汽化潜热按照IAPWS SR1-86(1992)计算，以Clapeyron关系Lv=T(dp_sat/dT)(1/ρv−1/ρl)求两相焓差，并与发布文件第7页三个温度的检查表对照通过。Lv(28°C)=__LV28__ MJ/kg，Lv(50°C)=__LV50__ MJ/kg。[IAPWS官方文件](https://iapws.org/technical-guidance/release/Supp-sat)。

| 问题 | 固定Lv，同网格/h | IAPWS Lv(Ts)，同网格/h | 变化/min |
| --- | ---: | ---: | ---: |
__VARIABLE__

这只是纯水汽化潜热的温度依赖，未包含药材结合水解吸热。题给有效比热仍沿用守恒修正版的解释，不能将这项物性改进等同于完整多相热力学模型。

## 7 复算与结果定位

code/study_latent.py实际运行本次计算，water_thermo.py为经表值核验的水物性实现，latent_model.py提供温变潜热对照。运行环境为本机Python，不是COMSOL/CST/MATLAB。本目录包含输入副本，默认不会覆盖原模型文件。

```text
python code/study_latent.py --cache <温变潜热临时缓存> --baseline-cache <模型临时缓存>
python code/check_first_question.py --cache <模型临时缓存>
python code/make_latent_report.py
```

模型临时缓存可复用上一轮缓存，也可为空目录。第一次运行会计算全部对照。本次交付为相变热影响与边界诊断，不输出论文，也不把条件性结果替换为题设唯一答案。
'''
    for key,rows in [('COMPARE',compare),('CASES',cases),('THERMAL',thermal),('INTERVALS',intervals),('ENERGY',energy),('MESH',mesh),('VARIABLE',variable)]:text=text.replace('__'+key+'__','\n'.join(rows))
    for key,val in [('TA',f"{humidity['late_air_T_C']:.5f}"),('W',f"{humidity['late_air_W']:.8f}"),
                    ('RH',f"{humidity['late_air_RH']*100:.2f}"),('DEW',f"{humidity['late_air_dewpoint_C']:.2f}"),
                    ('LV28',f"{verification['latent_at_28C_J_kg']/1e6:.5f}"),('LV50',f"{verification['latent_at_50C_J_kg']/1e6:.5f}")]:text=text.replace('__'+key+'__',val)
    for key,col in [('Q1BC','without_latent_center_T_C'),('Q1LC','with_latent_center_T_C'),
                    ('Q1BS','without_latent_surface_T_C'),('Q1LS','with_latent_surface_T_C')]:text=text.replace('__'+key+'__',f"{q1[col]:.3f}")
    text=text.replace('__Q1CD__',f"{q1['max_C_difference']:.3e}")
    text=text.replace('nan','—')
    (ROOT/'平衡含水率与相变热分析.md').write_text(text,encoding='utf-8')
    print(json.dumps(s['crossing'],ensure_ascii=False))


if __name__=='__main__':main()
