import json
from run_structure_test import ROOT

def main():
    s=json.loads((ROOT/'results/study.json').read_text(encoding='utf-8'))
    a=json.loads((ROOT/'results/conservation_checks.json').read_text(encoding='utf-8'))
    c=json.loads((ROOT/'results/reduction_scales.json').read_text(encoding='utf-8'))
    assert s['status']=='COMPUTED' and a['status']=='PASS'
    mainrows=['| 问题 | 全部表面汽化/h | 全部内部汽化/h | 时长变化 | 最大采样温差/°C |',
              '| --- | ---: | ---: | ---: | ---: |']
    results={}
    for q in [3,4]:
        fine={r['eta']:r for r in s['mesh'] if r['q']==q}
        coarse=next(r for r in s['cases'] if r['q']==q and r['eta']==1 and r['shape_power']==0)
        delta=100*(fine[1]['end_h']/fine[0]['end_h']-1)
        results[q]={'surface_h':fine[0]['end_h'],'internal_h':fine[1]['end_h'],'change_pct':delta,
                    'max_sampled_temperature_change_C':coarse['max_sampled_T_difference_C']}
        mainrows.append(f"| 第{q}问 | {fine[0]['end_h']:.5f} | {fine[1]['end_h']:.5f} | +{delta:.3f}% | {coarse['max_sampled_T_difference_C']:.3f} |")
    scenarios=['| 问题 | 内部比例η | 分布权重w(x) | 时间/h（N=160） | 相对表面汽化变化 |',
               '| --- | ---: | --- | ---: | ---: |']
    for r in s['cases']:
        w='1' if r['shape_power']==0 else f"x^{r['shape_power']:g}"
        scenarios.append(f"| 第{r['q']}问 | {r['eta']:.2f} | {w} | {r['end_h']:.5f} | {r['change_pct']:+.3f}% |")
    auditrows=['| 问题 | η | 水质量相对积分残差 | 能量积分残差/J |','| --- | ---: | ---: | ---: |']
    for r in a['cases']:auditrows.append(f"| 第{r['q']}问 | {r['eta']:.0f} | {r['water_relative_residual']:.3e} | {r['energy_residual_J']:.6f} |")
    meshrows=['| 问题 | η | N=160时间/h | N=320时间/h | 加密变化/s |','| --- | ---: | ---: | ---: | ---: |']
    for r in s['mesh']:meshrows.append(f"| 第{r['q']}问 | {r['eta']:.0f} | {r['reference_h']:.6f} | {r['end_h']:.6f} | {r['change_h']*3600:.4f} |")
    text=r'''# 毛细与内部蒸发对现有模型结构的影响：条件性约化论证与受控检验

## 1 结论应当怎样表述

本次实际得到的支持是：**在有效输运闭合和局部相态关系可用的条件下，可以保留以温度T、总干基含水率C为主要状态变量的热湿耦合结构；保持题给总水分扩散规律和现有边界时，所测试的汽化位置重分配对第三、四问终止时间的影响分别约1.53%和3.81%。**

这不等于“毛细输运和内部蒸发的物理作用都很弱”，也不等于“所有温湿度结果都可忽略它们”。毛细作用可能占总输运的重要部分，但可由有效系数表征；内部相变在总水守恒中相消，却仍必须进入能量方程。局部温度的最大采样变化超过10°C，因而不能把终点的相对稳定推广到第一、二问的温度精度或品质预测。

这里区分三种证据：方程约化中的严格代数恒等式、附带条件的储量上界，以及实际计算的有限组机理对照。它们没有被混称为真实药材的实验验证。

## 2 毛细输运可以保持原来的扩散型结构

设干固体骨架速度为vs，液态干基含水率为Cl，孔隙蒸气在单位总体积的储量为v。液态、蒸气相对于骨架的通量分别为Jl、Jv，内部相变源为Γ，则

$$\partial_t(\rho_d C_l)+\nabla\cdot(\rho_d C_l\mathbf v_s+\mathbf J_l)=-\Gamma,$$
$$\partial_t v+\nabla\cdot(v\mathbf v_s+\mathbf J_v)=+\Gamma.$$

定义总干基含水率C=Cl+v/ρd，并利用干固体连续方程，严格得到

$$\rho_d\frac{DC}{Dt}=-\nabla\cdot(\mathbf J_l+\mathbf J_v).$$

**Γ的消去不要求Γ本身很小。** 它只是内部水的相态转移，不是药材总水量损失。这个恒等式证明：内部蒸发的存在本身不迫使我们给“总水分方程”增加净消耗源项。

以气压空间近似均匀的Darcy毛细关系为例，pc=pg−pl，

$$\mathbf J_l=\rho_l\frac{Kk_{rl}}{\mu_l}\nabla p_c.$$

若局部保水闭合为pc=pc(Cl,T)，且∂pc/∂Cl<0，那么

$$\mathbf J_l=-\rho_dD_{cap}\nabla C_l+B_T\nabla T,$$
$$D_{cap}=-\frac{\rho_lKk_{rl}}{\mu_l\rho_d}\frac{\partial p_c}{\partial C_l}\ge0.$$

毛细项在等温或温度交叉输运可忽略时，恰好具有扩散通量形式。其作用大小体现于系数Dcap，而非一定增加一个新的动态状态变量。若蒸气也能通过局部代数关系表示，液、气通量可合并为一个有效总通量。

在本题既定有效模型层次，把题给D理解为已经概括微观输运作用的系数，是保留原结构的合理闭合选择；再无依据地叠加一份Darcy液流可能重复计数。**但题给数据没有独立标定K、保水关系及气相系数，因而不能反过来证明D中每种机制的真实比例，更不能证明毛细流量本身一定很小。**

若保留∂pc/∂T和温度引起的蒸气压梯度，一般还会得到温度交叉通量。此时仍可保持T、C两场的耦合结构，但不一定与目前省略交叉通量的每个系数完全相同；交叉项大小需要另行估计。

## 3 内部蒸发为什么不一定需要独立动态气相方程

在局部相平衡近似下

$$p_v=a_w(C_l,T)p_{sat}(T),\qquad
v=\epsilon_g\frac{a_w(C_l,T)p_{sat}(T)}{R_vT_K}.$$

于是C=Cl+v(Cl,T)/ρd是局部代数关系。若其对Cl的导数为正，则可反解Cl=G(C,T)，再把它代入液、气通量和储能方程。内部相态不再引入第三个独立动态未知量，而体现于有效储量、有效通量、潜热输运与温度耦合系数。

这是一条**附带相态闭合条件的结构约化**，不是凭经验删去气相方程。[COMSOL官方理论](https://doc.comsol.com/6.3/doc/com.comsol.help.heat/heat_ug_theory.07.076.html)明确区分局部平衡下的单一总含水量描述与非平衡下分别求液、气两相的描述。

若相间交换很慢，或蒸气压力与局部平衡值相差明显，代数约化不能直接使用，应保留气相状态或准稳态气相约束。[非平衡理论](https://doc.comsol.com/6.4/doc/com.comsol.help.heat/heat_ug_theory.07.082.html)要求分别处理两相守恒与相间源。

不能仅比较1/ke与几十小时烘干时长。气相扩散也可能很快，局部平衡还应检查keℓ²/Dg及实际偏离；ℓ是局部传递长度，不必等于整根半径。题目缺少ke、气相参数和保水曲线，因此尚不能给本药材无条件的局部平衡证明。

## 4 蒸气库存可以给出多小的界限

若气孔体积分数≤1、蒸气不超饱和、材料T≤附件最高温50.246°C、干密度不低于初始值，则

$$\frac{M_v\text{的局部密度}}{M_w\text{的局部密度}}\le
\frac{\rho_{v,sat}(50.246)}{\rho_{d,0}C}.$$

利用已核验的IAPWS饱和物性和理想蒸气密度，ρv,sat约0.083781 kg/m³。在C=0.15处，第三、四问上界分别约**0.2031%、0.2004%**；在更干的C=0.05处分别约0.6092%、0.6012%。这组界限不需要猜一个具体孔隙率，第四问若收缩提高ρd，界限会进一步减小。

它支持“气相水的独立储量相对液态总水量较小”，但不支持“蒸气通量小”“蒸发热小”或“局部相平衡自动成立”。少量蒸气可以不断生成、排出，累计输运很多水。

附加假设aw对温度不显著变化时，Lv·∂v/∂T造成的热储量修正上界，在C=0.15处约为第三问显热容量的1.62%、第四问的1.36%。这仅是该温度依赖假设下的估计，未知的吸附热与aw温度依赖不能据此略去。

![储量与一致性](figures/02_库存界限与参数一致性.png)

## 5 为什么此前约105.7小时的双相试算不能否定现有结构

那组探索计算不仅增加相态，还使用了未标定的渗透率、保水曲线、气膜传质和气相扩散参数，并替换了题给有效D与水分边界。因此105.7 h与60.43 h之差，不是单独两个机制的边际影响。

本次进一步核对：把那组气相参数代入局部平衡通量，所得蒸气扩散贡献在部分低含水率状态已超过题给总D。此时再添加非负液态扩散系数，也不能匹配题给D。这直接表明其参数组合与本题有效闭合没有匹配；在完整采样范围C=0.05—2.55内的比值曲线见图。不是说实际药材参数就是这些值。

还检查了该探索解：其实际气相水量占总水量的最大采样比例约0.141%，但归一化平衡分压偏离|pv−aw psat|/psat最大约15.64%，体积平均偏离的最大值约5.17%。这正是“小气相库存不能自动证明相平衡”的反例，不能隐藏这项不利证据。

## 6 受控计算：只重新分配潜热位置

为避免同时改变多个参数，本次保持：题给D(T,C)、h、hm、原有效Ceq边界、初始干质量、比热和导热关系，以及第三问固定尺寸、第四问实测半径。每千克外排水对应的总潜热始终为Lv=2.38 MJ/kg。

这里沿用当前有效边界，因此此前发现的真实空气解释与露点冲突仍未解决。计算用于检验这套有效模型内部的终点敏感性，不把它称为真实多相热力学验证。

在忽略气相储量的相间分配对照中，设内部蒸气通量

$$\mathbf J_v=\eta w(x)\mathbf J_{total},\quad
\mathbf J_l=\mathbf J_{total}-\mathbf J_v,\quad
\Gamma=\nabla\cdot\mathbf J_v.$$

η∈[0,1]，主对照w=1；另外测试w=x、w=x²。它们是人为规定的相态分配情景，不是由实测气压推导出的物理比例，也未覆盖所有可能的相变位置。

离散时只在内部面总能量通量中增加LvηwFtotal；外边界总汽化热仍为LvFsurface。η=0对应全部表面汽化；η=1、w=1对应体内液态库存变化通过汽化后以蒸气输送，边界不再重复增加一份同量潜热。

**固定的是总扩散定律和每公斤水的潜热，而非强制两种解的排水历程相同。** 重新分配热量会改变T，再经D(T,C)改变实际水通量；不同终点的累积排水量也不被人为锁死。潜热不因改变位置而被删减或重复计算。

## 7 计算证据

下表时长使用N=320，局部温差来自N=160在多个时刻、全部该网格节点的比较。两项对照保持同一物性、几何与边界定义。

__MAIN__

从全部表面到全部内部的极端分配，时长分别增加约0.925 h和2.153 h；变化为1.53%和3.81%。在本次测试范围中，终止时间对汽化位置的敏感性低于此前从“不计潜热”到“计入潜热”的5.15%、10.55%变化。因此可以论证“潜热总效应需要保留；对停机时长而言，进一步细分其空间位置的影响较小”。

但局部温场改变超过10°C，不能写“场解误差均小于4%”，也不能据此保证问题1、2的温度结果准确。以低阶二场模型作停机分析，与用同一简化精确预测早期温度，是不同要求。

![位置敏感性](figures/01_相变位置敏感性.png)

完整情景表如下。此表展示有限组实际计算，不是对所有η、任意w或所有材料参数的数学最坏界。

__SCENARIOS__

## 8 数值与守恒复核

__MESH__

四组端点的末次网格时间变化均小于0.63 s，远小于小时级的相变位置影响。早期薄层的空间分布尚存在插值与离散误差，不能用时长收敛宣称每处温度完全准确。

独立Gauss积分重新核对水库存和边界总能量，结果如下。内部面能量通量相消，检查同时覆盖内部潜热输运项，不是只检查外边界公式。

__AUDIT__

瞬时质量残差不超过约10⁻²¹ kg/s，瞬时能量残差不超过约5×10⁻¹⁵ W；独立水质量相对积分残差小于9×10⁻⁹，能量积分残差绝对值小于0.005 J。这说明本次对照没有通过人为丢水或丢热获得小差异。

## 9 建议用于论文的论证表述

“本文以温度和总干基含水率为宏观状态变量。液态水与水蒸气守恒方程相加后，内部相变源项严格抵消；在局部保水与相平衡近似下，毛细输运及蒸气迁移可通过有效总水分通量表征，从而保留热湿二场耦合结构。相变的能量效应仍予以保留。为检验汽化空间位置的敏感性，在保持题给总水分扩散规律、边界定义和单位排水潜热不变的条件下，比较了表面汽化至内部汽化的多组分配情景；第三、四问终止时间最大观测变化分别约为1.53%和3.81%。该结果支持将相变空间分布视为停机时长分析中的次要修正，但不意味着早期局部温度变化可忽略。局部相平衡、保水关系和有效边界解释仍属于需要材料数据支持的假设。”

引用这段时应同时保留假设、情景范围、验证表和温场限制。当前不能写“已证明毛细与内部蒸发在真实药材中影响均小于4%”，因为没有对应材料参数与实验数据，也没有证明交叉输运和非平衡效应在所有状态下都小。

## 10 当前结构是否值得保留

对于本题数据约束下的T—C有效建模，保留原有结构、把毛细机制解释为有效输运组成，并将内部汽化位置作为能量分配敏感性，是有依据的研究路线。无需仅因为存在微观机制就机械增加大量不可识别参数。

若研究目标转为内部温度、组织损伤、凝结位置或真实工程预测，则目前论证不充分，需要解吸关系和气相输运/相变速率数据，必要时保留独立蒸气场。结构简洁的合理性与微观机制不重要是两回事。

## 11 数据与复算

code/redistribution_model.py保留总水输运闭合并重分配内部潜热；run_structure_test.py计算两问14组位置情景和网格端点；finish_structure_checks.py进行独立收支核验；check_reduction_scales.py核查库存上界及此前双相试算的一致性；make_figures.py、make_report.py生成图和本说明。

results/study.json保存全精度时长与差异，conservation_checks.json保存收支，reduction_scales.json保存带假设的界限和反例检查，history_*.json保存中心/表面/平均量的共同时间段对照。两幅图均由实际数据生成。

```text
python code/run_structure_test.py --cache <独立缓存目录>
python code/finish_structure_checks.py --cache <同一缓存目录>
python code/check_reduction_scales.py --two-phase-results <毛细与内部蒸发分析/results目录>
python code/make_figures.py
python code/make_report.py
```

尺度检查需要此前双相分析目录中的fields_coupled_fine.npz；结果JSON已随本专题提供，不重新计算也可审查。程序在本地Python运行，未称为MATLAB、COMSOL或实验复核。
'''
    for key,value in [('MAIN',mainrows),('SCENARIOS',scenarios),('MESH',meshrows),('AUDIT',auditrows)]:text=text.replace('__'+key+'__','\n'.join(value))
    assert '__MAIN__' not in text
    (ROOT/'结构保留论证与敏感性检验.md').write_text(text,encoding='utf-8')
    (ROOT/'results/conclusions.json').write_text(json.dumps({'status':'PASS','results':results,
         'conclusion':'Conditional support for retaining T-C macrostructure and modest stop-time sensitivity in tested latent-location family; not negligible physical mechanisms or proven local equilibrium.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
