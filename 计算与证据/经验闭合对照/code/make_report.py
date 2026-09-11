"""Create the technical handover note from computed evidence, not a competition paper."""
import json
from pathlib import Path
from model import ROOT


def table(headers,rows,digits=4):
    def fmt(v):
        if v is None:return '—'
        if isinstance(v,(float,int)):return f'{v:.{digits}f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])+'\n'


def main():
    v=json.loads((ROOT/'results/validation.json').read_text(encoding='utf-8'))
    geo=json.loads((ROOT/'results/geometry_validation.json').read_text(encoding='utf-8'))
    c=v['crossing'];audit=v['input_audit'];parts=[]
    add=parts.append
    add('# A题药材烘干的建模结果与验证说明\n')
    add('本说明交付可复现的模型、四问数值结果、误差证据和可用于后续论证的改进点，不采用竞赛论文格式。输入为用户提供的 CUMCM2026Problems.zip 内 A 题及附件。\n')
    add(f"**选定 A 题。** 在本文明确的边界与物理简化下，第 3 问临界烘干时间为 **{c['3']['critical_h']:.4f} h**，第 4 问为 **{c['4']['critical_h']:.4f} h**。按整分钟操作，分别取 **{c['3']['first_full_minute_h']:.4f} h** 与 **{c['4']['first_full_minute_h']:.4f} h**，此时未舍入的最大含水率严格低于 0.15。数值验证已经完成；附件没有药材内部温度、含水率或称重实测，因而不能宣称已经完成真实干燥实验验证。\n")
    add('## 1 选题与数据条件\n')
    add(table(['比较项','A 题','B 题'],[
        ['主问题','非线性热湿扩散与收缩','几何定位、搜索与路径决策'],
        ['验证条件','边界和半径数据齐备，可做解析与数值基准','第 3、4 问各需联网模拟器三次正式测试及日志'],
        ['适合本次目标的理由','四问递进明确，能在本地完成模型与验证闭环','还依赖登录、联网测试窗口和正式测试机会'],
        ['决定','优先深入完成 A 题','不同时分散求解 B 题']]))
    add('题面公式已逐项核对原 PDF。所有几何量在求解中换算为 m，时间为 s，状态温度保存为 °C，仅扩散公式使用 T+273.15。所有指数中的 C 位于分母。不能把 e^(−0.45/C) 写成 e^(−0.45C)。第 2、3 问从 t=0 起统一使用附录 3，不把第 1 问的 1800 s 结果作为其接续初值。\n')
    add(table(['数据项目','检查结果'],[
        ['烘房边界',f"{audit['ambient_rows']} 个时点，0—14400 s，间隔 60 s"],
        ['半径数据',f"{audit['radius_rows']} 个时点，0—259200 s，间隔 1800 s"],
        ['后期温度延拓',f"最后 1 h 均值 {audit['tail_mean'][0]:.8f} °C"],
        ['后期浓度延拓',f"最后 1 h 均值 {audit['tail_mean'][1]:.8f} kg/kg"],
        ['半径单调处理',f"保形插值前检查单调性，实际调整 {audit['radius_adjusted_points']} 个点"],
        ['数据追溯','data/inputs.json 保留全部数值；source_manifest.json 保存原 Excel 的 SHA-256']]))
    add('0—4 h 的烘房边界采用逐段线性插值，不抹去已给出的温度波动。4 h 后缺少观测，按恒温干燥阶段取最后 1 h 的均值延拓。在 4 h 处允许由末条观测切换到平台估计的微小边界跳变，积分器分段重启；不是把 4 h 认定为由数据识别出的唯一工艺切换时刻。另以最后一条数据延拓进行敏感性对照。半径采用单调 PCHIP；本题达标发生于 72 h 数据区间内，不依赖半径外推。\n')
    add('![输入与收缩](figures/01_输入与收缩.png)\n')
    add('## 2 主模型和必须说明的假设\n')
    add('### 2.1 固定圆柱\n')
    add('取轴向中截面的径向坐标 r∈[0,R]，R=0.02 m，圆柱长度 L=0.25 m。主模型假定环境沿侧面均匀、材料有效性质各向同性，先忽略端面影响，再用二维有限圆柱检验这项近似。C 为 kg 水/kg 干物质。采用题设经验系数的有效扩散闭合：\n')
    add(r'''
$$
\rho(C)c_p(C)\frac{\partial T}{\partial t}
=\frac1r\frac{\partial}{\partial r}\left(rk(C)\frac{\partial T}{\partial r}\right),
\qquad
\frac{\partial C}{\partial t}
=\frac1r\frac{\partial}{\partial r}\left(rD(T,C)\frac{\partial C}{\partial r}\right).
$$

$$
T(r,0)=28,\quad C(r,0)=2.55,\qquad
T_r(0,t)=C_r(0,t)=0.
$$

$$
-kT_r(R,t)=h[T(R,t)-T_a(t)],\qquad
-DC_r(R,t)=h_m[C(R,t)-C_a(t)],
$$

其中 h=25 W/(m²·K)，h_m=8×10⁻⁷ m/s。这里不能把变量系数扩散项写成 D∇²C 而漏掉系数随空间变化的贡献。
''')
    add(table(['问题','ρ','cₚ','k','D / m²·s⁻¹'],[
        ['1','820','2600','0.36','7×10⁻⁹ exp(−0.89/C)'],
        ['2、3','650+128C','1450+2736C/(C+1)','0.21+0.38C/(C+1)','2.4×10⁻³ exp(−0.45/C−3850/(T+273.15))'],
        ['4','760+90C','1850+2150C/(C+1)','0.12+0.20C/(C+1)','4.2×10⁻⁴ exp(−0.30/C−3850/(T+273.15))']]))
    add('ρ、cₚ、k 的单位分别为 kg/m³、J/(kg·K)、W/(m·K)。热量与水分通过性质和扩散系数双向耦合。主模型未另加蒸发潜热或体积相变源项，因为题目没有给出相变位置、吸附关系和含水率到水蒸气通量的转换。该选择是题设有效参数下的简化，不代表真实烘干没有蒸发冷却。第 6 节给出潜热情景对照。\n')
    add('**边界含水率的闭合。** 烘房给出的 kg/kg 与药材的干基 kg/kg，在真实物理中可能分别以干空气、干固体为基准，不能仅凭单位相同就视为相同热力学变量。本主模型按题面提供的“浓度”和传质系数，将 Cₐ 解释为已折算的有效环境驱动值，即有效分配系数取 1。若 Cₐ 实际只是空气湿度，则还需吸附等温线或平衡含水率关系，单凭当前附件不能唯一标定真实界面传质。这个识别限制必须保留。\n')
    add('### 2.2 收缩坐标\n')
    add('第 4 问使用实测 R(t)。在长度不变、横截面均匀径向收缩假设下，材料速度 vᵣ=(Ṙ/R)r。令 x=r/R(t)，u(x,t)=C(r,t)，θ(x,t)=T(r,t)。干基含水率随材料的变化满足 DₜC=扩散项，因此：\n')
    add(r'''
$$
\frac{\partial u}{\partial t}
=\frac{1}{R(t)^2x}\frac{\partial}{\partial x}
\left(xD(\theta,u)\frac{\partial u}{\partial x}\right),
$$

$$
\rho(u)c_p(u)\frac{\partial\theta}{\partial t}
=\frac{1}{R(t)^2x}\frac{\partial}{\partial x}
\left(xk(u)\frac{\partial\theta}{\partial x}\right).
$$

$$
-\frac{D}{R(t)}u_x(1,t)=h_m[u(1,t)-C_a(t)],\qquad
-\frac{k}{R(t)}\theta_x(1,t)=h[\theta(1,t)-T_a(t)].
$$
''')
    add('材料对流项与坐标变换项恰好抵消。不能再重复加入 xṘ/R·uₓ；也不能对干基 C 无依据地增加 −2ṘC/R 的“浓缩项”。后者适用于某些单位现体积的浓度表达，不适用于当前干基定义。取均匀参考干固体分布时，ρ_d(t)R(t)² 保持常数，水分收支由下式检查。题设 ρ(C) 作为热容量计算中的有效性质使用，不再同时强制它等于几何收缩和干固体质量守恒导出的局部湿基密度；这属于有效模型的适用范围。\n')
    add(r'''
$$
\bar C(t)=2\int_0^1xu(x,t)\,dx,\qquad
\frac{d\bar C}{dt}=-\frac{2h_m}{R(t)}[u(1,t)-C_a(t)].
$$
''')
    add('### 2.3 终点定义\n')
    add(r'定义 $t_*=\inf\{t:\max_{0\le x\le1}u(x,t)\le0.15\}$。程序对全体网格点取最大值触发事件，不直接假定中心就是最湿点。随后检查径向单调性和二维最大值位置。临界时刻对应等号；需要严格低于时，取紧接临界时刻后的整分钟。Excel 中按要求显示四位小数，显示成 0.1500 不等于未舍入数值恰好等于 0.15。'+'\n')
    add('## 3 数值方法与可复现性\n')
    add('采用包含中心和表面的有限体积节点 xᵢ=i/N。控制体面在相邻节点中点，权重 wᵢ=(x²ᵢ₊½−x²ᵢ₋½)/2。内部面通量被相邻控制体以相反符号共享，中心面通量为零，表面直接施加 Robin 通量，因此不需要在 r=0 上计算 1/r。热导率用谐均值。水分通量使用三点 Gauss 积分得到面有效扩散系数：\n')
    add(r'''
$$
D_{i+1/2}=\int_0^1D\big(T_i+s(T_{i+1}-T_i),\ C_i+s(C_{i+1}-C_i)\big)\,ds,
\quad
F^C_{i+1/2}=x_{i+1/2}D_{i+1/2}\frac{C_{i+1}-C_i}{\Delta x}.
$$

$$
\dot C_i=\frac{F^C_{i+1/2}-F^C_{i-1/2}}{R^2w_i},\quad
F^C_{-1/2}=0,\quad F^C_{N+1/2}=-Rh_m(C_N-C_a).
$$
''')
    add('恒温时，面通量对应 Kirchhoff 积分势 Φ(C)=∫D(C)dC 的差分，能更稳妥地处理干表层低扩散率造成的阻力。它是既有数学方法在本题中的组合使用，不宣称为新的扩散理论。\n')
    add('时间积分采用稀疏隐式 BDF，rtol=2×10⁻⁸、atol=2×10⁻¹⁰；0—4 h 最大内部步长 30 s，后期 600 s。输出间隔与内部步长不同：每秒或每分钟的数据来自积分器的连续插值。最终结果采用 N=1280，内部径向步长初始为 0.0015625 cm，明显小于交付表的 0.1 cm。检查网格为 80、160、320、640、1280。还用 Radau 独立时间积分器交叉核验。\n')
    add('完整实现见 code/model.py；code/run_study.py 一次运行完成四问、网格检验、解析基准、敏感性分析和数据导出。默认完整运行，--quick 只用于环境冒烟检查，不能代替最终结果。已保存的 results/validation.json 是本次实际运行证据，profiles_full_precision.npz 保存未舍入的代表性时空场。\n')
    add('## 4 四问计算结果\n')
    add(table(['问题','临界时间 / h','整分钟操作时间 / h','该分钟最大 C'],[
        [str(q),c[str(q)]['critical_h'],c[str(q)]['first_full_minute_h'],f"{c[str(q)]['max_C_first_full_minute']:.8f}"] for q in [3,4]]))
    for key,title,unit in [('Q1_T','问题 1 温度','s'),('Q1_C','问题 1 水分浓度','s'),('Q2_T','问题 2 温度','h'),('Q2_C','问题 2 水分浓度','h'),('Q3_C','问题 3 水分浓度','h')]:
        add('### '+title+'\n')
        add(table([f'时间 / {unit}','0 cm','0.5 cm','1.0 cm','1.5 cm','2.0 cm'],v['tables'][key]))
    add('### 问题 4 水分浓度\n')
    add(table(['时间 / h','0 cm','0.5 cm','1.0 cm','表面 C','半径 / cm'],[[r[0],r[1],r[2],r[3],r[-2],r[-1]] for r in v['tables']['Q4_C']]))
    add('第 3、4 问末行采用实际整分钟操作时刻。第 4 问早期完整结果仍包含 0—2 cm 的固定位置列，r>R(t) 的位置为空，绝不填 0，也不把这些列当作归一化坐标。另给“药材表面”和“当前半径/cm”两列，后者位于模板主结果之后，用于识别随时间改变的表面位置。\n')
    add('![热湿分布](figures/02_热湿时空分布.png)\n')
    add('## 5 验证证据\n')
    add('### 5.1 网格加密\n')
    rows=[r for r in v['mesh'] if r['problem']!=1]
    add(table(['问题','N','临界时间 / h','较前级时长差 / s','抽查最大 C 差'],[[r['problem'],r['n'],f"{r['critical_time_h']:.8f}",r.get('delta_time_s'),f"{r['max_C_difference']:.3e}" if 'max_C_difference' in r else '—'] for r in rows]))
    add('差值按抽查时间与 21 个位置计算：问题 1 使用题设展示时刻；问题 3、4 使用前 3 h 展示时刻及后期 40 个共同时间点。此指标不是遍历每一个输出时空点的严格误差上界，也不把最细网格当成解析真值。最后一次加密后，两道时长题的差均小于 0.13 s。四位小数是题目要求的显示格式，不能解释为同等精度的物理实验预测。\n')
    add('### 5.2 解析基准与时间积分交叉检查\n')
    add('在常系数、均匀初值、恒定环境的独立基准下，以圆柱 Robin 边界的 Bessel 级数为解析参照：特征根满足 λJ₁(λ)=Bi·J₀(λ)，系数 A=2J₁(λ)/[λ(J₀²+J₁²)]。取 40 个根，在 100、300、600、1800 s 对全体径向节点比较。\n')
    add(table(['N','最大无量纲误差','等效 22°C 阶跃最大误差 / °C'],[[r['n'],f"{r['max_dimensionless_error']:.3e}",f"{r['equivalent_22K_step_error_C']:.3e}"] for r in v['analytic']]))
    st=v['time_solver'];add(f"第 3 问 N=160 时，BDF 得 {st['BDF_h']:.9f} h，Radau 得 {st['Radau_h']:.9f} h，差 {st['difference_s']:.6f} s。这验证时间积分一致性，不能替代模型假设的验证。\n")
    add('### 5.3 收支与合理范围\n')
    add(table(['问题','C 最小值','C 最大值','独立通量积分收支误差','占初始含水率比例'],[[q,b['C_min'],b['C_max'],f"{b['independent_quadrature_balance_error']:.3e}",f"{b['relative_to_initial_C']:.3e}"] for q,b in v['balance'].items()]))
    add('独立收支核验使用另行采样的边界通量和复合梯形积分：问题 1 每 2 s，问题 3、4 每 10 s。它与半离散方程的瞬时代数守恒检查同时进行。数值含水率始终非负且未超初值，径向含水率的向外增加只在浮点舍入量级出现。温度保持在初始与给定环境的合理范围。这个检查核验的是所选干基有效模型中的归一化水量，并非使用题设湿密度计算出的实物公斤数。\n')
    add('### 5.4 有限长度和端面\n')
    add(table(['问题','2D 网格 Nr×Nz','同径向网格 1D / h','2D / h','差 / s','相对差 / %'],[[r['problem'],f"{r['nr']}×{r['nz']}",r['time_1d_same_nr_h'],r['time_2d_h'],r['time_difference_s'],f"{r['change_pct']:.6f}"] for r in geo]))
    add('二维模型覆盖 0≤r≤R(t)、0≤z≤L/2，中心面对称，侧面和两端暴露于同一烘房环境，采用同一套材料参数和有限体积原理。两种情形的最大 C 均在几何中心。结果支持一维模型用于最迟达标时间；它不表示所有端面附近的温湿场都可由一维模型准确代表，也不应把一维截面均值称为有限圆柱的全体积均值。该二维检查仅完成所示网格，不宣称二维自身已达到网格无关。\n')
    add('![数值收敛](figures/05_收敛与空间剖面.png)\n')
    add('## 6 参数、闭合和创新点的检验\n')
    add('### 6.1 平均达标会导致提前停机\n')
    add(table(['问题','平均 C=0.15 的时刻 / h','该时中心 C','较全部达标提前 / h'],[[q,b['mean_threshold_time_h'],b['center_C_at_mean_threshold'],b['premature_stop_h']] for q,b in v['stopping_comparison'].items()]))
    add('这项对照给出了实际决策差异：采用平均含水率停机时，中心仍未合格。主模型以全域最大值作为终点，直接对应题目的“各处”。\n')
    add('![停机标准](figures/03_达标与提前停机风险.png)\n')
    add('### 6.2 参数扰动与结构对照\n')
    names={'D -10%':'D×0.9','D +10%':'D×1.1','hm -20%':'hₘ×0.8','hm +20%':'hₘ×1.2',
           'tail T -1 C':'后期温度 −1°C','tail T +1 C':'后期温度 +1°C','tail C -0.01':'后期浓度 −0.01','tail C +0.01':'后期浓度 +0.01',
           'last datum tail':'最后单条数据延拓','latent fraction 1':'附加全部表面潜热情景','exponential radius':'指数半径对照',
           'fixed radius appendix4':'附录4参数 但固定半径2cm','radius -1%':'整条半径曲线×0.99','radius +1%':'整条半径曲线×1.01'}
    add(table(['问题','情景','临界时间 / h','相对同网格基准变化 / %'],[[r['problem'],names.get(r['case'],r['case']),r['critical_time_h'],r['change_pct']] for r in v['sensitivity']]))
    add('所有情景使用 N=160，与同 N 基准作比，以免把网格误差混入参数影响。参数范围是人为设置的压力情景，不是经测量标定的分布，因此不输出“95%置信区间”。温湿边界扰动只施加在 4 h 后；半径百分比扰动是整条输入曲线的校准误差情景，也改变初始半径，不属于主模型。\n')
    latent=[r for r in v['sensitivity'] if r['case']=='latent fraction 1']
    add('潜热情景在热边界中额外扣除 Lᵥ·[ρ(Cₛ)/(1+Cₛ)]·hₘ(Cₛ−Cₐ)，取代表性 Lᵥ=2.38×10⁶ J/kg，并假设所有离开表面的水都在该表面蒸发。潜热和这个质量通量换算均属新增假设；该对照用于说明主模型遗漏蒸发冷却可能造成的偏差，不把它当作已经实验标定的更正确答案。也不能从该模型推出真实烘房总能耗，因为设备散热、排风和装载量尚未给出。\n')
    fixed=next(r for r in v['sensitivity'] if r['case']=='fixed radius appendix4')
    add(f"保持附录 4 的材料参数，仅取消收缩，时长变为 {fixed['critical_time_h']:.4f} h。主收缩模型同网格时长为 {next(r['critical_time_h'] for r in v['mesh'] if r['problem']==4 and r['n']==160):.4f} h。这个对照才能解释收缩本身的作用；不能把第 3 问与第 4 问的总差异全部归因于收缩，因为两问材料经验公式也发生了变化。\n")
    add(f"半径指数对照 R(t)=R∞+(2−R∞)exp(−t/τ) 的全样本拟合给出 R∞={audit['radius_exp_parameters'][0]:.6f} cm、τ={audit['radius_exp_parameters'][1]:.2f} s，RMSE={audit['radius_exp_RMSE_cm']:.6f} cm。仅以前 12 h 拟合、以其后数据检验时，留出 RMSE={audit['radius_holdout_after12h_RMSE_cm']:.6f} cm。这是对半径输入规律的留出检验，不是对药材水分预测的留出检验。主模型保留全部实测半径的保形插值，指数拟合只作结构对照。\n")
    add('![参数敏感性](figures/04_参数敏感性.png)\n')
    add('### 6.3 可用于竞赛论证的改进\n')
    add('1. **干基定义一致的收缩模型。** 由材料导数出发推导变域方程，避免错误的浓缩源项，结合实测半径和共享面通量检查水分收支。\n2. **针对非线性低扩散表层的积分通量。** 用扩散势对应的积分面系数处理表面阻力，再通过解析基准和网格加密给出证据。\n3. **以全域达标为目标的停机判据。** 对照平均含水率策略，量化提前停机造成的中心不合格，并以二维端面模型检查最湿点。\n4. **有对照的收缩贡献分析。** 固定材料参数再关闭收缩，同时分析温湿边界和相变假设，避免只报告一个缺少适用范围的时长。\n')
    add('这些改进服务于正确性和决策解释，不以算法名称数量或复杂程度作为创新证据。不声称上述基本方法是文献中首次提出，也不以数值收敛、留出半径拟合或省赛获奖目标替代外部实验验证。\n')
    add('## 7 交付文件与复现\n')
    add(table(['文件','内容'],[
        ['result1.xlsx','1—1800 s，每秒、每 0.1 cm，温度和水分浓度两表'],
        ['result2.xlsx','题面展示的前 3 h 完整窗口，1—10800 s，每秒、每 0.1 cm，两表'],
        ['result3.xlsx',f"60—{int(c['3']['first_full_minute_s'])} s，每分钟、每 0.1 cm，直至整分钟安全终点"],
        ['result4.xlsx',f"60—{int(c['4']['first_full_minute_s'])} s，每分钟、每 0.1 cm，含表面值和当时半径"],
        ['results/validation.json','网格、解析、守恒、敏感性、停机、展示表的未舍入证据'],
        ['results/geometry_validation.json','二维端面效应检查'],
        ['results/profiles_full_precision.npz','代表性全精度温湿时空场，用于作图和再分析'],
        ['code/','求解器、验证程序、图表和说明生成程序'],
        ['data/','输入原数值、原模板和校验摘要'],
        ['figures/','5 张可直接使用的数值结果图']]))
    add('复现步骤见 README.md。问题 2 的 Excel 范围按题干“3 h 内”的展示区间理解；全烘干过程方程已经求解至第 3 问终点，其分钟级全程数据在 result3.xlsx。若另有官方补充说明要求 result2 也输出全程每秒温度和含水率，应扩展同一模型的采样区间，不能另换模型或凭插值外推初期结果。\n')
    add('### 参考依据\n')
    add('- 用户提供的 A 题 PDF 第 1—4 页、附件 1、附件 2、附件 3，提供全部主模型数值输入与结果格式。\n- [SciPy solve_ivp 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)：BDF、Radau、稀疏 Jacobian、连续输出及事件定位接口。\n- [Curcio 与 Aversa 的对流干燥传递和收缩模型](https://www.comsol.com/paper/download/63285/aversa_paper.pdf)：真实干燥可涉及热、水和蒸气多相耦合及 ALE 变域；本文只采用题设数据可支撑的简化闭合，不移用该文试验结果或参数。\n- [IAPWS 水和水蒸气性质公式](https://iapws.org/relguide/IF97-Rev.pdf)：附加潜热情景所涉及水物性的物理参考，不是题目提供的药材实验数据。\n')
    (ROOT/'建模与验证说明.md').write_text('\n'.join(parts),encoding='utf-8')
    print('report written')


if __name__=='__main__':main()
