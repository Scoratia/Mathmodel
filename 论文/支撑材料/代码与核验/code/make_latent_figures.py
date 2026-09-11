"""λ=1 主工况的论文插图：分辨率提高，字号与图幅比例与原图一致。

与原图的对应关系
  经验闭合对照/figures/02_热湿时空分布.png       → figures/02_热湿时空分布_计相变热.png
  经验闭合对照/figures/03_达标与提前停机风险.png → figures/03_达标与提前停机风险_计相变热.png
  守恒热湿模型/figures/前3小时温度对照.png       → figures/前3小时温度对照_计相变热.png

做法：figsize、字号、配色、子图布局与面板标题全部沿用原脚本，
只把 savefig.dpi 由 200/150 提高到 --dpi（默认 400），
因此"文字占图幅的比例"不变，而像素分辨率提高一倍以上。

同时导出 λ=1 的全精度场 results/profiles_full_precision_latent.npz 与
停机对照 results/stopping_latent.json，供其他插图或复核复用。

用法
  python make_latent_figures.py --cache <临时缓存目录> [--dpi 400]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
OUT_ROOT = HERE.parents[1]
EVIDENCE = HERE.parents[2]
CONSERVATIVE = EVIDENCE / '守恒热湿模型'
sys.path.insert(0, str(CONSERVATIVE / 'code'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TEMP', '/tmp')) / 'v5_mpl_cache'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from conservative_model import ConservativeSettings as Settings  # noqa: E402
from run_conservation_study import get_model  # noqa: E402

RESULTS = OUT_ROOT / 'results'
FIGURES = OUT_ROOT / 'figures'
COLORS = ['#184A67', '#C45E37', '#407B68', '#92624A']


def pick_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']:
        if name in available:
            return name
    return 'DejaVu Sans'


def load_models(cache, n):
    return {q: get_model(Settings(problem=q, n=n, latent_fraction=1.), cache) for q in (1, 3, 4)}


def export_fields(models):
    """λ=1 全精度场，键名与守恒热湿模型导出一致。"""
    fields = {}
    for q, m in models.items():
        ts = np.unique(np.r_[np.linspace(0, m.end, 501), 1800., 10800. if q != 1 else 1800.])
        yy = m.state(ts)
        rd, rw = [], []
        for t, C in zip(ts, yy[m.m:].T):
            a, b = m.densities(t, C)
            rd.append(a)
            rw.append(b)
        fields[f'q{q}_time_s'] = ts
        fields[f'q{q}_x'] = m.x
        fields[f'q{q}_T_C'] = yy[:m.m]
        fields[f'q{q}_C_dry_basis'] = yy[m.m:]
        fields[f'q{q}_radius_m'] = np.array([m.env.radius(t) for t in ts])
        fields[f'q{q}_dry_density'] = np.array(rd).T
        fields[f'q{q}_wet_density'] = np.array(rw).T
        fields[f'q{q}_dry_cell_mass_kg'] = m.dry_mass
    np.savez_compressed(RESULTS / 'profiles_full_precision_latent.npz', **fields)
    return fields


def stopping(models):
    """λ=1 的停机对照：平均含水率达标时刻与该时刻的中心含水率。"""
    out = {}
    for q in (3, 4):
        m = models[q]
        threshold = brentq(lambda t: float(m.mean_C(t)) - .15, 0., m.end)
        out[str(q)] = {
            'critical_h': m.end / 3600,
            'mean_threshold_h': threshold / 3600,
            'center_C_at_mean_threshold': float(m.state(threshold)[m.m]),
            'premature_stop_h': (m.end - threshold) / 3600,
        }
    (RESULTS / 'stopping_latent.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8', newline='\n')
    return out


def figure_spacetime(fields, stop, dpi):
    """对应原 02_热湿时空分布.png：2×2，材料坐标，时间-材料位置伪彩图。"""
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), layout='constrained')
    for col, q in enumerate([3, 4]):
        ts = fields[f'q{q}_time_s'] / 3600
        x = fields[f'q{q}_x']
        C = fields[f'q{q}_C_dry_basis']
        T = fields[f'q{q}_T_C']
        for row, mat, title, unit in [(0, T, '温度', '°C'), (1, C, '水分浓度', 'kg/kg')]:
            im = axs[row, col].pcolormesh(ts, x, mat, shading='auto', cmap='viridis')
            axs[row, col].set(xlabel='时间 / h', ylabel='材料位置 r/R(t)', title=f'问题 {q}  {title}')
            if row == 0:
                axs[row, col].set_xlim(0, 4)
                axs[row, col].set_title(f'问题 {q}  温度 前4小时')
            else:
                # 水分浓度的横轴终点即烘干终点，故用角标文字标注，不画贴边虚线。
                tc = stop[str(q)]['critical_h']
                axs[row, col].text(.985, .955, '终点 $t^*$=%.2f h' % tc, transform=axs[row, col].transAxes,
                                   ha='right', va='top', color='w', fontsize=8)
            fig.colorbar(im, ax=axs[row, col], label=unit, pad=.02)
    fig.savefig(FIGURES / '02_热湿时空分布_计相变热.png')
    plt.close(fig)


def figure_stopping(fields, stop, dpi):
    """对应原 03_达标与提前停机风险.png：中心与截面平均、停机附近放大。"""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for q, color in zip([3, 4], COLORS):
        ts = fields[f'q{q}_time_s'] / 3600
        C = fields[f'q{q}_C_dry_basis']
        x = fields[f'q{q}_x']
        faces = (x[1:] + x[:-1]) / 2
        w = np.diff(np.r_[0, faces, 1] ** 2) / 2
        axs[0].plot(ts, C[0], color=color, label=f'问题 {q} 中心')
        axs[0].plot(ts, 2 * w @ C, color=color, ls='--', label=f'问题 {q} 截面平均')
        mt = stop[str(q)]['mean_threshold_h']
        mc = stop[str(q)]['center_C_at_mean_threshold']
        axs[1].plot(ts, C[0], color=color, label=f'问题 {q} 中心')
        axs[1].scatter([mt], [mc], color=color, s=24, zorder=5)
    for ax in axs:
        ax.axhline(.15, color='#555', ls=':', label='要求上限 0.15')
        ax.set(xlabel='时间 / h', ylabel='水分浓度 / kg/kg')
        ax.legend(fontsize=8)
    axs[0].set_title('中心与平均含水率')
    axs[1].set(title='停机附近 放大', xlim=(30, 62), ylim=(.12, .26))
    fig.savefig(FIGURES / '03_达标与提前停机风险_计相变热.png')
    plt.close(fig)


def figure_first_three_hours(dpi):
    """对应原 前3小时温度对照.png：第二问前3小时，显热基准与计相变热对照。"""
    with (OUT_ROOT / 'results/result2.json').open(encoding='utf-8') as fh:
        latent = np.asarray(json.load(fh)['温度'][1:], float)
    with (CONSERVATIVE / 'results/result2.json').open(encoding='utf-8') as fh:
        sensible = np.asarray(json.load(fh)['温度'][1:], float)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout='constrained')
    for a, ls, tag in [(sensible, '--', '显热基准 λ=0'), (latent, '-', '计相变热 λ=1')]:
        ax.plot(a[:, 0] / 3600, a[:, 1], ls=ls, color='#1D4ED8', label=tag + '：中心')
        ax.plot(a[:, 0] / 3600, a[:, -1], ls=ls, color='#B45309', label=tag + '：表面')
    ax.set(xlabel='时间 / h', ylabel='温度 / °C', title='第二问前3小时温度对照（表格精度）')
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(FIGURES / '前3小时温度对照_计相变热.png', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--n', type=int, default=1280)
    parser.add_argument('--dpi', type=int, default=400)
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # 字号与原脚本一致：经验闭合对照 10 pt，守恒热湿模型 11 pt。
    plt.rcParams.update({'font.family': pick_font(), 'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 200, 'savefig.dpi': args.dpi})
    print('font=%s dpi=%d' % (plt.rcParams['font.family'], args.dpi), flush=True)

    models = load_models(Path(args.cache), args.n)
    fields = export_fields(models)
    stop = stopping(models)
    print('stopping:', json.dumps(stop, ensure_ascii=False), flush=True)

    figure_spacetime(fields, stop, args.dpi)
    plt.rcParams.update({'font.size': 11})
    figure_stopping(fields, stop, args.dpi)
    plt.rcParams.update({'figure.dpi': 150})
    figure_first_three_hours(args.dpi)

    for name in ['02_热湿时空分布_计相变热.png', '03_达标与提前停机风险_计相变热.png',
                 '前3小时温度对照_计相变热.png']:
        path = FIGURES / name
        print('%-40s %8d bytes' % (name, path.stat().st_size), flush=True)
    print('FIGURES OK')


if __name__ == '__main__':
    main()
