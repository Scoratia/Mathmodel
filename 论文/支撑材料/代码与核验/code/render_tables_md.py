"""把 results/tables_1_6.json 渲染为可直接粘进论文的 Markdown（tables_1_6.md）。

用法: python render_tables_md.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve()
OUT_ROOT = HERE.parents[1]
RESULTS = OUT_ROOT / 'results'


def cell(value):
    return '—' if value is None else ('%g' % value if isinstance(value, (int, float)) else str(value))


def grid(header, rows, labels, columns):
    lines = ['| ' + ' | '.join([header] + [cell(c) for c in columns]) + ' |',
             '| ' + ' | '.join(['---'] * (len(columns) + 1)) + ' |']
    for label, row in zip(labels, rows):
        lines.append('| ' + ' | '.join([str(label)] + [cell(v) for v in row]) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    data = json.loads((RESULTS / 'tables_1_6.json').read_text(encoding='utf-8'))
    tables = data['tables']
    out = ['# 表 1—表 6（主工况 λ=1，计全部表面汽化热）', '',
           '- 工况：干质量守恒热湿模型 + 表面全部汽化热，$\\lambda=1$，$L_v=2.38$ MJ/kg，$N=%s$'
           % data.get('N', 1280),
           '- 第三问临界时长 **%s h**，严格达标首个整分钟 **%s h**'
           % (round(data['critical_h']['3'], 6), round(data['first_full_minute_h']['3'], 6)),
           '- 第四问临界时长 **%s h**，严格达标首个整分钟 **%s h**'
           % (round(data['critical_h']['4'], 6), round(data['first_full_minute_h']['4'], 6)),
           '- 空白「—」表示该固定位置已在药材之外（第四问半径收缩所致），不是零，也不是漏算。', '']

    q1 = tables['表1_表2_问题1']
    out += ['## 表 1　30 分钟内药材的温度（单位：°C）', '',
            grid('时间/s', q1['温度/°C'], q1['时间/s'], q1['到药材中心的距离/cm'])]
    out += ['## 表 2　30 分钟内药材的水分浓度（干基，单位：kg/kg）', '',
            grid('时间/s', q1['水分浓度/(kg/kg)'], q1['时间/s'], q1['到药材中心的距离/cm'])]

    q2 = tables['表3_表4_问题2']
    out += ['## 表 3　3 小时内药材的温度（单位：°C）', '',
            grid('时间/h', q2['温度/°C'], ['%g' % t for t in q2['时间/h']], q2['到药材中心的距离/cm'])]
    out += ['## 表 4　3 小时内药材的水分浓度（干基，单位：kg/kg）', '',
            grid('时间/h', q2['水分浓度/(kg/kg)'], ['%g' % t for t in q2['时间/h']], q2['到药材中心的距离/cm'])]

    q3 = tables['表5_问题3']
    out += ['## 表 5　药材烘干过程的水分浓度（问题 3，单位：kg/kg）', '',
            grid('时间/h', q3['水分浓度/(kg/kg)'], q3['行标签'], q3['到药材中心的距离/cm'])]

    q4 = tables['表6_问题4']
    columns = list(q4['到药材中心的距离/cm']) + ['药材表面']
    rows = [list(r) + [s] for r, s in zip(q4['水分浓度/(kg/kg)'], q4['药材表面'])]
    out += ['## 表 6　药材烘干过程的水分浓度（问题 4，单位：kg/kg）', '',
            grid('时间/h', rows, q4['行标签'], columns), '',
            '问题 4 的当前半径（cm）随行给出，用于识别随时间改变的药材表面位置：', '',
            grid('时间/h', [[r] for r in q4['当前半径/cm']], q4['行标签'], ['当前半径/cm'])]

    (RESULTS / 'tables_1_6.md').write_text('\n'.join(out), encoding='utf-8', newline='\n')
    print('written', RESULTS / 'tables_1_6.md')


if __name__ == '__main__':
    main()
