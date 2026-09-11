"""λ=1 主工况（计全部表面汽化热）的四问交付导出与核验。

背景：论文主结果取"干质量守恒热湿模型 + 表面全部汽化热"（λ=1，L_v=2.38 MJ/kg），
第三、四问临界时长 60.430002 h / 56.501190 h。原 result1-4.xlsx 属 λ=0 显热基准，
不能与主工况数字混用，故本脚本按同一求解器重新求解四问并导出对应交付文件。

产物
  results/result1.json - result4.json   与 守恒热湿模型 同格式的结果表数据
  results/tables_1_6.json               论文表 1-6 的数值
  results/checks.json                   求解、守恒、逐格回读核验记录
  result1.xlsx - result4.xlsx           交付工作簿（表头/列样式克隆自显热基准交付文件）

用法
  python export_latent_books.py --cache <临时缓存目录>
可选参数
  --n 1280           求解网格；默认 1280，与 60.430002 h / 56.501190 h 同网格
  --style-source DIR 克隆表头与列样式的来源目录，默认 计算与证据/守恒热湿模型
  --skip-xlsx        只求解与核验，不写工作簿
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

HERE = Path(__file__).resolve()
OUT_ROOT = HERE.parents[1]                     # 计算与证据/相变热工况交付
EVIDENCE = HERE.parents[2]                     # 计算与证据
CONSERVATIVE_CODE = EVIDENCE / '守恒热湿模型' / 'code'
sys.path.insert(0, str(CONSERVATIVE_CODE))

from conservative_model import ConservativeSettings as Settings   # noqa: E402
from run_conservation_study import get_model, payload, extend, balance_checks  # noqa: E402

RESULTS = OUT_ROOT / 'results'
# 文档中记录的 λ=1、N=1280 临界时长，用于核对本次复算是否一致。
DOCUMENTED = {3: 60.430002, 4: 56.501190}
CRITICAL_TOLERANCE_H = 0.01
R_COLS_CM = np.round(np.arange(0, 2.0001, .1), 1)
TABLE_R_CM = [0., .5, 1., 1.5, 2.]


def save_lf(path, value):
    """写 JSON，换行统一为 LF，与成果中其余文本文件一致。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8', newline='\n')


# ---------------------------------------------------------------- 工作簿写入


def _col_letter(index):
    """0 -> A, 25 -> Z, 26 -> AA"""
    letters = ''
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _num(value):
    """数值单元格文本；避免科学计数法，保持最短往返表示。"""
    if isinstance(value, int):
        return str(value)
    text = json.dumps(float(value))
    if 'e' in text or 'E' in text:
        text = ('%.10f' % float(value)).rstrip('0').rstrip('.')
    return text


def build_rows_xml(rows, row_styles, header_xml):
    """由数据行生成 sheetData 内容；row_styles 为每列样式索引的列表。"""
    parts = [header_xml]
    for offset, row in enumerate(rows[1:], start=2):
        cells = []
        for col, value in enumerate(row):
            ref = '%s%d' % (_col_letter(col), offset)
            style = row_styles[col]
            if value is None:
                cells.append('<x:c r="%s" s="%s" />' % (ref, style))
            elif isinstance(value, str):
                cells.append('<x:c r="%s" s="%s" t="str"><x:v>%s</x:v></x:c>'
                             % (ref, style, escape(value)))
            else:
                cells.append('<x:c r="%s" s="%s" t="n"><x:v>%s</x:v></x:c>'
                             % (ref, style, _num(value)))
        parts.append('<x:row r="%d" ht="18" customHeight="1">%s</x:row>' % (offset, ''.join(cells)))
    return ''.join(parts)


def _sheet_parts(archive):
    """工作簿内各工作表名 -> 部件路径，按工作簿顺序。"""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
          'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
          'p': 'http://schemas.openxmlformats.org/package/2006/relationships'}
    import xml.etree.ElementTree as ET
    wb = ET.fromstring(archive.read('xl/workbook.xml'))
    rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    targets = {rel.get('Id'): rel.get('Target') for rel in rels.findall('p:Relationship', ns)}
    order = []
    for sheet in wb.find('m:sheets', ns):
        target = targets[sheet.get('{%s}id' % ns['r'])].lstrip('/')
        order.append((sheet.get('name'), target if target.startswith('xl/') else 'xl/' + target))
    return order


def clone_workbook(dst, style_source, sheets):
    """把 sheets{表名: 行列表} 写入 dst；表头与列样式取自 style_source 的同名工作表。"""
    src = zipfile.ZipFile(style_source)
    mapping = dict(_sheet_parts(src))
    missing = set(sheets) - set(mapping)
    if missing:
        raise KeyError('样式来源缺少工作表: %s' % sorted(missing))
    replaced = {}
    for name, rows in sheets.items():
        part = mapping[name]
        xml = src.read(part).decode('utf-8')
        start = xml.index('<x:sheetData>') + len('<x:sheetData>')
        end = xml.index('</x:sheetData>')
        head = xml[:start]
        tail = xml[end:]
        # 表头行与第二行样式模板按原样取用，保持与显热基准交付件一致的外观。
        row1_start = xml.index('<x:row r="1"')
        row1_end = xml.index('</x:row>', row1_start) + len('</x:row>')
        header_xml = xml[row1_start:row1_end]
        row2_start = xml.index('<x:row r="2"')
        row2_end = xml.index('</x:row>', row2_start) + len('</x:row>')
        row_styles = []
        import re
        for match in re.finditer(r'<x:c r="([A-Z]+)2"(?: s="(\d+)")?', xml[row2_start:row2_end]):
            row_styles.append(match.group(2) or '0')
        if len(row_styles) < len(rows[0]):
            raise ValueError('%s 样式列数不足: %d < %d' % (name, len(row_styles), len(rows[0])))
        replaced[part] = (head + build_rows_xml(rows, row_styles, header_xml) + tail).encode('utf-8')
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = replaced.get(item.filename, src.read(item.filename))
            out.writestr(item, data)
    src.close()


def read_book(path):
    """独立回读：工作表名 -> 取值后的行列表（空单元格为 None）。"""
    import xml.etree.ElementTree as ET
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    archive = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in archive.namelist():
        root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
        for si in root.findall('m:si', ns):
            shared.append(''.join(node.text or '' for node in si.iter(
                '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')))
    out = {}
    for name, part in _sheet_parts(archive):
        root = ET.fromstring(archive.read(part))
        rows = []
        for row in root.iter('{%s}row' % ns['m']):
            values = []
            for cell in row.findall('m:c', ns):
                kind = cell.get('t')
                node = cell.find('m:v', ns)
                text = None if node is None else node.text
                if text is None:
                    values.append(None)
                elif kind == 's':
                    values.append(shared[int(text)])
                elif kind in ('str', 'inlineStr'):
                    values.append(text)
                else:
                    values.append(float(text))
            rows.append(values)
        out[name] = rows
    archive.close()
    return out


# ---------------------------------------------------------------- 表格与核验


def _pick(model, t, r_cm, field):
    value = float(model.profile(t, np.array([r_cm]), field)[0])
    return None if np.isnan(value) else round(value, 4)


def tables(models, criticals):
    m1, m3, m4 = models[1], models[3], models[4]
    tables_out = {}
    q1_times = [100, 300, 600, 900, 1200, 1500, 1800]
    q1 = {'时间/s': q1_times, '到药材中心的距离/cm': TABLE_R_CM}
    q1['温度/°C'] = [[_pick(m1, t, r, 'T') for r in TABLE_R_CM] for t in q1_times]
    q1['水分浓度/(kg/kg)'] = [[_pick(m1, t, r, 'C') for r in TABLE_R_CM] for t in q1_times]
    tables_out['表1_表2_问题1'] = q1

    q2_times = [1800, 3600, 5400, 7200, 9000, 10800]
    q2 = {'时间/s': q2_times, '时间/h': [t / 3600 for t in q2_times], '到药材中心的距离/cm': TABLE_R_CM}
    q2['温度/°C'] = [[_pick(m3, t, r, 'T') for r in TABLE_R_CM] for t in q2_times]
    q2['水分浓度/(kg/kg)'] = [[_pick(m3, t, r, 'C') for r in TABLE_R_CM] for t in q2_times]
    tables_out['表3_表4_问题2'] = q2

    for key, model in (('表5_问题3', m3), ('表6_问题4', m4)):
        q = model.p.problem
        critical = criticals[q]['critical_s']
        end_minute = criticals[q]['first_full_minute_s']
        grid = [t for t in np.arange(21600., critical + 1e-9, 21600.)]
        times = list(grid) + [end_minute]
        entry = {'时间/h': [round(t / 3600, 6) for t in times],
                 '时间/s': [int(t) for t in times],
                 '到药材中心的距离/cm': TABLE_R_CM,
                 '行标签': ['%g' % (t / 3600) for t in grid] + ['烘干结束时间'],
                 '水分浓度/(kg/kg)': [[_pick(model, t, r, 'C') for r in TABLE_R_CM] for t in times]}
        if q == 4:
            entry['药材表面'] = [round(float(model.state(t)[-1]), 4) for t in times]
            entry['当前半径/cm'] = [round(float(model.env.radius(t)) * 100, 6) for t in times]
        tables_out[key] = entry
    return tables_out


def verify_book(path, expected, label):
    """逐格回读核验：表头、行列数、每个数值与空白位置。"""
    book = read_book(path)
    report = {'file': path.name, 'sheets': {}, 'status': 'PASS'}
    for name, rows in expected.items():
        got = book[name]
        if len(got) != len(rows):
            raise AssertionError('%s/%s 行数 %d != %d' % (label, name, len(got), len(rows)))
        numeric = 0
        blank = 0
        for i, (got_row, exp_row) in enumerate(zip(got, rows)):
            if len(got_row) < len(exp_row):
                raise AssertionError('%s/%s 第%d行列数 %d < %d' % (label, name, i + 1, len(got_row), len(exp_row)))
            if i == 0:
                if [str(v) for v in got_row[:len(exp_row)]] != [str(v) for v in exp_row]:
                    raise AssertionError('%s/%s 表头不一致: %s' % (label, name, got_row[:6]))
                continue
            for j, exp in enumerate(exp_row):
                value = got_row[j]
                if exp is None:
                    if value is not None:
                        raise AssertionError('%s/%s 第%d行第%d列应为空白，实为 %r'
                                             % (label, name, i + 1, j + 1, value))
                    blank += 1
                else:
                    if value is None or abs(float(value) - float(exp)) > 1e-12:
                        raise AssertionError('%s/%s 第%d行第%d列 %r != %r'
                                             % (label, name, i + 1, j + 1, value, exp))
                    numeric += 1
        report['sheets'][name] = {'rows': len(got), 'numeric_cells': numeric, 'blank_cells': blank}
    report['total_cells'] = sum(v['rows'] * len(expected[k][0]) for k, v in report['sheets'].items())
    report['numeric_cells'] = sum(v['numeric_cells'] for v in report['sheets'].values())
    report['blank_cells'] = sum(v['blank_cells'] for v in report['sheets'].values())
    return report


def legacy_reader_check(style_source):
    """用同一读取器核对显热基准交付件，确认读取器与文档记录的单元格总数一致。"""
    names = ['result1.xlsx', 'result2.xlsx', 'result3.xlsx', 'result4.xlsx']
    total = 0
    detail = {}
    for name in names:
        book = read_book(style_source / name)
        cells = 0
        for sheet, rows in book.items():
            width = max(len(r) for r in rows)
            cells += len(rows) * width
        last_times = []
        for sheet, rows in book.items():
            last_times.append(int(rows[-1][0]))
        detail[name] = {'cells': cells, 'last_time_s': last_times}
        total += cells
    return {'total_cells': total, 'detail': detail,
            'documented_total_cells': 704020, 'matches_documented': total == 704020}


# ---------------------------------------------------------------- 主流程


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--n', type=int, default=1280)
    parser.add_argument('--style-source', default=str(EVIDENCE / '守恒热湿模型'))
    parser.add_argument('--skip-xlsx', action='store_true')
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    style_source = Path(args.style_source)
    RESULTS.mkdir(parents=True, exist_ok=True)
    wall = {}
    checks = {'mesh': args.n, 'latent_fraction': 1., 'latent_heat_J_kg': 2.38e6}

    models = {}
    for q in (1, 3, 4):
        tic = time.perf_counter()
        models[q] = get_model(Settings(problem=q, n=args.n, latent_fraction=1.), cache)
        wall['solve_q%d' % q] = round(time.perf_counter() - tic, 1)

    criticals = {}
    for q in (3, 4):
        model = models[q]
        critical = float(model.end)
        minute = float((np.floor(critical / 60) + 1) * 60)
        max_before = float(np.max(model.state(critical)[model.m:]))
        models[q].parts_backup = len(model.parts)
        extend(model, minute)
        criticals[q] = {
            'critical_s': critical,
            'critical_h': critical / 3600,
            'documented_h': DOCUMENTED[q],
            'difference_vs_documented_s': critical - DOCUMENTED[q] * 3600,
            'first_full_minute_s': minute,
            'first_full_minute_h': minute / 3600,
            'max_C_at_critical': max_before,
            'max_C_at_first_full_minute': float(np.max(model.state(minute)[model.m:])),
            'surface_C_at_first_full_minute': float(model.state(minute)[-1]),
        }
        drift = abs(criticals[q]['difference_vs_documented_s'])
        print('CRITICAL q%d %.6f h (documented %.6f, drift %.3f s) minute %.0f s maxC %.8f'
              % (q, criticals[q]['critical_h'], DOCUMENTED[q], drift, minute,
                 criticals[q]['max_C_at_first_full_minute']), flush=True)
        if drift > CRITICAL_TOLERANCE_H * 3600:
            raise AssertionError('第%d问临界时长与文档记录相差 %.1f s，超过 %.0f s 容差'
                                 % (q, drift, CRITICAL_TOLERANCE_H * 3600))
        if not criticals[q]['max_C_at_first_full_minute'] < 0.15:
            raise AssertionError('第%d问首个整分钟最大含水率未严格低于 0.15' % q)

    books = {
        'result1': {'温度': payload(models[1], range(1, 1801), 'T'),
                    '水分浓度': payload(models[1], range(1, 1801), 'C')},
        'result2': {'温度': payload(models[3], range(1, 10801), 'T'),
                    '水分浓度': payload(models[3], range(1, 10801), 'C')},
        'result3': {'Sheet1': payload(models[3], np.arange(60, models[3].end + 1, 60))},
        'result4': {'Sheet1': payload(models[4], np.arange(60, models[4].end + 1, 60), shrink=True)},
    }
    for name, book in books.items():
        save_lf(RESULTS / (name + '.json'), book)

    tables_out = tables(models, criticals)
    save_lf(RESULTS / 'tables_1_6.json', {
        'condition': 'dry-mass-conserving model with all net drainage evaporated at the surface, '
                     'lambda=1, Lv=2.38 MJ/kg, N=%d' % args.n,
        'critical_h': {str(q): criticals[q]['critical_h'] for q in (3, 4)},
        'first_full_minute_h': {str(q): criticals[q]['first_full_minute_h'] for q in (3, 4)},
        'tables': tables_out})

    checks['critical'] = {str(q): criticals[q] for q in (3, 4)}
    checks['balance'] = {}
    for q in (1, 3, 4):
        checks['balance'][str(q)] = balance_checks(models[q])
        row = checks['balance'][str(q)]
        print('BALANCE q%d dry %.3e water %.3e energy %.3e' % (
            q, row['max_global_dry_mass_relative_error'],
            row['independent_time_integrals'][-1]['relative_water_residual'],
            row['independent_time_integrals'][-1]['relative_energy_residual']), flush=True)

    checks['legacy_reader'] = legacy_reader_check(style_source)
    print('LEGACY reader total cells %d (documented 704020, match %s)'
          % (checks['legacy_reader']['total_cells'], checks['legacy_reader']['matches_documented']), flush=True)

    if not args.skip_xlsx:
        checks['books'] = {}
        for name, sheets in books.items():
            dst = OUT_ROOT / (name + '.xlsx')
            tic = time.perf_counter()
            clone_workbook(dst, style_source / (name + '.xlsx'), sheets)
            report = verify_book(dst, sheets, name)
            report['write_and_verify_s'] = round(time.perf_counter() - tic, 1)
            report['bytes'] = dst.stat().st_size
            checks['books'][name] = report
            print('BOOK %s rows %s cells %d numeric %d blank %d'
                  % (name, {k: v['rows'] for k, v in report['sheets'].items()},
                     report['total_cells'], report['numeric_cells'], report['blank_cells']), flush=True)

    total_numeric = sum(v['numeric_cells'] for v in checks.get('books', {}).values())
    checks['wall_seconds'] = wall
    checks['total_numeric_cells'] = total_numeric
    checks['status'] = 'PASS'
    save_lf(RESULTS / 'checks.json', checks)
    print('FINISHED numeric_cells %d wall %s' % (total_numeric, wall), flush=True)


if __name__ == '__main__':
    main()
