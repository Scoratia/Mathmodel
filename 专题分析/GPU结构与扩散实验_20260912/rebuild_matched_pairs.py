"""Rebuild descriptive Q3/Q4 pairs from existing CSV; no simulation/imports."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.summary.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    keys = ('perturb_kind', 'logD_delta', 'region_width', 'C_cut', 'C_smooth')
    q3 = [r for r in rows if r['task'] == '2' and r['question'] == '3']
    q4 = [r for r in rows if r['task'] == '2' and r['question'] == '4']
    if len(q3) != 45 or len(q4) != 45:
        raise ValueError('This archived design requires 45 cases per question')
    output = []
    for right in q4:
        refs = [left for left in q3 if all(left[k] == right[k] for k in keys)]
        if len(refs) != 1:
            raise ValueError('Missing or duplicate Q3 counterpart')
        left = refs[0]
        if left['status'] != 'reached' or right['status'] != 'reached':
            raise ValueError('This table requires attained endpoints')
        a, b = float(left['tcritical_s']) / 3600, float(right['tcritical_s']) / 3600
        output.append([left['id'], right['id'], *[right[k] for k in keys], a, b, a-b])
    with args.output.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['q3_id', 'q4_id', *keys, 'q3_tcritical_h', 'q4_tcritical_h', 'q3_minus_q4_h'])
        writer.writerows(output)
    print('Wrote 45 descriptive pairs:', args.output)


if __name__ == '__main__':
    main()
