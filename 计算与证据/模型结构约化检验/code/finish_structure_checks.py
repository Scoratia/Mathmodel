from pathlib import Path
import argparse,json
from run_structure_test import get,audit,Settings,ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);a=ap.parse_args();cache=Path(a.cache)
    result={'cases':[]}
    for q in [3,4]:
        for eta in [0.,1.]:
            m=get(Settings(problem=q,n=160,internal_fraction=eta),cache)
            row={'q':q,'eta':eta,**audit(m)};result['cases'].append(row)
            (ROOT/'results/conservation_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
    assert all(r['water_relative_residual']<1e-5 and abs(r['energy_residual_J'])<1. for r in result['cases'])
    assert all(r['instant_water_error_kg_s']<1e-12 and r['instant_energy_error_W']<1e-7 for r in result['cases'])
    result['status']='PASS'
    (ROOT/'results/conservation_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
