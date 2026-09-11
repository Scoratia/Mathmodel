"""Second review: quantify closure assumptions; do not overwrite nominal answers.

Run with the same dependencies as A题建模成果. These are hypothetical stress
scenarios, not additional observations or calibrated confidence intervals.
"""
from pathlib import Path
import sys,json,time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'经验闭合对照'
sys.path.insert(0,str(BASE/'code'))
from model import Settings,RadialModel,properties


def main():
    baseline=json.loads((BASE/'results/validation.json').read_text(encoding='utf-8'))
    fields=np.load(BASE/'results/profiles_full_precision.npz')
    result={'description':'Conditional physical-closure audit; not a replacement solution or an award prediction.',
            'stress_scenarios':[],'density_interpretation':{},'published_precision':{}}
    # If the empirical rho(C) is interpreted literally as wet bulk density,
    # compute the dry-matter mass it implies. This is a diagnostic of that
    # interpretation, not the dry mass definition of the nominal effective model.
    for q in [3,4]:
        x=fields[f'q{q}_x'];tt=fields[f'q{q}_time_s'];r=fields[f'q{q}_radius_m']
        C=fields[f'q{q}_C_dry_basis'];T=fields[f'q{q}_T_C']
        rho=properties(T,C,q)[0]
        volume_mean_dry_rho=2*np.trapezoid(x[:,None]*rho/(1+C),x=x,axis=0)
        implied_mass=np.pi*r*r*.25*volume_mean_dry_rho
        ratio=implied_mass/implied_mass[0]
        result['density_interpretation'][str(q)]={
            'assumption':'rho(C) treated as actual wet bulk density, axial uniformity and L=0.25m',
            'initial_dry_mass_kg':float(implied_mass[0]),'minimum_mass_ratio':float(ratio.min()),
            'time_of_minimum_h':float(tt[ratio.argmin()]/3600),
            'end_mass_ratio':float(ratio[-1]),
            'meaning':'Deviation from 1 diagnoses incompatibility of this literal density interpretation with dry-matter conservation.'}
        last=[r for r in baseline['mesh'] if r['problem']==q][-1]
        result['published_precision'][str(q)]={'finest_grid_change_s':last['delta_time_s'],
            'baseline_h':baseline['crossing'][str(q)]['critical_h'],
            'latent_scenario_h':next(r['critical_time_h'] for r in baseline['sensitivity'] if r['problem']==q and r['case']=='latent fraction 1')}
    cases=[('boundary effective Ceq=0.10 after4h',{'tail_moist_shift':.10-baseline['input_audit']['tail_mean'][1]}),
           ('boundary effective Ceq=0.14 after4h',{'tail_moist_shift':.14-baseline['input_audit']['tail_mean'][1]}),
           ('joint D-10%, hm-20%, lateT-1C, lateC+0.01',
            {'d_scale':.9,'hm':6.4e-7,'tail_temp_shift':-1.,'tail_moist_shift':.01})]
    for q in [3,4]:
        base160=next(r['critical_time_h'] for r in baseline['mesh'] if r['problem']==q and r['n']==160)
        for label,kw in cases:
            tic=time.perf_counter()
            m=RadialModel(Settings(problem=q,n=160,**kw)).solve()
            row={'problem':q,'scenario':label,'settings':kw,'critical_h':m.end/3600,
                 'change_h_vs_same_grid':m.end/3600-base160,
                 'change_pct_vs_same_grid':100*(m.end/3600/base160-1),
                 'radius_extrapolation_needed':bool(q==4 and m.end>259200),
                 'minimum_C_at4h':float(m.state(14400)[m.m:].min()),
                 'wall_s':time.perf_counter()-tic}
            result['stress_scenarios'].append(row)
            print(json.dumps(row),flush=True)
            (HERE/'审查计算证据.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('AUDIT FINISHED',flush=True)


if __name__=='__main__':main()
