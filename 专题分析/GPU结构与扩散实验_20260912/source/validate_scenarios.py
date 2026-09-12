"""Extra grid checks for the newly localized diffusivity perturbations."""
from pathlib import Path
import json
from cpu_solver import solve_case


def main():
    cases=[{'id':'q3_outer_thin','question':3,'moving_radius':False,'perturb_kind':'outer',
            'region_width':.05,'logD_delta':-.1},
           {'id':'q4_lowC_sharp','question':4,'moving_radius':True,'perturb_kind':'lowC',
            'C_cut':.2,'C_smooth':.01,'logD_delta':.1}]
    checks=[]
    for case in cases:
        coarse,_=solve_case(case,N=160);fine,_=solve_case(case,N=320)
        dt=abs(coarse['tcritical_s']-fine['tcritical_s']);dm=abs(coarse['tmean_s']-fine['tmean_s'])
        trials=[{'coarse_N':160,'fine_N':320,'delta_critical_s':dt,'delta_mean_s':dm,'passed':dt<2 and dm<2}]
        if not trials[-1]['passed']:
            coarse=fine;fine,_=solve_case(case,N=640)
            dt=abs(coarse['tcritical_s']-fine['tcritical_s']);dm=abs(coarse['tmean_s']-fine['tmean_s'])
            trials.append({'coarse_N':320,'fine_N':640,'delta_critical_s':dt,'delta_mean_s':dm,'passed':dt<2 and dm<2})
        checks.append({'case':case,'trials':trials,'supported_N':trials[-1]['coarse_N'],
                       'delta_critical_s':dt,'delta_mean_s':dm,'passed':trials[-1]['passed']})
        print(checks[-1],flush=True)
    result={'passed':all(c['passed'] for c in checks),'checks':checks,'scope':'Representative localization grid checks; not a uniform error bound for every full-suite case.'}
    out=Path('validation_cpu');out.mkdir(exist_ok=True)
    (out/'localization_grid_checks.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('Localization grid check failed')


if __name__=='__main__':main()
