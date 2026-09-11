"""Question 1: D(C) has no temperature dependence; isolate its thermal response."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from run_conservation_study import get_model,save
from conservative_model import ConservativeSettings as Settings


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);args=ap.parse_args()
    b=get_model(Settings(problem=1,n=1280),Path(args.cache))
    m=get_model(Settings(problem=1,n=1280,latent_fraction=1.),Path(args.cache))
    ts=np.arange(0,1801.);a=m.state(ts);z=b.state(ts)
    out={'problem':1,'n':1280,'time_s':1800,
        'without_latent_center_T_C':float(z[0,-1]),'with_latent_center_T_C':float(a[0,-1]),
        'without_latent_surface_T_C':float(z[b.m-1,-1]),'with_latent_surface_T_C':float(a[m.m-1,-1]),
        'max_C_difference':float(np.max(abs(a[m.m:]-z[b.m:]))),
        'minimum_T_C':float(a[:m.m].min()),
        'interpretation':'D(C) independent of T in appendix 2; the moisture equation is unchanged by latent heat coupling.'}
    assert out['max_C_difference']<1e-6
    save(Path(__file__).resolve().parents[1]/'results/question1_comparison.json',out)
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
