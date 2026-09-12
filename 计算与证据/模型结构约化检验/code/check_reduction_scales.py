"""Conditional vapor inventory/storage bounds and a falsification check for equilibrium."""
from pathlib import Path
import argparse,json
import numpy as np
from water_thermo import saturation,vapor_pressure_from_humidity_ratio
from reference_model import Environment,Settings,diffusivity

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--two-phase-results',required=True);args=ap.parse_args()
    env=Environment(Settings());Tmax=float(max(env.a[:,1].max(),env.tail[0]))
    p,dp,*_=saturation(Tmax);TK=Tmax+273.15;rv=p/(461.5*TK)
    drv=dp/(461.5*TK)-p/(461.5*TK**2)
    rows=[]
    for q in [3,4]:
        rd=(650+128*2.55)/3.55 if q==3 else (760+90*2.55)/3.55
        cs,cw=(1450.,4186.) if q==3 else (1850.,4000.)
        for C in [.05,.15,2.55]:
            rows.append({'q':q,'C':C,'vapor_mass_fraction_bound':rv/(rd*C),
                         'vapor_latent_capacity_fraction_bound':2.38e6*drv/(rd*(cs+cw*C)),
                         'assumption':'gas volume fraction<=1, aw<=1, aw temperature-independent for heat-capacity estimate, no vapor supersaturation, T<=Tmax, rho_d>=initial.'})
    prior=Path(args.two_phase_results)
    z=np.load(prior/'fields_coupled_fine.npz');f=z['values'];t=z['time_s'];r=z['r_m'];T,C,v=f[:,0],f[:,1],f[:,2]
    rd=(650+128*2.55)/3.55;eps=1-rd/1500;gas=eps-rd*C/1000.
    RH=float(vapor_pressure_from_humidity_ratio(env.tail[1])/saturation(env.tail[0])[0]);b=-.1*np.log(RH)
    ps=np.array([saturation(temp)[0] for temp in T])
    pv=v/gas*461.5*(T+273.15);peq=np.exp(-b/C)*ps
    deviation=np.abs(pv-peq)/ps
    weights=np.diff(np.linspace(0,.02,len(r)+1)**2);weights/=weights.sum()
    volmean=deviation@weights
    # Test whether the illustrative prior gas coefficient could share the supplied D budget.
    cc=np.geomspace(.05,2.55,200);temps=np.full_like(cc,50.);gg=eps-rd*cc/1000.
    a=np.exp(-b/cc);daw=a*b/cc**2
    Dv=8e-6*gg/(rd*461.5*(50+273.15))*saturation(50.)[0]*daw
    Dgiven=diffusivity(temps,cc,3)
    data={'Tmax_C':Tmax,'saturated_vapor_density_kg_m3':rv,'d_rho_vsat_dT':drv,'bounds':rows,
          'nonequilibrium_prior_fine':{'max_normalized_pressure_deficit':float(deviation.max()),
              'max_volume_mean_normalized_pressure_deficit':float(volmean.max()),
              'max_actual_vapor_fraction_of_water':float(np.max(v/(rd*C+v))),
              'definition':'|pv-aw psat|/psat, over saved prior illustrative coupled fields; inventory small does NOT imply equilibrium.'},
          'prior_coefficient_consistency':{'C_liquid':cc.tolist(),'D_vapor_equilibrium':Dv.tolist(),'D_given':Dgiven.tolist(),
                 'max_Dv_over_Dgiven':float(max(Dv/Dgiven)),
                 'meaning':'If this ratio exceeds 1, keeping these illustrative gas parameters and matching the given total D by adding a nonnegative liquid diffusivity is impossible at those states.'},
          'time_scale_note':'1/(ke*t_dry) alone does not establish local phase equilibrium. Need transport-scale Damkohler ke*ell^2/Dg and actual equilibrium departure; ell is local, not necessarily full radius.'}
    (ROOT/'results/reduction_scales.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in data.items() if k not in ['prior_coefficient_consistency','bounds']},ensure_ascii=False))

if __name__=='__main__':main()
