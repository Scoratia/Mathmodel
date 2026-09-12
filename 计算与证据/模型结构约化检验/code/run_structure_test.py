from pathlib import Path
from dataclasses import asdict
import json,hashlib,pickle,argparse
import numpy as np
from scipy.integrate import simpson
from redistribution_model import Settings,RedistributionModel
from reference_model import diffusivity
from water_thermo import saturation

ROOT=Path(__file__).resolve().parents[1]
def save(obj):
    (ROOT/'results/study.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def get(p,cache):
    code=b''.join((ROOT/'code'/s).read_bytes() for s in ['redistribution_model.py','conservative_model.py','reference_model.py'])
    code+=(ROOT/'data/inputs.json').read_bytes()
    key=hashlib.sha256(code+json.dumps(asdict(p),sort_keys=True).encode()).hexdigest()[:24]
    f=cache/(key+'.pkl')
    if f.exists():
        with f.open('rb') as stream:m=pickle.load(stream)
    else:
        m=RedistributionModel(p).solve()
        with f.open('wb') as stream:pickle.dump(m,stream)
    print(json.dumps({'q':p.problem,'n':p.n,'eta':p.internal_fraction,'shape':p.shape_power,'time_h':m.end/3600}),flush=True)
    return m

def compare(m,b):
    ts=np.unique(np.r_[0,np.geomspace(.01,60,30),np.linspace(60,14400,100),np.linspace(14460,min(m.end,b.end),160)])
    maxT=maxC=0.;rows=[]
    for t in ts:
        y=m.state(t);z=b.state(t);T,C=y[:m.m],y[m.m:]
        Tb=np.interp(m.x,b.x,z[:b.m]);Cb=np.interp(m.x,b.x,z[b.m:])
        maxT=max(maxT,float(max(abs(T-Tb))));maxC=max(maxC,float(max(abs(C-Cb))))
        rows.append([float(t),float(T[0]),float(T[-1]),float(C[0]),float(C[-1]),float(m.mean_C(t)),float(Tb[0]),float(Tb[-1])])
    return {'end_h':m.end/3600,'reference_h':b.end/3600,'change_h':(m.end-b.end)/3600,'change_pct':100*(m.end/b.end-1),
            'max_sampled_T_difference_C':maxT,'max_sampled_C_difference':maxC},rows

def audit(m):
    worstM=worstE=0.;stored=[]
    for t in np.linspace(0,m.end,151):
        y=m.state(t);T,C=y[:m.m],y[m.m:];der=m.rhs(t,y);F,Q,E,_=m.fluxes(t,y)
        dE=m.dry_mass@((m.cs+m.cw*C)*der[:m.m]+m.cw*T*der[m.m:])
        worstM=max(worstM,abs(m.dry_mass@der[m.m:]+F[-1]));worstE=max(worstE,abs(dE+Q[-1]+E[-1]))
    # Composite independent Gauss integration across accepted step intervals and ambient knots.
    from numpy.polynomial.legendre import leggauss
    nodes,weights=leggauss(3);integ=np.zeros(2)
    # Reference solver stores individual solve_ivp objects in .parts.
    for sol in m.parts:
        knots=np.arange(0,14401,60.)
        edges=np.unique(np.r_[sol.t,knots[(knots>=sol.t[0])&(knots<=sol.t[-1])]])
        for lo,hi in zip(edges[:-1],edges[1:]):
            for x,w in zip(nodes,weights):
                t=(lo+hi)/2+(hi-lo)*x/2;y=sol.sol(t);F,Q,E,_=m.fluxes(t,y)
                integ+=w*(hi-lo)/2*np.array([F[-1],Q[-1]+E[-1]])
    w0=m.masses(0)[1];we=m.masses(m.end)[1];h0=m.sensible_energy(0);he=m.sensible_energy(m.end)
    return {'instant_water_error_kg_s':worstM,'instant_energy_error_W':worstE,
            'water_relative_residual':float(abs(we-w0+integ[0])/w0),'energy_residual_J':float(he-h0+integ[1])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);args=ap.parse_args();cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    result={'status':'RUNNING','cases':[],'mesh':[],'bounds':{}}
    for q in [3,4]:
        b=get(Settings(problem=q,n=160,internal_fraction=0),cache)
        for eta in [0.,.25,.5,.75,1.]:
            m=get(Settings(problem=q,n=160,internal_fraction=eta),cache)
            row,history=compare(m,b);row.update({'q':q,'n':160,'eta':eta,'shape_power':0})
            result['cases'].append(row)
            (ROOT/'results'/f'history_q{q}_eta{eta:g}.json').write_text(json.dumps({'columns':['time_s','center_T','surface_T','center_C','surface_C','mean_C','reference_center_T','reference_surface_T'],'rows':history}),encoding='utf-8')
            save(result)
        for pwr in [1.,2.]:
            m=get(Settings(problem=q,n=160,internal_fraction=1,shape_power=pwr),cache)
            row,_=compare(m,b);row.update({'q':q,'n':160,'eta':1.,'shape_power':pwr})
            result['cases'].append(row);save(result)
        # Coarse-to-fine verification of the two endpoints, not a claim of engineering accuracy.
        for eta in [0.,1.]:
            m=get(Settings(problem=q,n=320,internal_fraction=eta),cache)
            coarse=get(Settings(problem=q,n=160,internal_fraction=eta),cache)
            row,_=compare(m,coarse);row.update({'q':q,'eta':eta,'comparison':'N320 minus N160'})
            result['mesh'].append(row);save(result)
        rd0=m.rho_d0
        Tmax=max(m.env.a[:,1].max(),m.env.tail[0])
        pmax=saturation(Tmax)[0];rvmax=pmax/(461.5*(Tmax+273.15))
        result['bounds'][str(q)]={'temperature_assumption_max_C':float(Tmax),'saturated_vapor_density_kg_m3':float(rvmax),
            'minimum_dry_density_kg_m3':rd0,'vapor_to_water_at_C015_upper_bound':rvmax/(rd0*.15),
            'vapor_to_water_at_C005_upper_bound':rvmax/(rd0*.05),
            'assumptions':'epsilon_g<=1, vapor not supersaturated, T<=maximum supplied air T, dry density at least its initial value. Inventory bound, not vapor flux bound.'}
        save(result)
    result['status']='COMPUTED';save(result)
    print('COMPUTATION_DONE',flush=True)

if __name__=='__main__':main()
