"""Targeted validation for the new geometry and mass-flux definitions."""
from pathlib import Path
import sys,json
import numpy as np
from physics import *
from cpu_solver import solve_case


def run(out_dir='validation_cpu'):
    out=Path(out_dir);out.mkdir(exist_ok=True,parents=True)
    inp=Inputs();rows=[]
    sys.path.insert(0,str(Path(__file__).parent/'reference'))
    from baseline_model import RadialModel,ModelInputs,simulate
    oldinp=ModelInputs(inp.boundary_time,inp.ambient_temperature,inp.ambient_moisture,inp.radius_time,inp.radius_values)
    for q in (3,4):
        c={'id':f'q{q}','question':q,'moving_radius':q==4}
        g=make_grid(80);b=prepare_cases([c]);a=g['a'];y=np.stack((35+3*a*a,1-.3*a*a),axis=-1)[None]
        old=RadialModel(q,oldinp,N=80,gauss_order=5,grid_kind='surface_clustered')
        actual=rhs(10000,y,b,g,inp)[0]
        ref=old.rhs(10000,np.r_[y[0,:,0],y[0,:,1]]).reshape(2,-1).T
        err=float(np.max(abs(actual-ref)));rows.append({'check':f'affine_rhs_q{q}','error':err,'passed':err<1e-8})
    for eps in (-.15,0,.15):
        for norm in ('current_dry_density','initial_dry_density'):
            c={'question':4,'eps_final':eps,'boundary_norm':norm}
            g=make_grid(160);b=prepare_cases([c]);y=np.stack((35+3*g['a']**2,1-.3*g['a']**2),axis=-1)[None]
            t=180000.;geo=geometry(t,b,g,inp);dy=rhs(t,y,b,g,inp);d=diagnostics(t,y,b,g,inp)
            residual=float(abs(2*np.dot(g['weights'],dy[0,:,1])+d['outflow_kg_s'][0]/d['dry_mass_kg'][0]))
            rows.append({'check':f'mass_identity_eps{eps}_{norm}','error':residual,'passed':residual<1e-12})
            assert np.all(geo['ga']>0) and np.all(geo['volume']>0)
    # Interior differential operator consistency for a smooth non-affine field.
    errors=[]
    for N in (40,80,160):
        g=make_grid(N,power=1);c={'question':4,'eps_final':.15};b=prepare_cases([c]);a=g['a'];t=180000.
        amp=.1;C=.8+amp*a*a;T=np.full_like(a,40.);y=np.stack((T,C),axis=-1)[None]
        geo=geometry(t,b,g,inp);R=geo['R'][0,0];eps=geo['eps'][0,0];ga=geo['ga'][0]
        D=properties(T,C,np.full_like(a,4))[3]
        exact=(4*amp*D+4*amp*amp*a*a*D*.3/C**2+24*amp*eps*a*a*D/ga)/(R*R*ga*ga)
        errors.append(float(np.max(abs(rhs(t,y,b,g,inp)[0,3:-3,1]-exact[3:-3]))))
    order=float(np.log(errors[-2]/errors[-1])/np.log(2))
    rows.append({'check':'nonaffine_operator_order','errors':errors,'order':order,'passed':order>1.8})
    # Full trajectory regression plus representative geometry grid refinements.
    for q in (3,4):
        c={'id':f'baseline_q{q}','task':0,'question':q,'moving_radius':q==4}
        row,prof=solve_case(c,N=160,inputs=inp)
        old=simulate(q,oldinp,N=160,gauss_order=5,grid_kind='surface_clustered',rtol=2e-8,atol=2e-10)
        dt=abs(row['tcritical_s']-old.t_critical)
        rows.append({'check':f'affine_endpoint_q{q}','error_s':dt,'passed':dt<.2})
        np.savez_compressed(out/f'baseline_q{q}.npz',**prof)
    for eps,norm in ((.15,'current_dry_density'),(-.15,'initial_dry_density')):
        c={'id':f'grid_eps{eps}_{norm}','question':4,'moving_radius':True,'eps_final':eps,'boundary_norm':norm}
        coarse,p=solve_case(c,N=160,inputs=inp);fine,p=solve_case(c,N=320,inputs=inp)
        if coarse['tcritical_s'] is None or fine['tcritical_s'] is None:
            dc=abs(coarse['max_C_end']-fine['max_C_end']);passed=coarse['status']==fine['status'] and dc<1e-4;dt=None
        else:dt=abs(coarse['tcritical_s']-fine['tcritical_s']);dc=None;passed=dt<2.
        rows.append({'check':c['id'],'delta_time_s':dt,'delta_max_C':dc,'passed':passed})
    result={'passed':all(r['passed'] for r in rows),'checks':rows,
            'scope':'New equations and selected trajectories; not experimental validation; no CUDA execution.'}
    (out/'checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit('Validation failed')
    return result


if __name__=='__main__':run()
