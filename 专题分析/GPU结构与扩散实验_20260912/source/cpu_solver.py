"""Independent adaptive SciPy BDF reference for shared task 1/2 equations."""
import time
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.sparse import lil_matrix
from physics import (Inputs,make_grid,prepare_cases,rhs,diagnostics,geometry,
                     perturbation_mask,T0,C0,THRESHOLD)


def horizon(case,inputs):
    return min(72*3600.,inputs.radius_end) if case.get('moving_radius',case['question']==4) else (168*3600. if case['question']==4 else 72*3600.)


def solve_case(case,N=320,rtol=2e-8,atol=2e-10,t_end=None,inputs=None,profile_count=401):
    started=time.perf_counter();inputs=inputs or Inputs();g=make_grid(N);b=prepare_cases([case]);n=N+1
    end=horizon(case,inputs) if t_end is None else float(t_end)
    if case.get('moving_radius',case['question']==4) and end>inputs.radius_end:raise ValueError('Radius horizon exceeded')
    y0=np.empty((1,n,2));y0[...,0]=T0;y0[...,1]=C0
    sparsity=lil_matrix((2*n,2*n),dtype=int)
    for i in range(n):sparsity[2*i:2*i+2,2*max(i-1,0):2*min(i+2,n)]=1
    sparsity=sparsity.tocsc()
    segments=[];critical=mean_hit=None
    def event_max(t,y):return y.reshape(n,2)[:,1].max()-THRESHOLD
    def event_mean(t,y):return 2*np.dot(g['weights'],y.reshape(n,2)[:,1])-THRESHOLD
    event_max.direction=-1;event_max.terminal=True
    event_mean.direction=-1;event_mean.terminal=False
    current=0.;state=y0.ravel()
    while current<end:
        phase='observed' if current<inputs.boundary_end else 'plateau'
        bound=min(end,inputs.boundary_end) if phase=='observed' else end
        fun=lambda t,y:rhs(t,y.reshape(1,n,2),b,g,inputs,np,phase).ravel()
        sol=solve_ivp(fun,(current,bound),state,method='BDF',rtol=rtol,atol=atol,
                      jac_sparsity=sparsity,dense_output=True,first_step=min(1e-4,bound-current),
                      max_step=30. if phase=='observed' else 300.,events=(event_max,event_mean))
        if not sol.success:raise RuntimeError(sol.message)
        if np.any(sol.y.reshape(n,2,-1)[:,1,:]<=0):raise RuntimeError('Nonpositive accepted moisture')
        segments.append((current,float(sol.t[-1]),sol,phase))
        if len(sol.t_events[1]) and mean_hit is None:mean_hit=float(sol.t_events[1][0])
        current=float(sol.t[-1]);state=sol.y[:,-1]
        if len(sol.t_events[0]):critical=float(sol.t_events[0][0]);break
    actual_end=current
    def evaluate(ts):
        ts=np.asarray(ts);out=np.empty((len(ts),n,2));assigned=np.zeros(len(ts),bool)
        for start,stop,sol,phase in segments:
            mask=(ts>=start)&(ts<=stop)&~assigned
            if mask.any():out[mask]=sol.sol(ts[mask]).T.reshape(-1,n,2);assigned[mask]=True
        if not assigned.all():raise ValueError('Requested samples outside integrated interval')
        return out
    ts=np.unique(np.r_[0.,np.geomspace(.001,min(actual_end,14400.),100),
                        np.linspace(0,actual_end,profile_count),
                        [v for v in (mean_hit,critical,inputs.boundary_end) if v is not None and v<=actual_end]])
    ys=evaluate(ts);diags={};masks=[];radii=[]
    for t,y in zip(ts,ys):
        d=diagnostics(float(t),y[None,:,:],b,g,inputs)
        for key,val in d.items():diags.setdefault(key,[]).append(float(val[0]))
        masks.append(perturbation_mask(g['a'][None,:],y[None,:,1],b)[0])
        radii.append(geometry(float(t),b,g,inputs)['r'][0])
    diags={k:np.asarray(v) for k,v in diags.items()};masks=np.asarray(masks)
    # Independent quadrature of boundary flux, split exactly at the observed /
    # plateau discontinuity. Two audit resolutions expose quadrature sensitivity.
    audits=[]
    for resolution in (1000,2000):
        integral=0.
        for start,stop,sol,phase in segments:
            at=np.unique(np.r_[np.linspace(start,stop,resolution+1),
                                np.geomspace(.0001,min(stop,100),200) if start==0 else []])
            ay=sol.sol(at).T.reshape(-1,n,2)
            rates=np.array([diagnostics(float(t),y[None],b,g,inputs,phase=phase)['outflow_kg_s'][0] for t,y in zip(at,ay)])
            integral+=float(trapezoid(rates,at))
        initial=diags['water_mass_kg'][0];remaining=diags['water_mass_kg'][-1]
        audits.append((remaining-initial+integral)/initial)
    support=2*np.sum(g['weights'][None,:]*(masks>.5),axis=1)
    row={**case,'backend':'cpu_bdf','N':N,'rtol':rtol,'atol':atol,
         'status':'reached' if critical is not None else 'not_reached_by_horizon',
         'tcritical_s':critical,'tmean_s':mean_hit,
         'tail_s':None if critical is None or mean_hit is None else critical-mean_hit,
         't_end_s':actual_end,'horizon_s':end,'mass_audit_rel':audits[-1],
         'mass_audit_coarse_rel':audits[0],'mass_audit_refinement_difference':abs(audits[1]-audits[0]),
         'max_C_end':diags['max_C'][-1],'mean_C_end':diags['mean_C'][-1],
         'initial_dry_mass_kg':diags['dry_mass_kg'][0],
         'empirical_dry_mass_ratio_end':diags['empirical_dry_mass_ratio'][-1],
         'perturb_support_dry_mass_fraction':float(trapezoid(support,ts)/actual_end),
         'perturb_weighted_dry_mass_fraction':float(trapezoid(diags['mask_dry_mass_fraction'],ts)/actual_end),
         'wall_s':time.perf_counter()-started,
         'nfev':sum(s[2].nfev for s in segments),'nlu':sum(s[2].nlu for s in segments)}
    profile={'t_s':ts,'a':g['a'],'x':g['a'],'weights':g['weights'],
             'dry_mass_weights':2*g['weights'],'T':ys[...,0],'C':ys[...,1],
             'mask':masks,'zone_weight':masks,'r_m':np.asarray(radii),**diags}
    return row,profile
