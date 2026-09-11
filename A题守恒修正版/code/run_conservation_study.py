"""Reproduce the corrected model, ablations, conservation and convergence checks."""
import argparse, hashlib, json, pickle, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq
from scipy.special import j0,j1
from conservative_model import ConservativeSettings as Settings, ConservativeModel as Model
from reference_model import properties

ROOT=Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def get_model(p, cache):
    signature=b''.join((ROOT/'code'/f).read_bytes() for f in ['conservative_model.py','reference_model.py'])
    key=hashlib.sha256(signature+json.dumps(asdict(p),sort_keys=True).encode()).hexdigest()[:24]
    file=cache/(key+'.pkl')
    tic=time.perf_counter()
    if file.exists():
        with file.open('rb') as f: m=pickle.load(f)
    else:
        m=Model(p).solve()
        with file.open('wb') as f: pickle.dump(m,f)
    print(json.dumps({'q':p.problem,'n':p.n,'closure':p.thermal_closure,'latent_fraction':p.latent_fraction,
                      'method':p.method,'h':m.end/3600,'wall_s':round(time.perf_counter()-tic,2)}),flush=True)
    return m


def balance_checks(m):
    dry_errors=[]; local_errors=[]; ode_water=[]; ode_energy=[]
    for t in np.linspace(0,m.end,101):
        y=m.state(t);T,C=y[:m.m],y[m.m:]
        rd,rw=m.densities(t,C);V=m.volumes(t)
        reconstructed=rw/(1+C)*V
        dry_errors.append(abs(reconstructed.sum()/m.Md-1))
        local_errors.append(float(np.max(abs(reconstructed/m.dry_mass-1))))
        der=m.rhs(t,y);Fw,Q,Ew,_=m.fluxes(t,y)
        ode_water.append(abs(m.dry_mass@der[m.m:]+Fw[-1]))
        edot=m.dry_mass@((m.cs+m.cw*C)*der[:m.m]+m.cw*T*der[m.m:])
        ode_energy.append(abs(edot+Q[-1]+Ew[-1]))
    checks=[]
    for step in [10.,5.]:
        integrals=np.zeros(3)
        ends=[0.,14400.,m.end] if m.end>14400 else [0.,m.end]
        for lo,hi in zip(ends[:-1],ends[1:]):
            ts=np.unique(np.r_[np.arange(lo,hi,step),hi]); fs=[]
            for start in range(0,len(ts),256):
                tc=ts[start:start+256];yy=m.state(tc)
                for t,y in zip(tc,yy.T):
                    Fw,Q,Ew,_=m.fluxes(t+1e-7 if t==14400 and lo==14400 else t,y)
                    energy_flux=Q[-1]+Ew[-1]
                    fs.append([Fw[-1],energy_flux,abs(energy_flux)])
            integrals+=simpson(np.array(fs),x=ts,axis=0)
        water0=m.masses(0)[1];water1=m.masses(m.end)[1]
        water_error=float(water1-water0+integrals[0])
        E0=m.sensible_energy(0);E1=m.sensible_energy(m.end)
        energy_error=float(E1-E0+integrals[1])
        checks.append({'sampling_step_s':step,'outward_water_kg':float(integrals[0]),
                       'water_residual_kg':water_error,'relative_water_residual':abs(water_error)/water0,
                       'outward_sensible_energy_J':float(integrals[1]),'energy_residual_J':energy_error,
                       'relative_energy_residual':abs(energy_error)/max(abs(E0),integrals[2],1.)})
    ts=np.linspace(0,m.end,301); yy=m.state(ts);C=yy[m.m:];T=yy[:m.m]
    return {'dry_mass_kg':m.Md,'water_initial_kg':m.masses(0)[1],'water_end_kg':m.masses(m.end)[1],
            'wet_mass_end_kg':m.masses(m.end)[2],'max_global_dry_mass_relative_error':max(dry_errors),
            'max_cell_dry_mass_relative_error':max(local_errors),'max_water_rate_residual_kg_s':max(ode_water),
            'max_energy_rate_residual_W':max(ode_energy),'independent_time_integrals':checks,
            'C_min':float(C.min()),'C_max':float(C.max()),'T_min':float(T.min()),'T_max':float(T.max()),
            'max_radial_C_increase':float(np.max(np.diff(C,axis=0)))}


def heat_benchmark(n):
    m=Model(Settings(problem=1,n=n,hm=0,rtol=2e-10,atol=2e-12,max_step=30))
    m.env.ambient=lambda t:(50.,2.55)
    m.solve(end=1800,event=False)
    Bi=25*.02/.36;alpha=.36/(820*2600);roots=[]
    f=lambda z:z*j1(z)-Bi*j0(z)
    scan=np.linspace(.001,150,20000)
    for a,b in zip(scan[:-1],scan[1:]):
        if f(a)*f(b)<0:roots.append(brentq(f,a,b))
        if len(roots)==40:break
    lam=np.array(roots);coef=2*j1(lam)/(lam*(j0(lam)**2+j1(lam)**2))
    ts=[100.,300.,600.,1800.]
    exact=np.array([50-22*(coef*np.exp(-lam**2*alpha*t/.02**2))@j0(lam[:,None]*m.x) for t in ts]).T
    return {'n':n,'max_temperature_error_C':float(np.max(abs(m.state(ts)[:m.m]-exact)))}


def sealed_shrinkage():
    m=Model(Settings(problem=4,n=32,hm=0,h=0)).solve(end=72*3600,event=False)
    yy=m.state(np.linspace(0,m.end,101))
    return {'radius_ratio':float(m.env.radius(m.end)/m.radius0),
            'dry_density_ratio':float(m.densities(m.end,yy[m.m:,-1])[0][0]/m.rho_d0),
            'max_C_change':float(np.max(abs(yy[m.m:]-2.55))),
            'max_T_change_C':float(np.max(abs(yy[:m.m]-28))),
            'water_mass_relative_change':float(m.masses(m.end)[1]/m.masses(0)[1]-1)}


def extend(m,end):
    critical=m.end
    sol=solve_ivp(m.rhs,(critical,end),m.state(critical),method=m.p.method,rtol=m.p.rtol,atol=m.p.atol,
                  dense_output=True,jac_sparsity=m.sparsity,max_step=10.)
    if not sol.success:raise RuntimeError(sol.message)
    m.parts.append(sol);m.end=end


def payload(m, ts, field='C', shrink=False):
    rr=np.round(np.arange(0,2.0001,.1),1)
    rows=[['时间\\到药材中心的距离']+rr.tolist()+(['药材表面','当前半径/cm'] if shrink else [])]
    for t in ts:
        vals=m.profile(t,rr,field)
        row=[int(t)]+[None if np.isnan(v) else round(float(v),4) for v in vals]
        if shrink:row += [round(float(m.state(t)[-1]),4),round(float(m.env.radius(t))*100,6)]
        rows.append(row)
    return rows


def export_grid_comparison(m, prev):
    """Compare every requested export time/radius on the two finest meshes."""
    q=m.p.problem
    saved=[(model,model.end,len(model.parts)) for model in [m,prev]]
    if q!=1:
        # Include the exported first strict full-minute row, beyond the event.
        common_end=float((np.floor(max(m.end,prev.end)/60)+1)*60)
        for model in [m,prev]:extend(model,common_end)
    if q==1:ts=np.arange(1,1801.)
    elif q==3:ts=np.unique(np.r_[np.arange(1,10801.),np.arange(60,m.end+1,60)])
    else:ts=np.arange(60,m.end+1,60)
    maxima=np.zeros(2);rr=np.arange(0,2.0001,.1)/100
    for start in range(0,len(ts),128):
        tc=ts[start:start+128];vals=[]
        for model in [m,prev]:
            yy=model.state(tc);profiles=[]
            for j,t in enumerate(tc):
                R=float(model.env.radius(t));xp=rr/R;valid=xp<=1+1e-12
                profile=np.stack([np.where(valid,np.interp(xp,model.x,yy[k*model.m:(k+1)*model.m,j]),np.nan) for k in [0,1]])
                if q==4:profile=np.column_stack([profile,[yy[model.m-1,j],yy[-1,j]]])
                profiles.append(profile)
            vals.append(np.asarray(profiles))
        diff=abs(vals[0]-vals[1])
        maxima=np.maximum(maxima,np.nanmax(diff,axis=(0,2)))
    for model,end,count in saved:
        model.parts=model.parts[:count];model.end=end
    return {'times_checked':len(ts),'includes_strict_full_minute':q!=1,'includes_actual_moving_surface':q==4,
            'max_T_difference_C':float(maxima[0]),'max_C_difference':float(maxima[1])}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    results=ROOT/'results';results.mkdir(exist_ok=True)
    study={'model':'conserved dry material cells, conservative water and sensible enthalpy, no explicit latent heat',
           'mesh':[],'balance':{},'ablations':[],'analytic':[],'crossing':{},'solver_comparison':[]}
    models={};meshes=[80,160] if args.quick else [80,160,320,640,1280]
    for q in [1,3,4]:
        prev=None
        for n in meshes:
            m=get_model(Settings(problem=q,n=n),cache)
            row={'problem':q,'n':n,'time_h':m.end/3600}
            if prev is not None:
                early=[60.,120.] if q==4 else [1.,2.,5.,10.,20.,30.,60.]
                ts=np.unique(np.r_[np.linspace(0,min(m.end,prev.end),31),early,[100.,300.,600.,900.,1200.,1500.,1800.],
                                   np.arange(3600.,10801.,1800.) if q!=1 else []])
                ds=[]
                for t in ts:
                    rr=np.linspace(0,float(m.env.radius(t))*100,21)
                    ds.append([max(abs(m.profile(t,rr,k)-prev.profile(t,rr,k))) for k in ['T','C']])
                row.update(delta_time_s=m.end-prev.end,max_T_delta_C=float(np.max(ds,axis=0)[0]),
                           max_C_delta=float(np.max(ds,axis=0)[1]))
                if n==meshes[-1]:row['all_export_points']=export_grid_comparison(m,prev)
            study['mesh'].append(row);prev=m
            save(results/'validation.json',study)
        models[q]=m
        study['balance'][str(q)]=balance_checks(m)
        print('BALANCE',q,json.dumps(study['balance'][str(q)]),flush=True)
        save(results/'validation.json',study)
    for n in ([80] if args.quick else [80,320,1280]):study['analytic'].append(heat_benchmark(n))
    study['sealed_shrinkage']=sealed_shrinkage()
    for q in [3,4]:
        nominal=get_model(Settings(problem=q,n=160),cache)
        cases=[('empirical thermal density, dry-mass accounting only',{'thermal_closure':'empirical'}),
               ('conserved wet density, heat capacity only',{'thermal_closure':'capacity'})]
        if not args.quick:cases += [('conservative surface latent heat sensitivity',{'latent_fraction':1.})]
        for label,kw in cases:
            alt=get_model(Settings(problem=q,n=160,**kw),cache)
            study['ablations'].append({'problem':q,'case':label,'time_h':alt.end/3600,
                                       'delta_vs_selected_s':alt.end-nominal.end,'settings':kw})
            save(results/'validation.json',study)
        if not args.quick:
            alt=get_model(Settings(problem=q,n=160,method='Radau',rtol=5e-9,atol=5e-11,max_step=300),cache)
            study['solver_comparison'].append({'problem':q,'BDF_h':nominal.end/3600,'Radau_h':alt.end/3600,
                                              'difference_s':alt.end-nominal.end})
    fields={}
    for q,m in models.items():
        critical=m.end
        if q!=1:
            minute=float((np.floor(critical/60)+1)*60)
            mean_time=brentq(lambda t:float(m.mean_C(t))-.15,0,critical)
            study['crossing'][str(q)]={'critical_s':critical,'critical_h':critical/3600,
                    'mean_C':float(m.mean_C(critical)),'surface_C':float(m.state(critical)[-1]),
                    'mean_threshold_h':mean_time/3600,'premature_stop_h':(critical-mean_time)/3600,
                    'first_full_minute_s':minute}
            extend(m,minute)
            study['crossing'][str(q)]['max_C_first_full_minute']=float(np.max(m.state(minute)[m.m:]))
        ts=np.unique(np.r_[np.linspace(0,m.end,501),1800.,10800. if q!=1 else 1800.])
        yy=m.state(ts)
        rd=[];rw=[]
        for t,C in zip(ts,yy[m.m:].T):
            a,b=m.densities(t,C);rd.append(a);rw.append(b)
        for name,value in [('time_s',ts),('x',m.x),('T_C',yy[:m.m]),('C_dry_basis',yy[m.m:]),
                           ('radius_m',np.array([m.env.radius(t) for t in ts])),('dry_density',np.array(rd).T),
                           ('wet_density',np.array(rw).T),('dry_cell_mass_kg',m.dry_mass)]:fields[f'q{q}_{name}']=value
        rows=[]
        for t in np.unique(np.r_[np.arange(0,m.end,3600.),m.end]):
            md,mw,mt=m.masses(t)
            rows.append([float(t),md,float(mw),float(mt),float(m.env.radius(t)),float(m.mean_C(t))])
        save(results/f'mass_history_q{q}.json',{'columns':['time_s','dry_mass_kg','water_mass_kg','wet_mass_kg','radius_m','mean_C'],'rows':rows})
    np.savez_compressed(results/'profiles_full_precision.npz',**fields)
    books={
        'result1':{'温度':payload(models[1],range(1,1801),'T'),'水分浓度':payload(models[1],range(1,1801),'C')},
        'result2':{'温度':payload(models[3],range(1,10801),'T'),'水分浓度':payload(models[3],range(1,10801),'C')},
        'result3':{'Sheet1':payload(models[3],np.arange(60,models[3].end+1,60))},
        'result4':{'Sheet1':payload(models[4],np.arange(60,models[4].end+1,60),shrink=True)}}
    for name,book in books.items():save(results/(name+'.json'),book)
    save(results/'validation.json',study)
    print('FINISHED',json.dumps(study['crossing']),flush=True)


if __name__=='__main__':main()
