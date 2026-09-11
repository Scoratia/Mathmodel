"""Solve all four questions; write numerical evidence and spreadsheet payloads.

Usage: python run_study.py [--quick] [--cache PATH]
Full study uses mesh refinement and sensitivity runs; --quick only checks a coarse solution.
"""
import argparse, json, pickle, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import brentq, least_squares
from scipy.special import j0,j1
from scipy.sparse import diags
from model import ROOT, Settings, RadialModel, Environment, diffusivity, properties


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def get_model(p,cache):
    key='_'.join(f'{k}-{v}' for k,v in asdict(p).items() if v!=getattr(Settings(),k)) or 'default'
    import hashlib
    f=cache/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.pkl')
    if f.exists():
        with f.open('rb') as stream:m=pickle.load(stream)
        print('cached',p.problem,p.n,round(m.end/3600,7),flush=True)
        return m
    tick=time.perf_counter()
    m=RadialModel(p).solve()
    with f.open('wb') as stream:pickle.dump(m,stream)
    print('solved',p.problem,p.n,round(m.end/3600,7),'wall',round(time.perf_counter()-tick,2),flush=True)
    return m


def constant_benchmark(n):
    """Independent Robin-Bessel solution for a cylinder, compared with FV heat equation."""
    R=.02; D=.36/(820*2600); h=25.; k=.36; Bi=h*R/k
    roots=[]
    scan=np.linspace(.001,150,20000)
    fun=lambda z:z*j1(z)-Bi*j0(z)
    for a,b in zip(scan[:-1],scan[1:]):
        if fun(a)*fun(b)<0:roots.append(brentq(fun,a,b))
        if len(roots)==40:break
    roots=np.array(roots)
    coeff=2*j1(roots)/(roots*(j0(roots)**2+j1(roots)**2))
    x=np.linspace(0,1,n+1); f=(x[1:]+x[:-1])/2
    w=np.diff(np.r_[0,f,1]**2)/2
    def rhs(t,y):
        q=f*D*np.diff(y)*n
        return np.diff(np.r_[0,q,-h/(820*2600)*R*y[-1]])/(R*R*w)
    ts=np.array([100.,300.,600.,1800.])
    s=solve_ivp(rhs,(0,ts[-1]),np.ones(n+1),method='BDF',t_eval=ts,
                rtol=1e-10,atol=1e-12,jac_sparsity=diags([np.ones(n),np.ones(n+1),np.ones(n)],[-1,0,1]))
    exact=np.array([(coeff*np.exp(-roots**2*D*t/R**2))@j0(roots[:,None]*x) for t in ts]).T
    return {'n':n,'Bi_heat':Bi,'max_dimensionless_error':float(np.max(np.abs(s.y-exact))),
            'equivalent_22K_step_error_C':float(22*np.max(np.abs(s.y-exact)))}


def balance_and_bounds(m):
    ts=np.unique(np.r_[np.arange(0,m.end,10.),m.end,14400.] if m.end>14400 else np.r_[np.arange(0,m.end,2.),m.end])
    ys=m.state(ts)
    C=ys[m.m:];T=ys[:m.m]
    mean=2*m.w@C
    flux=np.array([2*m.p.hm/float(m.env.radius(t))*(c-m.env.ambient(t)[1]) for t,c in zip(ts,C[-1])])
    # Composite trapezoids on independently sampled boundary flux. Refine to assess quadrature error.
    from scipy.integrate import trapezoid
    loss=trapezoid(flux,ts)
    residual=mean[-1]-2.55+loss
    max_ode_balance=0.
    for t in np.linspace(0,m.end,41):
        y=m.state(t)
        der=m.rhs(t,y)[m.m:]
        outward=2*m.p.hm/float(m.env.radius(t))*(y[-1]-m.env.ambient(t)[1])
        max_ode_balance=max(max_ode_balance,abs(2*m.w@der+outward))
    return {'C_min':float(C.min()),'C_max':float(C.max()),'T_min_C':float(T.min()),'T_max_C':float(T.max()),
            'max_radial_C_increase':float(np.max(np.diff(C,axis=0))),
            'mean_C_end':float(mean[-1]),'integrated_normalized_moisture_loss':float(loss),
            'independent_quadrature_balance_error':float(residual),
            'relative_to_initial_C':float(abs(residual)/2.55),
            'semidiscrete_balance_residual_per_s':max_ode_balance}


def extend(m,end):
    """Continue to the first full minute after critical crossing; preserve the crossing separately."""
    m.critical_time=m.end
    if end<=m.end:return
    s=solve_ivp(m.rhs,(m.end,end),m.state(m.end),method=m.p.method,
                rtol=m.p.rtol,atol=m.p.atol,jac_sparsity=m.sparsity,dense_output=True,max_step=10.)
    if not s.success:raise RuntimeError(s.message)
    m.parts.append(s);m.end=end


def payload(m,ts,field,shrink=False):
    radii=np.round(np.arange(0,2.0001,.1),1)
    header=['时间\\到药材中心的距离']+radii.tolist()
    if shrink:header+=['药材表面','当前半径/cm']
    rows=[header]
    for t in ts:
        vals=m.profile(t,radii,field)
        row=[int(t) if float(t).is_integer() else float(t)]+[None if np.isnan(v) else round(float(v),4) for v in vals]
        if shrink:
            row+=[round(float(m.state(t)[-1]),4),round(float(m.env.radius(t))*100,6)]
        rows.append(row)
    return rows


def sparse_table(m,ts,field,shrink=False):
    rr=np.arange(0,2.01,.5)
    data=[]
    for t in ts:
        vals=m.profile(t,rr,field)
        row=[float(t/3600 if m.p.problem!=1 else t)]+[None if np.isnan(v) else float(v) for v in vals]
        if shrink:row += [float(m.state(t)[-1]),float(m.env.radius(t)*100)]
        data.append(row)
    return data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--cache',default=str(ROOT/'cache'))
    args=ap.parse_args();cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
    results=ROOT/'results';results.mkdir(exist_ok=True)
    meshes=[80] if args.quick else [80,160,320,640,1280]
    study={'mesh':[],'analytic':[],'balance':{},'sensitivity':[],'tables':{}}
    models={}
    for problem in [1,3,4]:
        prev=None
        for n in meshes:
            m=get_model(Settings(problem=problem,n=n),cache)
            row={'problem':problem,'n':n,'critical_time_h':m.end/3600}
            if prev is not None:
                times=np.array([100,300,600,900,1200,1500,1800]) if problem==1 else np.unique(np.r_[np.arange(1800,10801,1800),np.linspace(14400,min(m.end,prev.end),40)])
                diffs=[[],[]]
                for t in times:
                    for i,field in enumerate(['T','C']):
                        radii=np.linspace(0,min(float(m.env.radius(t)),float(prev.env.radius(t)))*100,21)
                        diffs[i].extend(abs(m.profile(t,radii,field)-prev.profile(t,radii,field)))
                row.update(delta_time_s=m.end-prev.end,max_T_difference_C=float(np.max(diffs[0])),max_C_difference=float(np.max(diffs[1])))
            study['mesh'].append(row);prev=m
            write_json(results/'validation.json',study)
        models[problem]=m
        study['balance'][str(problem)]=balance_and_bounds(m)
    for n in meshes:study['analytic'].append(constant_benchmark(n))
    # Compare a different implicit time integrator on the same spatial grid.
    if not args.quick:
        base=get_model(Settings(problem=3,n=160),cache)
        alt=get_model(Settings(problem=3,n=160,method='Radau',rtol=5e-9,atol=5e-11,max_step=300.),cache)
        study['time_solver']={'BDF_h':base.end/3600,'Radau_h':alt.end/3600,'difference_s':alt.end-base.end}
        cases=[('D -10%',{'d_scale':.9}),('D +10%',{'d_scale':1.1}),
               ('hm -20%',{'hm':6.4e-7}),('hm +20%',{'hm':9.6e-7}),
               ('tail T -1 C',{'tail_temp_shift':-1.}),('tail T +1 C',{'tail_temp_shift':1.}),
               ('tail C -0.01',{'tail_moist_shift':-.01}),('tail C +0.01',{'tail_moist_shift':.01}),
               ('last datum tail',{'tail_mode':'last'}),
               ('latent fraction 1',{'latent_fraction':1.})]
        for problem in [3,4]:
            cases_p=cases+([('exponential radius',{'radius_mode':'exp'}),('fixed radius appendix4',{'radius_mode':'fixed'}),
                            ('radius -1%',{'r_scale':.99}),('radius +1%',{'r_scale':1.01})] if problem==4 else [])
            b=get_model(Settings(problem=problem,n=160),cache)
            for label,kw in cases_p:
                m=get_model(Settings(problem=problem,n=160,**kw),cache)
                study['sensitivity'].append({'problem':problem,'case':label,'critical_time_h':m.end/3600,
                                              'change_pct':100*(m.end/b.end-1),'settings':kw})
                write_json(results/'validation.json',study)
    env=models[4].env
    rad=env.rad
    fit=lambda p,t:p[0]+(2-p[0])*np.exp(-t/p[1])
    cut=rad[:,0]<=12*3600
    tr=least_squares(lambda p:fit(p,rad[cut,0])-rad[cut,1],[1.198,11000],bounds=([.5,100],[1.5,1e5])).x
    study['input_audit']={'ambient_rows':len(env.a),'radius_rows':len(rad),'ambient_last_s':float(env.a[-1,0]),
            'tail_mean':env.tail.tolist(),'tail_std':env.a[env.a[:,0]>=10800,1:].std(axis=0,ddof=1).tolist(),
            'radius_adjusted_points':int(np.sum(abs(env.clean_radius-rad[:,1])>1e-10)),
            'radius_exp_parameters':env.r_fit.tolist(),'radius_exp_RMSE_cm':float(np.sqrt(np.mean((fit(env.r_fit,rad[:,0])-rad[:,1])**2))),
            'radius_train_first12h_parameters':tr.tolist(),'radius_holdout_after12h_RMSE_cm':float(np.sqrt(np.mean((fit(tr,rad[~cut,0])-rad[~cut,1])**2)))}
    m1,m3,m4=models[1],models[3],models[4]
    crossing={}
    for q,m in [(3,m3),(4,m4)]:
        crit=m.end;oper=float((np.floor(crit/60)+1)*60)
        crossing[str(q)]={'critical_s':crit,'critical_h':crit/3600,'first_full_minute_s':oper,'first_full_minute_h':oper/3600,
                          'surface_C_at_critical':float(m.state(crit)[-1]),'mean_C_at_critical':float(m.mean_C(crit))}
        extend(m,oper)
        crossing[str(q)]['max_C_first_full_minute']=float(np.max(m.state(oper)[m.m:]))
    study['crossing']=crossing
    study['stopping_comparison']={}
    for q,m in [(3,m3),(4,m4)]:
        tmean=brentq(lambda t:float(m.mean_C(t))-.15,0,m.critical_time)
        study['stopping_comparison'][str(q)]={'mean_threshold_time_h':tmean/3600,
                'center_C_at_mean_threshold':float(m.state(tmean)[m.m]),
                'premature_stop_h':(m.critical_time-tmean)/3600,
                'surface_D_at_critical_m2_s':float(diffusivity(m.state(m.critical_time)[m.m-1],m.state(m.critical_time)[-1],q)),
                'center_D_at_critical_m2_s':float(diffusivity(m.state(m.critical_time)[0],m.state(m.critical_time)[m.m],q))}
    q1t=np.array([100,300,600,900,1200,1500,1800.]);q2t=np.arange(1800,10801,1800.)
    for field in ['T','C']:
        study['tables']['Q1_'+field]=sparse_table(m1,q1t,field)
        study['tables']['Q2_'+field]=sparse_table(m3,q2t,field)
    for q,m in [(3,m3),(4,m4)]:
        ts=np.r_[np.arange(21600,m.end,21600.),m.end]
        study['tables'][f'Q{q}_C']=sparse_table(m,ts,'C',q==4)
    books={
        'result1':{'温度':payload(m1,np.arange(1,1801),'T'),'水分浓度':payload(m1,np.arange(1,1801),'C')},
        'result2':{'温度':payload(m3,np.arange(1,10801),'T'),'水分浓度':payload(m3,np.arange(1,10801),'C')},
        'result3':{'Sheet1':payload(m3,np.arange(60,m3.end+1,60),'C')},
        'result4':{'Sheet1':payload(m4,np.arange(60,m4.end+1,60),'C',True)}}
    for name,sheets in books.items():write_json(results/(name+'.json'),sheets)
    # Portable full-precision arrays for independent analysis, plotting, and cross-checks.
    bundle={}
    for q,m in [(1,m1),(3,m3),(4,m4)]:
        ts=np.unique(np.r_[np.linspace(0,m.end,501),1800.,10800. if q!=1 else 1800.])
        yy=m.state(ts)
        for key,val in [('time_s',ts),('x',m.x),('T_C',yy[:m.m]),('C_dry_basis',yy[m.m:]),
                        ('radius_m',np.array([m.env.radius(t) for t in ts]))]:bundle[f'q{q}_{key}']=val
    np.savez_compressed(results/'profiles_full_precision.npz',**bundle)
    write_json(results/'validation.json',study)
    print('FINISHED',json.dumps(crossing),flush=True)


if __name__=='__main__':main()
