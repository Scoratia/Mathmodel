"""Portable resumable CPU / genuine CUDA runner. Run --help for options."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,platform,time
import numpy as np
import pandas as pd
from cases import make_cases,physical_key
from physics import Inputs,make_grid,prepare_cases,diagnostics,geometry,perturbation_mask
from cpu_solver import solve_case,horizon


def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj


def write_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(data),indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)


def fingerprint(backend,N,rtol,atol):
    root=Path(__file__).parent
    names=['physics.py','cases.py','cpu_solver.py','data/inputs.npz']
    if backend!='cpu':names+=['gpu_solver.py']
    return {'backend':backend,'N':N,'rtol':rtol,'atol':atol,
            'hashes':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in names}}


def _cpu_job(case,N,rtol,atol):return solve_case(case,N=N,rtol=rtol,atol=atol)


def batch_rows(batch,result,grid,inputs,backend,N,rtol,atol):
    """Turn solver samples into common diagnostics; no hidden CPU reintegration."""
    from scipy.integrate import trapezoid
    rows=[]
    for j,c in enumerate(batch):
        valid=result.sample_valid[j]
        ts=result.sample_times[valid];T=result.samples_T[j,valid];C=result.samples_C[j,valid]
        final_time=float(result.final_times[j])
        if len(ts)==0 or final_time>ts[-1]+1e-9:
            ts=np.r_[ts,final_time];T=np.concatenate([T,result.final_state[j:j+1,:,0]])
            C=np.concatenate([C,result.final_state[j:j+1,:,1]])
        b=prepare_cases([c]);ds={};masks=[];radii=[]
        for t,tt,cc in zip(ts,T,C):
            y=np.stack((tt,cc),axis=-1)[None]
            d=diagnostics(float(t),y,b,grid,inputs)
            for k,v in d.items():ds.setdefault(k,[]).append(float(v[0]))
            masks.append(perturbation_mask(grid['a'][None],cc[None],b)[0]);radii.append(geometry(float(t),b,grid,inputs)['r'][0])
        ds={k:np.asarray(v) for k,v in ds.items()};masks=np.asarray(masks)
        # The saved sample grid contains the boundary break. Integrate each side
        # using its own limiting flux instead of smearing the discontinuity.
        integral=0.
        for i in range(1,len(ts)):
            left=ds['outflow_kg_s'][i-1];right=ds['outflow_kg_s'][i]
            if ts[i-1]==inputs.boundary_end:
                y=np.stack((T[i-1],C[i-1]),axis=-1)[None]
                left=diagnostics(ts[i-1],y,b,grid,inputs,phase='plateau')['outflow_kg_s'][0]
            integral+=(left+right)*(ts[i]-ts[i-1])/2
        residual=(ds['water_mass_kg'][-1]-ds['water_mass_kg'][0]+integral)/ds['water_mass_kg'][0]
        tc=float(result.t_critical[j]);tm=float(result.t_mean_threshold[j]);tc=tc if np.isfinite(tc) else None;tm=tm if np.isfinite(tm) else None
        support=2*np.sum(grid['weights'][None]*(masks>.5),axis=1)
        row={**c,'backend':backend,'N':N,'rtol':rtol,'atol':atol,
             'status':'reached' if tc is not None else 'not_reached_by_horizon',
             'solver_status':str(result.status[j]),'tcritical_s':tc,'tmean_s':tm,
             'tail_s':None if tc is None or tm is None else tc-tm,'t_end_s':final_time,
             'horizon_s':horizon(c,inputs),'mass_audit_rel':residual,
             'mass_audit_note':'Independent trapezoid on saved GPU samples; includes sampling error',
             'max_C_end':ds['max_C'][-1],'mean_C_end':ds['mean_C'][-1],
             'initial_dry_mass_kg':ds['dry_mass_kg'][0],
             'empirical_dry_mass_ratio_end':ds['empirical_dry_mass_ratio'][-1],
             'perturb_support_dry_mass_fraction':float(trapezoid(support,ts)/final_time),
             'perturb_weighted_dry_mass_fraction':float(trapezoid(ds['mask_dry_mass_fraction'],ts)/final_time),
             'event_bracket_s':np.asarray(result.max_event_brackets[j]).tolist(),
             'batch_wall_s':result.stats.get('wall_s',result.stats.get('wall_time_s'))}
        profile={'t_s':ts,'a':grid['a'],'x':grid['a'],'weights':grid['weights'],
                 'dry_mass_weights':2*grid['weights'],'T':T,'C':C,'mask':masks,
                 'zone_weight':masks,'r_m':np.asarray(radii),**ds}
        rows.append((row,profile))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['cpu','numpy','cupy'],default='cpu')
    p.add_argument('--suite',choices=['pilot','full'],default='pilot');p.add_argument('--task',type=int,choices=[1,2])
    p.add_argument('--N',type=int,default=320);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--rtol',type=float)
    p.add_argument('--atol',type=float);p.add_argument('--output',default='results_run');p.add_argument('--limit',type=int)
    a=p.parse_args();a.rtol=a.rtol or (2e-8 if a.backend=='cpu' else 2e-5);a.atol=a.atol or (2e-10 if a.backend=='cpu' else 1e-7)
    out=Path(a.output);out.mkdir(exist_ok=True,parents=True);(out/'cases').mkdir(exist_ok=True);(out/'profiles').mkdir(exist_ok=True)
    cases=make_cases(a.suite,a.task);cases=cases[:a.limit] if a.limit else cases
    fp=fingerprint(a.backend,a.N,a.rtol,a.atol);meta=out/'configuration.json'
    if meta.exists() and json.loads(meta.read_text())!=fp:raise SystemExit('Output configuration/source hash mismatch; select a fresh output directory')
    write_json(meta,fp);write_json(out/'cases_requested.json',cases)
    started=time.perf_counter();done={};pending=[]
    for c in cases:
        f=out/'cases'/(c['id']+'.json')
        if f.exists() and (out/'profiles'/(c['id']+'.npz')).exists():done[c['id']]=json.loads(f.read_text())
        else:pending.append(c)
    def save(row,profile):
        write_json(out/'cases'/(row['id']+'.json'),row)
        np.savez_compressed(out/'profiles'/(row['id']+'.npz'),**profile)
        done[row['id']]=row
        pd.DataFrame([done[c['id']] for c in cases if c['id'] in done]).to_csv(out/'summary.csv',index=False)
        print(f"{len(done)}/{len(cases)} {row['id']} {row['status']} t_h={None if row['tcritical_s'] is None else row['tcritical_s']/3600}",flush=True)
    if a.backend=='cpu':
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(_cpu_job,c,a.N,a.rtol,a.atol):c for c in pending}
            for future in as_completed(futures):
                c=futures[future]
                try:save(*future.result())
                except Exception as exc:
                    write_json(out/(c['id']+'.failure.json'),{'case':c,'error':repr(exc)});raise
    else:
        from gpu_solver import solve_batch,select_backend
        xp,device=select_backend(a.backend);write_json(out/'device.json',device)
        inputs=Inputs()
        # Avoid forcing 72 h cases to share a 168 h horizon, and keep task/physics
        # together so easy cases are not mixed arbitrarily with stiff ones.
        pending.sort(key=lambda c:(horizon(c,inputs),c['task'],c['question'],c['boundary_norm']))
        groups={}
        for c in pending:groups.setdefault(horizon(c,inputs),[]).append(c)
        for end,group in groups.items():
            for start in range(0,len(group),a.batch_size):
                batch=group[start:start+a.batch_size]
                sample=np.unique(np.r_[0.,np.geomspace(.001,14400,140),np.linspace(0,end,301),inputs.boundary_end])
                result=solve_batch(prepare_cases(batch,xp),make_grid(a.N,xp),inputs,backend=a.backend,
                                   t_end=end,sample_times=sample,rtol=a.rtol,atol=(max(a.atol,1e-5),a.atol))
                write_json(out/f'batch_{batch[0]["id"]}.json',result.stats)
                if result.stats.get('failed_cases',0):
                    write_json(out/'GPU_FAILURE.json',{'cases':[c['id'] for c in batch],
                               'status':result.status,'stats':result.stats})
                    raise RuntimeError('GPU integration failed; no successful-run manifest will be emitted')
                for row,prof in batch_rows(batch,result,make_grid(a.N),inputs,a.backend,a.N,a.rtol,a.atol):save(row,prof)
    frame=pd.DataFrame([done[c['id']] for c in cases]);frame.to_csv(out/'summary.csv',index=False)
    write_json(out/'run_manifest.json',{'completed':True,'case_count':len(frame),'requested_count':len(cases),
        'elapsed_wall_s':time.perf_counter()-started,'python':platform.python_version(),'platform':platform.platform(),
        'configuration':fp,'cuda_executed':a.backend=='cupy','arguments':vars(a)})
    from analyze import write_analysis
    write_analysis(frame,cases,out/'analysis')


if __name__=='__main__':main()
