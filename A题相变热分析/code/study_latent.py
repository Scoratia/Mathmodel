"""Surface phase-change sensitivity, energy balance, and conditional dew-point audit."""
from pathlib import Path
import json,hashlib,pickle,time,argparse
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
from conservative_model import ConservativeSettings as Settings
from latent_model import TemperatureLatentModel,TemperatureLatentSettings
from water_thermo import saturation,vapor_pressure_from_humidity_ratio,dewpoint,verification
from run_conservation_study import get_model,save

ROOT=Path(__file__).resolve().parents[1]


def variable_model(q,cache):
    p=TemperatureLatentSettings(problem=q,n=160,latent_fraction=1.)
    signature=b''.join((ROOT/'code'/f).read_bytes() for f in ['latent_model.py','water_thermo.py','conservative_model.py','reference_model.py'])
    f=cache/(hashlib.sha256(signature+str(q).encode()).hexdigest()[:24]+'.pkl')
    if f.exists():
        with f.open('rb') as stream:m=pickle.load(stream)
    else:
        m=TemperatureLatentModel(p).solve()
        with f.open('wb') as stream:pickle.dump(m,stream)
    return m


def energy_audit(m):
    results=[]
    for step in [10.,5.]:
        integrals=np.zeros(5)
        for lo,hi in [(0.,14400.),(14400.,m.end)]:
            ts=np.unique(np.r_[np.arange(lo,hi,step),hi]); vals=[]
            for start in range(0,len(ts),256):
                tc=ts[start:start+256];yy=m.state(tc)
                for t,y in zip(tc,yy.T):
                    tt=t+1e-7 if t==14400 and lo==14400 else t
                    Fw,Q,Ew,_=m.fluxes(tt,y)
                    Ts=y[m.m-1];Ta=m.env.ambient(tt)[0]
                    area=2*np.pi*float(m.env.radius(tt))*m.length
                    heatin=area*m.p.h*(Ta-Ts)
                    latent=m.p.latent_fraction*m.p.latent_heat_J_kg*Fw[-1]
                    vals.append([Fw[-1],heatin,latent,Ew[-1],abs(heatin)])
            integrals+=simpson(np.asarray(vals),x=ts,axis=0)
        deltaE=m.sensible_energy(m.end)-m.sensible_energy(0)
        residual=float(deltaE-integrals[1]+integrals[2]+integrals[3])
        water=m.masses(m.end)[1]-m.masses(0)[1]+integrals[0]
        results.append({'step_s':step,'water_out_kg':float(integrals[0]),'convective_heat_input_J':float(integrals[1]),
            'latent_energy_J':float(integrals[2]),'outward_water_sensible_energy_J':float(integrals[3]),
            'stored_sensible_energy_change_J':float(deltaE),'energy_residual_J':residual,
            'energy_relative_residual':abs(residual)/integrals[4],
            'water_residual_kg':float(water),'water_relative_residual':float(abs(water)/m.masses(0)[1])})
    return results


def sample(m,baseline):
    ts=np.unique(np.r_[np.arange(0,14401,30.),np.arange(14460,m.end,60.),m.end])
    rows=[];min_surface=(1e9,0.);max_gap=(0.,0.);max_reverse=0.;md_error=0.
    for start in range(0,len(ts),256):
        tc=ts[start:start+256];yy=m.state(tc)
        for t,y in zip(tc,yy.T):
            T,C=y[:m.m],y[m.m:];Ta,W=m.env.ambient(t)
            Ts=float(T[-1]);pv=float(vapor_pressure_from_humidity_ratio(W));ps=saturation(Ts)[0]
            td=dewpoint(W);Fw=m.fluxes(t,y)[0][-1]
            rows.append([float(t),float(T[0]),Ts,Ta,float(C[0]),float(C[-1]),float(m.mean_C(t)),
                         float(Fw),float(m.env.radius(t)),td,pv/ps])
            if Ts<min_surface[0]:min_surface=(Ts,t)
            if Ta-Ts>max_gap[0]:max_gap=(Ta-Ts,t)
            if pv>ps and Fw>0:max_reverse=max(max_reverse,pv-ps)
            rd,rw=m.densities(t,C)
            md_error=max(md_error,abs(np.sum(rw/(1+C)*m.volumes(t))/m.Md-1))
    a=np.asarray(rows)
    # Locate sign changes of saturation pressure at the surface minus ambient vapor pressure.
    gap=a[:,10]-1
    crossings=[]
    def fun(t):
        Ts=float(m.state(t)[m.m-1]);W=m.env.ambient(t)[1]
        return float(vapor_pressure_from_humidity_ratio(W)/saturation(Ts)[0]-1)
    for i in np.where(gap[:-1]*gap[1:]<0)[0]:crossings.append(brentq(fun,float(a[i,0]),float(a[i+1,0])))
    # Condition assessed only at sampled points; intervals refined at crossing roots.
    cuts=[0.]+crossings+[m.end];intervals=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        mid=(lo+hi)/2
        if fun(mid)>0 and m.fluxes(mid,m.state(mid))[0][-1]>0:intervals.append([lo,hi])
    compare_times=np.array([1800.,3600.,10800.,14400.,21600.,43200.,86400.])
    temperature=[]
    for t in compare_times:
        y=m.state(t);b=baseline.state(t)
        temperature.append({'time_h':t/3600,'center_T_C':float(y[0]),'surface_T_C':float(y[m.m-1]),
             'baseline_center_T_C':float(b[0]),'baseline_surface_T_C':float(b[baseline.m-1]),
             'center_C':float(y[m.m]),'surface_C':float(y[-1])})
    return {'columns':['time_s','center_T_C','surface_T_C','ambient_T_C','center_C','surface_C','mean_C',
             'water_out_kg_s','radius_m','conditional_air_dewpoint_C','conditional_pv_over_surface_psat'],
            'rows':rows}, {'surface_min_C':min_surface[0],'surface_min_time_h':min_surface[1]/3600,
              'maximum_air_surface_temperature_gap_C':max_gap[0],'gap_time_h':max_gap[1]/3600,
              'dry_mass_max_relative_error':md_error,'temperature_comparison':temperature,
              'conditional_evaporation_below_air_dewpoint_intervals_s':intervals,
              'conditional_max_reverse_pressure_difference_Pa':float(max_reverse),
              'conditional_max_required_water_activity':float(a[:,10].max())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--baseline-cache',required=True)
    args=ap.parse_args();cache=Path(args.cache);bc=Path(args.baseline_cache)
    cache.mkdir(parents=True,exist_ok=True)
    result={'thermophysical_verification':verification(),'cases':[],'mesh':[],'validation':{},'crossing':{},'solver':[]}
    for q in [3,4]:
        b=get_model(Settings(problem=q,n=1280),bc)
        b160=get_model(Settings(problem=q,n=160),bc)
        for fraction in [0.,.25,.5,.75,1.]:
            m=get_model(Settings(problem=q,n=160,latent_fraction=fraction),bc)
            result['cases'].append({'problem':q,'fraction':fraction,'law':'constant 2.38 MJ/kg','time_h':m.end/3600,
                'change_h':(m.end-b160.end)/3600,'change_pct':100*(m.end/b160.end-1)})
            save(ROOT/'results/study.json',result)
        prev=None
        for n in [160,320,640,1280]:
            m=get_model(Settings(problem=q,n=n,latent_fraction=1.),bc)
            row={'problem':q,'n':n,'time_h':m.end/3600}
            if prev is not None:
                row['delta_time_s']=m.end-prev.end
                times=np.unique(np.r_[[1.,10.,60.,300.,1800.,3600.,10800.,14400.],np.linspace(14460,min(m.end,prev.end),31)])
                diff=[]
                for t in times:
                    rr=np.linspace(0,float(m.env.radius(t))*100,31)
                    diff.append([max(abs(m.profile(t,rr,f)-prev.profile(t,rr,f))) for f in ['T','C']])
                row['max_T_delta_C']=float(np.max(diff,axis=0)[0]);row['max_C_delta']=float(np.max(diff,axis=0)[1])
            result['mesh'].append(row);prev=m
            save(ROOT/'results/study.json',result)
        result['crossing'][str(q)]={'without_latent_h':b.end/3600,'with_latent_h':m.end/3600,
             'change_h':(m.end-b.end)/3600,'change_pct':100*(m.end/b.end-1),
             'critical_time_s':m.end,'first_full_minute_s':float((np.floor(m.end/60)+1)*60)}
        curves,diagnostic=sample(m,b)
        save(ROOT/f'results/history_q{q}.json',curves)
        result['validation'][str(q)]={'energy':energy_audit(m),'diagnostics':diagnostic}
        save(ROOT/'results/study.json',result)
        var=variable_model(q,cache)
        result['cases'].append({'problem':q,'fraction':1.,'law':'IAPWS saturation Lv(Ts)','time_h':var.end/3600,
             'constant_same_grid_h':next(x['time_h'] for x in result['cases'] if x['problem']==q and x['fraction']==1 and x['law']=='constant 2.38 MJ/kg')})
        alt=get_model(Settings(problem=q,n=160,latent_fraction=1.,method='Radau',rtol=5e-9,atol=5e-11,max_step=300),bc)
        nom=next(x['time_h'] for x in result['cases'] if x['problem']==q and x['fraction']==1 and x['law']=='constant 2.38 MJ/kg')
        result['solver'].append({'problem':q,'BDF_h':nom,'Radau_h':alt.end/3600,'difference_s':alt.end-nom*3600})
        save(ROOT/'results/study.json',result)
        print('QUESTION_DONE',q,json.dumps(result['crossing'][str(q)]),flush=True)
    env=b.env;Ta,W=env.tail
    pv=float(vapor_pressure_from_humidity_ratio(W));ps=saturation(Ta)[0]
    result['conditional_humidity_audit']={'assumptions':'W is kg water vapor/kg dry air; total pressure=101325 Pa; ideal moist air',
        'late_air_T_C':float(Ta),'late_air_W':float(W),'late_air_RH':pv/ps,'late_air_dewpoint_C':dewpoint(W),
        'late_vapor_pressure_Pa':pv,'initial_RH':float(vapor_pressure_from_humidity_ratio(env.a[0,2])/saturation(env.a[0,1])[0])}
    for q in [3,4]:
        r=result['validation'][str(q)]
        assert r['diagnostics']['dry_mass_max_relative_error']<1e-13
        assert r['energy'][-1]['energy_relative_residual']<2e-5
        assert r['energy'][-1]['water_relative_residual']<2e-5
        last=[a for a in result['mesh'] if a['problem']==q][-1]
        assert abs(last['delta_time_s'])<.3
    assert all(abs(a['difference_s'])<.1 for a in result['solver'])
    result['numerical_status']='PASS'
    result['physical_status']='Conditional effective-boundary model; inspect dew-point audit before engineering interpretation'
    save(ROOT/'results/study.json',result)
    print('FINISHED',json.dumps(result['crossing']),flush=True)


if __name__=='__main__':main()
