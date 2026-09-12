from pathlib import Path
from dataclasses import asdict,replace
import argparse,hashlib,json,pickle,time
import numpy as np
from scipy.integrate import simpson
from two_phase import Model,Parameters,ROOT

def save(name,obj):
    (ROOT/'results'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def get(p,cache):
    signature=(ROOT/'code/two_phase.py').read_bytes()+(ROOT/'data/inputs.json').read_bytes()+(ROOT/'code/water_thermo.py').read_bytes()
    key=hashlib.sha256(signature+json.dumps(asdict(p),sort_keys=True).encode()).hexdigest()[:24]
    path=cache/(key+'.pkl')
    if path.exists():
        with path.open('rb') as f:m=pickle.load(f)
    else:
        m=Model(p).solve()
        with path.open('wb') as f:pickle.dump(m,f)
    print(json.dumps({'parameters':asdict(p),'end_h':m.end/3600,'reached':m.reached}),flush=True)
    return m

def summary(m):
    out={'parameters':asdict(m.p),'end_h':m.end/3600,'threshold_reached':m.reached,
         'initial_porosity':m.eps,'initial_dry_mass_kg':m.rho_d*m.V.sum(),'water_activity_b':m.b,'snapshots':{}}
    for h in [0,.5,1,4,12,24,48,72,min(120,m.end/3600)]:
        if h*3600>m.end+1e-6:continue
        y=m.state(h*3600);T,C,v,gas,pv,S=m.fields(y)
        FL,FV,Q,E,source,pc=m.fluxes(h*3600,y)
        out['snapshots'][f'{h:.6f}']={'time_h':h,'mean_total_C':m.inventories(y)[0]/(m.rho_d*m.V.sum()),
             'max_total_C':float(max(C+v/m.rho_d)),'center_T_C':float(T[0]),'surface_cell_T_C':float(T[-1]),
             'water_lost_kg':m.inventories(m.y0)[0]-m.inventories(y)[0],
             'net_internal_evaporation_kg_s':float(m.V@source),
             'surface_liquid_evaporation_kg_s':float(FL[-1]),'vapor_out_kg_s':float(FV[-1]),
             'surface_liquid_evaporation_fraction':float(FL[-1]/(FL[-1]+FV[-1])) if FL[-1]+FV[-1]!=0 else None}
    return out

def history(m,label):
    ts=np.unique(np.r_[0.,np.geomspace(.01,60,36),np.arange(120,min(m.end,14400)+1,120.),np.arange(14700,m.end,300.),m.end])
    rows=[];fields=[]
    for t in ts:
        y=m.state(t);T,C,v,gas,pv,S=m.fields(y)
        FL,FV,Q,E,source,pc=m.fluxes(t,y)
        mw,H=m.inventories(y)
        rows.append([t,mw/(m.rho_d*m.V.sum()),max(C+v/m.rho_d),T[0],T[-1],mw,FL[-1],FV[-1],m.V@source,
                     np.min(C),np.min(gas),np.max(pv),np.min(v)])
        fields.append([T,C,v,source,pc])
    np.savez_compressed(ROOT/'results'/f'fields_{label}.npz',time_s=ts,r_m=m.r,values=np.array(fields),
                        variable_names=np.array(['T_C','C_liquid_dry_basis','vapor_bulk_kg_m3','phase_source_kg_m3_s','capillary_pressure_Pa']))
    out={'columns':['time_s','mean_total_C','max_total_C','center_T_C','surface_cell_T_C','water_kg',
                    'surface_liquid_evap_kg_s','vapor_out_kg_s','net_internal_evap_kg_s','min_liquid_C','min_gas_porosity','max_vapor_pressure_Pa','min_vapor_bulk'],
         'rows':np.array(rows).tolist()}
    save(f'history_{label}.json',out)
    a=np.array(rows)
    assert np.min(a[:,9])>=-1e-8 and np.min(a[:,10])>0 and np.min(a[:,12])>=-1e-8
    return {'min_T_C':float(np.min(np.array(fields)[:,0])), 'min_liquid_C':float(a[:,9].min()),
            'min_gas_porosity':float(a[:,10].min()),'max_vapor_pressure_fraction_of_total':float(a[:,11].max()/101325)}

def audit(m):
    rows=[]
    for step in [120.,60.]:
        integral=np.zeros(5)
        for lo,hi in [(0.,min(14400.,m.end)),(14400.,m.end)]:
            if hi<=lo:continue
            ts=np.unique(np.r_[lo,np.arange(lo,hi,step),hi,np.geomspace(.0001,min(60.,hi),60) if lo==0 else []])
            vals=[]
            for t in ts:
                tt=t+1e-7 if lo==14400 and t==lo else t
                FL,FV,Q,E,S,_=m.fluxes(tt,m.state(t))
                vals.append([FL[-1]+FV[-1],-Q[-1],E[-1],m.V@S,FL[-1]])
            integral+=simpson(np.array(vals),x=ts,axis=0)
        w0,h0=m.inventories(m.y0);we,he=m.inventories(m.state(m.end))
        rows.append({'step_s':step,'water_residual_kg':we-w0+integral[0],
                     'water_relative_residual':abs(we-w0+integral[0])/w0,
                     'energy_residual_J':he-h0-integral[1]+integral[2],
                     'energy_relative_residual':abs(he-h0-integral[1]+integral[2])/max(abs(integral[1]),1),
                     'net_internal_evaporation_kg':integral[3],'surface_liquid_evaporation_kg':integral[4],
                     'water_out_kg':integral[0],'convective_heat_in_J':integral[1],'water_enthalpy_out_J':integral[2]})
    # Instantaneous independent inventory derivative identity.
    worstM=worstE=0.
    for t in np.linspace(0,m.end,101):
        y=m.state(t);dy=m.rhs(t,y);T,C,v,*_=m.fields(y)
        FL,FV,Q,E,*_=m.fluxes(t,y)
        dH=m.V@((m.rho_d*(m.cs+m.cl*C)+m.cv*v)*dy[:m.n]+m.rho_d*m.cl*T*dy[m.n:2*m.n]+(m.Lv0+m.cv*T)*dy[2*m.n:])
        dW=m.V@(m.rho_d*dy[m.n:2*m.n]+dy[2*m.n:])
        worstM=max(worstM,abs(dW+FL[-1]+FV[-1]));worstE=max(worstE,abs(dH+Q[-1]+E[-1]))
    return {'integrated':rows,'instantaneous_water_residual_kg_s':worstM,'instantaneous_energy_residual_W':worstE}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);args=ap.parse_args()
    cache=Path(args.cache);cache.mkdir(exist_ok=True,parents=True)
    out={'status':'RUNNING','scope':'Uncalibrated Q3 fixed geometry, separate liquid and vapor, true-humidity interpretation.',
         'cases':{},'mesh':[],'sensitivity':[]}
    models={}
    for label,k,e in [('neither',0.,0.),('capillary',3e-19,0.),('internal',0.,1.),('coupled',3e-19,1.)]:
        m=get(Parameters(n=80,permeability=k,evaporation_rate=e),cache);models[label]=m
        out['cases'][label]=summary(m)
        out['cases'][label]['ranges']=history(m,label)
        out['cases'][label]['audit']=audit(m)
        save('study.json',out)
    prev=None
    for n in [40,80,160]:
        m=get(Parameters(n=n),cache)
        row={'n':n,'end_h':m.end/3600,'reached':m.reached}
        if prev is not None:
            row['time_change_h']=(m.end-prev.end)/3600
            dif=[]
            for t in [3600,14400,86400,259200]:
                if t>min(m.end,prev.end):continue
                y=m.state(t);old=prev.state(t)
                dif.append([max(abs(np.interp(prev.r,m.r,y[:m.n])-old[:prev.n])),
                            max(abs(np.interp(prev.r,m.r,y[m.n:2*m.n])-old[prev.n:2*prev.n]))])
            row['max_T_difference_C']=float(np.max(dif,axis=0)[0]);row['max_C_difference']=float(np.max(dif,axis=0)[1])
        out['mesh'].append(row);prev=m;save('study.json',out)
    out['fine_coupled']=summary(m);out['fine_coupled']['ranges']=history(m,'coupled_fine');out['fine_coupled']['audit']=audit(m)
    for name,kwargs in [('K/10',{'permeability':3e-20}),('K*10',{'permeability':3e-18}),
                         ('evap/10',{'evaporation_rate':.1}),('evap*10',{'evaporation_rate':10.}),
                         ('Ceq=.05',{'equilibrium_C':.05}),('Ceq=.14',{'equilibrium_C':.14})]:
        sm=get(Parameters(n=80,**kwargs),cache)
        out['sensitivity'].append({'name':name,**summary(sm)});save('study.json',out)
    alt=get(Parameters(n=40,method='Radau',rtol=5e-8,atol=5e-10),cache)
    base=get(Parameters(n=40),cache)
    out['solver']={'n':40,'BDF_h':base.end/3600,'Radau_h':alt.end/3600,'difference_s':alt.end-base.end}
    sealed=get(Parameters(n=20,sealed=True,hours=4),cache)
    out['sealed']={'max_state_change':float(max(abs(sealed.state(sealed.end)-sealed.y0))),
                   'inventory_change':(np.array(sealed.inventories(sealed.state(sealed.end)))-sealed.inventories(sealed.y0)).tolist()}
    for row in out['cases'].values():
        assert row['audit']['integrated'][-1]['water_relative_residual']<2e-4
        assert row['audit']['integrated'][-1]['energy_relative_residual']<2e-3
    assert abs(out['solver']['difference_s'])<2.
    assert out['sealed']['max_state_change']<1e-5
    out['status']='PASS';save('study.json',out)
    print('ALL_DONE',flush=True)

if __name__=='__main__':main()
