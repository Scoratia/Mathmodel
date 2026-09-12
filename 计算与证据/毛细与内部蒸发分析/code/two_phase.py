"""Exploratory fixed-cylinder liquid/vapor/thermal model; NOT calibrated drug properties."""
from pathlib import Path
from dataclasses import dataclass,asdict
import json,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags,bmat
from water_thermo import saturation,vapor_pressure_from_humidity_ratio

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Parameters:
    n:int=60
    permeability:float=3e-19
    evaporation_rate:float=1.
    gas_diffusion:float=8e-6
    equilibrium_C:float=.10
    h:float=25.
    gas_film:float=.005
    hours:float=120.
    method:str='BDF'
    rtol:float=2e-7
    atol:float=2e-9
    sealed:bool=False

class Model:
    R=.02; length=.25; rho_l=1000.; mu_l=.001
    rho_d=(650+128*2.55)/3.55
    solid_density=1500.; Rv=461.5
    cs=1450.; cl=4186.; cv=1850.; Lv0=2.5008e6
    def __init__(self,p=Parameters()):
        self.p=p;self.n=p.n
        # Cell-centered annular finite volumes, uniform radial faces.
        self.edges=np.linspace(0,self.R,p.n+1)
        self.r=(self.edges[:-1]+self.edges[1:])/2
        self.V=np.pi*self.length*np.diff(self.edges**2)
        self.A=2*np.pi*self.length*self.edges
        self.dr=self.R/p.n
        self.eps=1-self.rho_d/self.solid_density
        self.a=np.array(json.loads((ROOT/'data/inputs.json').read_text(encoding='utf-8'))['ambient'])
        self.tail=self.a[self.a[:,0]>=10800,1:].mean(axis=0)
        RH=float(vapor_pressure_from_humidity_ratio(self.tail[1]))/saturation(self.tail[0])[0]
        self.b=-p.equilibrium_C*np.log(RH)
        T=np.full(p.n,28.);C=np.full(p.n,2.55)
        # Split given TOTAL water C0 into liquid and local-equilibrium vapor.
        for _ in range(8):
            vapor=(self.eps-self.rho_d*C/self.rho_l)*self.aw(C)*saturation(T)[0]/(self.Rv*(T+273.15))
            C=2.55-vapor/self.rho_d
        self.y0=np.r_[T,C,vapor]
        tri=diags([np.ones(p.n-1),np.ones(p.n),np.ones(p.n-1)],[-1,0,1],format='csc')
        self.sparsity=bmat([[tri]*3]*3,format='csc')
    def ambient(self,t):
        if t<=14400:return tuple(np.interp(t,self.a[:,0],self.a[:,j]) for j in (1,2))
        return tuple(self.tail)
    def aw(self,C):return np.exp(-self.b/np.maximum(C,1e-8))
    def fields(self,y):
        T,C,v=y[:self.n],y[self.n:2*self.n],y[2*self.n:]
        gas=self.eps-self.rho_d*C/self.rho_l
        pv=v/np.maximum(gas,1e-8)*self.Rv*(T+273.15)
        S=self.rho_d*C/(self.rho_l*self.eps)
        return T,C,v,gas,pv,S
    def fluxes(self,t,y):
        T,C,v,gas,pv,S=self.fields(y)
        TK=T+273.15
        ps=saturation(T)[0];aw=self.aw(C)
        pc=self.rho_l*self.Rv*TK*self.b/np.maximum(C,1e-8)
        kr=np.maximum(S,0.)**3
        krf=2*kr[:-1]*kr[1:]/np.maximum(kr[:-1]+kr[1:],1e-30)
        liquid=self.A[1:-1]*self.rho_l*self.p.permeability/self.mu_l*krf*np.diff(pc)/self.dr
        gf=2*gas[:-1]*gas[1:]/np.maximum(gas[:-1]+gas[1:],1e-8)
        tf=(T[:-1]+T[1:])/2
        vapor=-self.A[1:-1]*self.p.gas_diffusion*gf/(self.Rv*(tf+273.15))*np.diff(pv)/self.dr
        Ta,W=self.ambient(t);pva=float(vapor_pressure_from_humidity_ratio(W))
        # Disjoint exposed liquid and gas surface areas; no duplicated exchange area.
        fl=float(np.clip(S[-1],0.,1.));fg=1-fl
        beta=self.p.gas_film/(self.Rv*(T[-1]+273.15))
        Lout=self.A[-1]*fl*beta*(aw[-1]*ps[-1]-pva)
        Vout=self.A[-1]*fg*beta*(pv[-1]-pva)
        if self.p.sealed:Lout=Vout=0.
        FL=np.r_[0.,liquid,Lout];FV=np.r_[0.,vapor,Vout]
        k=.21+.38*C/(1+C)
        kf=2*k[:-1]*k[1:]/(k[:-1]+k[1:])
        Q=np.r_[0.,-self.A[1:-1]*kf*np.diff(T)/self.dr,self.A[-1]*self.p.h*(T[-1]-Ta)]
        if self.p.sealed:Q[-1]=0.
        # Enthalpy flux: internal liquid carries liquid enthalpy, outgoing water is vapor.
        hliquid=self.cl*tf;hvapor=self.Lv0+self.cv*tf
        hvL=self.Lv0+self.cv*(T[-1] if Lout>=0 else Ta)
        hvV=self.Lv0+self.cv*(T[-1] if Vout>=0 else Ta)
        E=np.r_[0.,hliquid*liquid+hvapor*vapor,hvL*Lout+hvV*Vout]
        source=self.p.evaporation_rate*gas*(aw*ps-pv)/(self.Rv*TK)
        return FL,FV,Q,E,source,pc
    def rhs(self,t,y):
        T,C,v,gas,pv,S=self.fields(y)
        FL,FV,Q,E,source,_=self.fluxes(t,y)
        dC=(-np.diff(FL)/self.V-source)/self.rho_d
        dv=-np.diff(FV)/self.V+source
        cap=self.rho_d*(self.cs+self.cl*C)+self.cv*v
        dT=(-np.diff(Q+E)/self.V-self.rho_d*self.cl*T*dC-(self.Lv0+self.cv*T)*dv)/cap
        return np.r_[dT,dC,dv]
    def inventories(self,y):
        T,C,v,*_=self.fields(y)
        water=self.V@(self.rho_d*C+v)
        energy=self.V@(self.rho_d*(self.cs+self.cl*C)*T+v*(self.Lv0+self.cv*T))
        return float(water),float(energy)
    def solve(self):
        self.solutions=[];y=self.y0.copy()
        def event(t,y):return np.max(y[self.n:2*self.n]+y[2*self.n:]/self.rho_d)-.15
        event.terminal=True;event.direction=-1
        for lo,hi in [(0.,min(14400.,self.p.hours*3600)),(14400.,self.p.hours*3600)]:
            if hi<=lo:continue
            def fun(t,y):return self.rhs(t+1e-7 if lo==14400 and t==lo else t,y)
            sol=solve_ivp(fun,(lo,hi),y,method=self.p.method,rtol=self.p.rtol,atol=self.p.atol,
                          jac_sparsity=self.sparsity,dense_output=True,max_step=60. if lo==0 else 900.,events=event)
            if not sol.success:raise RuntimeError(sol.message)
            self.solutions.append(sol);y=sol.y[:,-1]
            if sol.t_events[0].size:break
        self.end=float(sol.t[-1]);self.reached=bool(sol.t_events[0].size)
        return self
    def state(self,t):
        for sol in self.solutions:
            if t<=sol.t[-1]+1e-7:return sol.sol(t)
        raise ValueError('Requested beyond computed interval')

if __name__=='__main__':
    import argparse,pickle
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=40);a=ap.parse_args()
    for label,k,e in [('surface',3e-19,0.),('internal',0.,1.),('coupled',3e-19,1.)]:
        t=time.perf_counter();m=Model(Parameters(n=a.n,permeability=k,evaporation_rate=e)).solve()
        print(label,m.end/3600,m.reached,'wall',time.perf_counter()-t,flush=True)
        with (ROOT/'results'/f'pilot_{label}.pkl').open('wb') as f:pickle.dump(m,f)
