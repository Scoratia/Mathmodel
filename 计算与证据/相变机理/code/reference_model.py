"""A题：圆柱药材非线性热湿扩散与均匀径向收缩模型。

Units: seconds, metres, Celsius in state; Kelvin only in diffusivity.
C is kg water / kg dry solid. Run from any directory; data are package-relative.
"""
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression, least_squares
from scipy.sparse import diags, bmat

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    problem: int = 3
    n: int = 160
    hm: float = 8e-7
    h: float = 25.
    d_scale: float = 1.
    r_scale: float = 1.
    radius_mode: str = 'pchip'
    tail_mode: str = 'plateau'
    tail_temp_shift: float = 0.
    tail_moist_shift: float = 0.
    rtol: float = 2e-8
    atol: float = 2e-10
    method: str = 'BDF'
    max_step: float = 600.
    threshold: float = .15
    # Diagnostic only: evaporation fraction times latent heat, zero in supplied-data model.
    latent_fraction: float = 0.


class Environment:
    def __init__(self, settings, data=None):
        self.p = settings
        if data is None:
            data = json.loads((ROOT/'data/inputs.json').read_text(encoding='utf-8'))
        self.a = np.asarray(data['ambient'],float)
        self.rad = np.asarray(data['radius'],float)
        self.tail = self.a[self.a[:,0]>=10800,1:].mean(axis=0)
        clean = isotonic_regression(self.rad[:,1], increasing=False).x
        self.r_interp = PchipInterpolator(self.rad[:,0],clean/100,extrapolate=False)
        self.clean_radius = clean
        def residual(par):
            return par[0]+(2-par[0])*np.exp(-self.rad[:,0]/par[1])-self.rad[:,1]
        self.r_fit = least_squares(residual,[1.198,11000],bounds=([.5,100],[1.5,1e5])).x

    def ambient(self,t):
        if t <= self.a[-1,0]:
            return tuple(np.interp(t,self.a[:,0],self.a[:,j]) for j in (1,2))
        tail = self.a[-1,1:] if self.p.tail_mode=='last' else self.tail
        return tail[0]+self.p.tail_temp_shift,tail[1]+self.p.tail_moist_shift

    def radius(self,t):
        if self.p.problem!=4 or self.p.radius_mode=='fixed':
            return .02*self.p.r_scale
        if self.p.radius_mode=='exp':
            rinf,tau=self.r_fit
            r=(rinf+(2-rinf)*np.exp(-np.asarray(t)/tau))/100
        else:
            r=self.r_interp(np.clip(t,0,self.rad[-1,0]))
        return r*self.p.r_scale


def properties(T,C,problem):
    c=np.maximum(C,1e-9)
    if problem==1:
        return np.full_like(c,820.),np.full_like(c,2600.),np.full_like(c,.36)
    if problem==4:
        return 760+90*c,1850+2150*c/(c+1),.12+.20*c/(c+1)
    return 650+128*c,1450+2736*c/(c+1),.21+.38*c/(c+1)


def diffusivity(T,C,problem):
    c=np.maximum(C,1e-9)
    if problem==1:
        return 7e-9*np.exp(-.89/c)
    a,b=(4.2e-4,.30) if problem==4 else (2.4e-3,.45)
    return a*np.exp(-b/c-3850/(T+273.15))


def face_D(T0,T1,C0,C1,problem):
    """Three-point Gauss integration of D along the line joining nodal states.

    At constant T, this approximates [Phi(C1)-Phi(C0)]/(C1-C0),
    preserving nonlinear diffusive resistance without evaluating derivatives of D.
    """
    result=0.
    for a,w in ((.5-np.sqrt(15)/10,5/18),(.5,4/9),(.5+np.sqrt(15)/10,5/18)):
        result=result+w*diffusivity(T0+a*(T1-T0),C0+a*(C1-C0),problem)
    return result


class RadialModel:
    def __init__(self,settings=Settings(),data=None):
        self.p=settings
        self.env=Environment(settings,data)
        self.x=np.linspace(0,1,settings.n+1)
        self.dx=1/settings.n
        self.faces=(self.x[:-1]+self.x[1:])/2
        edges=np.r_[0,self.faces,1]
        self.w=np.diff(edges**2)/2
        self.m=len(self.x)
        tri=diags([np.ones(self.m-1),np.ones(self.m),np.ones(self.m-1)],[-1,0,1],format='csc')
        self.sparsity=bmat([[tri,tri],[tri,tri]],format='csc')

    def rhs(self,t,y):
        T,C=y[:self.m],y[self.m:]
        R=float(self.env.radius(t))
        Ta,Ca=self.env.ambient(t)
        rho,cp,k=properties(T,C,self.p.problem)
        kf=2*k[:-1]*k[1:]/(k[:-1]+k[1:])
        df=face_D(T[:-1],T[1:],C[:-1],C[1:],self.p.problem)*self.p.d_scale
        qT=self.faces*kf*np.diff(T)/self.dx
        qC=self.faces*df*np.diff(C)/self.dx
        outC=R*self.p.hm*(C[-1]-Ca)
        outT=R*self.p.h*(T[-1]-Ta)
        if self.p.latent_fraction:
            # Extra assumption for sensitivity only: surface dry bulk density rho/(1+C).
            outT+=self.p.latent_fraction*2.38e6*rho[-1]/(1+C[-1])*outC
        dT=np.diff(np.r_[0,qT,-outT])/(R*R*self.w*rho*cp)
        dC=np.diff(np.r_[0,qC,-outC])/(R*R*self.w)
        return np.r_[dT,dC]

    def solve(self,end=None,event=True):
        if end is None: end=1800. if self.p.problem==1 else 10*86400.
        if self.p.problem==1:event=False
        def done(t,y):return np.max(y[self.m:])-self.p.threshold
        done.terminal=True
        done.direction=-1
        y=np.r_[np.full(self.m,28.),np.full(self.m,2.55)]
        self.parts=[]
        bounds=sorted(set([0.,min(14400.,end),float(end)]))
        for lo,hi in zip(bounds[:-1],bounds[1:]):
            # Input interpolation has 60 s knots, resolve it before the constant stage.
            rhs=self.rhs
            if lo==14400:
                # Right limit at stage start prevents using the old datum at the jump.
                rhs=lambda t,y:self.rhs(t+1e-7 if t==lo else t,y)
            s=solve_ivp(rhs,(lo,hi),y,method=self.p.method,
                        rtol=self.p.rtol,atol=self.p.atol,
                        jac_sparsity=self.sparsity,dense_output=True,
                        max_step=min(30.,self.p.max_step) if hi<=14400 else self.p.max_step,
                        events=done if event else None)
            if not s.success:raise RuntimeError(s.message)
            self.parts.append(s)
            y=s.y[:,-1]
            if s.status==1:break
        self.end=float(self.parts[-1].t[-1])
        self.event_found=bool(self.parts[-1].status==1)
        self.nfev=sum(s.nfev for s in self.parts)
        if event and not self.event_found:raise RuntimeError('Drying threshold not reached within horizon')
        return self

    def state(self,t):
        scalar=np.ndim(t)==0
        ts=np.atleast_1d(t).astype(float)
        if np.any(ts<0) or np.any(ts>self.end+1e-6):raise ValueError('Time outside solved interval')
        out=np.empty((2*self.m,len(ts)))
        for i,s in enumerate(self.parts):
            mask=(ts>=s.t[0])&(ts<=s.t[-1]+1e-6)
            if np.any(mask):out[:,mask]=s.sol(ts[mask])
        return out[:,0] if scalar else out

    def profile(self,t,r_cm,field='C'):
        y=self.state(t)
        vals=y[self.m:] if field=='C' else y[:self.m]
        r=np.asarray(r_cm)/100
        R=float(self.env.radius(t))
        v=np.interp(r/R,self.x,vals)
        return np.where(r<=R+1e-12,v,np.nan)

    def mean_C(self,t):
        return 2*self.w@self.state(t)[self.m:]
