"""Task 1/2 shared finite-volume equations; NumPy or CuPy, SI units, FP64.

Temperature is the declared effective heat model, not a mixture enthalpy model.
State layout: (case, material node, [temperature Celsius, dry-basis moisture]).
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

R0, LENGTH, T0, C0, H, HM = .02, .25, 28., 2.55, 25., 8e-7
THRESHOLD = .15


class Inputs:
    def __init__(self, path=None):
        path = Path(path or Path(__file__).parent/'data'/'inputs.npz')
        with np.load(path, allow_pickle=False) as z:
            for key in ('boundary_time','ambient_temperature','ambient_moisture',
                        'radius_time','radius_values'):
                setattr(self, key, z[key].copy())
        for name in ('boundary_time','radius_time'):
            v=getattr(self,name)
            if v[0]!=0 or np.any(np.diff(v)<=0): raise ValueError(name)
        self.boundary_end=float(self.boundary_time[-1])
        self.radius_end=float(self.radius_time[-1])
        self.radius_last=float(self.radius_values[-1])
        self._r=PchipInterpolator(self.radius_time,self.radius_values,extrapolate=False)
        mask=self.boundary_time>=self.boundary_end-3600
        self.plateau=(float(self.ambient_temperature[mask].mean()),
                      float(self.ambient_moisture[mask].mean()))

    def radius(self,t):
        if np.any(np.asarray(t)<0) or np.any(np.asarray(t)>self.radius_end):
            raise ValueError('Radius extrapolation is forbidden')
        return self._r(t)

    def boundary(self,t,phase=None):
        if phase=='plateau' or (phase is None and t>self.boundary_end): return self.plateau
        return (float(np.interp(t,self.boundary_time,self.ambient_temperature)),
                float(np.interp(t,self.boundary_time,self.ambient_moisture)))


def make_grid(N=320,xp=np,power=1.5):
    s=np.linspace(0,1,N+1);a=1-(1-s)**power
    faces=np.r_[0.,(a[:-1]+a[1:])/2,1.]
    gs,gw=np.polynomial.legendre.leggauss(5)
    d={'a':a,'faces':faces,'weights':np.diff(faces**2)/2,'da':np.diff(a),
       'gauss_s':(gs+1)/2,'gauss_w':gw/2}
    return {k:xp.asarray(v,dtype=xp.float64) for k,v in d.items()}


def prepare_cases(cases,xp=np):
    kinds={'none':0,'outer':1,'inner':2,'lowC':3,'global':4}
    defaults={'eps_final':0.,'logD_delta':0.,'region_width':.1,'C_cut':.2,'C_smooth':.02}
    out={}
    for key,default in defaults.items():out[key]=xp.asarray([c.get(key,default) for c in cases],dtype=xp.float64)
    out['question']=xp.asarray([c['question'] for c in cases])
    out['moving_radius']=xp.asarray([c.get('moving_radius',c['question']==4) for c in cases])
    out['boundary_initial']=xp.asarray([c.get('boundary_norm','current_dry_density')=='initial_dry_density' for c in cases])
    out['perturb_kind']=xp.asarray([kinds[c.get('perturb_kind','none')] for c in cases])
    for c in cases:
        if c['question'] not in (3,4):raise ValueError('Only Q3/Q4 material properties supported')
        if not -.5<c.get('eps_final',0)<.4:raise ValueError('Non-crossing deformation range exceeded')
        if not c.get('moving_radius',c['question']==4) and c.get('eps_final',0)!=0:raise ValueError('Fixed geometry requires epsilon=0')
        if c.get('boundary_norm','current_dry_density') not in ('current_dry_density','initial_dry_density'):raise ValueError('Unknown boundary norm')
        if not 0<c.get('region_width',.1)<1 or c.get('C_smooth',.02)<=0:raise ValueError('Invalid perturbation window')
    return out


def geometry(t,case_arrays,grid,inputs,xp=np):
    b=case_arrays
    # Only evaluate the measured curve when time lies in its domain. Any moving
    # case beyond that domain is rejected by the integration wrapper.
    Rmeas=float(inputs.radius(t)) if t<=inputs.radius_end else R0
    R=xp.where(b['moving_radius'],Rmeas,R0)[:,None]
    progress=(1-R/R0)/(1-inputs.radius_last/R0)
    eps=b['eps_final'][:,None]*progress
    a=grid['a'][None,:];f=grid['faces'][None,:]
    g=a*(1+eps*(1-a*a));ga=1+eps-3*eps*a*a
    gf=f*(1+eps*(1-f*f));gaf=1+eps-3*eps*f*f
    rho0=xp.where(b['question']==4,760+90*C0,650+128*C0)/(1+C0)
    # Analytic a=0 limit avoids 0/0 at the axis.
    jac=(R/R0)**2*(1+eps*(1-a*a))*ga
    rho_d=rho0[:,None]/jac
    volumes=xp.diff((R*gf)**2,axis=1)/2
    return {'R':R,'eps':eps,'g':g,'ga':ga,'gf':gf,'gaf':gaf,
            'rho0':rho0,'rho_d':rho_d,'volume':volumes,'r':R*g}


def properties(T,C,q,xp=np):
    rho=xp.where(q==4,760+90*C,650+128*C)
    cp=xp.where(q==4,1850+2150*C/(1+C),1450+2736*C/(1+C))
    k=xp.where(q==4,.12+.20*C/(1+C),.21+.38*C/(1+C))
    D=xp.where(q==4,4.2e-4,2.4e-3)*xp.exp(-xp.where(q==4,.30,.45)/C-3850/(T+273.15))
    return rho,cp,k,D


def perturbation_mask(a,C,b,xp=np):
    # a,C have shape (batch, quadrature, edge) or (batch,node).
    expand=(slice(None),)+(None,)*(C.ndim-1)
    width=b['region_width'][expand]
    outer=.5*(1+xp.tanh((a-(1-width))/(width/10)))
    low=.5*(1-xp.tanh((C-b['C_cut'][expand])/b['C_smooth'][expand]))
    kind=b['perturb_kind'][expand]
    return xp.where(kind==1,outer,xp.where(kind==2,1-outer,xp.where(kind==3,low,xp.where(kind==4,1.,0.))))


def rhs(t,y,case_arrays,grid,inputs,xp=np,phase=None):
    b=case_arrays;T=y[...,0];C=y[...,1]
    geom=geometry(t,b,grid,inputs,xp);q=b['question'][:,None]
    rho,cp,k,_=properties(T,C,q,xp)
    gs=grid['gauss_s'][None,:,None];gw=grid['gauss_w'][None,:,None]
    Ti=T[:,None,:-1]+gs*xp.diff(T,axis=1)[:,None,:]
    Ci=C[:,None,:-1]+gs*xp.diff(C,axis=1)[:,None,:]
    ai=grid['a'][None,None,:-1]+gs*grid['da'][None,None,:]
    Di=properties(Ti,Ci,b['question'][:,None,None],xp)[3]
    Di=Di*xp.exp(b['logD_delta'][:,None,None]*perturbation_mask(ai,Ci,b,xp))
    Df=xp.sum(gw*Di,axis=1)
    kf=2*k[:,:-1]*k[:,1:]/(k[:,:-1]+k[:,1:])
    ta,ce=inputs.boundary(t,phase)
    density=xp.where(b['boundary_initial'],geom['rho0'],geom['rho_d'][:,-1])
    jw=density*HM*(C[:,-1]-ce)
    # F is inward-signed r*j divided by initial dry density*R0^2.
    F=xp.zeros((y.shape[0],y.shape[1]+1),dtype=y.dtype)
    F[:,1:-1]=grid['faces'][None,1:-1]*Df*xp.diff(C,axis=1)/(geom['R']**2*geom['gaf'][:,1:-1]**2*grid['da'][None,:])
    F[:,-1]=-geom['R'][:,0]*jw/(geom['rho0']*R0**2)
    dC=xp.diff(F,axis=1)/grid['weights'][None,:]
    Q=xp.zeros_like(F)
    Q[:,1:-1]=geom['gf'][:,1:-1]/geom['gaf'][:,1:-1]*kf*xp.diff(T,axis=1)/grid['da'][None,:]
    Q[:,-1]=geom['R'][:,0]*H*(ta-T[:,-1])
    dT=xp.diff(Q,axis=1)/(geom['volume']*rho*cp)
    return xp.stack((dT,dC),axis=-1)


def diagnostics(t,y,b,grid,inputs,xp=np,phase=None):
    geom=geometry(t,b,grid,inputs,xp);T=y[...,0];C=y[...,1]
    _,ce=inputs.boundary(t,phase)
    density=xp.where(b['boundary_initial'],geom['rho0'],geom['rho_d'][:,-1])
    jw=density*HM*(C[:,-1]-ce)
    mean=2*xp.sum(grid['weights'][None,:]*C,axis=1)
    md=geom['rho0']*np.pi*R0**2*LENGTH
    rho=properties(T,C,b['question'][:,None],xp)[0]
    empirical_md=2*np.pi*LENGTH*xp.sum(rho/(1+C)*geom['volume'],axis=1)
    mask=perturbation_mask(grid['a'][None,:],C,b,xp)
    return {'mean_C':mean,'max_C':xp.max(C,axis=1),'surface_flux_kg_m2_s':jw,
            'outflow_kg_s':2*np.pi*LENGTH*geom['R'][:,0]*jw,'water_mass_kg':md*mean,
            'dry_mass_kg':md,'empirical_dry_mass_ratio':empirical_md/md,
            'mask_dry_mass_fraction':2*xp.sum(grid['weights'][None,:]*mask,axis=1),
            'radius_m':geom['R'][:,0]}
