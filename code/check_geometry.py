"""Independent 2D axisymmetric check of the 1D approximation, with both end faces exposed."""
import json,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags,kron,eye,bmat
from model import ROOT,Settings,Environment,properties,face_D,RadialModel


def solve_2d(problem,nr=40,nz=40):
    p=Settings(problem=problem,n=nr,rtol=2e-7,atol=2e-9)
    env=Environment(p)
    x=np.linspace(0,1,nr+1);f=(x[:-1]+x[1:])/2
    wr=np.diff(np.r_[0,f,1]**2)/2
    z=np.linspace(0,.125,nz+1);dz=z[1]
    wz=np.diff(np.r_[0,(z[:-1]+z[1:])/2,.125])
    shape=(nr+1,nz+1);M=(nr+1)*(nz+1)
    rtri=diags([np.ones(nr),np.ones(nr+1),np.ones(nr)],[-1,0,1])
    ztri=diags([np.ones(nz),np.ones(nz+1),np.ones(nz)],[-1,0,1])
    adj=kron(rtri,eye(nz+1))+kron(eye(nr+1),ztri)
    sparse=bmat([[adj,adj],[adj,adj]],format='csc')
    def rhs(t,y):
        T=y[:M].reshape(shape);C=y[M:].reshape(shape)
        R=float(env.radius(t));Ta,Ca=env.ambient(t)
        rho,cp,k=properties(T,C,problem)
        dr=face_D(T[:-1],T[1:],C[:-1],C[1:],problem)
        dzf=face_D(T[:,:-1],T[:,1:],C[:,:-1],C[:,1:],problem)
        kr=2*k[:-1]*k[1:]/(k[:-1]+k[1:]);kz=2*k[:,:-1]*k[:,1:]/(k[:,:-1]+k[:,1:])
        out=[]
        for v,fr,fz,coef in [(T,kr,kz,p.h),(C,dr,dzf,p.hm)]:
            ambient=Ta if coef==p.h else Ca
            qr=f[:,None]*fr*np.diff(v,axis=0)*nr
            qr=np.vstack([np.zeros((1,nz+1)),qr,-R*coef*(v[-1:]-ambient)])
            qz=fz*np.diff(v,axis=1)/dz
            qz=np.column_stack([np.zeros(nr+1),qz,-coef*(v[:,-1]-ambient)])
            div=np.diff(qr,axis=0)/(R*R*wr[:,None])+np.diff(qz,axis=1)/wz
            if coef==p.h:div/=rho*cp
            out.append(div.ravel())
        return np.r_[out[0],out[1]]
    def event(t,y):return y[M:].max()-.15
    event.terminal=True;event.direction=-1
    y=np.r_[np.full(M,28.),np.full(M,2.55)]
    tic=time.perf_counter();parts=[]
    for lo,hi in [(0.,14400.),(14400.,864000.)]:
        s=solve_ivp(rhs,(lo,hi),y,method='BDF',rtol=p.rtol,atol=p.atol,jac_sparsity=sparse,
                    events=event,max_step=60. if lo==0 else 1200.,dense_output=True)
        assert s.success,s.message
        parts.append(s);y=s.y[:,-1]
        if s.status==1:break
    assert s.status==1
    ref=RadialModel(p).solve()
    end=s.t[-1];C=y[M:].reshape(shape)
    row={'problem':problem,'nr':nr,'nz':nz,'time_2d_h':end/3600,'time_1d_same_nr_h':ref.end/3600,
         'time_difference_s':float(end-ref.end),'change_pct':100*(end/ref.end-1),
         'maximum_grid_location_r_z_m':[float(env.radius(end))*x[np.unravel_index(np.argmax(C),shape)[0]],z[np.unravel_index(np.argmax(C),shape)[1]]],
         'max_midplane_profile_difference_C':float(np.max(abs(C[:,0]-ref.state(end)[ref.m:]))),
         'mean_2d_C_at_crossing':float(np.sum(C*wr[:,None]*wz[None,:])/(.5*.125)),
         'end_face_center_C_at_crossing':float(C[0,-1]),'wall_s':time.perf_counter()-tic}
    print(json.dumps(row),flush=True)
    return row


if __name__=='__main__':
    rows=[]
    for q in [3,4]:
        rows.append(solve_2d(q))
        (ROOT/'results/geometry_validation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
