"""Independent Gauss integration on stored time segments; no inventory renormalization."""
import argparse,json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from two_phase import Model,Parameters,ROOT
from run_analysis import get,save

def check(m):
    out=[]
    for order in [2,4]:
        nodes,weights=leggauss(order);integ=np.zeros(5)
        # Add every piecewise-linear environment knot to the quadrature partition.
        for sol in m.solutions:
            edges=np.unique(np.r_[sol.t,np.arange(0,14401,60.)[(np.arange(0,14401,60.)>=sol.t[0])&(np.arange(0,14401,60.)<=sol.t[-1])]])
            for lo,hi in zip(edges[:-1],edges[1:]):
                for n,w in zip(nodes,weights):
                    t=(lo+hi)/2+(hi-lo)*n/2;y=sol.sol(t)
                    FL,FV,Q,E,S,_=m.fluxes(t,y)
                    integ+=w*(hi-lo)/2*np.array([FL[-1]+FV[-1],-Q[-1],E[-1],m.V@S,FL[-1]])
        w0,h0=m.inventories(m.y0);we,he=m.inventories(m.state(m.end))
        out.append({'gauss_order':order,'water_residual_kg':float(we-w0+integ[0]),'water_relative_residual':float(abs(we-w0+integ[0])/w0),
                    'energy_residual_J':float(he-h0-integ[1]+integ[2]),'energy_relative_residual':float(abs(he-h0-integ[1]+integ[2])/max(abs(integ[1]),1.)),
                    'net_internal_evaporation_kg':float(integ[3]),'surface_liquid_evaporation_kg':float(integ[4]),
                    'water_out_kg':float(integ[0]),'convective_heat_in_J':float(integ[1]),'water_enthalpy_out_J':float(integ[2])})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);args=ap.parse_args();cache=Path(args.cache)
    result={}
    for label,k,e in [('neither',0.,0.),('capillary',3e-19,0.),('internal',0.,1.),('coupled',3e-19,1.)]:
        m=get(Parameters(n=80,permeability=k,evaporation_rate=e),cache)
        result[label]=check(m)
        save('refined_conservation.json',result)
        print(label,result[label][-1],flush=True)
    assert all(v[-1]['water_relative_residual']<1e-5 and v[-1]['energy_relative_residual']<1e-5 for v in result.values())
    save('refined_conservation.json',{'status':'PASS','cases':result})

if __name__=='__main__':main()
