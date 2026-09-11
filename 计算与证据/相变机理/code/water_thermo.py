"""IAPWS SR1-86(1992), saturation equations (1)-(3), Clapeyron latent heat.
Primary source: https://iapws.org/technical-guidance/release/Supp-sat
T is Celsius externally; SI pressure, density, latent heat internally.
"""
import numpy as np
from scipy.optimize import brentq


def saturation(T_C):
    scalar=np.ndim(T_C)==0
    T=np.atleast_1d(T_C).astype(float)+273.15
    if np.any((T<273.16)|(T>647.096)):
        raise ValueError('IAPWS saturation temperature outside published range')
    tau=1-T/647.096
    a=np.array([-7.85951783,1.84408259,-11.7866497,22.6807411,-15.9618719,1.80122502])
    e=np.array([1.,1.5,3.,3.5,4.,7.5])
    f=np.sum(a[:,None]*tau**e[:,None],axis=0)
    fp=np.sum((a*e)[:,None]*tau**(e[:,None]-1),axis=0)
    p=22.064e6*np.exp(647.096/T*f)
    dp=p*(-647.096/T**2*f-fp/T)
    b=np.array([1.99274064,1.09965342,-.510839303,-1.75493479,-45.5170352,-6.74694450e5])
    eb=np.array([1/3,2/3,5/3,16/3,43/3,110/3])
    rl=322*(1+np.sum(b[:,None]*tau**eb[:,None],axis=0))
    c=np.array([-2.03150240,-2.68302940,-5.38626492,-17.2991605,-44.7586581,-63.9201063])
    ec=np.array([2,4,8,18,37,71])/6
    rv=322*np.exp(np.sum(c[:,None]*tau**ec[:,None],axis=0))
    latent=T*dp*(1/rv-1/rl)
    values=(p,dp,rl,rv,latent)
    return tuple(float(x[0]) for x in values) if scalar else values


def vapor_pressure_from_humidity_ratio(W,pressure=101325.):
    return pressure*np.asarray(W)/(.621945+np.asarray(W))


def dewpoint(W,pressure=101325.):
    pv=float(vapor_pressure_from_humidity_ratio(W,pressure))
    return brentq(lambda T:saturation(T)[0]-pv,.010001,100.)


def verification():
    # Rounded published verification values on page 7. Critical point handled too.
    rows=[]
    for T,refs in [(273.16,[611.657,44.436693,999.789,.00485426]),
                   (373.1243,[101325.,3616.,958.365,.597586]),
                   (647.096,[22.064e6,268000.,322.,322.])]:
        got=saturation(T-273.15)
        error=max(abs(np.asarray(got[:4])/refs-1))
        assert error<2e-4
        rows.append({'T_K':T,'p_Pa':got[0],'rho_liquid':got[2],'rho_vapor':got[3],
                     'latent_J_kg':got[4],'max_relative_difference_from_rounded_table':float(error)})
    return {'source':'IAPWS SR1-86(1992), page 7, Table 1','checks':rows,
            'latent_at_28C_J_kg':saturation(28.)[4],'latent_at_50C_J_kg':saturation(50.)[4],'status':'PASS'}
