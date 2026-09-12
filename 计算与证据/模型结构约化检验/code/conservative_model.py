"""Dry-solid material control volumes, conservative water and sensible energy.

This is a documented alternative closure, not an exact preservation of the
statement's empirical rho(C). cp, k, D and environmental data remain supplied.
No empirical density is silently reused as a conserved physical mass density.
"""
from dataclasses import dataclass
import numpy as np
from reference_model import Settings, RadialModel, properties, face_D


@dataclass
class ConservativeSettings(Settings):
    # empirical: diagnostic bookkeeping only, reproduces the original PDE.
    # capacity: actual wet density in heat capacity, neglect water enthalpy transport.
    # enthalpy: conservative sensible-energy flux, selected correction.
    thermal_closure: str = 'enthalpy'
    latent_heat_J_kg: float = 2.38e6
    grid_beta: float = 4.


def constituent_heat_capacities(problem):
    # Algebraic decomposition of the GIVEN cp(C) = (cs + cw*C)/(1+C).
    # For question 1 constant cp, this is an effective equal-cp decomposition.
    if problem == 1:
        return 2600., 2600.
    return (1850., 4000.) if problem == 4 else (1450., 4186.)


class ConservativeModel(RadialModel):
    def __init__(self, settings=None, data=None):
        super().__init__(settings or ConservativeSettings(), data)
        # Resolve the initial surface layer and late low-D surface region.
        u=np.linspace(0,1,self.p.n+1)
        beta=self.p.grid_beta
        self.x=u if beta==0 else 1-np.sinh(beta*(1-u))/np.sinh(beta)
        self.dx=np.diff(self.x)
        self.faces=(self.x[:-1]+self.x[1:])/2
        self.w=np.diff(np.r_[0,self.faces,1]**2)/2
        self.length = .25
        self.radius0 = float(self.env.radius(0))
        self.C0 = 2.55
        initial_rho = float(properties(np.array([28.]), np.array([self.C0]), self.p.problem)[0][0])
        self.rho_d0 = initial_rho / (1 + self.C0)
        self.volume0 = 2*np.pi*self.radius0**2*self.length*self.w
        # Each control volume follows the dry skeleton; these masses never change.
        self.dry_mass = self.rho_d0*self.volume0
        self.Md = float(self.dry_mass.sum())
        self.cs, self.cw = constituent_heat_capacities(self.p.problem)

    def volumes(self, t):
        R = float(self.env.radius(t))
        return 2*np.pi*R**2*self.length*self.w

    def densities(self, t, C):
        """Actual dry/wet bulk densities, obtained from conserved cell masses."""
        rd = self.dry_mass/self.volumes(t)
        return rd, rd*(1+np.asarray(C))

    def fluxes(self, t, y):
        """Outward fluxes at edges: water kg/s, conductive/sensible heat W."""
        T, C = y[:self.m], y[self.m:]
        R = float(self.env.radius(t))
        Ta, Ceq = self.env.ambient(t)
        rd, rw = self.densities(t, C)
        _, cp, k = properties(T, C, self.p.problem)
        kf = 2*k[:-1]*k[1:]/(k[:-1]+k[1:])
        rdf = 2*rd[:-1]*rd[1:]/(rd[:-1]+rd[1:])
        D = face_D(T[:-1], T[1:], C[:-1], C[1:], self.p.problem)*self.p.d_scale
        area = 2*np.pi*R*self.length*self.faces
        distance = R*self.dx
        Fw = np.r_[0., -area*rdf*D*np.diff(C)/distance,
                    2*np.pi*R*self.length*rd[-1]*self.p.hm*(C[-1]-Ceq)]
        Q = np.r_[0., -area*kf*np.diff(T)/distance,
                   2*np.pi*R*self.length*self.p.h*(T[-1]-Ta)]
        # Centered face temperature yields a second-order conservative energy flux.
        # At the external boundary incoming water uses ambient sensible enthalpy.
        Tface = np.r_[T[0], (T[:-1]+T[1:])/2, T[-1] if Fw[-1]>=0 else Ta]
        Ew = self.cw*Tface*Fw
        Q[-1] += self.p.latent_fraction*self.p.latent_heat_J_kg*Fw[-1]
        return Fw, Q, Ew, rw*cp

    def rhs(self, t, y):
        if self.p.thermal_closure == 'empirical':
            if self.p.latent_fraction:
                raise ValueError('Empirical diagnostic is only defined without latent heat.')
            return super().rhs(t, y)
        T, C = y[:self.m], y[self.m:]
        Fw, Q, Ew, capacity_volume = self.fluxes(t, y)
        dC = -np.diff(Fw)/self.dry_mass
        if self.p.thermal_closure == 'capacity':
            dT = -np.diff(Q)/(capacity_volume*self.volumes(t))
        elif self.p.thermal_closure == 'enthalpy':
            # d[md*(cs+cw*C)*T]/dt = -(energy_out-energy_in).
            energy_rate = -np.diff(Q+Ew)
            dT = (energy_rate-self.dry_mass*self.cw*T*dC)/(self.dry_mass*(self.cs+self.cw*C))
        else:
            raise ValueError('Unknown thermal closure: '+self.p.thermal_closure)
        return np.r_[dT, dC]

    def masses(self, t):
        C = self.state(t)[self.m:]
        Mw = self.dry_mass@C
        return self.Md, Mw, self.Md+Mw

    def sensible_energy(self, t):
        y = self.state(t)
        return self.dry_mass@((self.cs+self.cw*y[self.m:])*y[:self.m])

    def mean_C(self, t):
        return self.dry_mass@self.state(t)[self.m:]/self.Md
