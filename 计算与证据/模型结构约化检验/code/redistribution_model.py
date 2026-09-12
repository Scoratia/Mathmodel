"""Keep effective total-water law; relocate a fraction of latent release to bulk.

This is a conditional phase-location test, not a calibrated vapor constitutive model.
Vapor storage neglected; Jv=eta*w(x)*J_total internally and local source=div(Jv).
The total boundary latent power remains Lv*F_total in every case.
"""
from dataclasses import dataclass
from conservative_model import ConservativeSettings,ConservativeModel

@dataclass
class Settings(ConservativeSettings):
    latent_fraction:float=1.
    internal_fraction:float=0.
    shape_power:float=0.

class RedistributionModel(ConservativeModel):
    def fluxes(self,t,y):
        F,Q,E,cap=super().fluxes(t,y)
        Q[1:-1]+=self.p.latent_heat_J_kg*self.p.internal_fraction*self.faces**self.p.shape_power*F[1:-1]
        return F,Q,E,cap
