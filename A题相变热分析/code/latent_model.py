from dataclasses import dataclass
from conservative_model import ConservativeModel,ConservativeSettings
from water_thermo import saturation


@dataclass
class TemperatureLatentSettings(ConservativeSettings):
    latent_law:str='iapws_saturation'


class TemperatureLatentModel(ConservativeModel):
    def fluxes(self,t,y):
        Fw,Q,Ew,capacity=super().fluxes(t,y)
        Lv=saturation(float(y[self.m-1]))[4]
        Q[-1]+=self.p.latent_fraction*(Lv-self.p.latent_heat_J_kg)*Fw[-1]
        return Fw,Q,Ew,capacity
