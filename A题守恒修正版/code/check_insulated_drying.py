"""Nonzero water loss under adiabatic sensible-energy conditions must not cool/heat the body."""
from pathlib import Path
import json
import numpy as np
from conservative_model import ConservativeModel,ConservativeSettings
m=ConservativeModel(ConservativeSettings(problem=3,n=80,h=0)).solve(end=10800,event=False)
y=m.state(np.linspace(0,m.end,101))
result={'test':'adiabatic water loss, no explicit latent heat',
        'water_loss_kg':float(m.masses(0)[1]-m.masses(m.end)[1]),
        'max_T_change_C':float(np.max(abs(y[:m.m]-28))),
        'end_C_range':[float(y[m.m:,-1].min()),float(y[m.m:,-1].max())]}
assert result['water_loss_kg']>0.01
assert result['max_T_change_C']<1e-8
result['status']='PASS'
root=Path(__file__).resolve().parents[1]
(root/'results/insulated_drying_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
