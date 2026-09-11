"""Independent readback of all spreadsheet cells and density-field mass integrals."""
from pathlib import Path
import json,math,hashlib
import numpy as np
import openpyxl
from reference_model import properties

ROOT=Path(__file__).resolve().parents[1]


def main():
    evidence={'workbooks':[],'density_fields':{}}
    for q in range(1,5):
        f=ROOT/f'result{q}.xlsx'
        expected=json.loads((ROOT/f'results/result{q}.json').read_text(encoding='utf-8'))
        wb=openpyxl.load_workbook(f,data_only=True,read_only=True)
        assert wb.sheetnames==list(expected)
        cells=blanks=0
        for name,rows in expected.items():
            s=wb[name]
            if s.max_row is None or s.max_column is None:s.calculate_dimension(force=True)
            assert (s.max_row,s.max_column)==(len(rows),len(rows[0]))
            for i,(actual,want) in enumerate(zip(s.values,rows)):
                for j,(a,b) in enumerate(zip(actual,want)):
                    if b is None:assert a is None;blanks+=1
                    elif isinstance(b,(float,int)):
                        assert isinstance(a,(float,int)) and math.isfinite(a)
                        assert abs(a-b)<1e-9,(q,name,i,j,a,b)
                    else:assert a==b
                    cells+=1
                if i:
                    assert actual[0]==i*(1 if q<=2 else 60)
                    if q==4:
                        R=actual[-1]
                        for j,r in enumerate(np.arange(0,2.001,.1),1):
                            if r>R+1e-6:assert actual[j] is None
                            elif r<R-1e-6:assert actual[j] is not None
            assert s.cell(2,2).number_format=='0.0000'
        wb.close()
        evidence['workbooks'].append({'file':f.name,'cells_checked':cells,'blanks_outside_material':blanks,
                                      'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'status':'PASS'})
    arr=np.load(ROOT/'results/profiles_full_precision.npz')
    assert all(np.isfinite(arr[k]).all() for k in arr.files)
    for q in [1,3,4]:
        x=arr[f'q{q}_x'];C=arr[f'q{q}_C_dry_basis'];R=arr[f'q{q}_radius_m']
        rw=arr[f'q{q}_wet_density'];rd=arr[f'q{q}_dry_density']
        edges=np.r_[0,(x[:-1]+x[1:])/2,1]
        volumes=np.pi*.25*np.diff(edges**2)[:,None]*R[None,:]**2
        dry=np.sum(rw/(1+C)*volumes,axis=0)
        rho0=properties(np.array([28.]),np.array([2.55]),q)[0][0]
        md0=rho0/(1+2.55)*np.pi*R[0]**2*.25
        error=float(np.max(abs(dry/md0-1)))
        assert error<5e-13
        assert np.max(abs(rw-rd*(1+C)))<1e-9
        evidence['density_fields'][str(q)]={'initial_dry_mass_from_given_initial_density_kg':float(md0),
                                            'max_relative_mass_error_reconstructed_from_saved_fields':error,'status':'PASS'}
    evidence['status']='PASS'
    evidence['scope']='Saved-cell readback and conserved-density arithmetic; not experimental validation.'
    (ROOT/'results/delivery_checks.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
