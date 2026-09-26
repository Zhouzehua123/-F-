"""Read-only Q3 reproduction delta audit. Writes evidence only to an explicit independent output directory."""
from pathlib import Path
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import argparse

parser = argparse.ArgumentParser(description="Read-only Q3 numerical reproduction drift diagnosis")
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--reproduced', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.source.resolve()
WORKSPACE = args.reproduced.resolve()
HERE = args.output.resolve()
if HERE.is_relative_to(ROOT) or HERE.is_relative_to(WORKSPACE):
    parser.error('Output must be outside both input snapshots')
HERE.mkdir(parents=True, exist_ok=True)
RELATIVE = '求解/问题三/结果/预算扫略.csv'
K = 6 + 2e-4 * 2048
paths = [base/rel for base in [ROOT, WORKSPACE] for rel in [RELATIVE, '求解/广义标度律参数.csv', '求解/问题一_关键量.json', '求解/问题三/问题三.py']]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

before = {str(p): sha(p) for p in paths}
old = pd.read_csv(ROOT/RELATIVE)
new = pd.read_csv(WORKSPACE/RELATIVE)
assert old.shape == new.shape and list(old.columns) == list(new.columns)
assert np.array_equal(old.C, new.C)
params = [pd.read_csv(base/'求解/广义标度律参数.csv').set_index('param').value.to_dict() for base in [ROOT, WORKSPACE]]
q0s = [float(json.loads((base/'求解/问题一_关键量.json').read_text(encoding='utf-8'))['Q0_mix_weighted']) for base in [ROOT, WORKSPACE]]
assert q0s[0] == q0s[1]
cells=[]
for c in old.select_dtypes('number'):
    a,b=old[c].to_numpy(),new[c].to_numpy()
    for i in np.flatnonzero(~np.isclose(a,b,rtol=1e-6,atol=1e-8,equal_nan=True)):
        cells.append(dict(row_index=int(i),csv_line=int(i+2),C=float(old.C.iloc[i]),field=c,saved=float(a[i]),regenerated=float(b[i]),absolute_delta=float(abs(b[i]-a[i])),relative_delta_vs_saved=float(abs(b[i]-a[i])/abs(a[i])) if a[i] else None,absolute_delta_over_budget=float(abs(b[i]-a[i])/old.C.iloc[i]) if c.startswith('c_') else None))
pd.DataFrame(cells).to_csv(HERE/'failed_cells.csv',index=False,encoding='utf-8-sig')
indices=sorted({c['row_index'] for c in cells})

def loss(p,n,d,q):
    return p['E']+p['A']*n**-p['alpha']+p['B']*d**-p['beta']+p['C']*(1-q)**p['gamma']

def fixed_q(p,budget,q,q0):
    """Budget eliminate D, solve unique N stationary point in original bounds."""
    cb=budget/1e18
    s=5*(q**4-q0**4)
    lo=max(1e-4,(cb/1e7-s)/K)
    hi=min(1e6,(cb/1e-3-s)/K)
    assert hi>lo
    def derivative(z):
        n=np.exp(z);d=cb/(K*n+s)
        return -p['alpha']*p['A']*n**-p['alpha']+p['beta']*p['B']*d**-p['beta']*K*n/(K*n+s)
    zlo,zhi=np.log(lo),np.log(hi)
    z=zlo if derivative(zlo)>=0 else zhi if derivative(zhi)<=0 else brentq(derivative,zlo,zhi,xtol=1e-14)
    n=float(np.exp(z));d=float(cb/(K*n+s))
    return n,d,float(loss(p,n,d,q))

def profile(p,budget,q0):
    """Independent derivative-root profile, endpoints included; no SLSQP."""
    def dq(q):
        n,d,l=fixed_q(p,budget,q,q0)
        s=5*(q**4-q0**4)
        return p['beta']*p['B']*d**-p['beta']*20*q**3/(K*n+s)-p['C']*p['gamma']*(1-q)**(p['gamma']-1)
    grid=np.unique(np.r_[np.linspace(q0,1-1e-12,513),1-10.**np.arange(-15.,-1.)])
    grid=grid[(grid>=q0)&(grid<1)]
    vals=[dq(q) for q in grid]
    qs=[q0,1.]
    for a,b,va,vb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
        if va*vb<0: qs.append(float(brentq(dq,a,b,xtol=1e-14)))
    candidates=[]
    for q in qs:
        n,d,l=fixed_q(p,budget,q,q0)
        candidates.append(dict(N_B=n,D_B=d,Q=q,L=l))
    answer=min(candidates,key=lambda x:x['L'])
    return answer,candidates

checks=[]
profile_details=[]
for i in indices:
    r=dict(row_index=i,csv_line=i+2,C=float(old.C.iloc[i]))
    a,b=old.iloc[i],new.iloc[i]
    for c in ['N_B','D_B','Q','L','D_over_N','c_qual','f_qual']:
        r[c+'_saved']=float(a[c]);r[c+'_regenerated']=float(b[c]);r[c+'_delta']=float(b[c]-a[c])
    for label, row, p,q0 in zip(['saved','regenerated'],[a,b],params,q0s):
        cost=row.D_B*1e9*(K*row.N_B*1e9+max(5e9*(row.Q**4-q0**4),0.))
        r[label+'_budget_relative_residual']=float(cost/row.C-1)
        r[label+'_loss_formula_error']=float(loss(p,row.N_B,row.D_B,row.Q)-row.L)
        r[label+'_Q_minus_Q0']=float(row.Q-q0)
        r[label+'_bounds_ok']=bool(1e-4<=row.N_B<=1e6 and 1e-3<=row.D_B<=1e7 and q0-1e-15<=row.Q<=1+1e-15)
        pr,candidates=profile(p,row.C,q0)
        for c in ['N_B','D_B','Q','L']:
            r[label+'_profile_'+c]=pr[c]
        r[label+'_loss_minus_profile']=float(row.L-pr['L'])
        r[label+'_reevaluated_loss_minus_profile']=float(loss(p,row.N_B,row.D_B,row.Q)-pr['L'])
        r[label+'_profile_N_relative_error']=float((row.N_B-pr['N_B'])/pr['N_B'])
        r[label+'_profile_D_relative_error']=float((row.D_B-pr['D_B'])/pr['D_B'])
        profile_details.append(dict(row_index=i,parameter_set=label,candidates=candidates))
    # Compare both returned points under precisely the same saved coefficients.
    r['new_minus_saved_loss_same_old_parameters']=float(loss(params[0],b.N_B,b.D_B,b.Q)-loss(params[0],a.N_B,a.D_B,a.Q))
    checks.append(r)
df=pd.DataFrame(checks)
df.to_csv(HERE/'affected_budget_checks.csv',index=False,encoding='utf-8-sig')
param_rows=[dict(param=k,saved=params[0][k],regenerated=params[1][k],delta=params[1][k]-params[0][k],relative_delta=(params[1][k]-params[0][k])/params[0][k]) for k in params[0]]
pd.DataFrame(param_rows).to_csv(HERE/'parameter_deltas.csv',index=False,encoding='utf-8-sig')
all_differences={}
for c in old.select_dtypes('number'):
    delta=np.abs(new[c]-old[c]); finite=np.isfinite(delta)
    all_differences[c]=dict(max_absolute_delta=float(delta[finite].max()),max_relative_delta_nonzero=float((delta[finite & old[c].ne(0)]/old.loc[finite & old[c].ne(0),c].abs()).max()),nonfinite_pattern_equal=bool(np.array_equal(np.isfinite(old[c]),np.isfinite(new[c]))))
hashes_unchanged=all(sha(p)==before[str(p)] for p in paths)
assert hashes_unchanged
summary=dict(source=str(ROOT),workspace=str(WORKSPACE),shape=list(old.shape),failed_cells=len(cells),affected_rows=len(indices),failed_columns=sorted({c['field'] for c in cells}),rtol=1e-6,atol=1e-8,Q0=q0s[0],all_column_delta_maxima=all_differences,max_abs_budget_relative_residual=float(df[['saved_budget_relative_residual','regenerated_budget_relative_residual']].abs().to_numpy().max()),max_abs_recorded_loss_minus_profile=float(df[['saved_loss_minus_profile','regenerated_loss_minus_profile']].abs().to_numpy().max()),max_abs_reevaluated_loss_minus_profile=float(df[['saved_reevaluated_loss_minus_profile','regenerated_reevaluated_loss_minus_profile']].abs().to_numpy().max()),max_abs_loss_change_at_same_parameters=float(df.new_minus_saved_loss_same_old_parameters.abs().max()),all_affected_bounds_ok=bool(df.saved_bounds_ok.all() and df.regenerated_bounds_ok.all()),finite_loss_rows_saved=int(old.L.notna().sum()),finite_loss_rows_regenerated=int(new.L.notna().sum()),source_and_workspace_inputs_unchanged=hashes_unchanged,input_sha256=before,profile_method='Budget elimination; logN brentq; Q derivative sign-change roots on 513-point grid plus endpoint-tail grid; include both endpoints. Finite diagnostic, not global proof.',profile_candidates=profile_details)
(HERE/'q3_delta_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['input_sha256','profile_candidates','all_column_delta_maxima']},ensure_ascii=False,indent=2))
print(df[['row_index','C','Q_delta','L_delta','saved_loss_minus_profile','regenerated_loss_minus_profile','new_minus_saved_loss_same_old_parameters']].to_string(index=False))
print(pd.DataFrame(param_rows).to_string(index=False))
