"""Synthetic fill/completion dependence stress; no market observations."""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1]

def coupling_extrema(p,q,cost):
 n,m=cost.shape
 rows=[]
 for i in range(n):
  row=np.zeros((n,m));row[i,:]=1;rows.append(row.ravel())
 for j in range(m):
  row=np.zeros((n,m));row[:,j]=1;rows.append(row.ravel())
 vals=[]
 for sign in [1,-1]:
  sol=linprog(sign*cost.ravel(),A_eq=rows,b_eq=np.r_[p,q],bounds=(0,None),method='highs')
  assert sol.success,sol.message
  assert np.max(np.abs(np.asarray(rows)@sol.x-np.r_[p,q]))<1e-10
  vals.append(float(cost.ravel()@sol.x))
 return vals

def main():
 weights=[F(45,100),F(35,100),F(20,100)]
 fills=[F(1),F(1,2),F(0)]; prices=[F(2,100),F(8,100),F(30,100)]
 rebate=F(1,1000);market=F(6,100)
 ef=sum(p*f for p,f in zip(weights,fills));ek=sum(p*k for p,k in zip(weights,prices))
 paired=sum(p*(k*(1-f)-rebate*f) for p,f,k in zip(weights,fills,prices))
 independent=sum(p*q*(k*(1-f)-rebate*f) for p,f in zip(weights,fills) for q,k in zip(weights,prices))
 cov=sum(p*f*k for p,f,k in zip(weights,fills,prices))-ef*ek
 assert paired==ek*(1-ef)-rebate*ef-cov
 assert independent==F(3575,100000) and paired==F(73375,1000000)
 cross=(market-independent)/(paired-independent)
 for t in range(101):
  theta=F(t,100)
  mass=[[weights[i]*weights[j]*(1-theta)+(theta*weights[i] if i==j else 0) for j in range(3)] for i in range(3)]
  direct=sum(mass[i][j]*(prices[j]*(1-fills[i])-rebate*fills[i]) for i in range(3) for j in range(3))
  assert direct==independent+theta*(paired-independent)
  assert (direct<=market)==(theta<=cross)
 p=np.array([float(v) for v in weights]); f=np.array([float(v) for v in fills]);k=np.array([float(v) for v in prices])
 costs=k[None,:]*(1-f[:,None])-float(rebate)*f[:,None]
 extrema=coupling_extrema(p,p,costs)
 assert abs(extrema[1]-float(paired))<1e-12
 rng=np.random.default_rng(25092026);error=0.
 for _ in range(100):
  joint=rng.dirichlet(np.ones(12)).reshape(3,4);fv=rng.random(3);kv=rng.random(4)
  fp=joint.sum(1);kp=joint.sum(0);ef1=fp@fv;ek1=kp@kv
  cov1=np.sum(joint*fv[:,None]*kv[None,:])-ef1*ek1
  val=np.sum(joint*(kv[None,:]*(1-fv[:,None])-float(rebate)*fv[:,None]))
  error=max(error,abs(val-(ek1*(1-ef1)-float(rebate)*ef1-cov1)))
 assert error<1e-12
 rows=[('Independent',float(independent),'Passive'),('Paired',float(paired),'Market'),('Marginal-only worst case',extrema[1],'Market')]
 (ROOT/'tables/realism_rows.tex').write_text(''.join(f'{name} & {val:.6f} & {choice} \\\\\n' for name,val,choice in rows))
 result={'synthetic_only':True,'mean_fill':float(ef),'mean_completion_cost':float(ek),'independent_cost':float(independent),'paired_cost':float(paired),'market_cost':float(market),'covariance':float(cov),'marginal_cost_extrema':extrema,'switch_theta':float(cross),'wrong_action_loss':float(paired-market),'exact_mixture_cases':101,'random_covariance_checks':100,'max_identity_error':error}
 (ROOT/'results/realism_stress.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
