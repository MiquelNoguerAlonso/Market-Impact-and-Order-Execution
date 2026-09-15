"""Mathematical checks supporting the manuscript's explicit proofs."""
import json
import math
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_function, gamma
from experiments import ROOT
from signature_checks import signature_reduction,gaussian_information
from cross_impact_checks import cross_impact_checks, crossover_checks
from defect_checks import defect_checks
from orderbook_bridge import bridge_checks


def exact_signature_counterexample():
    laws=[[(F(0),F(1,16)),(F(2,5),F(10,16)),(F(4,5),F(5,16))],
          [(F(1,5),F(5,16)),(F(3,5),F(10,16)),(F(1),F(1,16))]]
    def coordinate(u,word):
        a=[u,F(0)];b=[1-u,1-u]
        return sum(math.prod([a[j] for j in word[:k]],start=F(1))/math.factorial(k)
                   *math.prod([b[j] for j in word[k:]],start=F(1))/math.factorial(len(word)-k)
                   for k in range(len(word)+1))
    count=0
    for n in range(5):
        for word in product(range(2),repeat=n):
            x=[sum(p*coordinate(u,word) for u,p in law) for law in laws]
            assert x[0]==x[1];count+=1
    costs=[]
    for law in laws:
        assert sum(p*(1-u)**2/2 for u,p in law)==F(3,20)
        assert sum(p*(1-u)**3/3 for u,p in law)==F(1,15)
        costs.append(sum(float(p)*((1-float(u))**3/3-(1-float(u))**2/2+1
                                 -(2-float(u))*math.exp(-(1-float(u)))) for u,p in law))
    assert abs(costs[0]-costs[1])>4e-5
    return {'exact_matching_coordinates':count,'matching_depth':4,
            'expected_executed_quantity':'3/20','expected_temporary_cost_eta_one':'1/15',
            'expected_costs':costs}


def impact_checks():
    matrix=np.array([[0.,1.],[-1.,0.]])
    vertices=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[0.,0.]])
    cost=sum((matrix@((a+b)/2))@(b-a) for a,b in zip(vertices[:-1],vertices[1:]))
    assert cost==-2
    k=np.array([[1,.95,.2],[.95,1,.95],[.2,.95,1.]])
    nu=np.array([.25,-.5,.25]);defect=-float(nu@k@nu)
    assert abs(defect-.075)<1e-14
    assert abs(4**2*defect/2-.6)<1e-14
    riesz=[]
    for beta in [.2,.5,.8]:
        a=(1+beta)/2
        expected=(.5)**(-beta)*gamma(1+beta/2)*gamma((1-beta)/2)/math.sqrt(math.pi)
        for t in [.2,.5,.8]:
            density=lambda x:(x*(1-x))**(a-1)/beta_function(a,a)
            # Both integrable endpoint singularities are handled by weighted quadrature.
            left=quad(lambda x:(1-x)**(a-1)/beta_function(a,a),0,t,
                      weight='alg',wvar=(a-1,-beta),epsabs=1e-10)[0]
            right=quad(lambda x:x**(a-1)/beta_function(a,a),t,1,
                       weight='alg',wvar=(-beta,a-1),epsabs=1e-10)[0]
            assert abs(left+right-expected)<1e-8
        n=40;h=1/n;denom=(1-beta)*(2-beta)
        row=np.empty(n);row[0]=2*h**(-beta)/denom
        j=np.arange(1,n,dtype=float)
        row[1:]=h**(-beta)*((j+1)**(2-beta)-2*j**(2-beta)+(j-1)**(2-beta))/denom
        cell_matrix=row[np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])]
        energy=float(np.ones(n)@cell_matrix@np.ones(n)/n**2)
        assert abs(energy-2/denom)<1e-10
        assert np.linalg.eigvalsh(cell_matrix).min()>0
        altered=cell_matrix.copy();np.fill_diagonal(altered,0.)
        bias=energy-float(np.ones(n)@altered@np.ones(n)/n**2)
        assert abs(bias-h*row[0])<1e-10
        riesz.append({'beta':beta,'constant_potential':expected,'cell_energy':energy,
                      'uniform_energy_exact':2/denom,'zero_diagonal_energy_bias':bias})
    boundary=[]
    for q in [1.,2.,4.,10.]:
        average=(1+2/3*(q**1.5-1))/q;permanent=2/3*math.sqrt(q)
        assert abs(average-permanent-1/(3*q))<1e-12
        boundary.append({'q':q,'fair_pricing_boundary_term':average-permanent})
    return {'signed_loop_cost':cost,'grid_defect':defect,'gross_volume_four_profit':4**2*defect/2,
            'riesz':riesz,'pareto_boundary':boundary,'crossover':crossover_checks(),
            'transient_cross_impact':cross_impact_checks()}


def information_checks():
    def entropy(p):return -p*math.log(p)-(1-p)*math.log1p(-p)
    rows=[]
    for u in [.1,.05,.01,.005]:
        information=math.log(2)-entropy(.5+u)
        bound=math.sqrt(information/2)
        assert u<=bound+1e-12
        rows.append({'u':u,'mutual_information':information,'saving_Delta_one':u,
                     'ratio_to_sharp_ceiling':u/bound})
    assert rows[-1]['ratio_to_sharp_ceiling']>.9999
    return {'binary_sharpness':rows,'gaussian_expansion':gaussian_information()}


def main():
    r={'orderbook_bridge':bridge_checks(),
       'three_point_defect_and_phantom_profit':defect_checks(),
       'ordinary_signature_counterexample':exact_signature_counterexample(),
       'exponential_signature_algebra':signature_reduction(),
       'impact':impact_checks(),'information':information_checks(),
       'status':'All mathematical checks passed; proofs and scope are stated in the manuscript.'}
    (ROOT/'audit/mathematical_checks.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['status'])


if __name__=='__main__':main()
