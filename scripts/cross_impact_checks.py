"""Independent checks of cross-impact energy, weighted area, and lift endpoints.

Exact integrals on constant-rate cells are compared with state-ODE integration.
The fixtures include permanent impact, scalar transient mixtures, and a
lag-dependent matrix kernel whose antisymmetric entries vary with the lag.
"""
from fractions import Fraction
from itertools import combinations
import math

import numpy as np
from scipy.integrate import solve_ivp


def crossover_checks():
    cases=[]
    for c,kappa in [(1,1),(2,3),(5,2)]:
        c,kappa=Fraction(c),Fraction(kappa)
        level_root=2*c/kappa
        slope_root=3*c/(2*kappa)
        assert kappa*level_root**4/2==c*level_root**3
        assert kappa*slope_root**2==3*c*slope_root/2
        assert level_root**2/slope_root**2==Fraction(16,9)
        cases.append({'c':str(c),'kappa':str(kappa),
                      'level_matching_size':str(level_root**2),
                      'slope_matching_size':str(slope_root**2)})
    return {'cases':cases,'ratio_level_to_slope_size':'16/9'}


def triangular_exponential_cells(rho,n,horizon):
    width=horizon/n
    cells=np.zeros((n,n))
    if rho==0:
        cells[np.tril_indices(n,-1)]=width**2
        np.fill_diagonal(cells,width**2/2)
        return cells
    lost=-math.expm1(-rho*width)
    for late in range(n):
        cells[late,late]=width/rho-lost/rho**2
        for early in range(late):
            cells[late,early]=math.exp(-rho*(late-early-1)*width)*lost**2/rho**2
    return cells


def exact_cell_split(rates,components,horizon=1.):
    n,d=rates.shape
    energy=pairing=direct=0.
    for rho,matrix in components:
        cells=triangular_exponential_cells(rho,n,horizon)
        symmetric=.5*(matrix+matrix.T)
        skew=.5*(matrix-matrix.T)
        direct+=float(np.sum(cells*(rates@matrix@rates.T)))
        energy+=.5*float(np.sum((cells+cells.T)*(rates@symmetric@rates.T)))
        for i,j in combinations(range(d),2):
            area=.5*sum(cells[t,s]*(rates[s,i]*rates[t,j]-rates[s,j]*rates[t,i])
                         for t in range(n) for s in range(t))
            pairing+=2*skew[i,j]*area
    return float(energy),float(pairing),float(direct)


def state_ode(rates,components,horizon=1.):
    n,d=rates.shape
    count=len(components)
    pairs=list(combinations(range(d),2))
    impacts=count*d
    dissipation_start=impacts+1
    areas_start=dissipation_start+count
    state=np.zeros(areas_start+count*len(pairs))
    for cell,rate in enumerate(rates):
        def rhs(t,z):
            out=np.zeros_like(z)
            for m,(rho,matrix) in enumerate(components):
                impact=z[m*d:(m+1)*d]
                derivative=rate-rho*impact
                out[m*d:(m+1)*d]=derivative
                out[impacts]+=rate@matrix@impact
                symmetric=.5*(matrix+matrix.T)
                out[dissipation_start+m]=impact@symmetric@impact
                for k,(i,j) in enumerate(pairs):
                    out[areas_start+m*len(pairs)+k]=.5*(impact[i]*derivative[j]-impact[j]*derivative[i])
            return out
        a,b=cell*horizon/n,(cell+1)*horizon/n
        sol=solve_ivp(rhs,(a,b),state,method='DOP853',rtol=1e-11,atol=1e-13)
        assert sol.success
        state=sol.y[:,-1]
    endpoint=dissipation=pairing=0.
    for m,(rho,matrix) in enumerate(components):
        impact=state[m*d:(m+1)*d]
        symmetric=.5*(matrix+matrix.T)
        skew=.5*(matrix-matrix.T)
        endpoint+=.5*impact@symmetric@impact
        dissipation+=rho*state[dissipation_start+m]
        for k,(i,j) in enumerate(pairs):
            pairing+=2*skew[i,j]*state[areas_start+m*len(pairs)+k]
    return {'direct_cost':float(state[impacts]),'symmetric_endpoint':float(endpoint),
            'symmetric_dissipation':float(dissipation),'area_pairing':float(pairing),
            'terminal_impact_states':state[:impacts].reshape(count,d).tolist()}


def cross_impact_checks():
    matrix=np.array([[1.,1.],[-1.,.7]])
    rectangle=np.array([[4.,0.],[0.,4.],[-4.,0.],[0.,-4.]])
    paths={
        'unit_square':rectangle,
        'mixed_closed':np.array([[1.,3.],[2.,-1.],[-1.,2.],[0.,-3.],[1.,-1.],[-3.,0.]]),
        'open_path':np.array([[1.,0.],[.5,1.],[0.,2.]])}
    models={
        'permanent':[(0.,matrix)],
        'scalar_exponential':[(1.,matrix)],
        'scalar_mixture':[(.6,.7*matrix),(1.7,.3*matrix)],
        'lag_dependent_matrix':[
            (.8,np.array([[1.,.7],[-.3,1.4]])),
            (2.2,np.array([[.4,-.8],[.2,.6]]))]}
    rows=[]
    max_error=0.
    for model,components in models.items():
        for path,rates in paths.items():
            energy,pairing,direct_cells=exact_cell_split(rates,components)
            ode=state_ode(rates,components)
            reversal=-rates[::-1]
            reverse_energy,reverse_pairing,reverse_cells=exact_cell_split(reversal,components)
            reverse_ode=state_ode(reversal,components)
            errors=[
                abs(direct_cells-(energy-pairing)),
                abs(ode['direct_cost']-(energy-pairing)),
                abs(energy-(ode['symmetric_endpoint']+ode['symmetric_dissipation'])),
                abs(pairing-ode['area_pairing']),
                abs(reverse_energy-energy),abs(reverse_pairing+pairing),
                abs(reverse_cells-(energy+pairing)),
                abs(reverse_ode['direct_cost']-(energy+pairing))]
            max_error=max(max_error,max(errors))
            assert max(errors)<1e-9,(model,path,errors)
            rows.append({'model':model,'path':path,'symmetric_energy':energy,
                         'weighted_area_pairing':pairing,'cost_from_cells':direct_cells,
                         'cost_from_ode':ode['direct_cost'],
                         'cost_reversed':reverse_ode['direct_cost'],
                         'endpoint':ode['symmetric_endpoint'],
                         'dissipation':ode['symmetric_dissipation'],
                         'maximum_identity_error':max(errors)})
    # The permanent unit square must give the signed-area cost -2.
    assert abs(rows[0]['cost_from_ode']+2)<1e-12
    # A closed inventory loop need not close its impact-state lift.
    lifted=state_ode(rectangle,[(1.,matrix)])
    assert np.linalg.norm(lifted['terminal_impact_states'][0])>.1
    assert lifted['symmetric_endpoint']>0
    short=[]
    for horizon in [.1,.01,.001]:
        rates=rectangle/horizon
        observed=state_ode(rates,[(1.,matrix)],horizon)['direct_cost']
        _,_,from_cells=exact_cell_split(rates,[(1.,matrix)],horizon)
        assert abs(observed-from_cells)<1e-9
        # Continuity bound on the cost difference from g(0)=1.
        limit_error_bound=.5*np.linalg.norm(matrix,2)*4**2*(-math.expm1(-horizon))
        assert abs(observed+2)<=limit_error_bound+1e-10
        assert observed<0
        short.append({'horizon':horizon,'cost':observed,
                      'permanent_limit':-2.,'continuity_error_bound':float(limit_error_bound)})
    assert abs(short[-1]['cost']+2)<.01
    # Counterexample to inferring A=0 from zero symmetric energy alone.
    skew=np.array([[0.,1.],[-1.,0.]])
    ramp=[]
    for name,rates in paths.items():
        n=len(rates);h=1/n
        cells=np.zeros((n,n))
        for t in range(n):
            cells[t,t]=h**3/6
            for s in range(t):
                cells[t,s]=(t-s)*h**3
        direct=float(np.sum(cells*(rates@skew@rates.T)))
        net=h*rates.sum(axis=0)
        first_moment=sum(rates[k]*((k+1)**2-k**2)*h*h/2 for k in range(n))
        factored=float(first_moment@skew@net)
        assert abs(direct-factored)<1e-12
        if name!='open_path':
            assert abs(direct)<1e-12
        ramp.append({'path':name,'net_vector':net.tolist(),
                     'g_r_cost':direct,'first_moment_times_A_times_net':factored})
    return {'model_path_pairs':len(rows),'maximum_identity_error':max_error,
            'checks':rows,'closed_inventory_terminal_impact':lifted['terminal_impact_states'],
            'short_loop_limit':short,'zero_symmetric_energy_counterexample':ramp,
            'conclusion':'Energy-area decomposition, matrix-lag reversal, and finite-exponential endpoint formula pass independent cell-integral and ODE checks.'}
