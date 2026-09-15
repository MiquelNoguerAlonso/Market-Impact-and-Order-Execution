"""Reproduce the completion-feature QP and deterministic error certificates.

The finite exogenous law is fully known here. Coefficient errors are measured
against it to validate a deterministic bound, not inferred as market confidence
intervals. Uses an independent state-ODE representation of transient cost.
"""
import json
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from experiments import ROOT,savefig,COLORS

M=np.array([1.,.5,.5]);R=2.0
SUPPORT=np.linspace(0,1,6);WEIGHTS=np.full(6,1/6)
ETA=.15


def features(t,u):
    z=max(t-u,0.)
    return np.array([1.,1-2*t,t-1.5*t*t,(1-t)*z-.5*z*z])


def scenario_coefficients(rho,u):
    # 4 impact states, 4x4 transient products, 4x4 temporary products,
    # 4 linear-charge integrals, and 4 executed-quantity integrals.
    state=np.zeros(44)
    def rhs(t,s):
        f=features(t,u);out=np.zeros_like(s)
        out[:4]=-rho*s[:4]+f
        out[4:20]=np.outer(f,s[:4]).ravel()
        out[20:36]=np.outer(f,f).ravel()
        out[36:40]=(-.3*max(t-u,0.))*f
        out[40:44]=f
        return out
    cuts=sorted(set([0.,float(u),1.]))
    for a,b in zip(cuts[:-1],cuts[1:]):
        sol=solve_ivp(rhs,(a,b),state,method='DOP853',rtol=1e-11,atol=1e-13)
        assert sol.success
        state=sol.y[:,-1]
    assert np.max(np.abs(state[40:44]-[1,0,0,0]))<1e-10
    trans=state[4:20].reshape(4,4);local=state[20:36].reshape(4,4)
    full=.5*(trans+trans.T)+ETA*local;linear=state[36:40]
    return full[1:,1:],2*full[0,1:]+linear[1:],float(full[0,0]+linear[0])


def combine(coefficients,weights):
    return tuple(np.tensordot(weights,np.array([c[k] for c in coefficients]),axes=(0,0)) for k in range(3))


def value(coef,ell):
    q,b,c=coef
    return float(ell@q@ell+b@ell+c)


def optimize(coef):
    q,b,c=coef
    def unpack(z):return z[:3]-z[3:]
    def obj(z):return value(coef,unpack(z))
    def grad(z):
        g=2*q@unpack(z)+b
        return np.r_[g,-g]
    weights=np.r_[M,M]
    sol=minimize(obj,np.zeros(6),jac=grad,method='SLSQP',bounds=[(0,None)]*6,
                 constraints={'type':'ineq','fun':lambda z:1-weights@z,
                              'jac':lambda z:-weights},
                 options={'ftol':1e-14,'maxiter':1000})
    assert sol.success,sol.message
    ell=unpack(sol.x)
    assert M@np.abs(ell)<=1+1e-9
    g=2*q@ell+b
    # Exact minimum of a linear objective on the weighted l1 ball.
    gap=float(g@ell+np.max(np.abs(g)/M))
    assert gap>=-1e-9 and gap<1e-7
    return ell,max(0.,gap)


def max_kernel_error(rho):
    candidates=[0.,1.]
    if rho!=1:
        t=math.log(rho)/(rho-1)
        if 0<t<1:candidates.append(t)
    return max(abs(math.exp(-t)-math.exp(-rho*t)) for t in candidates)


def main():
    truth_scenarios=[scenario_coefficients(1.,u) for u in SUPPORT]
    true=combine(truth_scenarios,WEIGHTS);ell_star,gap_star=optimize(true)
    rng=np.random.default_rng(525717)
    rows=[]
    for rho in [.9,1.,1.1]:
        scenarios=[scenario_coefficients(rho,u) for u in SUPPORT]
        envelope=1/rho-(1-math.exp(-rho))/rho**2+ETA
        linear_envelope=2*envelope+.3
        for scenario in scenarios:
            assert np.max(np.abs(scenario[0])-envelope*np.outer(M,M))<1e-10
            assert np.max(np.abs(scenario[1])-linear_envelope*M)<1e-10
        exact_approximation=combine(scenarios,WEIGHTS)
        for n in [100,1000,10000]:
            counts=rng.multinomial(n,WEIGHTS);empirical=counts/n
            estimated=combine(scenarios,empirical)
            ell,gap=optimize(estimated)
            assert np.linalg.eigvalsh(estimated[0]).min()>-1e-10
            eq=float(np.linalg.norm(estimated[0]-exact_approximation[0],ord=2))
            eb=float(np.linalg.norm(estimated[1]-exact_approximation[1]))
            ek=.5*max_kernel_error(rho)
            bound=2*ek+2*R*R*eq+2*R*eb+gap
            dimension=len(M)
            distinct_entries=dimension*(dimension+1)//2+dimension
            z=math.sqrt(2*math.log(2*distinct_entries/.05)/n)
            iid_q=envelope*float(M@M)*z
            iid_b=linear_envelope*float(np.linalg.norm(M))*z
            assert eq<=iid_q and eb<=iid_b
            regret=value(true,ell)-value(true,ell_star)
            assert regret>=-1e-9 and regret<=bound+1e-8
            for u in SUPPORT:
                rates=np.array([features(t,u)@np.r_[1,ell] for t in np.linspace(0,1,301)])
                assert rates.min()>-1e-8 and rates.max()<2+1e-8
            rows.append({'rho':rho,'sample_size':n,'sample_counts':counts.tolist(),
                         'coefficients':ell.tolist(),'true_objective':value(true,ell),
                         'regret_against_true_qp':max(0.,regret),
                         'kernel_cost_bound':ek,'matrix_error_op':eq,'linear_error_l2':eb,
                         'optimization_gap':gap,'regret_certificate':bound,
                         'iid_95_matrix_bound':iid_q,'iid_95_linear_bound':iid_b,
                         'iid_95_regret_bound':2*ek+2*R*R*iid_q+2*R*iid_b+gap})
    fig,axes=plt.subplots(1,2,figsize=(10,3.9))
    for u,color in zip([0.,.4,.8],COLORS):
        t=np.linspace(0,1,301);rate=np.array([features(z,u)@np.r_[1,ell_star] for z in t])
        axes[0].plot(t,rate,label=f'$U={u:.1f}$',color=color)
    axes[0].axhline(0,color='#9ca3af',lw=.7);axes[0].axhline(2,color='#9ca3af',lw=.7,ls='--')
    axes[0].set(xlabel='Time',ylabel='Rate (quantity / horizon)',ylim=(-.05,2.1),title='Exact completion; rates between 0 and 2')
    axes[0].legend(frameon=False,fontsize=8)
    labels=[f"{r['rho']:.1f}\n{r['sample_size']}" for r in rows];x=np.arange(len(rows))
    axes[1].plot(x,[max(r['regret_against_true_qp'],1e-12) for r in rows],'o-',label='Actual class regret')
    axes[1].plot(x,[r['regret_certificate'] for r in rows],'s--',label='Deterministic certificate',color=COLORS[2])
    axes[1].set_yscale('log');axes[1].set_xticks(x,labels,fontsize=7)
    axes[1].set(xlabel='Approximate decay rate / sample size',ylabel='Expected cost units',title='Measured errors under a known synthetic law')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    savefig(fig,'08_execution_certificate.png')
    result={'parameters':{'T':1,'X':1,'rate_cap':2,'temporary_eta':ETA,
              'linear_charge':'a(t,U)=-0.3*(t-U)_+', 'support':SUPPORT.tolist(),
              'probabilities':WEIGHTS.tolist(),'feature_bounds':M.tolist(),
              'coefficient_radius_bound':R,'seed':525717,
              'iid_confidence_level':.95,'distinct_coefficient_entries':distinct_entries},
            'true_optimum_coefficients':ell_star.tolist(),'true_optimum_objective':value(true,ell_star),
            'true_optimum_duality_gap':gap_star,'scenarios':rows,
            'scope':'Known finite synthetic law; measured deterministic errors, not market confidence intervals.'}
    (ROOT/'results/certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    tex=[]
    for r in rows:
        tex.append(f"{r['rho']:.1f} & {r['sample_size']:,} & {r['regret_against_true_qp']:.6f} & {r['regret_certificate']:.6f} \\\\")
    (ROOT/'tables/certificate_rows.tex').write_text('\n'.join(tex)+'\n')
    print('Completion QP and nine error-certificate scenarios passed.')
    print('True optimum:',value(true,ell_star),'coefficients:',ell_star)


if __name__=='__main__':main()
