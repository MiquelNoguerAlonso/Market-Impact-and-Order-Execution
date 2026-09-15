"""Regenerate the manuscript's synthetic execution calculations and PNG figures.

All parameters are stipulated. No exchange or market dataset is used.
The RNG construction and tie-breaking rule are explicit below.
"""
from pathlib import Path
import json
import math
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import binom, poisson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(seed=5257, paths=16000, units=20, shares_per_unit=100,
              waiting_intervals=12, immediate_linear=1.0,
              immediate_quadratic=0.08, inventory_weight=0.008,
              fill_probabilities=[[0.15,0.60],[0.05,0.75]],
              passive_charges=[-0.30,0.05], prior=0.5,
              belief_grid_points=101, refined_grid_points=201)
COLORS = ['#4739a6','#168a9a','#d78924','#59794a','#9e3e56','#6e7480']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':'#253047',
                     'text.color':'#253047','axes.prop_cycle':plt.cycler(color=COLORS),
                     'savefig.facecolor':'white','figure.facecolor':'white'})


def savefig(fig, name):
    fig.tight_layout(pad=1.6)
    fig.savefig(ROOT/'figures'/name, dpi=180)
    plt.close(fig)


def market(q):
    return q + .08*q*q


def solve_dp(points=101, frozen=False):
    beliefs = np.array([.5]) if frozen else np.linspace(0,1,points)
    nb = len(beliefs)
    values = np.zeros((13,21,nb))
    values[0] = market(np.arange(21))[:,None]
    actions = np.zeros((13,21,nb,3),dtype=np.int16)
    actions[...,2] = -1
    transitions = {}
    for ell in range(1,6):
        for venue in range(2):
            f=np.arange(ell+1)
            p0=binom.pmf(f,ell,PARAMS['fill_probabilities'][venue][0])
            p1=binom.pmf(f,ell,PARAMS['fill_probabilities'][venue][1])
            mixture=(1-beliefs[:,None])*p0+beliefs[:,None]*p1
            posterior=beliefs[:,None]*p1/mixture
            transitions[ell,venue]=mixture,posterior
    # Deterministic lexicographic ties: lower m, lower ell, lower venue index.
    for left in range(1,13):
        for q in range(1,21):
            best=np.full(nb,np.inf)
            selected=np.zeros((nb,3),dtype=np.int16)
            for m in range(q+1):
                rest=q-m
                for ell in sorted({0,min(1,rest),min(3,rest),min(5,rest)}):
                    for venue in ([-1] if ell==0 else [0,1]):
                        if ell==0:
                            candidate=market(m)+.008*rest**2+values[left-1,rest]
                        else:
                            mixture,posterior=transitions[ell,venue]
                            candidate=np.full(nb,float(market(m)))
                            for f in range(ell+1):
                                nq=rest-f
                                continuation=(values[left-1,nq] if frozen else
                                    np.interp(posterior[:,f],beliefs,values[left-1,nq]))
                                candidate += mixture[:,f]*(PARAMS['passive_charges'][venue]*f
                                                      +.008*nq**2+continuation)
                        improve=candidate<best-1e-12
                        best=np.where(improve,candidate,best)
                        selected[improve]=(m,ell,venue)
            values[left,q]=best
            actions[left,q]=selected
    return beliefs,values,actions


def simulate(beliefs,values,actions,frozen_actions):
    rng=np.random.default_rng(PARAMS['seed'])
    # The regime stream is consumed first, then the path-major fill uniforms.
    regimes=(rng.random(PARAMS['paths'])<.5).astype(np.int8)
    uniforms=rng.random((PARAMS['paths'],12))
    names=['Belief policy','Frozen prior','Regime oracle','TWAP market',
           'Market now','Midpoint first']
    results={}; arrays={}
    for name in names:
        q=np.full(PARAMS['paths'],20,dtype=np.int16)
        b=np.full(PARAMS['paths'],.5)
        charges=np.zeros(PARAMS['paths']); penalties=np.zeros(PARAMS['paths'])
        trajectory=[q.copy()]
        for k in range(12):
            left=12-k
            if name=='Belief policy':
                idx=np.rint(b*(len(beliefs)-1)).astype(int)
                a=actions[left,q,idx]
            elif name=='Frozen prior':
                a=frozen_actions[left,q,0]
            elif name=='Regime oracle':
                a=actions[left,q,regimes*(len(beliefs)-1)]
            else:
                a=np.zeros((len(q),3),dtype=np.int16);a[:,2]=-1
                if name=='TWAP market':a[:,0]=(q+left)//(left+1)
                elif name=='Market now':a[:,0]=q
                else:a[:,1]=np.minimum(5,q);a[:,2]=1
            m,ell,venue=a.T
            assert np.all(m>=0) and np.all(ell>=0) and np.all(m+ell<=q)
            fill=np.zeros(len(q),dtype=np.int16)
            active=ell>0
            prob=np.asarray(PARAMS['fill_probabilities'])[np.maximum(venue,0),regimes]
            fill[active]=binom.ppf(uniforms[active,k],ell[active],prob[active]).astype(np.int16)
            charges+=market(m)
            charges[active]+=np.asarray(PARAMS['passive_charges'])[venue[active]]*fill[active]
            q=q-m-fill
            assert np.all(q>=0)
            penalties+=.008*q*q
            trajectory.append(q.copy())
            if name=='Belief policy':
                p0=np.asarray(PARAMS['fill_probabilities'])[np.maximum(venue,0),0]
                p1=np.asarray(PARAMS['fill_probabilities'])[np.maximum(venue,0),1]
                l0=binom.pmf(fill,ell,p0);l1=binom.pmf(fill,ell,p1)
                denom=(1-b)*l0+b*l1
                assert np.all(denom>0)
                b=b*l1/denom
        terminal=q.copy()
        charges+=market(q)
        trajectory.append(np.zeros_like(q))
        objective=charges+penalties
        results[name]={'charge':float(charges.mean()),'penalty':float(penalties.mean()),
                       'objective':float(objective.mean()),
                       'mc_se':float(objective.std(ddof=1)/math.sqrt(len(q))),
                       'terminal_units':float(terminal.mean()),
                       'mean_inventory_path':np.mean(trajectory,axis=1).tolist()}
        arrays[name]=objective
    difference=arrays['Frozen prior']-arrays['Belief policy']
    oracle_gap=arrays['Belief policy']-arrays['Regime oracle']
    se=float(difference.std(ddof=1)/math.sqrt(len(difference)))
    paired={'frozen_minus_belief':float(difference.mean()),'standard_error':se,
            'normal_95_interval':[float(difference.mean()-1.96*se),float(difference.mean()+1.96*se)],
            'improvement_percent':float(100*difference.mean()/arrays['Frozen prior'].mean()),
            'belief_minus_oracle':float(oracle_gap.mean()),
            'oracle_gap_standard_error':float(oracle_gap.std(ddof=1)/math.sqrt(len(oracle_gap)))}
    np.savez_compressed(ROOT/'results/policy_scenarios.npz',regimes=regimes,
                        uniforms=uniforms,**{name.replace(' ','_'):a for name,a in arrays.items()})
    return results,paired


def analytic_figures():
    t=np.linspace(0,1,2001);rows=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for k in [0,1,3]:
        inventory=(1-t if k==0 else np.sinh(k*(1-t))/np.sinh(k))
        rate=(np.ones_like(t) if k==0 else k*np.cosh(k*(1-t))/np.sinh(k))
        impact=float(np.trapezoid(rate**2,t));exposure=float(np.trapezoid(inventory**2,t))
        rows.append({'kappa':k,'impact':impact,'exposure':exposure,'objective':impact+k*k*exposure})
        axes[0].plot(t,inventory,label=f'$\\kappa T={k}$')
        axes[1].plot(t,rate,label=f'$\\kappa T={k}$')
    axes[0].set(xlabel='Fraction of horizon',ylabel='Remaining quantity')
    axes[1].set(xlabel='Fraction of horizon',ylabel='Trading rate')
    for ax in axes:ax.legend(frameon=False);ax.grid(alpha=.15)
    savefig(fig,'01_schedules.png')
    a=np.array([.010,.012,.014]);b=np.array([.000080,.000025,.000012]);caps=np.array([120.,550.,900.])
    def route(size):
        if size==0:return np.zeros(3)
        lam=brentq(lambda z:np.clip((z-a)/b,0,caps).sum()-size,float(a.min()),.12)
        return np.clip((lam-a)/b,0,caps)
    quantities=np.linspace(0,1500,151);routes=np.array([route(q) for q in quantities])
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    axes[0].stackplot(quantities,routes.T,labels=['Venue A','Venue B','Venue C'],colors=COLORS[:3],alpha=.85)
    for j in range(3):
        x=np.linspace(0,caps[j],100);axes[1].plot(x,100*(a[j]+b[j]*x),label=f'Venue {"ABC"[j]}')
    axes[0].set(xlabel='Required shares',ylabel='Allocated shares')
    axes[1].set(xlabel='Venue allocation (shares)',ylabel='Marginal cost (cents/share)')
    for ax in axes:ax.legend(frameon=False);ax.grid(alpha=.15)
    savefig(fig,'02_routing.png')
    x=route(1000);routing={'shares':x.tolist(),'costs':(a*x+.5*b*x*x).tolist(),
                          'total':float(np.sum(a*x+.5*b*x*x))}
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    time=np.linspace(0,5,301)
    for ahead in [0,5,10]:axes[0].plot(time,poisson.sf(ahead,3*time),label=f'{ahead} units ahead')
    probability=np.linspace(0,1,301);passive=probability*(-.002)+(1-probability)*.025+.0005
    threshold=(.025+.0005-.010)/(.025+.002)
    axes[0].set(xlabel='Waiting time (seconds)',ylabel='First-fill probability',ylim=(0,1))
    axes[0].set_title('Poisson service: 3 units/second',fontsize=10)
    axes[1].plot(probability,passive,label='Passive-first cost');axes[1].axhline(.010,color=COLORS[1],label='Immediate cost')
    axes[1].axvline(threshold,color='#7b8290',ls='--',lw=1)
    axes[1].set(xlabel='Passive-fill probability',ylabel='Expected cost (USD/share)')
    for ax in axes:ax.legend(frameon=False);ax.grid(alpha=.15)
    savefig(fig,'03_queue_and_order_choice.png')
    prior=np.linspace(0,1,301);uninformed=np.minimum(1,1.8-1.6*prior);informed=1-.8*prior
    fig,ax=plt.subplots(figsize=(8.2,3.8))
    ax.plot(prior,uninformed,label='Best decision with current prior');ax.plot(prior,informed,label='Hidden regime observed before decision')
    ax.fill_between(prior,informed,uninformed,color=COLORS[0],alpha=.12,label='Value of information')
    ax.set(xlabel='Probability of high-liquidity regime',ylabel='Expected cost (USD)');ax.legend(frameon=False);ax.grid(alpha=.15)
    savefig(fig,'04_information_value.png')
    return {'schedules':rows,'routing':routing,'passive_threshold':threshold,
            'poisson_plot_parameters':{'service_units_per_second':3,'units_ahead':[0,5,10]},
            'separation_value_at_equal_prior':.4}


def policy_figures(policies,actions):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,(name,row) in enumerate(policies.items()):axes[0].plot(range(14),row['mean_inventory_path'],label=name,color=COLORS[i])
    axes[0].set(xlabel='Ledger event (13 = terminal trade)',ylabel='Mean remaining units',ylim=(0,20.5))
    axes[0].legend(frameon=False,fontsize=8);axes[0].grid(alpha=.15)
    names=list(policies);charge=[policies[n]['charge'] for n in names];penalty=[policies[n]['penalty'] for n in names]
    y=np.arange(len(names));axes[1].barh(y,charge,color=COLORS[0],label='Execution charge')
    axes[1].barh(y,penalty,left=charge,color=COLORS[2],label='Inventory penalty')
    axes[1].set_yticks(y, names,fontsize=8);axes[1].invert_yaxis();axes[1].set_xlabel('Objective (USD)');axes[1].legend(frameon=False,fontsize=8)
    savefig(fig,'05_policy_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,left,title in zip(axes,[12,1],['First decision','Last waiting decision']):
        im=ax.imshow(actions[left,:,:,0],origin='lower',aspect='auto',extent=(0,1,-.5,20.5),vmin=0,vmax=20,cmap='viridis')
        ax.set(title=title,xlabel='Probability of high liquidity',ylabel='Remaining units')
    fig.colorbar(im,ax=axes,label='Immediate units')
    fig.savefig(ROOT/'figures/06_urgency_and_belief.png',dpi=180);plt.close(fig)


def generated_tables(results):
    rows=[]
    for name,r in results['policies'].items():
        rows.append(f"{name} & {r['charge']:.3f} & {r['penalty']:.3f} & {r['objective']:.3f} & {r['mc_se']:.3f} & {r['terminal_units']:.2f} \\\\")
    (ROOT/'tables/policy_rows.tex').write_text('\n'.join(rows)+'\n')
    p=results['policies'];z=results['paired'];v=results['dynamic_program']
    macros={'InitialValue':f"{v['initial_101']:.4f}",'GridError':f"{v['grid_change']:.8f}",
      'BeliefObjective':f"{p['Belief policy']['objective']:.4f}",
      'FrozenObjective':f"{p['Frozen prior']['objective']:.4f}",
      'BeliefCharge':f"{p['Belief policy']['charge']:.4f}",
      'FrozenCharge':f"{p['Frozen prior']['charge']:.4f}",
      'BeliefPenalty':f"{p['Belief policy']['penalty']:.4f}",
      'FrozenPenalty':f"{p['Frozen prior']['penalty']:.4f}",
      'ImprovementPct':f"{z['improvement_percent']:.1f}",
      'PairedDifference':f"{z['frozen_minus_belief']:.4f}",
      'PairedSE':f"{z['standard_error']:.4f}",
      'CILower':f"{z['normal_95_interval'][0]:.3f}",'CIUpper':f"{z['normal_95_interval'][1]:.3f}",
      'OracleObjective':f"{p['Regime oracle']['objective']:.4f}",
      'OracleGap':f"{z['belief_minus_oracle']:.4f}",'OracleGapSE':f"{z['oracle_gap_standard_error']:.4f}",
      'BellmanZ':f"{abs(p['Belief policy']['objective']-v['initial_101'])/p['Belief policy']['mc_se']:.2f}"}
    (ROOT/'tables/experiment_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+val+'}' for k,val in macros.items())+'\n')


def main():
    for name in ['figures','results','tables']:(ROOT/name).mkdir(exist_ok=True)
    print('Solving belief dynamic programmes...',flush=True)
    beliefs,values,actions=solve_dp(101)
    _,refined,_=solve_dp(201)
    _,frozen_values,frozen_actions=solve_dp(frozen=True)
    assert abs(values[12,20,50]-10.109134201203524)<1e-9
    assert abs(refined[12,20,100]-10.109457688704305)<1e-9
    print('Simulating six policies on 16,000 common scenarios...',flush=True)
    policies,paired=simulate(beliefs,values,actions,frozen_actions)
    assert abs(policies['TWAP market']['objective']-32.12)<1e-10
    assert abs(policies['Market now']['objective']-52)<1e-10
    analytic=analytic_figures();policy_figures(policies,actions)
    results={'parameters':PARAMS,'environment':{'python':platform.python_version(),
             'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
             'scope':'Stipulated synthetic model; all Monte Carlo outputs regenerated by this implementation.',
             'dynamic_program':{'initial_101':float(values[12,20,50]),
                'initial_201':float(refined[12,20,100]),
                'grid_change':float(refined[12,20,100]-values[12,20,50]),
                'oracle_expected_value':float(.5*(values[12,20,0]+values[12,20,-1])),
                'frozen_model_value':float(frozen_values[12,20,0])},
             'policies':policies,'paired':paired,'analytic':analytic}
    (ROOT/'results/experiments.json').write_text(json.dumps(results,indent=2)+'\n')
    (ROOT/'results/parameters.json').write_text(json.dumps(PARAMS,indent=2)+'\n')
    np.savez_compressed(ROOT/'results/belief_dp.npz',beliefs=beliefs,values=values,
                       actions=actions,frozen_values=frozen_values,frozen_actions=frozen_actions)
    generated_tables(results)
    print(json.dumps({'belief':policies['Belief policy'],'paired':paired},indent=2),flush=True)


if __name__=='__main__':main()
