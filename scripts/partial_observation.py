"""Certified finite-belief planning for controlled hidden liquidity.

All prices, probabilities and regimes are synthetic. The planner never sees
the hidden regime. Integer Bayes rounding and outward-rounded, nonnegative
arithmetic enclose the finite-model values; no Monte Carlo estimate is used
for an optimality certificate. Run this file to regenerate all new outputs.
"""
from dataclasses import dataclass
from fractions import Fraction
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from functools import lru_cache
from math import comb
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEN = 100**4
CDEN = 1000
HORIZON = 12
PARENT = 10
SHARES_PER_UNIT = 100


@dataclass(frozen=True)
class Action:
    name: str
    market: int
    passive: int
    to_bad: int
    to_good: int
    fill_good: int
    fill_bad: int
    fee: int


ACTIONS = (
    Action('Wait', 0, 0, 5, 18, 0, 0, 0),
    Action('Lit 1', 0, 1, 12, 12, 80, 18, 10),
    Action('Lit 2', 0, 2, 28, 7, 65, 8, 10),
    Action('Dark 1', 0, 1, 4, 15, 70, 3, 25),
    Action('Market 1', 1, 0, 35, 6, 0, 0, 0),
    Action('Market 2', 2, 0, 60, 3, 0, 0, 0),
    Action('Split 1+1', 1, 1, 40, 5, 72, 12, 10),
)


def down(x):
    return np.maximum(0.0, np.nextafter(x, -np.inf))


def up(x):
    return np.nextafter(x, np.inf)


def rat_bounds(n, d):
    """Integer arguments are exactly representable in binary64 in this model."""
    n, d = np.asarray(n), np.asarray(d)
    assert np.max(np.abs(n)) < 2**53 and np.max(np.abs(d)) < 2**53
    z = n.astype(float) / d.astype(float)
    return down(z), up(z)


def add(a, b):
    return down(a[0] + b[0]), up(a[1] + b[1])


def mul(a, b):
    return down(a[0] * b[0]), up(a[1] * b[1])


def allowed(q):
    return [i for i, a in enumerate(ACTIONS) if a.market + a.passive <= q] if q else [0]


def charge(q, ai, fill):
    a = ACTIONS[ai]
    left = q - a.market - fill
    assert 0 <= fill <= a.passive and left >= 0
    return 1000*a.market + 180*a.market*a.market + 80*fill + a.fee + 40*left*left


def terminal(q):
    return 1800*q + 150*q*q


@lru_cache(None)
def kernel(ai, blind=False):
    a = ACTIONS[ai]
    aa = ACTIONS[0] if blind else a
    tr = np.array([[100-aa.to_bad, aa.to_bad], [aa.to_good, 100-aa.to_good]], dtype=np.int64)
    outcomes, counts = [], []
    for fill in range(a.passive+1):
        for signal in range(2):
            m = np.zeros((2, 2), dtype=np.int64)
            for h in range(2):
                for hp in range(2):
                    p = (a.fill_good, a.fill_bad)[hp]
                    f = comb(a.passive, fill) * p**fill * (100-p)**(a.passive-fill)
                    signal_one = (35, 65)[hp]
                    s = signal_one if signal else 100-signal_one
                    m[h, hp] = tr[h, hp]*f*s*100**(2-a.passive)
            outcomes.append((fill, signal))
            counts.append(m)
    counts = np.array(counts)
    assert np.all(counts.sum(axis=(0, 2)) == DEN)
    return tuple(outcomes), counts


@lru_cache(None)
def grid_kernel(m, ai, blind=False):
    assert 0 < m and 2*m*m*DEN+m*DEN < np.iinfo(np.int64).max
    outcomes, counts = kernel(ai, blind)
    j = np.arange(m+1, dtype=np.int64)
    num = (m-j)[None, :, None]*counts[:, 0, :][:, None, :] + j[None, :, None]*counts[:, 1, :][:, None, :]
    total = num.sum(axis=2)
    assert np.all(total > 0)
    scaled = m*num[:, :, 1]
    nearest = (2*scaled+total)//(2*total)  # exact, ties toward larger belief
    left = scaled//total
    rem = scaled % total
    right = np.minimum(left+1, m)
    p = rat_bounds(total, m*DEN)
    wl = rat_bounds(total-rem, m*DEN)
    wr = rat_bounds(rem, m*DEN)
    return outcomes, nearest, left, right, p, wl, wr


def solve_grid(m, horizon=HORIZON, parent=PARENT, blind=False):
    """Nearest-node finite MDP and concavity lower relaxation, with enclosures."""
    shape = (horizon+1, parent+1, m+1)
    wlo, whi, llo, lhi = [np.zeros(shape) for _ in range(4)]
    policy = np.zeros((horizon, parent+1, m+1), dtype=np.int8)
    for q in range(parent+1):
        lo, hi = rat_bounds(terminal(q), CDEN)
        wlo[-1, q] = llo[-1, q] = lo
        whi[-1, q] = lhi[-1, q] = hi
    residuals = np.zeros(horizon)
    for k in range(horizon-1, -1, -1):
        for q in range(1, parent+1):
            choices = allowed(q)
            qa, la = [], []
            for ai in choices:
                outcomes, near, left, right, p, wl, wr = grid_kernel(m, ai, blind)
                qw = (np.zeros(m+1), np.zeros(m+1))
                ql = (np.zeros(m+1), np.zeros(m+1))
                for o, (fill, signal) in enumerate(outcomes):
                    qp = q-ACTIONS[ai].market-fill
                    c = rat_bounds(charge(q, ai, fill), CDEN)
                    po = p[0][o], p[1][o]
                    future = wlo[k+1, qp, near[o]], whi[k+1, qp, near[o]]
                    qw = add(qw, mul(po, add(c, future)))
                    ql = add(ql, mul(po, c))
                    ql = add(ql, mul((wl[0][o], wl[1][o]), (llo[k+1, qp, left[o]], lhi[k+1, qp, left[o]])))
                    ql = add(ql, mul((wr[0][o], wr[1][o]), (llo[k+1, qp, right[o]], lhi[k+1, qp, right[o]])))
                qa.append(qw)
                la.append(ql)
            qlo, qhi = np.array([v[0] for v in qa]), np.array([v[1] for v in qa])
            selected = np.argmin(qhi, axis=0)
            policy[k, q] = np.array(choices)[selected]
            wlo[k, q], whi[k, q] = qlo.min(axis=0), qhi.min(axis=0)
            llo[k, q] = np.min([v[0] for v in la], axis=0)
            lhi[k, q] = np.min([v[1] for v in la], axis=0)
            residuals[k] = max(residuals[k], float(np.max(up(qhi[selected, np.arange(m+1)]-qlo.min(axis=0)))))
    tolerance=0.0
    for residual in residuals:
        tolerance=float(up(tolerance+residual))
    return {'policy':policy, 'nearest_lo':wlo, 'nearest_hi':whi,
            'lower_lo':llo, 'lower_hi':lhi,
            'optimization_tolerance':tolerance, 'stage_residuals':residuals.tolist()}


def evaluate(policy, m, planner_blind=False, physical_blind=False, weight=None, initial_bad=Fraction(1,2)):
    """True hidden-regime evaluation of a finite-memory controller, not resampling.

    Memory updates use the planner's kernel; physical transitions use the
    evaluation kernel. For finite weight w, preferred action has probability
    w/(w+n-1), each other admissible action 1/(w+n-1).
    """
    horizon, qdim, _ = policy.shape
    vlo, vhi = np.zeros((qdim, m+1, 2)), np.zeros((qdim, m+1, 2))
    for q in range(qdim):
        lo, hi = rat_bounds(terminal(q), CDEN)
        vlo[q], vhi[q] = lo, hi
    for k in range(horizon-1, -1, -1):
        nlo, nhi = np.zeros_like(vlo), np.zeros_like(vhi)
        for q in range(1, qdim):
            choices = allowed(q)
            qa = []
            for ai in choices:
                outcomes, counts = kernel(ai, physical_blind)
                _, near, *_ = grid_kernel(m, ai, planner_blind)
                out = (np.zeros((m+1, 2)), np.zeros((m+1, 2)))
                for o, (fill, signal) in enumerate(outcomes):
                    qp = q-ACTIONS[ai].market-fill
                    c = rat_bounds(charge(q, ai, fill), CDEN)
                    for hp in range(2):
                        po = rat_bounds(counts[o, :, hp], DEN)
                        future = vlo[qp, near[o], hp][:, None], vhi[qp, near[o], hp][:, None]
                        out = add(out, mul(po, add(c, future)))
                qa.append(out)
            if weight is None:
                lookup = np.full(len(ACTIONS), -1, dtype=int)
                lookup[choices] = np.arange(len(choices))
                idx = lookup[policy[k, q]]
                assert np.all(idx >= 0)
                nlo[q] = np.array([v[0] for v in qa])[idx, np.arange(m+1)]
                nhi[q] = np.array([v[1] for v in qa])[idx, np.arange(m+1)]
            else:
                acc = (np.zeros((m+1, 2)), np.zeros((m+1, 2)))
                for ai, out in zip(choices, qa):
                    numerator = np.where(policy[k, q] == ai, weight, 1)[:, None]
                    acc = add(acc, mul(rat_bounds(numerator, weight+len(choices)-1), out))
                nlo[q], nhi[q] = acc
        vlo, vhi = nlo, nhi
    initial_bad=Fraction(initial_bad)
    j0=m*initial_bad
    assert j0.denominator==1
    j0=int(j0)
    init = add(mul(rat_bounds(initial_bad.denominator-initial_bad.numerator, initial_bad.denominator), (vlo[-1,j0,0],vhi[-1,j0,0])),
               mul(rat_bounds(initial_bad.numerator, initial_bad.denominator), (vlo[-1,j0,1],vhi[-1,j0,1])))
    return float(init[0]), float(init[1])


def oracle(horizon=HORIZON, parent=PARENT):
    """Fully observed Bellman benchmark; its information is not used by planner."""
    value = np.zeros((horizon+1, parent+1, 2))
    pol = np.zeros((horizon, parent+1, 2), dtype=np.int8)
    value[-1] = np.array([terminal(q)/CDEN for q in range(parent+1)])[:, None]
    for k in range(horizon-1, -1, -1):
        for q in range(1, parent+1):
            qs = []
            for ai in allowed(q):
                out = np.zeros(2)
                outcomes, counts = kernel(ai)
                for o, (f, s) in enumerate(outcomes):
                    qp = q-ACTIONS[ai].market-f
                    out += counts[o] @ (charge(q, ai, f)/CDEN + value[k+1, qp])/DEN
                qs.append(out)
            qs = np.array(qs)
            pol[k, q] = np.array(allowed(q))[np.argmin(qs, axis=0)]
            value[k, q] = qs.min(axis=0)
    return float(value[0, parent].mean()), pol


def frozen_prior(m, horizon=HORIZON, parent=PARENT, initial_bad=Fraction(1,2)):
    value = np.zeros((horizon+1, parent+1))
    value[-1] = [terminal(q)/CDEN for q in range(parent+1)]
    policy = np.zeros((horizon, parent+1, m+1), dtype=np.int8)
    for k in range(horizon-1, -1, -1):
        for q in range(1, parent+1):
            qs=[]
            for ai in allowed(q):
                outcomes, counts = kernel(ai)
                qs.append(sum(float((1-initial_bad)*int(counts[o,0].sum())+initial_bad*int(counts[o,1].sum()))/DEN*(charge(q, ai, f)/CDEN+value[k+1, q-ACTIONS[ai].market-f]) for o,(f,s) in enumerate(outcomes)))
            a = allowed(q)[int(np.argmin(qs))]
            policy[k,q] = a
            value[k,q] = min(qs)
    return policy


def exact_tree(horizon=3, parent=2):
    """Independent exhaustive rational Bellman recursion at the true belief."""
    calls = 0
    @lru_cache(None)
    def rec(k, q, bad):
        nonlocal calls
        calls += 1
        if k==horizon or q==0:
            return Fraction(terminal(q), CDEN)
        best = None
        for ai in allowed(q):
            val = Fraction(0)
            outcomes, counts = kernel(ai)
            for o, (f, s) in enumerate(outcomes):
                nums = [(1-bad)*int(counts[o,0,hp])+bad*int(counts[o,1,hp]) for hp in range(2)]
                prob = sum(nums)/DEN
                if prob:
                    val += prob*(Fraction(charge(q,ai,f),CDEN)+rec(k+1,q-ACTIONS[ai].market-f,nums[1]/sum(nums)))
            best = val if best is None else min(best,val)
        return best
    value = rec(0,parent,Fraction(1,2))
    return value, calls


def exact_controller(policy, m, weight=None):
    """Independent Fraction evaluation, including exact posterior quantization."""
    horizon, qdim, _=policy.shape
    @lru_cache(None)
    def rec(k,q,j,h):
        if k==horizon or q==0:
            return Fraction(terminal(q),CDEN)
        ans=Fraction(0)
        choices=allowed(q) if weight is not None else [int(policy[k,q,j])]
        for ai in choices:
            pa=Fraction(weight if ai==int(policy[k,q,j]) else 1,weight+len(choices)-1) if weight is not None else Fraction(1)
            outcomes, counts=kernel(ai)
            for o,(fill,signal) in enumerate(outcomes):
                bad=Fraction(j,m)
                nums=[(1-bad)*int(counts[o,0,hp])+bad*int(counts[o,1,hp]) for hp in range(2)]
                pbad=nums[1]/sum(nums)
                shifted=m*pbad+Fraction(1,2)
                jp=shifted.numerator//shifted.denominator
                for hp in range(2):
                    ans+=pa*Fraction(int(counts[o,h,hp]),DEN)*(Fraction(charge(q,ai,fill),CDEN)+rec(k+1,q-ACTIONS[ai].market-fill,jp,hp))
        return ans
    return (rec(0,qdim-1,m//2,0)+rec(0,qdim-1,m//2,1))/2


def structural_checks(m):
    count=0
    for ai in range(len(ACTIONS)):
        outcomes,counts=kernel(ai)
        _,near,left,right,p,wl,wr=grid_kernel(m,ai)
        for o in range(len(outcomes)):
            for j in range(m+1):
                n0=(m-j)*int(counts[o,0,0])+j*int(counts[o,1,0])
                n1=(m-j)*int(counts[o,0,1])+j*int(counts[o,1,1])
                posterior=Fraction(n1,n0+n1)
                assert abs(Fraction(int(near[o,j]),m)-posterior)<=Fraction(1,2*m)
                assert Fraction(int(left[o,j]),m)<=posterior<=Fraction(int(right[o,j]),m)
                prob=Fraction(n0+n1,m*DEN)
                assert Fraction.from_float(float(p[0][o,j]))<=prob<=Fraction.from_float(float(p[1][o,j]))
                count+=1
    return count


class BeliefSignatureController:
    """Observable controller: hidden regime is not an argument to any method."""
    def __init__(self, policy, m, initial_bad=Fraction(1,2), weight=1048576, blind=False):
        self.policy=policy
        self.m=m
        self.q=policy.shape[1]-1
        initial=m*Fraction(initial_bad)
        assert initial.denominator==1 and isinstance(weight,int) and weight>=1
        self.node=int(initial)
        self.k=0
        self.weight=weight
        self.blind=blind

    def first_signature_level(self):
        """Endpoint of the one-hot summary path whose origin is zero."""
        out=np.zeros(self.policy.shape[1]*(self.m+1))
        out[self.q*(self.m+1)+self.node]=1
        return out

    def scores_from_signature(self):
        table=self.policy[self.k].ravel()
        coefficients=(np.arange(len(ACTIONS))[:,None]==table[None,:])*np.log(self.weight)
        return coefficients@self.first_signature_level()

    def integer_action_weights(self):
        preferred=int(self.policy[self.k,self.q,self.node])
        return {a:self.weight if a==preferred else 1 for a in allowed(self.q)}

    def act(self, rng):
        """Integer sampling realizes the stated rational softmax probabilities."""
        weights=self.integer_action_weights()
        ticket=rng.randrange(sum(weights.values()))
        for action,w in weights.items():
            if ticket<w:
                return action
            ticket-=w
        raise AssertionError('Unreachable sampling branch')

    def observe(self, action, fill, signal):
        assert action in allowed(self.q)
        outcomes,nearest,*_=grid_kernel(self.m,action,self.blind)
        outcome=outcomes.index((fill,signal))
        self.node=int(nearest[outcome,self.node])
        self.q-=ACTIONS[action].market+fill
        assert self.q>=0
        self.k+=1


def signature_controller_checks(policy,m):
    rng=random.Random(52572917)
    checked=0
    for _ in range(20):
        controller=BeliefSignatureController(policy,m,weight=1024)
        while controller.k<policy.shape[0]:
            scores=controller.scores_from_signature()
            weights=controller.integer_action_weights()
            mask=np.array([a in weights for a in range(len(ACTIONS))])
            p=np.zeros(len(ACTIONS))
            p[mask]=np.exp(scores[mask]-scores[mask].max())
            p/=p.sum()
            rational=np.array([weights.get(a,0)/sum(weights.values()) for a in range(len(ACTIONS))])
            assert np.max(np.abs(p-rational))<2e-15
            assert np.all(p[~mask]==0)
            assert np.count_nonzero(controller.first_signature_level())==1
            # Exercise all allowed branches; these are test reports, not MC values.
            action=rng.choice(allowed(controller.q))
            fill=rng.randrange(ACTIONS[action].passive+1)
            signal=rng.randrange(2)
            controller.observe(action,fill,signal)
            checked+=1
    return {'score_mask_and_memory_cases':checked,'hidden_regime_input':False,
            'signature_depth':1,'summary_dimension':policy.shape[1]*(m+1),
            'sampling':'Exact integer weights for the rational softmax law.'}


def forward_evaluate(policy,m):
    """Forward physical hidden-state occupancy, independent of value recursion."""
    horizon,qdim,_=policy.shape
    law=np.zeros((qdim,m+1,2))
    law[-1,m//2]=.5
    cost=0.0
    actions=np.zeros(len(ACTIONS))
    mass_errors=[]
    visited=0
    for k in range(horizon):
        nxt=np.zeros_like(law)
        nxt[0]=law[0]
        visited+=int(np.count_nonzero(law.sum(axis=2)>0))
        for q in range(1,qdim):
            for ai in allowed(q):
                active=policy[k,q]==ai
                mass=law[q]*active[:,None]
                actions[ai]+=mass.sum()
                outcomes,counts=kernel(ai)
                _,near,*_=grid_kernel(m,ai)
                for o,(fill,signal) in enumerate(outcomes):
                    qp=q-ACTIONS[ai].market-fill
                    new=mass@counts[o]/DEN
                    cost+=new.sum()*charge(q,ai,fill)/CDEN
                    for hp in range(2):
                        np.add.at(nxt[qp,:,hp],near[o],new[:,hp])
        law=nxt
        mass_errors.append(abs(float(law.sum())-1))
    cost+=sum(law[q].sum()*terminal(q)/CDEN for q in range(qdim))
    return {'value':cost,'maximum_mass_error':max(mass_errors),
            'positive_mass_stage_memory_states':visited,
            'expected_action_counts':dict(zip([a.name for a in ACTIONS],actions.tolist()))}


def export_outputs(report, policy, lower):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    rows=report['grid_results']
    def rounded(v,places=8,upper=True):
        return format(Decimal.from_float(float(v)).quantize(Decimal(1).scaleb(-places),rounding=ROUND_CEILING if upper else ROUND_FLOOR),'f')
    def scientific(v):
        d=Decimal.from_float(float(v))
        e=d.adjusted()
        mant=d.scaleb(-e).quantize(Decimal('.001'),rounding=ROUND_CEILING)
        return '$'+str(mant)+r'\times10^{'+str(e)+'}$'
    endrow = r" \\" + "\n"
    (ROOT/'tables/partial_grid_rows.tex').write_text(''.join(
        f"{r['belief_nodes']} & {rounded(r['lower_bound'],upper=False)} & {rounded(r['policy_value_hi'])} & {scientific(r['certified_gap'])}" + endrow for r in rows))
    base=report['policies']
    u=base['Certified belief controller'][1]
    (ROOT/'tables/partial_policy_rows.tex').write_text(''.join(
        f"{name} & {rounded(bounds[1],6)} & {rounded(bounds[0]-u,6,False) if name!='Certified belief controller' else '0'}" + endrow for name,bounds in base.items()))
    (ROOT/'tables/partial_robustness_rows.tex').write_text(''.join(
        f"{r['horizon']} & {r['initial_bad']:.2f} & {rounded(r['value_hi'],5)} & {scientific(r['certified_gap'])} & {rounded(r['frozen_excess_lower'],5,False)} & {rounded(r['blind_excess_lower'],5,False)}" + endrow for r in report['robustness']))
    (ROOT/'tables/partial_softmax_rows.tex').write_text(''.join(
        f"{r['preferred_weight']} & {rounded(r['value_hi'],8)} & {scientific(r['certified_gap'])}" + endrow for r in report['softmax_results']))
    macros={'PartialValue':rounded(u,8),'PartialGap':rounded(rows[-1]['certified_gap'],10),
            'PartialLower':rounded(rows[-1]['lower_bound'],8,False),
            'PartialSoftGap':rounded(report['softmax_results'][-1]['certified_gap'],8),
            'PartialFrozenPct':f"{100*(base['Frozen prior'][0]-u)/base['Frozen prior'][0]:.2f}",
            'PartialBlindPct':f"{100*(base['Action-independent transition'][0]-u)/base['Action-independent transition'][0]:.2f}",
            'PartialOracle':f"{report['fully_observed_oracle']:.6f}"}
    (ROOT/'tables/partial_macros.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(11.4,4.2),constrained_layout=True)
    ax[0].loglog([r['belief_nodes'] for r in rows],[r['certified_gap'] for r in rows],'o-',color='#225ea8')
    ax[0].loglog([r['belief_nodes'] for r in rows],[r['uniform_mesh_gap_bound'] for r in rows],'--',color='#747474',label='Uniform mesh guarantee')
    ax[0].lines[0].set_label('Computed lower/upper certificate')
    ax[0].legend(fontsize=8)
    ax[0].set(xlabel='Belief nodes',ylabel='Certified global gap (USD)',title='Partial observation: computed certificate')
    ax[0].grid(alpha=.2,which='both')
    names=['Belief controller','Frozen prior','Most-likely regime','Action-independent']
    values=[v[1]-u for v in base.values()]
    ax[1].barh(names,values,color=['#225ea8','#d69e2e','#747474','#b24745'])
    ax[1].invert_yaxis()
    ax[1].set(xlabel='Excess expected cost over belief controller (USD)',title='Same physical controlled model')
    fig.savefig(ROOT/'figures/10_partial_observation_certificate.png',dpi=180)
    plt.close(fig)
    colors=['#dddddd','#225ea8','#41b6c4','#6a51a3','#fec44f','#e34a33','#31a354']
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2),constrained_layout=True)
    cmap=ListedColormap(colors)
    norm=BoundaryNorm(np.arange(-.5,len(ACTIONS)+.5),len(ACTIONS))
    for k,ax in zip([0,HORIZON-1],axes):
        im=ax.imshow(policy[k,1:],origin='lower',aspect='auto',extent=[0,1,.5,PARENT+.5],cmap=cmap,norm=norm,interpolation='nearest')
        ax.set(xlabel='Rounded probability of bad liquidity',ylabel='Remaining 100-share units',title=f'Decision {k+1} of {HORIZON}')
    cb=fig.colorbar(im,ax=axes,ticks=range(len(ACTIONS)),shrink=.88)
    cb.ax.set_yticklabels([a.name for a in ACTIONS])
    fig.savefig(ROOT/'figures/11_partial_observation_policy.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    grids=[8,16,32,64] if args.quick else [8,16,32,64,128,256,512,1024]
    results=[]
    for m in grids:
        s=solve_grid(m)
        lo,hi=evaluate(s['policy'],m)
        lb=float(s['lower_lo'][0,PARENT,m//2])
        row={'subintervals':m,'belief_nodes':m+1,'controller_states':(PARENT+1)*(m+1),
             'lower_bound':lb,'policy_value_lo':lo,'policy_value_hi':hi,
             'certified_gap':float(up(hi-lb)), 'evaluation_width':hi-lo,
             'optimization_tolerance':s['optimization_tolerance']}
        stage_max=max(charge(q,ai,f)/CDEN for q in range(1,PARENT+1) for ai in allowed(q) for f in range(ACTIONS[ai].passive+1))
        spans=[(HORIZON-j)*stage_max+terminal(PARENT)/CDEN for j in range(HORIZON+1)]
        row['uniform_mesh_gap_bound']=3*sum(spans[1:])/(2*m)+s['optimization_tolerance']
        results.append(row)
        print(json.dumps(row),flush=True)
    m=grids[-1]
    baseline={}
    baseline['Certified belief controller']=list(evaluate(s['policy'],m))
    baseline['Frozen prior']=list(evaluate(frozen_prior(m),m))
    oracle_cost,oracle_pol=oracle()
    ce=oracle_pol[:,:, (np.arange(m+1)>m//2).astype(int)]
    baseline['Most-likely regime']=list(evaluate(ce,m))
    sb=solve_grid(m,blind=True)
    baseline['Action-independent transition']=list(evaluate(sb['policy'],m,planner_blind=True))
    soft=[]
    for weight in ([64,1024] if args.quick else [64,1024,16384,1048576]):
        lo,hi=evaluate(s['policy'],m,weight=weight)
        soft.append({'preferred_weight':weight,'value_lo':lo,'value_hi':hi,'certified_gap':float(up(hi-results[-1]['lower_bound']))})
    exact,calls=exact_tree()
    small=solve_grid(16,horizon=3,parent=2)
    small_eval=evaluate(small['policy'],16)
    assert Fraction.from_float(float(small['lower_lo'][0,2,8])) <= exact <= Fraction.from_float(small_eval[1])
    exact_eval=exact_controller(small['policy'],16)
    assert Fraction.from_float(small_eval[0])<=exact_eval<=Fraction.from_float(small_eval[1])
    soft_eval=evaluate(small['policy'],16,weight=64)
    exact_soft=exact_controller(small['policy'],16,weight=64)
    assert Fraction.from_float(soft_eval[0])<=exact_soft<=Fraction.from_float(soft_eval[1])
    structural_count=structural_checks(m)
    forward=forward_evaluate(s['policy'],m)
    assert abs(forward['value']-baseline['Certified belief controller'][1])<1e-10
    assert forward['maximum_mass_error']<1e-12
    signature_check=signature_controller_checks(s['policy'],m)
    robustness=[]
    if not args.quick:
        for horizon in [6,12]:
            ss=solve_grid(256,horizon=horizon)
            bb=solve_grid(256,horizon=horizon,blind=True)
            for bad in [Fraction(1,4),Fraction(1,2),Fraction(3,4)]:
                ev=evaluate(ss['policy'],256,initial_bad=bad)
                frozen=evaluate(frozen_prior(256,horizon=horizon,initial_bad=bad),256,initial_bad=bad)
                blind=evaluate(bb['policy'],256,planner_blind=True,initial_bad=bad)
                lb=float(ss['lower_lo'][0,PARENT,int(256*bad)])
                robustness.append({'horizon':horizon,'initial_bad':float(bad),'value_hi':ev[1],
                                   'certified_gap':float(up(ev[1]-lb)),
                                   'frozen_excess_lower':float(np.nextafter(frozen[0]-ev[1],-np.inf)),
                                   'blind_excess_lower':float(np.nextafter(blind[0]-ev[1],-np.inf))})
    report={'scope':'Synthetic, finite-observation, action-dependent hidden-liquidity execution; no hidden regime is provided to the certified controller.',
            'parameters':{'horizon':HORIZON,'parent_units':PARENT,'shares_per_unit':SHARES_PER_UNIT,'initial_bad_probability':.5,
                          'signal_probability_one_given_good_bad':[.35,.65],'probability_denominator':DEN,
                          'actions':[a.__dict__ for a in ACTIONS]},
            'grid_results':results,'policies':baseline,'fully_observed_oracle':oracle_cost,'softmax_results':soft,
            'independent_exact_tree':{'horizon':3,'parent':2,'rational_value':str(exact),'value':float(exact),'nodes':calls,
                                      'grid_lower':float(small['lower_lo'][0,2,8]),'controller_upper':small_eval[1]},
            'rational_controller_checks':{'deterministic_value':str(exact_eval),'softmax_value':str(exact_soft),'both_contained_in_outward_intervals':True},
            'integer_rounding_and_probability_enclosure_checks':structural_count,
            'forward_evaluation':forward,'robustness':robustness,
            'signature_controller_checks':signature_check,
            'numerical_method':'Integer likelihoods and Bayes quantization; elementary binary64 products, sums and divisions expanded with nextafter; all evaluated quantities are nonnegative.',
            'timing_scope':'Wall time is printed to the build log and excluded from deterministic results.'}
    print(json.dumps({'policies':baseline,'oracle':oracle_cost,'softmax':soft,'exact_tree':report['independent_exact_tree']},indent=2))
    out=ROOT/'results/partial_observation.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'results/partial_observation_policy.npz',policy=s['policy'],lower=s['lower_lo'],nearest=s['nearest_hi'])
    if not args.quick:
        export_outputs(report,s['policy'],s['lower_lo'])
        (ROOT/'audit/partial_observation_checks.json').write_text(json.dumps({k:report[k] for k in ['independent_exact_tree','rational_controller_checks','integer_rounding_and_probability_enclosure_checks','forward_evaluation','signature_controller_checks','numerical_method']},indent=2)+'\n')
    print(f'Elapsed seconds: {time.perf_counter()-start:.3f}',flush=True)


if __name__=='__main__':
    main()
