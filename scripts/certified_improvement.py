"""Synthetic common-model schedule certificates, checked by corner enumeration."""
from pathlib import Path
from itertools import product
import json
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def certificate(b, x, H, c, epsilon, B, rho):
    d = x-b
    A = d @ (H@b+c) + epsilon @ (b*np.abs(d)) + rho*np.linalg.norm(B.T@d)
    D = d@H@d + epsilon @ (np.sign(d)*d*d)
    assert D >= -1e-12
    t = 0. if A >= 0 else (min(1., -A/D) if D > 1e-14 else 1.)
    return float(A), float(D), float(t), float(max(0.,-(A*t+.5*D*t*t)))


def main():
    n = 6
    H = np.diag([1.,2.,4.,1.5,3.,2.5]) + .18*np.exp(-np.abs(np.arange(n)[:,None]-np.arange(n)))
    c = np.array([.04,.08,-.01,.06,-.03,.02])
    caps = np.array([.30,.22,.28,.25,.20,.28])
    b = np.full(n, 1/n)
    B = np.column_stack([np.linspace(-1,1,n), np.cos(np.arange(n))])
    opt = minimize(lambda x:.5*x@H@x+c@x, b, jac=lambda x:H@x+c,
                   bounds=list(zip(np.zeros(n),caps)),
                   constraints={'type':'eq','fun':lambda x:x.sum()-1.,'jac':lambda x:np.ones(n)},
                   method='SLSQP', options={'ftol':1e-14,'maxiter':500})
    assert opt.success, opt.message
    x = opt.x
    corners = np.array(list(product([-1.,1.],repeat=n)))
    rows = []
    error = 0.
    for scale in [0.,1.,2.,4.,8.]:
        epsilon = scale*.1*np.diag(H)
        rho = scale*.025
        assert np.linalg.eigvalsh(H-np.diag(epsilon)).min() > 0
        A,D,t,gain = certificate(b,x,H,c,epsilon,B,rho)
        def direct(step):
            y = b+step*(x-b)
            center = .5*y@H@y+c@y-(.5*b@H@b+c@b)
            return center + np.max(.5*(corners*epsilon)@(y*y-b*b)) + rho*np.linalg.norm(B.T@(y-b))
        for s in np.linspace(0,1,51):
            error = max(error,abs(direct(s)-(A*s+.5*D*s*s)))
        independent = minimize_scalar(direct,bounds=(0,1),method='bounded',options={'xatol':1e-13})
        independent_value = min(direct(0),direct(1),independent.fun)
        assert abs(independent_value+gain) < 1e-11
        accepted = b+t*(x-b)
        assert abs(accepted.sum()-1)<1e-12 and np.all(accepted>=0) and np.all(accepted<=caps+1e-12)
        assert direct(t) <= 1e-12
        assert certificate(b,b,H,c,epsilon,B,rho)[3] == 0.
        rows.append({'uncertainty_multiplier':scale,'full_proposal_worst_change':direct(1),
                     'accepted_fraction':t,'certified_saving':gain,'A':A,'D':D})
    assert error < 1e-12
    # Exhaustive viability check by a separately enumerated integer allocation.
    viability_cases = 0
    for remaining in range(9):
        for current_cap in range(5):
            for future in product(range(4),repeat=2):
                for y in range(6):
                    physical = y<=current_cap and y<=remaining and any(
                        y+f1+f2==remaining for f1 in range(future[0]+1) for f2 in range(future[1]+1))
                    interval = max(0,remaining-sum(future))<=y<=min(remaining,current_cap)
                    assert physical == interval
                    viability_cases += 1
    # Random schedule proposals exercise both positive and negative directions.
    rng=np.random.default_rng(5257)
    for _ in range(100):
        y = rng.dirichlet(np.ones(n))
        # No caps needed for this algebra check: both schedules are nonnegative and complete.
        eps=.05*np.diag(H)
        A,D,t,gain=certificate(b,y,H,c,eps,B,.03)
        s=float(rng.uniform())
        z=b+s*(y-b)
        exact=.5*z@H@z+c@z-(.5*b@H@b+c@b)+np.max(.5*(corners*eps)@(z*z-b*b))+.03*np.linalg.norm(B.T@(z-b))
        assert abs(exact-(A*s+.5*D*s*s))<1e-12
    report={'status':'all_passed','data':'Synthetic, normalized cost units; no market data.',
            'baseline':b.tolist(),'proposal':x.tolist(),'caps':caps.tolist(),
            'H':H.tolist(),'linear_charge':c.tolist(),'B':B.tolist(),
            'maximum_corner_identity_error':error,'corners_per_case':len(corners),
            'viability_cases':viability_cases,'random_proposal_checks':100,'rows':rows}
    (ROOT/'results/certified_improvement.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'tables/certified_improvement_rows.tex').write_text(''.join(
        f"{r['uncertainty_multiplier']:.0f} & {r['full_proposal_worst_change']:.6f} & {r['accepted_fraction']:.6f} & {r['certified_saving']:.6f} \\\\\n" for r in rows))
    fig,ax=plt.subplots(1,2,figsize=(10.8,3.8),layout='constrained')
    grid=np.linspace(0,1,201)
    for r in rows[::2]:
        ax[0].plot(grid,r['A']*grid+.5*r['D']*grid**2,label=f"Radius x{r['uncertainty_multiplier']:g}")
        ax[0].scatter(r['accepted_fraction'],-r['certified_saving'],s=28)
    ax[0].axhline(0,color='gray',lw=.8); ax[0].set(xlabel='Fraction of proposed change',ylabel='Worst-case cost change')
    ax[0].legend(frameon=False)
    ax[1].plot([r['uncertainty_multiplier'] for r in rows],[r['accepted_fraction'] for r in rows],'o-',color='#0057a6')
    ax[1].set(xlabel='Uncertainty multiplier',ylabel='Accepted fraction',ylim=(-.04,1.04))
    for a in ax:a.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'figures/12_certified_improvement.png',dpi=220,facecolor='white');plt.close(fig)
    print(json.dumps({'status':report['status'],'rows':rows,'maximum_corner_identity_error':error,'viability_cases':viability_cases},indent=2))


if __name__=='__main__': main()
