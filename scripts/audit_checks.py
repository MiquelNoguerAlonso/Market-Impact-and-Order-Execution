"""Independent accounting, posterior, and output-consistency checks."""
import json
import math
import numpy as np
from scipy.stats import binom
from experiments import ROOT


def ledger_checks():
    rng=np.random.default_rng(525723)
    largest=0.
    for _ in range(500):
        fills=rng.multinomial(2000,rng.dirichlet(np.ones(12)))
        midpoint=100+np.r_[0,np.cumsum(rng.normal(0,.02,12))]
        execution=midpoint[:-1]+rng.normal(.004,.003,12)
        fees=rng.normal(.0001,.0002,12)
        remaining=2000-np.cumsum(fills)
        direct=np.sum(fills*(execution-midpoint[0]+fees))
        decomposed=np.sum(fills*(execution-midpoint[:-1]+fees))+np.sum(remaining*np.diff(midpoint))
        error=abs(direct-decomposed);largest=max(largest,error)
        assert remaining[-1]==0 and error<1e-8
    return {'completed_ledgers':500,'largest_accounting_error':largest}


def posterior_checks():
    largest=0.;cases=0
    for prior in [0.,.1,.5,.9,1.]:
        for ell in range(1,6):
            for p0,p1 in [[.15,.60],[.05,.75]]:
                fill=np.arange(ell+1)
                l0=binom.pmf(fill,ell,p0);l1=binom.pmf(fill,ell,p1)
                mixture=(1-prior)*l0+prior*l1
                post=prior*l1/mixture
                error=abs(mixture@post-prior);largest=max(largest,error);cases+=1
                assert error<1e-13
    return {'binomial_posterior_cases':cases,'largest_posterior_mean_error':largest}


def reservation_check():
    # A pending cancel releases no capacity. Cumulative reports are idempotent.
    parent=1000;reported={'A':0,'B':0,'C':0};reserved={};events=[]
    def record(name):
        remaining=parent-sum(reported.values())
        assert sum(reserved.values())<=remaining
        events.append({'event':name,'remaining':remaining,'reservations':dict(reserved)})
    reserved['A']=600;record('send A for 600')
    reserved['B']=400;record('send B for 400')
    reported['A']=200;reserved['A']-=200;record('authoritative A cumulative fill 200')
    record('request cancellation of A; keep reservation')
    # 150 additional fills arrive with the final cancellation report.
    reported['A']=350;reserved.pop('A');record('terminal A report: cumulative fill 350')
    record('duplicate terminal A report: no quantity change')
    reserved['C']=250;record('send C for 250')
    reported['B']=400;reserved.pop('B');record('B completes 400')
    reported['C']=250;reserved.pop('C');record('C completes 250')
    assert sum(reported.values())==parent
    return {'events':events,'total_filled':sum(reported.values())}


def output_checks():
    r=json.loads((ROOT/'results/experiments.json').read_text())
    arrays=np.load(ROOT/'results/policy_scenarios.npz')
    for name,row in r['policies'].items():
        outcomes=arrays[name.replace(' ','_')]
        assert len(outcomes)==16000
        assert abs(outcomes.mean()-row['objective'])<1e-12
        assert abs(row['charge']+row['penalty']-row['objective'])<1e-11
        assert row['mean_inventory_path'][0]==20 and row['mean_inventory_path'][-1]==0
        assert np.max(np.diff(row['mean_inventory_path']))<=0
    difference=arrays['Frozen_prior']-arrays['Belief_policy']
    assert abs(difference.mean()-r['paired']['frozen_minus_belief'])<1e-12
    assert abs(difference.std(ddof=1)/math.sqrt(16000)-r['paired']['standard_error'])<1e-12
    text=(ROOT/'tables/policy_rows.tex').read_text()
    for name,row in r['policies'].items():assert f"{row['objective']:.3f}" in text
    cert=json.loads((ROOT/'results/certificate.json').read_text())
    for row in cert['scenarios']:
        assert row['regret_against_true_qp']+cert['true_optimum_duality_gap']<=row['regret_certificate']+1e-10
    assert (ROOT/'results/belief_dp.npz').exists()
    return {'policy_rows_verified':6,'paired_comparison_verified':True,
            'certificate_scenarios_verified':len(cert['scenarios'])}


def main():
    r={'ledgers':ledger_checks(),'posteriors':posterior_checks(),
       'reservation':reservation_check(),'outputs':output_checks(),
       'status':'All accounting and output checks passed.'}
    (ROOT/'audit/accounting_checks.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['status'])


if __name__=='__main__':main()
