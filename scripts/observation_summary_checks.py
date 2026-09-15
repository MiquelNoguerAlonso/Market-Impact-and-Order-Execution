"""Exact finite-tree checks of the observation-summary certificate.

The seven instruction parameters and charges are the manuscript benchmark.
This script independently constructs rational transition/fill/report laws,
enumerates ALL feasible fine-report histories (merging only identical
q, memory, full-posterior states), forms posterior-fiber envelopes, and
compares their bounds with exact full-information-history dynamic programming.
The physical policy is also evaluated by forward hidden-state occupancies.
No market calibration or formal proof checking is claimed.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from math import comb, erf, sqrt
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
# name, market, passive, T01, T10, fill-good, fill-bad, fee in 1/1000 USD
PARAMETERS = (
    ('Wait', 0, 0, 5, 18, 0, 0, 0),
    ('Lit 1', 0, 1, 12, 12, 80, 18, 10),
    ('Lit 2', 0, 2, 28, 7, 65, 8, 10),
    ('Dark 1', 0, 1, 4, 15, 70, 3, 25),
    ('Market 1', 1, 0, 35, 6, 0, 0, 0),
    ('Market 2', 2, 0, 60, 3, 0, 0, 0),
    ('Split 1+1', 1, 1, 40, 5, 72, 12, 10),
)


def allowed(q):
    return [i for i, a in enumerate(PARAMETERS) if a[1]+a[2] <= q] if q else [0]


def cost(q, ai, fill):
    _, market, _, *_, fee = PARAMETERS[ai]
    left = q-market-fill
    assert left >= 0
    return F(1000*market+180*market**2+80*fill+fee+40*left**2, 1000)


def terminal(q):
    return F(1800*q+150*q*q, 1000)


@lru_cache(None)
def kernel(ai, signals):
    _, _, passive, t01, t10, good, bad, _ = PARAMETERS[ai]
    tr = ((100-t01, t01), (t10, 100-t10))
    out = {}
    for fill in range(passive+1):
        for signal in range(2):
            m = []
            for h in range(2):
                row = []
                for hp in range(2):
                    p = F((good, bad)[hp], 100)
                    f = comb(passive, fill)*p**fill*(1-p)**(passive-fill)
                    s = F(signals[hp], 100)
                    row.append(F(tr[h][hp], 100)*f*(s if signal else 1-s))
                m.append(tuple(row))
            out[fill, signal] = tuple(m)
    for h in range(2):
        assert sum(m[h][hp] for m in out.values() for hp in range(2)) == 1
    return out


def posterior(b, mat):
    weights = tuple((1-b)*mat[0][hp]+b*mat[1][hp] for hp in range(2))
    prob = sum(weights)
    return prob, weights[1]/prob if prob else F(0)


def nearest(b, mesh):
    scaled = mesh*b
    return (2*scaled.numerator+scaled.denominator)//(2*scaled.denominator)


def record(x):
    return {'exact': str(x), 'decimal': float(x)}


def finite_tree_case(signals=(35, 65), mesh=8, horizon=3, parent=2):
    coarse = {}
    for ai in range(len(PARAMETERS)):
        fine = kernel(ai, signals)
        coarse[ai] = {
            f: tuple(tuple(sum(fine[f, y][h][hp] for y in range(2))
                           for hp in range(2)) for h in range(2))
            for f in range(PARAMETERS[ai][2]+1)
        }

    @lru_cache(None)
    def memory(ai, j, fill):
        _, b = posterior(F(j, mesh), coarse[ai][fill])
        return nearest(b, mesh)

    @lru_cache(None)
    def branches(q, j, b, ai):
        result = []
        for (fill, signal), mat in kernel(ai, signals).items():
            prob, bp = posterior(b, mat)
            if prob:
                child = q-PARAMETERS[ai][1]-fill, memory(ai, j, fill), bp
                result.append((prob, cost(q, ai, fill), child))
        assert sum(p for p, _, _ in result) == 1
        return tuple(result)

    start = parent, mesh//2, F(1, 2)
    states = [{start}]
    for k in range(horizon):
        nxt = set()
        for q, j, b in states[-1]:
            for ai in allowed(q):
                nxt.update(child for _, _, child in branches(q, j, b, ai))
        states.append(nxt)
    fibers = []
    for layer in states:
        groups = defaultdict(list)
        for q, j, b in layer:
            groups[q, j].append(b)
        fibers.append({z: (min(bs), max(bs)) for z, bs in groups.items()})

    lower, upper, selected, rich, selected_cost = [], [], [], [], []
    action_lower, action_upper, action_excess, action_regret = [], [], [], []
    for layer in states:
        lower.append({}); upper.append({}); selected.append({})
        rich.append({}); selected_cost.append({})
        action_lower.append({}); action_upper.append({})
        action_excess.append({}); action_regret.append({})
    for s in states[-1]:
        rich[-1][s] = selected_cost[-1][s] = terminal(s[0])
    for z in fibers[-1]:
        lower[-1][z] = upper[-1][z] = terminal(z[0])
        action_regret[-1][z] = F(0)
    verified = verified_actions = screened_actions = 0
    for k in range(horizon-1, -1, -1):
        for (q, j), endpoints in sorted(fibers[k].items()):
            la, ua = [], []
            for ai in allowed(q):
                lv, uv = [], []
                for b in endpoints:
                    lval = uval = F(0)
                    for fill, mat in coarse[ai].items():
                        prob, _ = posterior(b, mat)
                        if not prob:
                            continue
                        zp = q-PARAMETERS[ai][1]-fill, memory(ai, j, fill)
                        c = cost(q, ai, fill)
                        lval += prob*(c+lower[k+1][zp])
                        uval += prob*(c+upper[k+1][zp])
                    lv.append(lval); uv.append(uval)
                la.append((min(lv), ai)); ua.append((max(uv), ai))
            lower[k][q, j] = min(la)[0]
            upper[k][q, j], selected[k][q, j] = min(ua)
            z = q, j
            action_lower[k][z] = {ai: val for val, ai in la}
            action_upper[k][z] = {ai: val for val, ai in ua}
            excess = {}
            for ai in allowed(q):
                alternatives = [val for val, aj in la if aj != ai]
                excess[ai] = (max(F(0), action_upper[k][z][ai]-min(alternatives))
                              if alternatives else F(0))
            action_excess[k][z] = excess
            chosen = selected[k][z]
            continuation = []
            for b in endpoints:
                value = F(0)
                for fill, mat in coarse[chosen].items():
                    prob, _ = posterior(b, mat)
                    if prob:
                        zp = q-PARAMETERS[chosen][1]-fill, memory(chosen, j, fill)
                        value += prob*action_regret[k+1][zp]
                continuation.append(value)
            action_regret[k][z] = excess[chosen]+max(continuation)
        for s in states[k]:
            q, j, b = s
            rich[k][s] = min(sum(p*(c+rich[k+1][child])
                                 for p, c, child in branches(q, j, b, ai))
                             for ai in allowed(q))
            ai = selected[k][q, j]
            selected_cost[k][s] = sum(p*(c+selected_cost[k+1][child])
                                      for p, c, child in branches(q, j, b, ai))
            assert lower[k][q, j] <= rich[k][s] <= selected_cost[k][s] <= upper[k][q, j]
            assert selected_cost[k][s]-rich[k][s] <= action_regret[k][q, j]
            for aj in allowed(q):
                exact_q = sum(p*(c+rich[k+1][child])
                              for p, c, child in branches(q, j, b, aj))
                assert action_lower[k][q, j][aj] <= exact_q <= action_upper[k][q, j][aj]
                assert 0 <= exact_q-rich[k][s] <= action_excess[k][q, j][aj]
                if action_lower[k][q, j][aj] > min(action_upper[k][q, j].values()):
                    assert exact_q > rich[k][s]
                    screened_actions += 1
                verified_actions += 1
            verified += 1

    # Independent physical evaluation: forward occupancy over actual hidden
    # states, without computing a posterior or redrawing a hidden regime.
    mass = {(parent, mesh//2, 0): F(1, 2), (parent, mesh//2, 1): F(1, 2)}
    physical = F(0)
    for k in range(horizon):
        nxt = defaultdict(F)
        for (q, j, h), mu in mass.items():
            ai = selected[k][q, j]
            for (fill, signal), mat in kernel(ai, signals).items():
                qp, jp = q-PARAMETERS[ai][1]-fill, memory(ai, j, fill)
                for hp in range(2):
                    w = mu*mat[h][hp]
                    if w:
                        nxt[qp, jp, hp] += w
                        physical += w*cost(q, ai, fill)
        mass = dict(nxt)
        assert sum(mass.values()) == 1
    physical += sum(mu*terminal(q) for (q, j, h), mu in mass.items())
    assert physical == selected_cost[0][start]

    # Exact optimum using fills alone: this retains the entire coarse history,
    # not the rounded memory of the implemented summary policy.
    @lru_cache(None)
    def coarse_opt(k, q, b):
        if k == horizon:
            return terminal(q)
        vals = []
        for ai in allowed(q):
            val = F(0)
            for fill, mat in coarse[ai].items():
                prob, bp = posterior(b, mat)
                if prob:
                    qp = q-PARAMETERS[ai][1]-fill
                    val += prob*(cost(q, ai, fill)+coarse_opt(k+1, qp, bp))
            vals.append(val)
        return min(vals)

    v, l, u = rich[0][start], lower[0][start[:2]], upper[0][start[:2]]
    vc = coarse_opt(0, parent, F(1, 2))
    assert v <= vc <= physical <= u
    return {
        'signal_probabilities_percent': list(signals), 'mesh_subintervals': mesh,
        'horizon': horizon, 'parent_units': parent,
        'rich_optimum': record(v), 'coarse_history_optimum': record(vc),
        'lower_bound': record(l), 'upper_recursion': record(u),
        'physical_policy_cost': record(physical),
        'information_loss': record(vc-v), 'actual_policy_gap': record(physical-v),
        'certified_policy_gap': record(physical-l), 'envelope_width': record(u-l),
        'action_regret_certificate': record(action_regret[0][start[:2]]),
        'combined_certificate': record(min(physical-l, action_regret[0][start[:2]])),
        'full_states_by_stage': [len(s) for s in states],
        'summary_states_by_stage': [len(f) for f in fibers],
        'conditional_sandwich_checks': verified,
        'conditional_action_checks': verified_actions,
        'strict_action_screening_checks': screened_actions,
        'forward_hidden_state_evaluation_equals_history_evaluation': True,
        'maximum_posterior_memory_distance': record(max(abs(b-F(j, mesh))
            for layer in states for q, j, b in layer)),
    }


def probe_example():
    """Exact two-decision example: probe or wait, then market or passive."""
    prior = F(1, 2)
    accuracy, probe_fee = F(9, 10), F(1, 50)
    # b is probability of GOOD liquidity; passive failure costs 2 USD.
    passive = lambda b: 2*(1-(b*F(9, 10)+(1-b)*F(1, 10)))
    stage_value = lambda b: min(F(1), passive(b))
    rich_probe = probe_fee+(stage_value(accuracy)+stage_value(1-accuracy))/2
    rich = min(stage_value(prior), rich_probe)
    coarse = stage_value(prior)  # discarded report: paying to probe cannot help
    lower = min(stage_value(prior), probe_fee+min(stage_value(accuracy), stage_value(1-accuracy)))
    upper = min(stage_value(prior), probe_fee+min(F(1), max(passive(accuracy), passive(1-accuracy))))
    assert (rich, coarse, lower, upper) == (F(7,10), F(1), F(19,50), F(1))
    # Retaining the report makes every conditional envelope a singleton.
    sufficient_lower = sufficient_upper = rich
    assert sufficient_lower == sufficient_upper == F(7,10)
    return {k: record(v) for k, v in {
        'rich_optimum': rich, 'coarse_optimum': coarse, 'information_loss': coarse-rich,
        'lower_bound': lower, 'upper_bound': upper, 'certified_gap': upper-lower,
        'report_retained_lower': sufficient_lower, 'report_retained_upper': sufficient_upper,
    }.items()}


def non_singleton_decision_example():
    """Exact decision sufficiency despite a nonconstant richer value."""
    fee = F(1, 50)
    costs = {'L': (F(1, 10), F(1, 5)), 'M': (F(1, 2), F(1, 2))}
    qminus = {a: min(v) for a, v in costs.items()}
    qplus = {a: max(v) for a, v in costs.items()}
    excess = {a: max(F(0), qplus[a]-min(qminus[b] for b in costs if b != a))
              for a in costs}
    assert excess == {'L': F(0), 'M': F(2, 5)}
    rich = fee+sum(min(costs[a][h] for a in costs) for h in range(2))/2
    lower, upper = fee+min(qminus.values()), fee+min(qplus.values())
    assert (rich, lower, upper) == (F(17,100), F(3,25), F(11,50))
    rows = []
    for market_probability in (F(0), F(1,4), F(1)):
        probabilities = {'L': 1-market_probability, 'M': market_probability}
        actual = fee+sum(probabilities[a]*sum(costs[a])/2 for a in costs)
        certificate = sum(probabilities[a]*excess[a] for a in costs)
        assert 0 <= actual-rich <= certificate
        for h in range(2):
            conditional_gap = (sum(probabilities[a]*costs[a][h] for a in costs)
                               -min(costs[a][h] for a in costs))
            assert 0 <= conditional_gap <= certificate
        rows.append({'market_probability': record(market_probability),
                     'physical_cost': record(actual), 'actual_regret': record(actual-rich),
                     'cost_interval_certificate': record(actual-lower),
                     'action_certificate': record(certificate),
                     'combined_certificate': record(min(actual-lower, certificate))})
    assert qminus['M'] > min(qplus.values())
    assert costs['L'][0] != costs['L'][1]
    assert rows[0]['action_certificate']['exact'] == '0'
    return {'fee': record(fee), 'terminal_instruction_costs': {
                a: [record(v) for v in vals] for a, vals in costs.items()},
            'rich_optimum': record(rich), 'lower_bound': record(lower),
            'upper_bound': record(upper), 'envelope_width': record(upper-lower),
            'screened_instruction': 'M', 'policies': rows}


def grid_and_gaussian_checks():
    prior, m = F(1, 3), 8
    grid = sorted({F(i, m) for i in range(m+1)} | {prior})
    assert prior in grid and len(grid) <= m+2
    assert max(b-a for a, b in zip(grid, grid[1:])) <= F(1, m)
    checked = 0
    for left, right in zip(grid, grid[1:]):
        for theta in (F(0), F(1,4), F(1,2), F(3,4), F(1)):
            b = (1-theta)*left+theta*right
            mean_distance = (1-theta)*abs(b-left)+theta*abs(right-b)
            assert mean_distance == 2*theta*(1-theta)*(right-left) <= F(1,2*m)
            checked += 1
    gaussian_savings = {str(e): .5*erf(abs(e)/(2*sqrt(2))) for e in (-1, 0, 1)}
    assert gaussian_savings['-1'] == gaussian_savings['1'] > gaussian_savings['0'] == 0
    return {'initial_prior': str(prior), 'grid': [str(x) for x in grid],
            'interpolation_checks': checked, 'gaussian_savings_unit_loss': gaussian_savings}


def main():
    cases = [finite_tree_case(signals=s, mesh=m)
             for s, m in [((35,65),8), ((5,95),8), ((5,95),16)]]
    assert cases[1]['information_loss']['decimal'] > 0
    result = {'status': 'passed', 'arithmetic': 'fractions.Fraction (exact rational)',
              'scope': 'Finite examples; not a formal proof of the general theorem.',
              'controlled_liquidity_cases': cases, 'probe_example': probe_example(),
              'non_singleton_decision_example': non_singleton_decision_example(),
              'localized_corrections': grid_and_gaussian_checks()}
    for folder in ('results', 'audit'):
        (ROOT/folder/'observation_summary_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    rows, action_rows = [], []
    for c in cases:
        from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
        def fmt(key, upward=False):
            x = F(c[key]['exact'])
            with localcontext() as ctx:
                ctx.prec = 70
                val = Decimal(x.numerator)/Decimal(x.denominator)
                return str(val.quantize(Decimal('.000001'), rounding=ROUND_CEILING if upward else ROUND_FLOOR))
        sig = c['signal_probabilities_percent']
        rows.append(f"${sig[0]/100:.2f}/{sig[1]/100:.2f}$ & {c['mesh_subintervals']} & "
                    f"{fmt('rich_optimum')} & {fmt('coarse_history_optimum')} & "
                    f"{fmt('physical_policy_cost', True)} & {fmt('lower_bound')} & "
                    f"{fmt('certified_policy_gap', True)} " + r'\\')
        action_rows.append(f"${sig[0]/100:.2f}/{sig[1]/100:.2f}$ & {c['mesh_subintervals']} & "
                           f"{fmt('actual_policy_gap', True)} & {fmt('certified_policy_gap', True)} & "
                           f"{fmt('action_regret_certificate', True)} & "
                           f"{fmt('combined_certificate', True)} " + r'\\')
    (ROOT/'tables/observation_summary_rows.tex').write_text('\n'.join(rows)+'\n')
    (ROOT/'tables/action_separation_rows.tex').write_text('\n'.join(action_rows)+'\n')
    print(json.dumps({'status': 'passed', 'cases': cases, 'probe_example': result['probe_example']}, indent=2))


if __name__ == '__main__':
    main()
