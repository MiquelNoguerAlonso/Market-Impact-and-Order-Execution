"""Exact checks of the three-point defect and estimated-impact profit.

The independent minimizer uses the six edges of the TV-unit polygon in
(u, v), evaluating each edge's stationary point and its endpoints. It does
not use the manuscript's p,d reduction or its closed-form defect formula.
The zero vector covers the nonnegative homogeneous-quadratic case.
"""
from fractions import Fraction as F
from itertools import product
import json


def impact_matrix(g):
    return tuple(tuple(g[abs(i - j)] for j in range(3)) for i in range(3))


def energy(k, x):
    return sum((k[i][j] * x[i] * x[j]
                for i in range(3) for j in range(3)), F(0))


def zero_mass_trade(u, v):
    return (u, -u - v, v)


def polygon_minimum(g):
    """Globally minimize x'Kx with sum(x)=0 and ||x||_1 <= 1."""
    k = impact_matrix(g)
    half = F(1, 2)
    vertices = ((half, -half), (half, F(0)), (F(0), half),
                (-half, half), (-half, F(0)), (F(0), -half))
    candidates = [(F(0), (F(0), F(0), F(0)))]
    stationary_points = 0
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        start = zero_mass_trade(*a)
        direction = zero_mass_trade(b[0] - a[0], b[1] - a[1])
        constant = energy(k, start)
        quadratic = energy(k, direction)
        linear = (energy(k, tuple(start[i] + direction[i] for i in range(3)))
                  - constant - quadratic)
        times = [F(0), F(1)]
        if quadratic > 0:
            stationary = -linear / (2 * quadratic)
            if 0 < stationary < 1:
                times.append(stationary)
                stationary_points += 1
        for t in times:
            x = tuple(start[i] + t * direction[i] for i in range(3))
            assert sum(x) == 0 and sum(map(abs, x)) <= 1
            candidates.append((energy(k, x), x))
    best = min(candidates, key=lambda item: item[0])
    return best[0], best[1], stationary_points


def claimed_defect(g):
    assert g[0] >= g[2]
    return max(F(0), (4 * g[1] - 3 * g[0] - g[2]) / 8)


def defect_checks():
    g = (F(1), F(19, 20), F(1, 5))
    k = impact_matrix(g)
    # Direct matrix evaluations check the displayed coefficients separately.
    d_axis = zero_mass_trade(F(1, 2), -F(1, 2))
    p_axis = zero_mass_trade(F(1, 2), F(1, 2))
    assert energy(k, d_axis) == F(2, 5)
    assert energy(k, p_axis) == -F(3, 10)
    loop = (F(1), F(-2), F(1))
    assert energy(k, loop) == -F(6, 5)
    normalized = (F(1, 4), -F(1, 2), F(1, 4))
    assert energy(k, normalized) == -F(3, 40)
    assert -energy(k, loop) / 2 == F(3, 5)

    values = tuple(map(F, ('-1', '0', '1/5', '3/5', '7/10',
                          '4/5', '19/20', '1', '2')))
    count = stationary_count = expansion_count = 0
    positive_count = zero_count = 0
    for triple in product(values, repeat=3):
        if triple[0] < triple[2]:
            continue
        minimum, witness, stationary = polygon_minimum(triple)
        expected = claimed_defect(triple)
        assert -minimum == expected, (triple, minimum, expected)
        assert energy(impact_matrix(triple), witness) == minimum
        stationary_count += stationary
        count += 1
        positive_count += int(expected > 0)
        zero_count += int(expected == 0)
        if expected > 0:
            assert -energy(impact_matrix(triple), normalized) == expected
        if triple[0] - 2 * triple[1] + triple[2] >= 0:
            assert expected == 0
            if triple[0] > triple[2]:
                assert 4 * triple[1] < 3 * triple[0] + triple[2]
        for u, v in ((F(2, 7), -F(1, 3)), (F(-3, 5), F(4, 9)),
                     (F(1, 4), F(1, 4))):
            p, d = u + v, u - v
            reduced = ((triple[0] - triple[2]) * d**2 / 2
                       + (3 * triple[0] - 4 * triple[1] + triple[2]) * p**2 / 2)
            assert energy(impact_matrix(triple), zero_mass_trade(u, v)) == reduced
            expansion_count += 1

    # The assumption G0 >= G2 is necessary for the stated formula.
    excluded = (F(0), F(0), F(1))
    excluded_min, _, _ = polygon_minimum(excluded)
    assert excluded_min == -F(1, 2)
    assert max(F(0), (4 * excluded[1] - 3 * excluded[0] - excluded[2]) / 8) == 0

    examples = []
    for middle, expected in ((F(3, 5), F(0)), (F(7, 10), F(0)),
                             (F(4, 5), F(0)), (F(19, 20), F(3, 40))):
        triple = (F(1), middle, F(1, 5))
        minimum, _, _ = polygon_minimum(triple)
        assert -minimum == expected
        examples.append({'kernel': list(map(str, triple)),
                         'discretely_convex': triple[0] - 2 * middle + triple[2] >= 0,
                         'defect': str(-minimum)})

    h = F(1, 100)
    perturbed = (g[0] - h, g[1] + h, g[2] - h)
    assert all(a >= b for a, b in zip(perturbed, perturbed[1:]))
    assert min(perturbed) >= 0
    base_min, _, _ = polygon_minimum(g)
    perturbed_min, _, _ = polygon_minimum(perturbed)
    assert base_min - perturbed_min == h
    assert max(abs(a - b) for a, b in zip(g, perturbed)) == h

    # An admissible boundary kernel and an estimate attain the error bound.
    true_g = (F(1), F(4, 5), F(1, 5))
    estimated_g = (true_g[0] - h, true_g[1] + h, true_g[2] - h)
    true_min, _, _ = polygon_minimum(true_g)
    estimated_min, _, _ = polygon_minimum(estimated_g)
    assert true_min == 0 and -estimated_min == h
    true_k, estimated_k = impact_matrix(true_g), impact_matrix(estimated_g)
    error = max(abs(true_k[i][j] - estimated_k[i][j])
                for i in range(3) for j in range(3))
    assert error == h
    profit_cases = []
    for volume in (F(0), F(1), F(4), F(10)):
        trade = tuple(volume * a for a in normalized)
        assert sum(trade) == 0 and sum(map(abs, trade)) == volume
        gain = -energy(estimated_k, trade) / 2
        assert gain == volume**2 * h / 2
        assert energy(true_k, trade) == 0
        profit_cases.append({'gross_volume': str(volume),
                             'phantom_profit': str(gain),
                             'half_Q_squared_error': str(volume**2 * error / 2)})

    # Zero round-trip defect does not exclude opposite-side mandate trades.
    oscillatory_g = (F(1), F(7, 10), F(1, 5))
    oscillatory_k = impact_matrix(oscillatory_g)
    signed_optimum = (F(3, 4), -F(1, 2), F(3, 4))
    buy_optimum = (F(1, 2), F(0), F(1, 2))
    signed_potential = tuple(sum(oscillatory_k[i][j] * signed_optimum[j]
                                 for j in range(3)) for i in range(3))
    buy_potential = tuple(sum(oscillatory_k[i][j] * buy_optimum[j]
                              for j in range(3)) for i in range(3))
    assert -polygon_minimum(oscillatory_g)[0] == 0
    assert signed_potential == (F(11, 20),) * 3  # Signed stationarity.
    assert buy_potential == (F(3, 5), F(7, 10), F(3, 5))  # Buy-only KKT.
    assert energy(oscillatory_k, signed_optimum) / 2 == F(11, 40)
    assert energy(oscillatory_k, buy_optimum) / 2 == F(3, 10)
    for a in (F(-1), F(0), F(1, 2), F(3, 4), F(1)):
        x = (a, 1 - 2*a, a)
        assert energy(oscillatory_k, x) / 2 == F(1, 2) - 3*a/5 + 2*a*a/5

    # Euclidean projection of sampled kernel values may increase sup error.
    normal = (F(-3), F(4), F(-1))
    noisy_g = (true_g[0] + h, true_g[1] + h, true_g[2] - h)
    violation = sum(a*b for a, b in zip(normal, noisy_g))
    assert violation == 2*h
    projected_g = tuple(a - violation*b/26 for a, b in zip(noisy_g, normal))
    assert sum(a*b for a, b in zip(normal, projected_g)) == 0
    assert projected_g[0] > projected_g[2]
    assert -polygon_minimum(projected_g)[0] == 0
    before = max(abs(a-b) for a, b in zip(noisy_g, true_g))
    after = max(abs(a-b) for a, b in zip(projected_g, true_g))
    assert before == h and after == F(16, 13)*h

    # Exact uniform distance to the admissible cone on the stated domain.
    repair_size = claimed_defect(g)
    nearest_g = (g[0] + repair_size, g[1] - repair_size, g[2] + repair_size)
    assert -polygon_minimum(nearest_g)[0] == 0
    assert max(abs(a-b) for a, b in zip(nearest_g, g)) == repair_size
    return {
        'status': 'passed',
        'arithmetic': 'exact fractions; no floating-point tolerance',
        'independent_optimizer': 'six TV-polygon edges and zero trade',
        'tested_kernel_triples': count,
        'positive_defect_cases': positive_count,
        'zero_defect_cases': zero_count,
        'interior_edge_stationary_points_checked': stationary_count,
        'energy_expansion_checks': expansion_count,
        'example_coefficients': {'d_squared': '2/5', 'p_squared': '-3/10'},
        'example_defect': '3/40',
        'gross_volume_four_profit': '3/5',
        'threshold_examples': examples,
        'excluded_assumption_counterexample': {
            'kernel': list(map(str, excluded)), 'actual_defect': str(-excluded_min),
            'reason': 'G0 < G2 lies outside the proposition'},
        'lipschitz_sharpness': {
            'base_kernel': list(map(str, g)),
            'perturbed_kernel': list(map(str, perturbed)),
            'uniform_distance': str(h), 'defect_increase': str(h),
            'ratio': '1'},
        'phantom_bound_sharpness': {
            'true_admissible_kernel': list(map(str, true_g)),
            'estimated_kernel': list(map(str, estimated_g)),
            'entrywise_error': str(error), 'cases': profit_cases},
        'zero_defect_opposite_side_optimum': {
            'kernel': list(map(str, oscillatory_g)),
            'signed_optimum': list(map(str, signed_optimum)), 'signed_cost': '11/40',
            'buy_only_optimum': list(map(str, buy_optimum)), 'buy_only_cost': '3/10',
            'signed_potential': list(map(str, signed_potential)),
            'buy_only_potential': list(map(str, buy_potential))},
        'projection_error_counterexample': {
            'metric': 'Euclidean norm on the three sampled kernel values',
            'true_kernel': list(map(str, true_g)),
            'estimate': list(map(str, noisy_g)),
            'projection': list(map(str, projected_g)),
            'uniform_error_before': str(before), 'uniform_error_after': str(after),
            'error_ratio': '16/13', 'projected_defect': '0'},
        'uniform_cone_distance_example': {
            'kernel': list(map(str, g)), 'admissible_repair': list(map(str, nearest_g)),
            'uniform_distance': str(repair_size), 'defect': str(repair_size)},
        'scope': 'Fixed three-point impact models; tests corroborate the manuscript proofs.'
    }


if __name__ == '__main__':
    print(json.dumps(defect_checks(), indent=2))
