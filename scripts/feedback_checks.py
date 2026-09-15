"""Signature feedback, observable masks, hybrid coupling and reference models.

Synthetic calculations for the repaired controlled-book results. The finite
book uses a fully observed state and a Bellman oracle; it does not implement
the nonconstructive Lusin approximation for general hidden-state histories.
Run directly to regenerate the JSON report and manuscript table. Numerical
assertions support these examples; they are not formal proofs of universality.
"""
from itertools import product
from pathlib import Path
import json
import math

import numpy as np

from orderbook_bridge import (
    ACTIONS, BID, DONE, INITIAL, N, STATES, TERMINAL, intervention,
    solve, transition,
)

ROOT = Path(__file__).resolve().parents[1]
LOSS_WIDTH = TERMINAL - BID


def signature_two(nodes):
    """Exact piecewise-linear signature through level two, via Chen's rule."""
    nodes = np.asarray(nodes, dtype=float)
    first = np.zeros(nodes.shape[1])
    second = np.zeros((nodes.shape[1], nodes.shape[1]))
    for inc in np.diff(nodes, axis=0):
        second += np.outer(first, inc) + np.outer(inc, inc) / 2
        first += inc
    return first, second


def transcript_lift(records, alphabet_size):
    """Records include the initial observation, even on a zero-event path."""
    assert records and records[0][0] == 0
    m = len(records)
    nodes = np.zeros((m + 1, 2 + alphabet_size))
    counts = np.zeros(alphabet_size)
    for i, (timestamp, letter) in enumerate(records, 1):
        counts[letter] += 1
        nodes[i] = np.r_[i / m, timestamp, counts]
    assert np.all(np.diff(nodes[:, 0]) > 0)
    return nodes


def mask_softmax(scores, mask):
    scores = np.asarray(scores, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    assert mask.any() and np.isfinite(scores).all()
    out = np.zeros_like(scores)
    z = scores[mask] - np.max(scores[mask])
    out[mask] = np.exp(z) / np.exp(z).sum()
    return out


def encoding_checks():
    init0 = transcript_lift([(0, 0)], 4)
    init1 = transcript_lift([(0, 1)], 4)
    sig0, _ = signature_two(init0)
    sig1, _ = signature_two(init1)
    assert not np.array_equal(sig0, sig1)
    # Merely translating a path by its initial observation loses that value.
    line = np.array([[0., 0.], [1., 0.]])
    assert np.array_equal(signature_two(line)[0],
                          signature_two(line + np.array([0., 1.]))[0])
    # Co-timestamped reports remain ordered; first levels coincide but areas do not.
    ab = transcript_lift([(0, 0), (.5, 2), (.5, 3)], 4)
    ba = transcript_lift([(0, 0), (.5, 3), (.5, 2)], 4)
    f_ab, s_ab = signature_two(ab)
    f_ba, s_ba = signature_two(ba)
    assert np.array_equal(f_ab, f_ba) and not np.allclose(s_ab, s_ba)
    count = 0
    for m in range(1, 6):
        for letters in product(range(2), repeat=m - 1):
            records = [(0, 0)] + [(i / m, 2 + a) for i, a in enumerate(letters, 1)]
            nodes = transcript_lift(records, 4)
            assert int(nodes[-1, 2:].sum()) == m
            recovered = [(float(nodes[i, 1]), int(np.argmax(nodes[i, 2:] - nodes[i-1, 2:])))
                         for i in range(1, m + 1)]
            assert recovered == records
            count += 1
    missing_signal_cost = [.5 * p + .5 * (1 - p) for p in (0, .25, .5, .75, 1)]
    assert missing_signal_cost == [.5] * 5
    return {
        "transcripts_reconstructed": count,
        "initial_signal_distinguished_at_depth_one": True,
        "same_timestamp_order_distinguished_at_depth_two": True,
        "omitted_initial_bit_infimum": .5,
        "observed_initial_bit_infimum": 0.,
    }


def score_checks():
    rng = np.random.default_rng(52572910)
    checks = 0
    for mask_tuple in product((False, True), repeat=3):
        mask = np.array(mask_tuple)
        if not mask.any():
            continue
        for alpha in (.001, .01, .2):
            for b in (.001, .05, .3):
                for _ in range(20):
                    target = np.zeros(3)
                    target[rng.choice(np.flatnonzero(mask))] = 1
                    floor = (1 - alpha) * target + alpha * mask / mask.sum()
                    scores = np.zeros(3)
                    scores[mask] = np.log(floor[mask])
                    recovered = mask_softmax(scores, mask)
                    assert np.max(np.abs(recovered - floor)) < 1e-13
                    perturbed = mask_softmax(scores + rng.uniform(-b, b, 3), mask)
                    assert np.abs(perturbed - floor).sum() / 2 <= b + 1e-13
                    assert np.abs(perturbed - target).sum() / 2 <= alpha + b + 1e-13
                    assert np.all(perturbed[~mask] == 0)
                    checks += 1
    return {"finite_score_floor_and_mask_cases": checks,
            "inadmissible_probabilities_exactly_zero": True,
            "no_logarithm_of_inadmissible_zero_probabilities": True}


def admissible_mask(i):
    if i == DONE:
        return np.array([True, False, False])  # 'post' is a no-op after completion.
    return np.array([intervention(i, a)[0] is not None for a in ACTIONS])


def depth_one_state_feature(history):
    """A based, observable state-indicator channel ends at the current state."""
    nodes = np.zeros((len(history) + 1, 1 + len(STATES)))
    for j, state in enumerate(history, 1):
        nodes[j, 0] = j / len(history)
        nodes[j, 1 + state] = 1
    first, _ = signature_two(nodes)
    return first[1:]


def feedback_policy(optimal_actions, logit_size):
    coefficients = np.zeros((N, len(ACTIONS), len(STATES)))
    for k in range(N):
        for i in range(len(STATES)):
            coefficients[k, optimal_actions[k, i], i] = logit_size
    policy = np.zeros((N, len(STATES), len(ACTIONS)))
    for k in range(N):
        for i in range(len(STATES)):
            feature = depth_one_state_feature([INITIAL, i])
            assert np.array_equal(feature, np.eye(len(STATES))[i])
            scores = coefficients[k] @ feature
            policy[k, i] = mask_softmax(scores, admissible_mask(i))
    return policy


def deterministic_policy(actions):
    policy = np.zeros((N, len(STATES), len(ACTIONS)))
    for k in range(N):
        for i in range(len(STATES)):
            policy[k, i, actions[k, i]] = 1
    return policy


def evaluate(policy, rate):
    """Backward expectation for randomized actions, separate from the optimizer."""
    p = transition(rate)
    value = np.zeros((N + 1, len(STATES)))
    value[N, :DONE] = LOSS_WIDTH
    for k in range(N - 1, -1, -1):
        for i in range(len(STATES)):
            assert abs(policy[k, i].sum() - 1) < 1e-13
            for a, probability in enumerate(policy[k, i]):
                if probability:
                    j, charge = intervention(i, ACTIONS[a])
                    assert j is not None
                    value[k, i] += probability * (charge + p[j] @ value[k + 1])
    return value


def policy_transition(policy_stage, rate):
    p = transition(rate)
    matrix = np.zeros_like(p)
    for i in range(len(STATES)):
        for a, probability in enumerate(policy_stage[i]):
            if probability:
                j, _ = intervention(i, ACTIONS[a])
                matrix[i] += probability * p[j]
    return matrix


def hybrid_checks(original, replacement, rate):
    """Use the actual laws under earlier replacements for each decision mismatch."""
    distribution = np.eye(len(STATES))
    total_mismatch = np.zeros(len(STATES))
    hybrid = original.copy()
    previous = evaluate(hybrid, rate)[0]
    count = 0
    for k in range(N):
        tv = np.abs(original[k] - replacement[k]).sum(axis=1) / 2
        expected_mismatch = distribution @ tv
        total_mismatch += expected_mismatch
        hybrid[k] = replacement[k]
        following = evaluate(hybrid, rate)[0]
        assert np.max(np.abs(following - previous) - LOSS_WIDTH * expected_mismatch) < 2e-12
        previous = following
        distribution = distribution @ policy_transition(replacement[k], rate)
        count += len(STATES)
    overall = np.abs(evaluate(replacement, rate)[0] - evaluate(original, rate)[0])
    assert np.max(overall - LOSS_WIDTH * total_mismatch) < 2e-12
    return count


def feedback_experiment():
    true_rate = 1.2
    true_value, _ = solve(true_rate)
    rows = []
    hybrid_count = 0
    stage_state_count = 0
    for fitted_rate in (1.2, 1.4):
        model_value, optimal = solve(fitted_rate)
        reference = deterministic_policy(optimal)
        assert np.max(np.abs(evaluate(reference, fitted_rate) - model_value)) < 1e-13
        for r in (0., 1., 2., 4., 8., 12., 20.):
            policy = feedback_policy(optimal, r)
            model_softmax = evaluate(policy, fitted_rate)
            true_softmax = evaluate(policy, true_rate)
            p_error = (len(ACTIONS) - 1) / (math.exp(r) + len(ACTIONS) - 1)
            for k in range(N):
                tau_k = min(LOSS_WIDTH, (N - k) * LOSS_WIDTH * p_error)
                regret = model_softmax[k] - model_value[k]
                assert regret.min() > -2e-12 and regret.max() <= tau_k + 2e-12
                stage_state_count += len(STATES)
            tau_bound = min(LOSS_WIDTH, N * LOSS_WIDTH * p_error)
            model_error = 2 * LOSS_WIDTH * (-math.expm1(-abs(fitted_rate - true_rate)))
            total_bound = min(LOSS_WIDTH, model_error + tau_bound)
            assert np.max(true_softmax[0] - true_value[0]) <= total_bound + 2e-12
            hybrid_count += hybrid_checks(reference, policy, fitted_rate)
            rows.append({
                "fitted_sell_rate": fitted_rate,
                "signature_depth": 1,
                "logit_size": r,
                "model_optimal_value": float(model_value[0, INITIAL]),
                "model_policy_value": float(model_softmax[0, INITIAL]),
                "model_global_gap_at_initial_state": float(model_softmax[0, INITIAL] - model_value[0, INITIAL]),
                "tau_bound": tau_bound,
                "true_policy_value": float(true_softmax[0, INITIAL]),
                "true_regret": float(true_softmax[0, INITIAL] - true_value[0, INITIAL]),
                "combined_regret_bound": total_bound,
            })
    return {"rows": rows, "stage_state_optimality_comparisons": stage_state_count,
            "hybrid_stage_state_comparisons": hybrid_count,
            "scope": "Fully observed 19-state book; a Bellman oracle constructs depth-one logits. No generic depth rate or local-optimization guarantee is inferred."}


def reference_checks():
    # At the same state, freezing the action label changes no rates or outputs.
    gamma_mode_freezing = 0.
    probabilities = [-math.expm1(-(1 + a)) for a in (0, 1)]
    assert probabilities[1] > probabilities[0]
    # A genuinely autonomous two-state reference and a controlled perturbation.
    # The event takes 0 to absorbing state 1; the terminal loss is the state.
    stages, horizon, base = 3, 1., .7
    checks = 0
    max_ratio = 0.
    for theta in (0., .1, .5, 1.):
        bound = -math.expm1(-theta * horizon)
        for actions_flat in product((0, 1), repeat=2 * stages):
            actions = np.array(actions_flat).reshape(stages, 2)
            values = []
            for perturbation in (0., theta):
                value = np.array([0., 1.])
                for k in range(stages - 1, -1, -1):
                    prob = -math.expm1(-(base + perturbation * actions[k, 0]) * horizon / stages)
                    value = np.array([(1 - prob) * value[0] + prob * value[1], value[1]])
                values.append(value[0])
            defect = abs(values[1] - values[0])
            assert defect <= bound + 1e-13
            if bound:
                max_ratio = max(max_ratio, defect / bound)
            checks += 1
    nonconvexity = []
    for ell in (-2., 2.):
        p = 1 / (1 + math.exp(-ell))
        nonconvexity.append(p * (1 - p) * (1 - 2 * p))
    assert nonconvexity[0] > 0 > nonconvexity[1]
    return {
        "frozen_mode_gamma": gamma_mode_freezing,
        "report_probabilities_after_actions_zero_one": probabilities,
        "changed_reset_intervention_kappa": 1.,
        "autonomous_reference_controlled_policy_comparisons": checks,
        "maximum_endpoint_defect_to_path_bound_ratio": max_ratio,
        "one_decision_softmax_second_derivatives_at_minus_two_plus_two": nonconvexity,
    }


def write_outputs(result):
    (ROOT / "results/feedback_certificate.json").write_text(json.dumps(result, indent=2) + "\n")
    audit = {key: value for key, value in result.items() if key != "feedback"}
    audit["feedback"] = {
        "policies_evaluated": len(result["feedback"]["rows"]),
        "stage_state_optimality_comparisons": result["feedback"]["stage_state_optimality_comparisons"],
        "hybrid_stage_state_comparisons": result["feedback"]["hybrid_stage_state_comparisons"],
        "scope": result["feedback"]["scope"],
        "full_results": "results/feedback_certificate.json",
    }
    (ROOT / "audit/feedback_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    indexed = {(row["fitted_sell_rate"], row["logit_size"]): row for row in result["feedback"]["rows"]}
    def tex_number(x):
        if x and abs(x) < 1e-5:
            exponent = math.floor(math.log10(abs(x)))
            return f"${x / 10**exponent:.2f}\\times10^{{{exponent}}}$"
        return f"{x:.6f}"
    rows = []
    for r in (0., 2., 4., 8., 12., 20.):
        true, fitted = indexed[(1.2, r)], indexed[(1.4, r)]
        numbers = [tex_number(x) for x in (true['true_regret'], true['tau_bound'],
                                         fitted['true_regret'], fitted['combined_regret_bound'])]
        rows.append(f"{r:.0f} & " + " & ".join(numbers) + " \\\\")
    (ROOT / "tables/feedback_rows.tex").write_text("\n".join(rows) + "\n")


def main():
    result = {"encoding": encoding_checks(), "scores": score_checks(),
              "feedback": feedback_experiment(), "reference": reference_checks(),
              "status": "All feedback, encoding, mask, hybrid and reference-model checks passed."}
    write_outputs(result)
    print(result["status"])
    print(json.dumps({"encoding": result["encoding"], "scores": result["scores"],
                      "hybrid_comparisons": result["feedback"]["hybrid_stage_state_comparisons"],
                      "stage_state_comparisons": result["feedback"]["stage_state_optimality_comparisons"],
                      "reference": result["reference"]}, indent=2))


if __name__ == "__main__":
    main()
