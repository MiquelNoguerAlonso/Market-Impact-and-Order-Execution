"""A tagged FIFO book, controlled CTMC, and primitive-rate regret certificate.

All numerical inputs are synthetic. Prices are 100/101, with a stipulated
terminal facility at 102. The parent buys exactly one share. Ordered queues,
not aggregate depth alone, determine private fills. Run this script to
regenerate its JSON results, LaTeX inputs, and figure; --check verifies the
construction without rewriting the generated illustration.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import argparse
import json
import math

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
BID, ASK, TERMINAL = 100, 101, 102
EXT, OWN = "E", "O"
DEPTH = 2
ACTIONS = ("post", "cross", "wait")  # deterministic tie breaking
MARKS = ("bid", "ask", "sell", "buy", "cancel0", "cancel1")
N, H = 4, 0.25
SEED = 4304211


@dataclass(frozen=True)
class Order:
    side: str
    price: int
    owner: str
    quantity: int = 1


def match_order(book, incoming, ioc=False):
    """One arrival into an uncrossed book; best price then original list order.

    Execute at resting prices. The original list order breaks same-price
    ties. IOC residuals expire; other residuals join the end of their queue.
    Return (book, trade records, expired quantity). No hidden replenishment.
    """
    assert incoming.side in ("B", "S") and incoming.quantity > 0
    orders = list(book)
    left = incoming.quantity
    trades = []
    while left:
        eligible = [(i, o) for i, o in enumerate(orders)
                    if o.side != incoming.side and
                    (o.price <= incoming.price if incoming.side == "B"
                     else o.price >= incoming.price)]
        if not eligible:
            break
        idx, resting = min(eligible, key=lambda z:
                           ((z[1].price if incoming.side == "B" else -z[1].price), z[0]))
        volume = min(left, resting.quantity)
        buy_owner = incoming.owner if incoming.side == "B" else resting.owner
        sell_owner = resting.owner if incoming.side == "B" else incoming.owner
        trades.append((volume, resting.price, buy_owner, sell_owner))
        left -= volume
        if volume == resting.quantity:
            orders.pop(idx)
        else:
            orders[idx] = Order(resting.side, resting.price, resting.owner,
                                resting.quantity - volume)
    expired = left if ioc else 0
    if left and not ioc:
        orders.append(Order(incoming.side, incoming.price, incoming.owner, left))
    return tuple(orders), tuple(trades), expired


# An active state is (bid queue, ask occupancy). Completion is one absorbing
# state: market dynamics after completion have no further payoff in this toy.
QUEUES = [tuple([EXT] * n) for n in range(DEPTH + 1)]
QUEUES += [tuple([EXT] * p + [OWN] + [EXT] * (n - p))
           for n in range(DEPTH + 1) for p in range(n + 1)]
STATES = [(b, ask) for b in QUEUES for ask in (0, 1)] + [None]
INDEX = {s: i for i, s in enumerate(STATES)}
DONE = INDEX[None]
INITIAL = INDEX[((EXT, EXT), 1)]


def as_book(state):
    bids, ask = state
    return tuple(Order("B", BID, owner) for owner in bids) + \
        ((Order("S", ASK, EXT),) if ask else ())


def from_book(book, trades):
    own_fill = sum(q for q, _, buyer, _ in trades if buyer == OWN)
    assert own_fill in (0, 1)
    if own_fill:
        return DONE
    bids = tuple(o.owner for o in book if o.side == "B")
    ask = sum(o.quantity for o in book if o.side == "S")
    assert bids.count(EXT) <= DEPTH and bids.count(OWN) <= 1 and ask <= 1
    return INDEX[(bids, ask)]


def event(index, mark):
    """Deterministic book + execution/report pushforward for each raw mark."""
    if index == DONE:
        return DONE, ("stopped", (), 0)
    state = STATES[index]
    bids, ask = state
    book = as_book(state)
    if mark.startswith("cancel"):
        rank = int(mark[-1])
        locations = [i for i, o in enumerate(book) if o.side == "B" and o.owner == EXT]
        if rank >= len(locations):
            return index, ("cancel-miss", (), 0)
        k = locations[rank]
        out = book[:k] + book[k + 1:]
        return from_book(out, ()), ("cancel", (), 0)
    if mark == "bid" and bids.count(EXT) == DEPTH:
        return index, ("capacity-reject", (), 0)
    if mark == "ask" and ask:
        return index, ("capacity-reject", (), 0)
    incoming = {"bid": Order("B", BID, EXT), "ask": Order("S", ASK, EXT),
                "sell": Order("S", BID, EXT), "buy": Order("B", ASK, EXT)}[mark]
    out, trades, expired = match_order(book, incoming, ioc=mark in ("sell", "buy"))
    return from_book(out, trades), (mark, trades, expired)


def intervention(index, action):
    """Exact decision-time messages, including cancellation before crossing."""
    if index == DONE:
        return DONE, 0.0
    bids, ask = STATES[index]
    if action == "post":
        if OWN in bids:
            return index, 0.0  # retain priority
        out, trades, _ = match_order(as_book(STATES[index]), Order("B", BID, OWN))
        return from_book(out, trades), 0.0
    unposted = tuple(x for x in bids if x != OWN)
    if action == "wait":
        return INDEX[(unposted, ask)], 0.0
    if action == "cross":
        if not ask:
            return None, math.inf
        out, trades, _ = match_order(as_book((unposted, ask)), Order("B", ASK, OWN), ioc=True)
        assert sum(q for q, _, buyer, _ in trades if buyer == OWN) == 1
        return from_book(out, trades), float(ASK - BID)
    raise ValueError(action)


def rates(sell_rate):
    return np.array([0.4, 0.6, sell_rate, 0.5, 0.15, 0.15])


def generator(sell_rate):
    q = np.zeros((len(STATES), len(STATES)))
    for i in range(DONE):
        for mark, rate in zip(MARKS, rates(sell_rate)):
            j, _ = event(i, mark)
            q[i, j] += rate
            q[i, i] -= rate
    assert np.max(np.abs(q.sum(axis=1))) < 1e-13
    assert np.min(q - np.diag(np.diag(q))) >= 0
    return q


def transition(sell_rate, horizon=H):
    return expm(horizon * generator(sell_rate))


def solve(sell_rate, policy=None):
    """Backward induction, or exact evaluation of a frozen feedback policy."""
    p = transition(sell_rate)
    value = np.zeros((N + 1, len(STATES)))
    value[N, :DONE] = TERMINAL - BID
    selected = np.zeros((N, len(STATES)), dtype=int)
    for k in range(N - 1, -1, -1):
        for i in range(len(STATES)):
            options = []
            for action in ACTIONS:
                j, cost = intervention(i, action)
                options.append(math.inf if j is None else cost + p[j] @ value[k + 1])
            a = int(np.argmin(options)) if policy is None else int(policy[k, i])
            assert math.isfinite(options[a])
            selected[k, i] = a
            value[k, i] = options[a]
    return value, selected


def gillespie(policy, sell_rate, paths=40000, seed=SEED):
    """Independent event-time simulation, using raw marks, never expm."""
    rng = np.random.default_rng(seed)
    r = rates(sell_rate)
    cumulative = np.cumsum(r) / r.sum()
    costs = np.zeros(paths)
    outcomes = np.zeros(3, dtype=int)  # passive, aggressive, terminal
    for n in range(paths):
        state = INITIAL
        for k in range(N):
            state, charge = intervention(state, ACTIONS[int(policy[k, state])])
            costs[n] += charge
            if charge:
                outcomes[1] += 1
            if state == DONE:
                break
            elapsed = 0.0
            while True:
                elapsed += rng.exponential(1.0 / r.sum())
                if elapsed >= H:
                    break
                mark = MARKS[int(np.searchsorted(cumulative, rng.random()))]
                state, _ = event(state, mark)
                if state == DONE:
                    outcomes[0] += 1
                    break
            if state == DONE:
                break
        if state != DONE:
            costs[n] += TERMINAL - BID
            outcomes[2] += 1
    assert outcomes.sum() == paths and costs.min() >= 0 and costs.max() <= 2
    return {"paths": paths, "seed": seed, "mean": float(costs.mean()),
            "standard_error": float(costs.std(ddof=1) / math.sqrt(paths)),
            "passive_aggressive_terminal_counts": outcomes.tolist()}


def construction_checks():
    """Check price priority, FIFO, conservation and an aggregation obstruction."""
    checks = 0
    # Exhaustive small two-price books on each side, variable incoming sizes.
    for bids in product((0, 1, 2), repeat=2):
        for asks in product((0, 1, 2), repeat=2):
            book = tuple(Order("B", 98 + p, EXT, q) for p, q in enumerate(bids) if q) + \
                tuple(Order("S", 101 + p, EXT, q) for p, q in enumerate(asks) if q)
            for side, price, size, ioc in product(("B", "S"), (97, 99, 101, 103),
                                                  (1, 2, 4), (False, True)):
                incoming = Order(side, price, OWN, size)
                out, trades, expired = match_order(book, incoming, ioc)
                volume = sum(t[0] for t in trades)
                assert sum(o.quantity for o in out) == sum(o.quantity for o in book) + size - 2 * volume - expired
                out_bid = max([o.price for o in out if o.side == "B"], default=-math.inf)
                out_ask = min([o.price for o in out if o.side == "S"], default=math.inf)
                assert out_bid < out_ask
                execution_prices = [t[1] for t in trades]
                assert execution_prices == sorted(execution_prices, reverse=side == "S")
                checks += 1
    # Same anonymous depths, different tagged order position, different fill.
    front = INDEX[((OWN, EXT), 1)]
    back = INDEX[((EXT, OWN), 1)]
    assert event(front, "sell")[0] == DONE
    assert event(back, "sell")[0] == INDEX[((OWN,), 1)]
    # Equal price: the first order must be fully consumed before the second.
    book = (Order("B", 100, EXT, 2), Order("B", 100, OWN, 2))
    out, trades, expired = match_order(book, Order("S", 99, EXT, 3), ioc=True)
    assert trades == ((2, 100, EXT, EXT), (1, 100, OWN, EXT)) and expired == 0
    assert out == (Order("B", 100, OWN, 1),)
    for i in range(len(STATES)):
        for mark in MARKS:
            j, _ = event(i, mark)
            assert 0 <= j < len(STATES)
        for action in ACTIONS:
            j, cost = intervention(i, action)
            assert j is None or (0 <= j < len(STATES) and cost in (0, 1))
    return {"exhaustive_matching_cases": checks, "states": len(STATES),
            "raw_mark_types": len(MARKS), "non_lumpability_counterexample": "passed",
            "fifo_variable_size_check": "passed"}


def uniformization_checks(sell_rate=1.2, approximate_rate=1.4):
    q, qhat = generator(sell_rate), generator(approximate_rate)
    clock = float(max(rates(sell_rate).sum(), rates(approximate_rate).sum()))
    raw = np.append(rates(sell_rate) / clock, 1 - rates(sell_rate).sum() / clock)
    raw_hat = np.append(rates(approximate_rate) / clock, 1 - rates(approximate_rate).sum() / clock)
    eta = float(np.abs(raw - raw_hat).sum() / 2)
    assert abs(clock * eta - abs(sell_rate - approximate_rate)) < 1e-13
    pstep = np.eye(len(STATES)) + q / clock
    cutoff = int(poisson.isf(1e-14, clock * H))
    power = np.eye(len(STATES))
    uniformized = np.zeros_like(q)
    for k in range(cutoff + 1):
        uniformized += poisson.pmf(k, clock * H) * power
        power = power @ pstep
    tail = float(poisson.sf(cutoff, clock * H))
    # Put the omitted Poisson tail on the identity: a stochastic kernel with
    # an explicit TV error at most the omitted probability.
    uniformized += tail * np.eye(len(STATES))
    exact = expm(H * q)
    error = float(np.max(np.sum(np.abs(uniformized - exact), axis=1) / 2))
    tv = float(np.max(np.sum(np.abs(exact - expm(H * qhat)), axis=1) / 2))
    bound = -math.expm1(-clock * eta * H)
    assert error <= tail + 1e-13 and tv <= bound + 1e-13
    assert np.max(np.abs(uniformized.sum(axis=1) - 1)) < 1e-13
    # State AND execution/report marks obey contraction; test every row.
    joint_max = 0.0
    for i in range(DONE):
        outcomes = {}
        for mu, sign in ((raw, 1), (raw_hat, -1)):
            for k, mark in enumerate(MARKS):
                key = event(i, mark)
                outcomes[key] = outcomes.get(key, 0.0) + sign * mu[k]
            key = (i, ("dummy", (), 0))
            outcomes[key] = outcomes.get(key, 0.0) + sign * mu[-1]
        joint_tv = sum(abs(z) for z in outcomes.values()) / 2
        joint_max = max(joint_max, joint_tv)
        assert joint_tv <= eta + 1e-13
    return {"clock_rate": clock, "primitive_tv": eta,
            "mismatch_rate": clock * eta, "poisson_cutoff": cutoff,
            "poisson_tail": tail, "uniformization_max_row_tv_error": error,
            "endpoint_max_row_tv": tv, "endpoint_tv_bound": bound,
            "joint_mark_max_tv": joint_max}


def linear_response_checks():
    """Independent matrix checks of the response derivative and its remainder."""
    l0 = np.array([[-1., 1., 0.], [1., -3., 2.], [0., 2., -2.]])
    b = np.array([[-1., 1., 0.], [-1., 0., 1.], [0., -1., 1.]])
    pi = np.ones(3) / 3
    observable = np.array([-1., 0., 1.])
    block = np.block([[l0, b], [np.zeros_like(l0), l0]])
    integral = expm(block)[:3, 3:]
    derivative = float(pi @ integral @ observable)
    rows = []
    for eps in (0.1, 0.05, 0.025):
        exact = float(pi @ expm(l0 + eps * b) @ observable)
        error = abs(exact - eps * derivative)
        bound = 0.5 * eps**2 * np.linalg.norm(b, np.inf)**2 * np.max(np.abs(observable))
        assert error <= bound + 1e-14
        finite_difference = float(pi @ (expm(l0 + eps * b) - expm(l0 - eps * b)) @ observable / (2 * eps))
        rows.append({"amplitude": eps, "remainder": error, "bound": float(bound),
                     "symmetric_derivative_error": abs(finite_difference - derivative)})
    piecewise = []
    durations, inputs = (0.2, 0.35, 0.45), (0.6, -0.4, 1.1)
    nominal, tangent = np.eye(3), np.zeros((3, 3))
    for h, u in zip(durations, inputs):
        p = expm(h * l0)
        d = expm(h * np.block([[l0, u * b], [np.zeros_like(l0), l0]]))[:3, 3:]
        tangent = tangent @ p + nominal @ d
        nominal = nominal @ p
    for eps in (0.1, 0.05):
        perturbed = np.eye(3)
        for h, u in zip(durations, inputs):
            perturbed = perturbed @ expm(h * (l0 + eps * u * b))
        remainder = abs(float(pi @ (perturbed - nominal - eps * tangent) @ observable))
        limit = 0.5 * eps**2 * np.linalg.norm(b, np.inf)**2 * sum(h * abs(u) for h, u in zip(durations, inputs))**2
        assert remainder <= limit + 1e-14
        piecewise.append({"amplitude": eps, "remainder": remainder, "bound": float(limit)})
    # Exact two-state exponential propagator for all admissible deterministic
    # constant controls: response solves a scalar first-order ODE.
    kappa, beta, delta = 0.7, 0.4, 0.5
    l2 = np.array([[-kappa, kappa], [kappa, -kappa]])
    b2 = np.array([[-beta, beta], [-beta, beta]])
    largest_error = 0.0
    for amplitude, horizon in product((-0.8, 0.2, 0.9), (0.1, 0.8, 2.0)):
        direct = float(np.array([0.5, 0.5]) @ expm(horizon * (l2 + amplitude * b2)) @ np.array([-delta, delta]))
        formula = amplitude * beta * delta / kappa * (1 - math.exp(-2 * kappa * horizon))
        largest_error = max(largest_error, abs(direct - formula))
        assert abs(direct - formula) < 1e-13
    return {"three_state_derivative": derivative, "duhamel_remainders": rows,
            "piecewise_signed_input_remainders": piecewise,
            "two_state_exact_exponential_max_error": largest_error}


def bridge_checks():
    checks = {"construction": construction_checks(),
              "uniformization": uniformization_checks(),
              "linear_response": linear_response_checks()}
    true_rate = 1.2
    truth, _ = solve(true_rate)
    max_fixed_error, max_regret, cases = 0.0, 0.0, 0
    # Uniform in all structural states, not just the displayed initial state.
    for estimate in (0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.4):
        approximate, policy = solve(estimate)
        realized, _ = solve(true_rate, policy)
        for k in range(N):
            fixed_error = np.abs(realized[k] - approximate[k])
            regret = realized[k] - truth[k]
            probability = -math.expm1(-abs(estimate - true_rate) * (N - k) * H)
            assert fixed_error.max() <= 2 * probability + 1e-12
            assert regret.min() >= -1e-12
            assert regret.max() <= min(2.0, 4 * probability) + 1e-12
            max_fixed_error = max(max_fixed_error, float(fixed_error.max()))
            max_regret = max(max_regret, float(regret.max()))
            cases += len(STATES)
    checks["policy_bounds"] = {"state_stage_model_cases": cases,
                               "largest_fixed_policy_error": max_fixed_error,
                               "largest_regret": max_regret,
                               "all_bounds_passed": True}
    return checks


def illustration(checks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true_rate, learned_rate = 1.2, 1.4
    truth, true_policy = solve(true_rate)
    best = float(truth[0, INITIAL])
    curve = []
    true_endpoint = transition(true_rate)
    for estimate in np.linspace(0.6, 2.4, 91):
        approximate, policy = solve(float(estimate))
        realized, _ = solve(true_rate, policy)
        predicted_value = float(approximate[0, INITIAL])
        realized_value = float(realized[0, INITIAL])
        error = abs(float(estimate) - true_rate)
        endpoint_tv = float(np.max(np.sum(np.abs(true_endpoint - transition(float(estimate))), axis=1) / 2))
        endpoint_bound = -math.expm1(-error * H)
        assert endpoint_tv <= endpoint_bound + 1e-12
        curve.append({"estimate": float(estimate), "model_value": predicted_value,
                      "true_value": realized_value, "initial_action": ACTIONS[policy[0, INITIAL]],
                      "regret": max(0.0, realized_value - best),
                      "regret_bound": min(2.0, -4 * math.expm1(-error * N * H)),
                      "endpoint_tv": endpoint_tv, "endpoint_bound": endpoint_bound})
    table = []
    for estimate in (1.0, 1.2, 1.4, 1.6, 2.0, 2.4):
        row = min(curve, key=lambda r: abs(r["estimate"] - estimate)).copy()
        assert abs(row["estimate"] - estimate) < 1e-12
        table.append(row)
    learned_values, learned_policy = solve(learned_rate)
    true_learned_values, _ = solve(true_rate, learned_policy)
    model_mc = gillespie(learned_policy, learned_rate, seed=SEED)
    true_mc = gillespie(learned_policy, true_rate, seed=SEED + 1)
    for mc, target in ((model_mc, learned_values[0, INITIAL]),
                       (true_mc, true_learned_values[0, INITIAL])):
        mc["exact_value"] = float(target)
        mc["standardized_difference"] = float((mc["mean"] - target) / mc["standard_error"])
        # A diagnostic, not a deterministic correctness theorem for an MC draw.
        assert abs(mc["standardized_difference"]) < 5
    result = {"description": "Synthetic tagged FIFO book; one-share parent with guaranteed terminal facility.",
              "parameters": {"true_sell_rate": true_rate, "learned_sell_rate": learned_rate,
                             "bid_arrival_rate": 0.4, "ask_arrival_rate": 0.6,
                             "external_buy_rate": 0.5, "cancellation_rate_per_external_bid": 0.15,
                             "bid_price": BID, "ask_price": ASK, "terminal_price": TERMINAL,
                             "external_bid_capacity": DEPTH, "ask_capacity": 1,
                             "horizon": N * H, "decisions": N, "interval": H,
                             "initial_bid_queue": [EXT, EXT], "initial_ask_depth": 1,
                             "action_tie_breaking": list(ACTIONS),
                             "impact_cost_error": 0.0, "total_loss_range_width": 2.0},
              "checks": checks, "true_optimum": best,
              "true_initial_action": ACTIONS[true_policy[0, INITIAL]],
              "learned_policy": learned_policy.tolist(),
              "state_order": [None if s is None else [list(s[0]), s[1]] for s in STATES],
              "table": table, "curve": curve,
              "independent_event_simulation": {"learned_environment": model_mc,
                                                "true_environment": true_mc}}
    (ROOT / "results" / "orderbook_bridge.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for row in table:
        rows.append(f"{row['estimate']:.1f} & {row['initial_action'].capitalize()} & "
                    f"{row['model_value']:.6f} & {row['true_value']:.6f} & "
                    f"{row['regret']:.6f} & {row['regret_bound']:.6f} " + r"\\")
    (ROOT / "tables" / "orderbook_bridge_rows.tex").write_text("\n".join(rows) + "\n")
    macros = {"BridgeTrueValue": f"{true_learned_values[0, INITIAL]:.6f}",
              "BridgeModelValue": f"{learned_values[0, INITIAL]:.6f}",
              "BridgeRegret": f"{true_learned_values[0, INITIAL] - best:.6f}",
              "BridgeRateBound": f"{-4 * math.expm1(-0.2):.6f}",
              "BridgeTrueMC": f"{true_mc['mean']:.6f}",
              "BridgeTrueSE": f"{true_mc['standard_error']:.6f}",
              "BridgeModelMC": f"{model_mc['mean']:.6f}",
              "BridgeModelSE": f"{model_mc['standard_error']:.6f}",
              "BridgeMCMaxZ": f"{max(abs(true_mc['standardized_difference']), abs(model_mc['standardized_difference'])):.2f}"}
    (ROOT / "tables" / "orderbook_bridge_macros.tex").write_text(
        "% Generated by scripts/orderbook_bridge.py\n" +
        "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items()) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 10, "legend.fontsize": 8.5, "savefig.dpi": 220})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), layout="constrained")
    x = [r["estimate"] for r in curve]
    axes[0].plot(x, [r["model_value"] for r in curve], color="#28618d", label="Value in estimated model")
    axes[0].plot(x, [r["true_value"] for r in curve], color="#b04c31", label="Same policy in true model")
    axes[0].axhline(best, color="#3d7b5a", linestyle=":", label="True optimum")
    axes[0].set_ylabel("Expected cost above the bid")
    axes[0].set_title("Choosing a policy from estimated flow", loc="left", fontsize=10.5)
    axes[1].plot(x, [r["endpoint_bound"] for r in curve], color="#28618d", label=r"Rate bound: $1-e^{-|\hat\lambda-\lambda|h}$")
    axes[1].plot(x, [r["endpoint_tv"] for r in curve], color="#b04c31", label="Exact maximum endpoint TV")
    axes[1].set_ylabel("State transition error over 0.25 seconds")
    axes[1].set_title("A bound derived before solving the policy", loc="left", fontsize=10.5)
    for ax in axes:
        ax.axvline(true_rate, color="#6e7680", linewidth=0.9, linestyle="--")
        ax.set_xlabel("Estimated sell-arrival rate (per second)")
        ax.grid(alpha=0.18)
        ax.legend(loc="best", frameon=False)
    fig.savefig(ROOT / "figures" / "09_orderbook_bridge.png")
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    checks = bridge_checks()
    (ROOT / "audit" / "orderbook_bridge_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    if args.check:
        print(json.dumps(checks, indent=2))
        return
    result = illustration(checks)
    print(json.dumps({"checks": checks, "table": result["table"],
                      "independent_event_simulation": result["independent_event_simulation"]}, indent=2))


if __name__ == "__main__":
    main()
