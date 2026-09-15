"""Independent signature/ODE and Gaussian-information checks for the paper."""

import argparse

from collections import defaultdict

from functools import lru_cache

import json

import math

from pathlib import Path

import numpy as np

from scipy.integrate import quad, solve_ivp

from scipy.special import ndtr

@lru_cache(None)
def shuffle_words(a, b):
    """Return shuffles with multiplicities, including repeated letters."""
    if not a:
        return ((b, 1),)
    if not b:
        return ((a, 1),)
    out = defaultdict(int)
    for word, count in shuffle_words(a[1:], b):
        out[(a[0],) + word] += count
    for word, count in shuffle_words(a, b[1:]):
        out[(b[0],) + word] += count
    return tuple(out.items())

def shuffle(a, b):
    out = defaultdict(float)
    for wa, ca in a.items():
        for wb, cb in b.items():
            for word, count in shuffle_words(wa, wb):
                out[word] += ca * cb * count
    return dict(out)

def append_letter(poly, letter, scale=1.0):
    return {word + (letter,): scale * coefficient
            for word, coefficient in poly.items()}

def evaluate(poly, state, index):
    return sum(c * state[index[word]] for word, c in poly.items())

def signature_reduction():
    # Alphabet: time, (t-U)_+, exp(rho*t), exp(-rho*t).
    rho, eta = 1.3, 0.4
    policy = {(): 0.3, (0,): 0.2, (1,): 1.1,
              (2,): 0.15, (3,): -0.05}
    ep, em = {(): 1.0, (2,): 1.0}, {(): 1.0, (3,): 1.0}
    # The expanded multiplication construction: depth N+3 for I, 2N+4 for cost.
    impact = shuffle(em, append_letter(shuffle(ep, policy), 0))
    original_cost = append_letter(shuffle(impact, policy), 0)
    temporary = append_letter(shuffle(policy, policy), 0)
    # Sharper exact identity: dt*exp(rho*t)=d exp(rho*t)/rho,
    # and dt*exp(-rho*t)=-d exp(-rho*t)/rho.
    short_cost = append_letter(
        shuffle(append_letter(policy, 2), policy), 3, -1.0 / rho**2
    )
    quantity = append_letter(policy, 0)
    polys = [impact, original_cost, temporary, short_cost, quantity]
    needed = {()}
    for poly in polys:
        for word in poly:
            needed.update(word[:k] for k in range(len(word) + 1))
    words = sorted(needed, key=lambda word: (len(word), word))
    index = {word: i for i, word in enumerate(words)}
    parent = np.array([index[word[:-1]] for word in words[1:]])
    letter = np.array([word[-1] for word in words[1:]])
    n = len(words)
    checks = []
    for u in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        state = np.zeros(n + 4)
        state[0] = 1.0
        cuts = sorted(set([0.0, u, 1.0]))
        for start, end in zip(cuts[:-1], cuts[1:]):
            active = float(start >= u)

            def rhs(t, s):
                derivatives = np.array([1.0, active,
                    rho * np.exp(rho * t), -rho * np.exp(-rho * t)])
                out = np.zeros_like(s)
                out[1:n] = s[parent] * derivatives[letter]
                v = (0.3 + 0.2 * t + 1.1 * max(t-u, 0.0)
                     + 0.15 * np.expm1(rho*t)
                     - 0.05 * np.expm1(-rho*t))
                out[n] = -rho * s[n] + v
                out[n+1] = s[n] * v
                out[n+2] = v * v
                out[n+3] = v
                return out

            solution = solve_ivp(rhs, (start, end), state, method='DOP853',
                                 rtol=2e-11, atol=2e-13)
            assert solution.success, solution.message
            state = solution.y[:, -1]
        direct_cost = state[n+1] + eta * state[n+2]
        expanded_cost = evaluate(original_cost, state, index) + eta * evaluate(temporary, state, index)
        reduced_cost = evaluate(short_cost, state, index) + eta * evaluate(temporary, state, index)
        impact_error = abs(state[n] - evaluate(impact, state, index))
        assert abs(direct_cost - expanded_cost) < 2e-9
        assert abs(direct_cost - reduced_cost) < 2e-9
        assert impact_error < 2e-9
        checks.append({'u': u, 'direct_ode_cost': float(direct_cost),
                       'expanded_shuffle_cost': expanded_cost,
                       'shorter_shuffle_cost': reduced_cost,
                       'impact_state_error': impact_error})
    return {'policy_depth': 1, 'rho': rho, 'eta': eta,
            'expanded_maximum_word_depth': max(map(len, original_cost)),
            'exact_maximum_word_depth': max(map(len, short_cost)),
            'signature_coordinates_integrated': n, 'checks': checks,
            'conclusion': 'Both constructions agree with an independent ODE. Exponential differentials give the exact sufficient depth 2N+2.'}

def gaussian_information():
    rows = []
    normalizer = math.sqrt(2.0*math.pi)
    for epsilon in (0.2, 0.1, 0.05, 0.02):
        def integrand(z):
            # KL(N(0,1) || .5*N(0,1)+.5*N(epsilon,1)),
            # equal to the binary mutual information by symmetry.
            log_ratio = math.log(2.0) - np.logaddexp(0.0, epsilon*z-epsilon**2/2)
            return np.exp(-z*z/2)/normalizer * log_ratio
        information = quad(integrand, -12.0, 12.0, epsabs=1e-13)[0]
        ratio = information/epsilon**2
        saving = ndtr(epsilon/2)-0.5  # Delta=1
        assert abs(ratio-1/8) < 0.001
        rows.append({'epsilon': epsilon, 'mutual_information': information,
                     'I_over_epsilon_squared': ratio, 'saving_Delta_one': float(saving)})
    return {'rows': rows, 'limiting_I_over_epsilon_squared': 0.125,
            'finding': 'Supports the new Gaussian I=epsilon^2/8+o(epsilon^2) expansion.'}
