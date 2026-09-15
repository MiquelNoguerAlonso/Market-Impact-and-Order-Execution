"""Independent finite checks of the schedule-sensitive analytic crossover.

The manuscript supplies the proof and domain assumptions. These checks compare
ODE energy evaluation with exact cell integrals and independently optimize the
transformed objective on finite measure grids. They do not assert L2 attainment.
"""
from pathlib import Path
import json

import numpy as np
from scipy.optimize import minimize

from cross_impact_checks import triangular_exponential_cells, state_ode

ROOT = Path(__file__).resolve().parents[1]


def phi(energy, scale):
    return scale * np.expm1(np.longdouble(0.75) * np.log1p(energy / scale))


def main():
    energy_cases = []
    for rho in (0.4, 1.0, 3.0):
        rates = np.array([1.0, 3.0, -1.0, 1.0])
        cells = triangular_exponential_cells(rho, len(rates), 1.0)
        energy = float(rates @ (cells + cells.T) @ rates)
        ode_energy = 2 * state_ode(rates[:, None], [(rho, np.ones((1, 1)))])['direct_cost']
        assert abs(energy - ode_energy) < 1e-10
        uniform = np.ones(4)
        concentrated = np.array([4.0, 0.0, 0.0, 0.0])
        eu = float(uniform @ (cells + cells.T) @ uniform)
        ec = float(concentrated @ (cells + cells.T) @ concentrated)
        assert ec > eu > 0
        energy_cases.append({'rho': rho, 'energy': energy, 'ode_energy': ode_energy,
                             'uniform_energy': eu, 'concentrated_energy': ec})

    optimization_cases = []
    n = 9
    times = np.linspace(0.0, 1.0, n)
    for rho in (0.4, 1.0, 3.0):
        matrix = np.exp(-rho * np.abs(times[:, None] - times[None, :]))
        r = np.exp(-rho / (n - 1))
        weights = np.full(n, (1 - r) / (n - (n - 2) * r))
        weights[[0, -1]] = 1 / (n - (n - 2) * r)
        assert abs(weights.sum() - 1) < 1e-14
        assert np.ptp(matrix @ weights) < 1e-14
        for mass in (0.1, 1.0, 10.0):
            for scale in (0.3, 3.0):
                normalization = float(phi(np.longdouble(mass * mass), scale))

                def objective(w):
                    energy = np.longdouble(mass * mass * (w @ matrix @ w))
                    return float(phi(energy, scale)) / normalization

                fit = minimize(objective, np.full(n, 1 / n), method='SLSQP',
                               bounds=[(0.0, 1.0)] * n,
                               constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                               options={'ftol': 1e-13, 'maxiter': 1000})
                assert fit.success, fit.message
                error = float(np.max(np.abs(fit.x - weights)))
                assert error < 2e-5
                assert abs(objective(fit.x) - objective(weights)) < 1e-9
                optimization_cases.append({'rho': rho, 'mass': mass, 'scale': scale,
                                           'maximum_weight_error': error})

    regime_cases = []
    for eu in (0.2, 1.0, 4.0):
        for scale in (0.3, 3.0):
            eu, scale = np.longdouble(eu), np.longdouble(scale)
            x = np.longdouble(0.001)
            value = phi(eu * x * x, scale)
            quadratic = np.longdouble(0.75) * eu * x * x
            quartic = -np.longdouble(3) / 32 * eu * eu / scale * x**4
            quartic_ratio = (value - quadratic) / quartic
            assert abs(quartic_ratio - 1) < 1e-5
            large = np.longdouble(1e6)
            asymptotic = scale**np.longdouble(0.25) * eu**np.longdouble(0.75) * large**np.longdouble(1.5)
            large_ratio = phi(eu * large * large, scale) / asymptotic
            assert abs(large_ratio - 1) < 1e-7
            regime_cases.append({'unit_energy': float(eu), 'scale': float(scale),
                                 'quartic_ratio': float(quartic_ratio),
                                 'large_size_ratio': float(large_ratio)})

    report = {'scope': 'Synthetic finite checks; proofs and attainment qualifications are in the manuscript.',
              'energy_ode_comparisons': energy_cases,
              'independent_transformed_optimization_cases': optimization_cases,
              'small_and_large_size_cases': regime_cases,
              'status': 'All analytic crossover checks passed.'}
    (ROOT / 'audit/analytic_crossover_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Crossover checks passed: {len(energy_cases)} energy/ODE cases, "
          f"{len(optimization_cases)} independent optimizations and {len(regime_cases)} regime cases.")


if __name__ == '__main__':
    main()
