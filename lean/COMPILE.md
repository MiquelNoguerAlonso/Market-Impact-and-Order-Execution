# Verified build

The final v12 companion was compiled from a clean project build on September
14, 2026. `lake build` and the axiom audit both passed with zero warnings and
zero errors. The active Lean sources contain no admitted proofs, user-defined
axioms, `native_decide`, or unsafe declarations.

- Lean: **4.19.0**, release commit `6caaee842e94`.
- Mathlib: **v4.19.0**, commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`.
- Exact dependencies: `lean-toolchain`, `lakefile.lean`, `lake-manifest.json`.
- Evidence: `../audit/lean_build.log`, `../audit/lean_axioms.log`, and
  `../audit/lean_verification.json`, including hashes of the checked sources.

Each of the six audited declarations reports exactly `propext`,
`Classical.choice`, and `Quot.sound`. There is no `sorryAx` or added axiom in
those dependencies. This is ordinary Lean kernel verification with its
standard logical axioms, not an axiom-free claim.

## Reproduce

Install Lean through the official instructions at
https://leanprover-community.github.io/install/project.html, then run from
this directory. The included toolchain and dependency manifest fix the
versions; there is no need to run `lake update`.

```sh
lake exe cache get \
  Mathlib.Algebra.BigOperators.Group.Finset.Basic \
  Mathlib.Algebra.BigOperators.Intervals \
  Mathlib.Analysis.Convex.Basic \
  Mathlib.Analysis.Convex.Function \
  Mathlib.Analysis.SpecificLimits.Basic \
  Mathlib.Data.Matrix.Basic \
  Mathlib.Data.Real.Basic \
  Mathlib.LinearAlgebra.Matrix.Trace \
  Mathlib.Order.Filter.Basic \
  Mathlib.Tactic.Linarith \
  Mathlib.Tactic.Positivity \
  Mathlib.Tactic.Push \
  Mathlib.Tactic.Ring \
  Mathlib.Topology.Algebra.Order.LiminfLimsup
lake build
lake env lean MarketImpactCore/Audit.lean
```

For an automated clean build, source-admission scan, version check and axiom
check, run from the source archive's root:

```sh
python3 scripts/verify_lean.py --clean
```

That command removes only this project's generated `lean/.lake/build`, keeps
the downloaded dependency cache, and rewrites the three audit records above.
The delivering container required a self-executable path accommodation; its
source and exact scope are documented in `../audit/environment/README.md`.
Normal Lean installations do not require it. Downloaded dependencies and
compiled build products are excluded from the archive.

## Scope

| File | Checked statement | Assumed or outside the formalization |
| --- | --- | --- |
| `MarketImpactCore/LoopCost.lean` | Theorem 13.1 on a finite grid with trapezoidal cost, plus Corollary 13.2 in one asset | Continuum passage, the cost convention, and the self-adjointness converse |
| `MarketImpactCore/QuarticSign.lean` | Corollary 12.8's scalar ray sign argument | The scalar expansion and order-six remainder bound are supplied; Banach-space analyticity is not formalized |
| `MarketImpactCore/VenueConvex.lean` | The convexity conclusion of Theorem 27.1 on feasible quantities | Finite greatest lower bounds and endpoint feasibility are hypotheses; attainment, uniqueness and KKT conclusions are not formalized |
| `MarketImpactCore/Audit.lean` | Axiom dependencies of four principal declarations and two consistency lemmas | No claim that the whole monograph is verified |

The venue theorem requires convexity only on each venue's capacity interval.
Its greatest-lower-bound hypothesis applies only where the feasible cost set
is nonempty. The zero-cost lemmas verify that its value hypothesis is
satisfiable and that nonnegative capacities admit the zero split.

Appendix E's `\leanverifiedtrue` switch is enabled after this successful build
and audit. The general execution certificates and the numerical experiment
code remain outside the Lean formalization.
