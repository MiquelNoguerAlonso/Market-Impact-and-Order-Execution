/-
  QuarticSign.lean — Corollary 12.8, the ray argument.

  Along a fixed zero-mass direction ν, side symmetry kills the odd layers, so
  the cost of the ray ε ↦ C[εν] is a₂ε² + a₄ε⁴ + r(ε) with the remainder
  controlled at order ε⁶ on a neighbourhood of zero.  Nonnegativity of the
  round trip on that ray forces a₂ ≥ 0, and forces a₄ ≥ 0 on the null set of
  the quadratic layer.

  This is the whole content of the "divide by ε², then by ε⁴" step in the
  paper's proof, with the remainder bound made explicit rather than left as
  an o(·).  It is a statement about real functions; nothing about `C` being a
  functional on L² enters.  That is the point — the analytic input is
  Assumption 12.1 and is quoted, not reproved.

  Verification commands, axiom audit and scope are documented in COMPILE.md.
-/

import Mathlib.Order.Filter.Basic
import Mathlib.Topology.Algebra.Order.LiminfLimsup
import Mathlib.Analysis.SpecificLimits.Basic

import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Push

open Filter Topology

namespace MarketImpact

/-- The ray expansion: `f ε = a₂ε² + a₄ε⁴ + r ε` with `|r ε| ≤ C ε⁶` on `(0, δ)`. -/
structure RayData where
  a₂ : ℝ
  a₄ : ℝ
  C : ℝ
  δ : ℝ
  r : ℝ → ℝ
  hδ : 0 < δ
  hC : 0 ≤ C
  hr : ∀ ε, 0 < ε → ε < δ → |r ε| ≤ C * ε ^ 6

namespace RayData

variable (R : RayData)

/-- The cost along the ray. -/
def f (ε : ℝ) : ℝ := R.a₂ * ε ^ 2 + R.a₄ * ε ^ 4 + R.r ε

/-- **The quadratic layer is nonnegative.**  If the round trip costs nothing
negative on the ray, then `a₂ ≥ 0`. -/
theorem quadratic_nonneg (hpos : ∀ ε, 0 < ε → ε < R.δ → 0 ≤ R.f ε) :
    0 ≤ R.a₂ := by
  by_contra hneg
  push_neg at hneg
  -- choose ε small enough that |a₄|ε² + Cε⁴ < -a₂
  obtain ⟨ε, hε0, hεδ, hsmall⟩ :
      ∃ ε : ℝ, 0 < ε ∧ ε < R.δ ∧ |R.a₄| * ε ^ 2 + R.C * ε ^ 4 < -R.a₂ := by
    have hlim : Filter.Tendsto (fun ε : ℝ => |R.a₄| * ε ^ 2 + R.C * ε ^ 4)
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
      have : Filter.Tendsto (fun ε : ℝ => |R.a₄| * ε ^ 2 + R.C * ε ^ 4)
          (nhds 0) (nhds (|R.a₄| * 0 ^ 2 + R.C * 0 ^ 4)) := by
        exact ((continuous_const.mul (continuous_id.pow 2)).add
          (continuous_const.mul (continuous_id.pow 4))).continuousAt
      simpa using this.mono_left nhdsWithin_le_nhds
    have hev := hlim.eventually (eventually_lt_nhds (by linarith : (0:ℝ) < -R.a₂))
    have hev' : ∀ᶠ ε in nhdsWithin (0:ℝ) (Set.Ioi 0), ε < R.δ := by
      exact eventually_nhdsWithin_of_eventually_nhds (eventually_lt_nhds R.hδ)
    obtain ⟨ε, ⟨hsmall, hεδ⟩, hε0⟩ :=
      ((hev.and hev').and self_mem_nhdsWithin).exists
    exact ⟨ε, hε0, hεδ, hsmall⟩
  have h := hpos ε hε0 hεδ
  have hrb : |R.r ε| ≤ R.C * ε ^ 6 := R.hr ε hε0 hεδ
  have hrl : -(R.C * ε ^ 6) ≤ R.r ε := by
    have := abs_le.mp hrb
    linarith [this.1]
  -- divide the inequality by ε² and bound the tail
  have hε2 : 0 < ε ^ 2 := by positivity
  have hchain : 0 ≤ R.a₂ * ε ^ 2 + R.a₄ * ε ^ 4 + R.r ε := h
  have hbound : R.a₄ * ε ^ 4 + R.r ε ≤ |R.a₄| * ε ^ 4 + R.C * ε ^ 6 := by
    have h1 : R.a₄ * ε ^ 4 ≤ |R.a₄| * ε ^ 4 := by
      have : R.a₄ ≤ |R.a₄| := le_abs_self _
      nlinarith [pow_pos hε0 4]
    have h2 : R.r ε ≤ R.C * ε ^ 6 := (abs_le.mp hrb).2
    linarith
  -- so a₂ε² ≥ -(|a₄|ε⁴ + Cε⁶) = -ε²(|a₄|ε² + Cε⁴)
  have : 0 ≤ R.a₂ * ε ^ 2 + (|R.a₄| * ε ^ 4 + R.C * ε ^ 6) := by linarith
  have hfactor : R.a₂ * ε ^ 2 + (|R.a₄| * ε ^ 4 + R.C * ε ^ 6)
      = ε ^ 2 * (R.a₂ + (|R.a₄| * ε ^ 2 + R.C * ε ^ 4)) := by ring
  rw [hfactor] at this
  have hinner : 0 ≤ R.a₂ + (|R.a₄| * ε ^ 2 + R.C * ε ^ 4) :=
    nonneg_of_mul_nonneg_right this hε2
  linarith

/-- **The quartic layer is nonnegative on the null set of the quadratic one.**
If `a₂ = 0` and the round trip costs nothing negative on the ray, `a₄ ≥ 0`. -/
theorem quartic_nonneg_on_null (h₂ : R.a₂ = 0)
    (hpos : ∀ ε, 0 < ε → ε < R.δ → 0 ≤ R.f ε) :
    0 ≤ R.a₄ := by
  by_contra hneg
  push_neg at hneg
  obtain ⟨ε, hε0, hεδ, hsmall⟩ :
      ∃ ε : ℝ, 0 < ε ∧ ε < R.δ ∧ R.C * ε ^ 2 < -R.a₄ := by
    have hlim : Filter.Tendsto (fun ε : ℝ => R.C * ε ^ 2)
        (nhdsWithin 0 (Set.Ioi 0)) (nhds 0) := by
      have : Filter.Tendsto (fun ε : ℝ => R.C * ε ^ 2) (nhds 0) (nhds (R.C * 0 ^ 2)) := by
        exact (continuous_const.mul (continuous_id.pow 2)).continuousAt
      simpa using this.mono_left nhdsWithin_le_nhds
    have hev := hlim.eventually (eventually_lt_nhds (by linarith : (0:ℝ) < -R.a₄))
    have hev' : ∀ᶠ ε in nhdsWithin (0:ℝ) (Set.Ioi 0), ε < R.δ :=
      eventually_nhdsWithin_of_eventually_nhds (eventually_lt_nhds R.hδ)
    obtain ⟨ε, ⟨hsmall, hεδ⟩, hε0⟩ := ((hev.and hev').and self_mem_nhdsWithin).exists
    exact ⟨ε, hε0, hεδ, hsmall⟩
  have h := hpos ε hε0 hεδ
  rw [f, h₂] at h
  have hrb := (abs_le.mp (R.hr ε hε0 hεδ)).2
  have hε4 : 0 < ε ^ 4 := by positivity
  have : 0 ≤ ε ^ 4 * (R.a₄ + R.C * ε ^ 2) := by nlinarith
  have hinner : 0 ≤ R.a₄ + R.C * ε ^ 2 := nonneg_of_mul_nonneg_right this hε4
  linarith

/-- **Corollary 12.8 (ray form).**  Both sign conditions together. -/
theorem ray_sign_conditions (hpos : ∀ ε, 0 < ε → ε < R.δ → 0 ≤ R.f ε) :
    0 ≤ R.a₂ ∧ (R.a₂ = 0 → 0 ≤ R.a₄) :=
  ⟨R.quadratic_nonneg hpos, fun h => R.quartic_nonneg_on_null h hpos⟩

end RayData

end MarketImpact
