/-
  VenueConvex.lean — Theorem 27.1, convex venue aggregation.

  Each venue `v` has a cost `Cv : ℝ → ℝ`, convex on its capacity interval.
  The aggregate value of executing `q` shares is the infimum of `∑ᵥ Cv (xᵥ)`
  over feasible splits `0 ≤ xᵥ ≤ capᵥ`, `∑ᵥ xᵥ = q`.  The claim is that this
  value function is convex in `q`.

  The proof is the ε-argument: take near-optimal splits for `q₁` and `q₂`, note
  that their convex combination is feasible for the convex combination of the
  quantities (the constraint set is convex), and apply convexity venue by venue.

  Formalised on a finite venue set with the value function given as a hypothesis
  (`IsGLB` at feasible quantities). Attainment, strict-convexity uniqueness
  and KKT multiplier conclusions are outside this file.

  Verification commands, axiom audit and scope are documented in COMPILE.md.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Analysis.Convex.Function
import Mathlib.Analysis.Convex.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Push

open Finset

namespace MarketImpact

variable {ι : Type*} [Fintype ι]

/-- A feasible split of `q` shares across venues with capacities `cap`. -/
def Feasible (cap : ι → ℝ) (q : ℝ) (x : ι → ℝ) : Prop :=
  (∀ v, 0 ≤ x v) ∧ (∀ v, x v ≤ cap v) ∧ (∑ v, x v) = q

/-- The set of achievable aggregate costs at quantity `q`. -/
def CostSet (Cv : ι → ℝ → ℝ) (cap : ι → ℝ) (q : ℝ) : Set ℝ :=
  {c | ∃ x, Feasible cap q x ∧ c = ∑ v, Cv v (x v)}

/-- The feasible set is convex in the split, at the convex combination of the
quantities.  This is the only structural fact the theorem needs. -/
lemma feasible_convex {cap : ι → ℝ} {q₁ q₂ : ℝ} {x y : ι → ℝ}
    (hx : Feasible cap q₁ x) (hy : Feasible cap q₂ y)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    Feasible cap (a * q₁ + b * q₂) (fun v => a * x v + b * y v) := by
  obtain ⟨hx0, hxc, hxs⟩ := hx
  obtain ⟨hy0, hyc, hys⟩ := hy
  refine ⟨fun v => add_nonneg (mul_nonneg ha (hx0 v))
    (mul_nonneg hb (hy0 v)), fun v => ?_, ?_⟩
  · have h1 : a * x v ≤ a * cap v := mul_le_mul_of_nonneg_left (hxc v) ha
    have h2 : b * y v ≤ b * cap v := mul_le_mul_of_nonneg_left (hyc v) hb
    calc
      a * x v + b * y v ≤ a * cap v + b * cap v := add_le_add h1 h2
      _ = (a + b) * cap v := (add_mul a b (cap v)).symm
      _ = cap v := by rw [hab, one_mul]
  · rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hxs, hys]

/-- **Theorem 27.1 (aggregation).**  If every venue cost is convex on the
capacity interval, then the aggregate value function is convex, in the sense
that any value achievable at `q₁` and any value achievable at `q₂` combine to
bound a value achievable at the convex combination. -/
theorem aggregate_convex
    {Cv : ι → ℝ → ℝ} {cap : ι → ℝ}
    (hconv : ∀ v, ConvexOn ℝ (Set.Icc (0:ℝ) (cap v)) (Cv v))
    {q₁ q₂ : ℝ} {x y : ι → ℝ}
    (hx : Feasible cap q₁ x) (hy : Feasible cap q₂ y)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    ∃ c ∈ CostSet Cv cap (a * q₁ + b * q₂),
      c ≤ a * (∑ v, Cv v (x v)) + b * (∑ v, Cv v (y v)) := by
  refine ⟨∑ v, Cv v (a * x v + b * y v),
    ⟨fun v => a * x v + b * y v, feasible_convex hx hy ha hb hab, rfl⟩, ?_⟩
  have hbound : ∀ v ∈ (Finset.univ : Finset ι),
      Cv v (a * x v + b * y v) ≤ a * Cv v (x v) + b * Cv v (y v) := by
    intro v _
    exact (hconv v).2 ⟨hx.1 v, hx.2.1 v⟩ ⟨hy.1 v, hy.2.1 v⟩ ha hb hab
  calc ∑ v, Cv v (a * x v + b * y v)
      ≤ ∑ v, (a * Cv v (x v) + b * Cv v (y v)) := Finset.sum_le_sum hbound
    _ = a * (∑ v, Cv v (x v)) + b * (∑ v, Cv v (y v)) := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- **Theorem 27.1 (value-function form).**  If `V q` is a lower bound for the
cost set at `q` and is attained at `q₁` and `q₂` up to `ε`, then `V` satisfies
the convexity inequality up to `ε`.  Letting `ε ↓ 0` gives convexity of `V`;
that last step needs `V` to be the actual infimum, which is stated as the
hypothesis `hV`. -/
theorem value_convex
    {Cv : ι → ℝ → ℝ} {cap : ι → ℝ} {V : ℝ → ℝ}
    (hconv : ∀ v, ConvexOn ℝ (Set.Icc (0:ℝ) (cap v)) (Cv v))
    (hV : ∀ q, (CostSet Cv cap q).Nonempty → IsGLB (CostSet Cv cap q) (V q))
    {q₁ q₂ : ℝ}
    (hne₁ : (CostSet Cv cap q₁).Nonempty) (hne₂ : (CostSet Cv cap q₂).Nonempty)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    V (a * q₁ + b * q₂) ≤ a * V q₁ + b * V q₂ := by
  by_contra hgt
  push_neg at hgt
  set ε : ℝ := V (a * q₁ + b * q₂) - (a * V q₁ + b * V q₂) with hε
  have hεpos : 0 < ε := by simp only [hε]; linarith
  -- pick near-optimal splits at q₁ and q₂
  have hsplit : ∀ (q : ℝ) (δ : ℝ), 0 < δ → (CostSet Cv cap q).Nonempty →
      ∃ x, Feasible cap q x ∧ (∑ v, Cv v (x v)) < V q + δ := by
    intro q δ hδ hne
    by_contra hno
    push_neg at hno
    have : V q + δ ≤ V q := by
      refine (hV q hne).2 ?_
      rintro c ⟨x, hfx, rfl⟩
      exact hno x hfx
    linarith
  have hδpos : (0:ℝ) < ε / 2 := by linarith
  obtain ⟨x, hfx, hxlt⟩ := hsplit q₁ (ε / 2) hδpos hne₁
  obtain ⟨y, hfy, hylt⟩ := hsplit q₂ (ε / 2) hδpos hne₂
  obtain ⟨c, hcmem, hcle⟩ := aggregate_convex hconv hfx hfy ha hb hab
  have hVle : V (a * q₁ + b * q₂) ≤ c := (hV _ ⟨c, hcmem⟩).1 hcmem
  have hcomb : a * (∑ v, Cv v (x v)) + b * (∑ v, Cv v (y v))
      ≤ a * V q₁ + b * V q₂ + ε / 2 := by
    have h1 : a * (∑ v, Cv v (x v)) ≤ a * (V q₁ + ε / 2) :=
      mul_le_mul_of_nonneg_left hxlt.le ha
    have h2 : b * (∑ v, Cv v (y v)) ≤ b * (V q₂ + ε / 2) :=
      mul_le_mul_of_nonneg_left hylt.le hb
    nlinarith
  have : V (a * q₁ + b * q₂) ≤ a * V q₁ + b * V q₂ + ε / 2 := by linarith
  simp only [hε] at this
  linarith

/-- A concrete consistency check: zero venue costs have greatest lower bound
zero at every feasible quantity. No condition is imposed at infeasible quantities. -/
lemma zero_cost_isGLB {cap : ι → ℝ} {q : ℝ}
    (hne : (CostSet (fun _ _ => (0 : ℝ)) cap q).Nonempty) :
    IsGLB (CostSet (fun _ _ => (0 : ℝ)) cap q) 0 := by
  refine ⟨?_, ?_⟩
  · rintro c ⟨x, hx, rfl⟩
    simp
  · intro b hb
    obtain ⟨c, x, hx, rfl⟩ := hne
    have h := hb ⟨x, hx, rfl⟩
    simpa using h

/-- Nonnegative capacities admit the zero split at quantity zero. -/
lemma zero_cost_nonempty {cap : ι → ℝ} (hcap : ∀ v, 0 ≤ cap v) :
    (CostSet (fun _ _ => (0 : ℝ)) cap 0).Nonempty := by
  refine ⟨0, (fun _ => 0), ?_, ?_⟩
  · exact ⟨fun _ => le_rfl, hcap, by simp⟩
  · simp

end MarketImpact
