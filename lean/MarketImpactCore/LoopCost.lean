/-
  LoopCost.lean — finite-grid form of Theorem 13.1.

  Linear permanent impact `p = p₀ + M q`, trapezoidal (midpoint) cost
  convention, closed inventory loop on a finite grid.  The claim: the benchmark
  term `p₀` and the symmetric part `S` telescope to zero, and the entire
  execution cost of the loop is the trace pairing of the antisymmetric part `A`
  with the discrete Lévy area of the loop:

      cost n p₀ (S + A) q = trace (A * levyArea n q).

  Orientation.  `levyArea j k = ½ ∑ᵢ (qᵢ_j Δᵢ_k − qᵢ_k Δᵢ_j)` matches (13.1)
  index-for-index.  The identity comes out with the TRACE pairing
  `trace (A L) = ∑_{j,k} A_{jk} L_{kj}`.  Since both A and L are antisymmetric,
  `trace (A L) = −∑_{j,k} A_{jk} L_{jk} = −2 ∑_{j<k} A_{jk} L_{jk}`, which is
  exactly the right-hand side of the paper's (13.2).  No orientation change.

  Verification commands, axiom audit and scope are documented in COMPILE.md.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Algebra.BigOperators.Intervals

import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

open Matrix Finset

namespace MarketImpact

noncomputable section

variable {d : ℕ}

abbrev Vec (d : ℕ) := Fin d → ℝ
abbrev Mat (d : ℕ) := Matrix (Fin d) (Fin d) ℝ

/-! ## 1. A double-sum bilinear form and its symmetry lemmas -/

/-- `B M x y = ⟨M x, y⟩`, written as a double sum so every proof below is
elementary algebra plus index permutation. -/
def B (M : Mat d) (x y : Vec d) : ℝ := ∑ j, ∑ k, M j k * x k * y j

lemma B_eq_dot (M : Mat d) (x y : Vec d) : B M x y = (M *ᵥ x) ⬝ᵥ y := by
  simp only [B, Matrix.mulVec, dotProduct, Finset.sum_mul]

lemma B_add_mat (M N : Mat d) (x y : Vec d) : B (M + N) x y = B M x y + B N x y := by
  simp only [B, Matrix.add_apply, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by ring

lemma B_add_left (M : Mat d) (x y z : Vec d) : B M (x + y) z = B M x z + B M y z := by
  simp only [B, Pi.add_apply, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by ring

lemma B_sub_right (M : Mat d) (x y z : Vec d) : B M x (y - z) = B M x y - B M x z := by
  simp only [B, Pi.sub_apply, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by ring

lemma B_smul (c : ℝ) (M : Mat d) (x y : Vec d) : B M (c • x) y = c * B M x y := by
  simp only [B, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by ring

/-- Symmetric matrices give a symmetric form. -/
lemma B_comm_of_symm {S : Mat d} (hS : ∀ j k, S k j = S j k) (x y : Vec d) :
    B S x y = B S y x := by
  simp only [B]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun k _ => by
    rw [hS j k]; ring

/-- Antisymmetric matrices give an antisymmetric form. -/
lemma B_neg_of_antisymm {A : Mat d} (hA : ∀ j k, A k j = -A j k) (x y : Vec d) :
    B A x y = - B A y x := by
  simp only [B]
  rw [Finset.sum_comm, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun k _ => by rw [hA j k]; ring

/-- An antisymmetric form vanishes on the diagonal. -/
lemma B_self_of_antisymm {A : Mat d} (hA : ∀ j k, A k j = -A j k) (x : Vec d) :
    B A x x = 0 := by
  have h := B_neg_of_antisymm hA x x
  linarith

/-! ## 2. Cost of a discrete loop -/

/-- Trapezoidal execution cost of one step from `x` to `y`. -/
def stepCost (p₀ : Vec d) (M : Mat d) (x y : Vec d) : ℝ :=
  (∑ j, p₀ j * (y j - x j)) + B M ((2⁻¹ : ℝ) • (x + y)) (y - x)

/-- Total cost of the discrete inventory path `q 0, q 1, …, q n`. -/
def cost (n : ℕ) (p₀ : Vec d) (M : Mat d) (q : ℕ → Vec d) : ℝ :=
  ∑ i ∈ range n, stepCost p₀ M (q i) (q (i + 1))

/-- The unsymmetrised left-endpoint pair sums, `P j k = ∑ᵢ qᵢ_k Δᵢ_j`. -/
def P (n : ℕ) (q : ℕ → Vec d) (j k : Fin d) : ℝ :=
  ∑ i ∈ range n, q i k * (q (i + 1) j - q i j)

/-- Discrete Lévy area, matching (13.1) index-for-index. -/
def levyArea (n : ℕ) (q : ℕ → Vec d) : Mat d :=
  Matrix.of fun j k => (2⁻¹ : ℝ) * (P n q k j - P n q j k)

/-! ## 3. The two vanishing terms -/

lemma benchmark_vanishes (n : ℕ) (p₀ : Vec d) {q : ℕ → Vec d} (hq : q n = q 0) :
    ∑ i ∈ range n, ∑ j, p₀ j * (q (i + 1) j - q i j) = 0 := by
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero fun j _ => ?_
  rw [← Finset.mul_sum, Finset.sum_range_sub (fun i => q i j) n, hq, sub_self, mul_zero]

lemma symmetric_vanishes (n : ℕ) {S : Mat d} (hS : ∀ j k, S k j = S j k)
    {q : ℕ → Vec d} (hq : q n = q 0) :
    ∑ i ∈ range n, B S ((2⁻¹ : ℝ) • (q i + q (i + 1))) (q (i + 1) - q i) = 0 := by
  have step : ∀ i : ℕ,
      B S ((2⁻¹ : ℝ) • (q i + q (i + 1))) (q (i + 1) - q i)
        = (fun m => (2⁻¹ : ℝ) * B S (q m) (q m)) (i + 1)
          - (fun m => (2⁻¹ : ℝ) * B S (q m) (q m)) i := by
    intro i
    rw [B_smul, B_add_left, B_sub_right, B_sub_right]
    have hcross : B S (q i) (q (i + 1)) = B S (q (i + 1)) (q i) :=
      B_comm_of_symm hS _ _
    simp only []
    rw [hcross]; ring
  rw [Finset.sum_congr rfl (fun i _ => step i),
      Finset.sum_range_sub (fun m => (2⁻¹ : ℝ) * B S (q m) (q m)) n, hq, sub_self]

/-! ## 4. The antisymmetric part is the area pairing -/

/-- One step of the antisymmetric part is the left-endpoint pairing. -/
lemma antisymm_step {A : Mat d} (hA : ∀ j k, A k j = -A j k) (x y : Vec d) :
    B A ((2⁻¹ : ℝ) • (x + y)) (y - x) = B A x (y - x) := by
  rw [B_smul, B_add_left, B_sub_right, B_sub_right,
      B_self_of_antisymm hA x, B_self_of_antisymm hA y]
  have hswap : B A y x = - B A x y := by
    rw [B_neg_of_antisymm hA y x]
  rw [hswap]; ring

/-- Summing the left-endpoint pairing over the path gives `∑_{j,k} A_{jk} P_{jk}`. -/
lemma sum_antisymm_eq (n : ℕ) (A : Mat d) (q : ℕ → Vec d) :
    (∑ i ∈ range n, B A (q i) (q (i + 1) - q i))
      = ∑ j, ∑ k, A j k * P n q j k := by
  simp only [B, P, Pi.sub_apply, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun i _ => by ring

/-- Relabelling the index pair against antisymmetry of `A`. -/
lemma pair_swap {A : Mat d} (hA : ∀ j k, A k j = -A j k) (n : ℕ) (q : ℕ → Vec d) :
    (∑ j, ∑ k, A j k * P n q k j) = - ∑ j, ∑ k, A j k * P n q j k := by
  rw [Finset.sum_comm, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl fun k _ => by rw [hA j k]; ring

/-- The trace pairing expands as `∑_{j,k} A_{jk} L_{kj}`. -/
lemma trace_mul_expand (A L : Mat d) :
    Matrix.trace (A * L) = ∑ j, ∑ k, A j k * L k j := by
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply]

/-! ## 5. Theorem 13.1, finite-grid form -/

/-- **Theorem 13.1 (finite grid).**  For a closed inventory loop under linear
permanent impact `M = S + A`, with `S` symmetric and `A` antisymmetric, the
arrival benchmark and the symmetric part contribute exactly nothing, and the
whole execution cost of the loop is the trace pairing of `A` with the discrete
Lévy area of the loop. -/
theorem loop_cost_eq_area_pairing
    (n : ℕ) (p₀ : Vec d) (S A : Mat d)
    (hS : ∀ j k, S k j = S j k) (hA : ∀ j k, A k j = -A j k)
    {q : ℕ → Vec d} (hq : q n = q 0) :
    cost n p₀ (S + A) q = Matrix.trace (A * levyArea n q) := by
  have hsplit : ∀ i : ℕ,
      stepCost p₀ (S + A) (q i) (q (i + 1))
        = (∑ j, p₀ j * (q (i + 1) j - q i j))
          + B S ((2⁻¹ : ℝ) • (q i + q (i + 1))) (q (i + 1) - q i)
          + B A (q i) (q (i + 1) - q i) := by
    intro i
    rw [stepCost, B_add_mat, antisymm_step hA]
    ring
  rw [cost, Finset.sum_congr rfl (fun i _ => hsplit i),
      Finset.sum_add_distrib, Finset.sum_add_distrib,
      benchmark_vanishes n p₀ hq, symmetric_vanishes n hS hq, zero_add, zero_add,
      sum_antisymm_eq n A q, trace_mul_expand]
  -- right-hand side: ∑_{j,k} A_{jk} · ½ (P_{jk} − P_{kj})
  have hL : ∀ j k : Fin d,
      A j k * (levyArea n q) k j
        = (2⁻¹ : ℝ) * (A j k * P n q j k) - (2⁻¹ : ℝ) * (A j k * P n q k j) := by
    intro j k
    simp only [levyArea, Matrix.of_apply]
    ring
  rw [Finset.sum_congr rfl (fun j _ => Finset.sum_congr rfl (fun k _ => hL j k))]
  simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [pair_swap hA n q]
  ring

/-- **Corollary 13.2 (finite grid).**  In one asset every closed loop has zero
cost under linear permanent impact: the antisymmetric part of a `1 × 1` matrix
is zero, so there is nothing for the area to pair with. -/
theorem loop_cost_one_asset
    (n : ℕ) (p₀ : Vec 1) (S A : Mat 1)
    (hS : ∀ j k, S k j = S j k) (hA : ∀ j k, A k j = -A j k)
    {q : ℕ → Vec 1} (hq : q n = q 0) :
    cost n p₀ (S + A) q = 0 := by
  have hA0 : A = 0 := by
    ext j k
    have h := hA j k
    have hjk : j = k := Subsingleton.elim j k
    subst hjk
    simp only [Matrix.zero_apply]
    linarith
  rw [loop_cost_eq_area_pairing n p₀ S A hS hA hq, hA0, Matrix.zero_mul,
      Matrix.trace_zero]

/- The converse from nonnegative loop costs to symmetry of cross-impact
   (Corollary 13.3) is outside the scope of this file. -/

end

end MarketImpact
