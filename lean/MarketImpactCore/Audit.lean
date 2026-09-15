/-
  Audit.lean — run `lake env lean MarketImpactCore/Audit.lean` after a
  successful `lake build`.  Each line must report exactly
  `[propext, Classical.choice, Quot.sound]` (order may vary).  Any other
  axiom, or any `sorryAx`, means the artifact is not what Appendix E says.
-/
import MarketImpactCore.LoopCost
import MarketImpactCore.QuarticSign
import MarketImpactCore.VenueConvex

#print axioms MarketImpact.loop_cost_eq_area_pairing
#print axioms MarketImpact.loop_cost_one_asset
#print axioms MarketImpact.RayData.ray_sign_conditions
#print axioms MarketImpact.value_convex

#print axioms MarketImpact.zero_cost_isGLB
#print axioms MarketImpact.zero_cost_nonempty
