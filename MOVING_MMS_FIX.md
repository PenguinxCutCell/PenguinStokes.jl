# Moving MMS Convergence Fix

## Symptoms
- Velocity L2 order ~0.74–0.99 (target >1.5)
- `p_max` growing from 9 → 142 → 150 as mesh refines

## Root Cause

Two bugs in `_stokes_row_activity` for `MovingStokesModelMono` and `MovingStokesModelTwoPhase`:

### Bug 1 — mismatched gradient/divergence operators
The momentum rows use `grad = op_p_slab.G + op_p_slab.H` (time-integrated slab operators, scale `O(dt·h)`).
The pressure rows used `div = op_p_end.G + op_p_end.H` (end-time spatial operators, scale `O(h)`).

These two operators are inconsistent: they discretise different bilinear forms, breaking saddle-point consistency and causing the pressure to be underdetermined.

### Bug 2 — pressure activity based on end-time volume
`_pressure_activity(cap_p_end)` marks a cell active when `V_end > 0`.
Sliver cells nearly swept away by `t_{n+1}` have `V_end → 0` as `h → 0`, giving near-zero gradient coupling in the momentum rows. Their pressure rows are essentially unconstrained, causing `p → ∞`.

## Fix

1. Use `op_p_slab` for `div_omega`/`div_gamma` — grad and div now use the same time-integrated operators.
2. Use `cap_p_slab` for pressure activity with a minimum volume threshold `1e-3 * h^N` — sliver cells below this are deactivated.

## Result

| | Before | After |
|---|---|---|
| BE order | 0.74 → 0.99 | 1.66 → 1.87 |
| CN order | 0.74 → 0.99 | 2.59 |
| p_max | 9 → 142 → 150 | ≈ 0 |
| L2 error | ~0.022 | ~6.7e-5 |
