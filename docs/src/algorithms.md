# Algorithms

This page summarizes current assembly/solve algorithms as implemented.

## 1. Unknown Layouts

### Monophasic (`StokesLayout`)

```text
x = [uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]
```

### Two-phase (`StokesLayoutTwoPhase`)

```text
x = [uomega1_1; ...; uomega1_N; uomega2_1; ...; uomega2_N; ugamma_1; ...; ugamma_N; pomega1; pomega2]
```

`omega` denotes cell/face unknowns, `gamma` denotes cut/interface trace unknowns.

## 2. Discrete Operators

At high level:

- velocity diffusion-style blocks use cut-cell operator products
  (`G' * Winv * G`, `G' * Winv * H`),
- pressure coupling is split into gradient/divergence blocks,
- interface/cut equations are imposed by row overwrite/insertion,
- boundary side laws (traction/symmetry/outlet) overwrite boundary-adjacent
  momentum rows,
- active/inactive row masks are computed from capacity support.

## 3. Steady Monophasic Assembly

`assemble_steady!(sys, model::StokesModelMono, t)`:

1. Build momentum, trace-tie, and continuity blocks.
2. Insert forcing and cut Dirichlet trace RHS.
3. Apply side-level traction/symmetry overwrite (if present).
4. Apply component-wise velocity BC row modifications.
5. Apply optional pressure wall BC rows.
6. Apply pressure gauge row replacement.
7. Apply identity-row regularization on inactive/halo rows.

## 4. Unsteady Monophasic Assembly

`assemble_unsteady!(..., scheme)`:

- theta scheme (`:BE`, `:CN`, numeric theta),
- adds `(rho/dt) * V` mass term on momentum omega rows,
- adds history contribution from previous step (`x_prev`),
- keeps continuity/gauge algebraic,
- applies boundary/gauge/row-activity logic at `t_{n+1}`.

## 5. Moving Monophasic Assembly

`assemble_unsteady_moving!`:

1. Build slab and end-time capacities/operators (`_build_moving_slab!`).
2. Assemble slab-integrated momentum terms and end-time pressure coupling.
3. Enforce cut trace at end-time via `bc_cut_u`.
4. Apply end-time box BCs and gauge.
5. Mask inactive rows from end-time supports and regularize.

Interpretation:

- geometry/mass/diffusion terms come from slab integration,
- cut trace and outer BCs are strongly imposed at end-time rows.

## 6. Two-Phase Fixed-Interface Assembly

`assemble_steady!` / `assemble_unsteady!` for `StokesModelTwoPhase`:

- separate phase-1 and phase-2 momentum/continuity blocks,
- shared `ugamma` rows used for traction/interface-force equations,
- pressure coupling from both phase pressure blocks into traction rows,
- gauge applied on phase-1 pressure block row replacement,
- same row-activity/identity regularization strategy.

## 7. FSI Stepping Algorithms

### Split (loose) coupling

`step_fsi!`:

1. Predict rigid state over slab.
2. Solve moving-boundary fluid step.
3. Build end-time static model and compute integrated hydrodynamic force/torque.
4. Advance rigid-body ODE state (`:symplectic_euler` or `:forward_euler`).

### Strong partitioned coupling

`step_fsi_strong!`:

1. Initialize state guess at `t_{n+1}`.
2. Alternate fluid solve and rigid ODE update.
3. Relax updates (`:none`, `:constant`, `:aitken`).
4. Stop by absolute/relative residual tolerance or `maxiter`.

## 8. Regularization Strategy

Rows with no physical support (inactive/halo/uncoupled rows after masking) are
set to identity with zero RHS.

Why this is used:

- keeps linear systems nonsingular,
- preserves fixed global indexing/layout,
- simplifies coupling between assembly paths and postprocessing.

## 9. Practical Notes

- First-order loss can appear near cut/wall closure depending on collocation and
  BC type.
- Traction/symmetry/outlet rows are side-level constraints and must be declared
  consistently across velocity components on a side.
- `bc_p` cannot coexist with traction/symmetry side laws on the same side.
- For pressure errors, compare shift-invariant quantities when gauge differs.
