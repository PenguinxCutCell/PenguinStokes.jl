# Models and Equations

This page documents the PDE models represented by the current implementation.
The source/tests/examples are treated as the ground truth for supported scope.

## 1. Monophasic Steady Stokes

For velocity `u` and pressure `p` in fluid region `Ω`:

```math
-\nabla\cdot\left(2\mu D(u)\right) + \nabla p = f, \qquad
\nabla\cdot u = 0,
```

with `D(u) = (\nabla u + \nabla u^T)/2`.

Discrete representation in `StokesModelMono`:

- MAC staggered unknowns on per-component velocity grids (`uomega`) and pressure
  grid (`pomega`),
- cut/interface trace unknowns `ugamma`,
- momentum diffusion-like blocks from cut-cell operators,
- split pressure-gradient/divergence coupling,
- row overwrite for boundary/interface constraints,
- pressure gauge replacement to remove nullspace.

## 2. Monophasic Unsteady Stokes

```math
\rho \partial_t u - \nabla\cdot\left(2\mu D(u)\right) + \nabla p = f,
\qquad \nabla\cdot u = 0.
```

Implemented via `assemble_unsteady!` / `solve_unsteady!` with theta scheme:

- `:BE` (`\theta = 1`),
- `:CN` (`\theta = 1/2`),
- numeric `theta in [0,1]`.

Mass terms are added on momentum omega rows; continuity and gauge remain
algebraic constraints.

## 3. Moving-Boundary Monophasic Stokes

`MovingStokesModelMono` supports prescribed moving embedded boundaries:

- geometry callback `body(x..., t)` (or static `body(x...)`),
- per-component cut velocity BC `bc_cut_u`,
- slab-based assembly over `[t_n, t_{n+1}]` using `SpaceTimeCartesianGrid`
  reduction,
- strong end-time trace enforcement on `ugamma` rows,
- activity masking based on end-time support.

This is a prescribed-motion moving-boundary formulation, not deformable-body FSI.

## 4. Fixed-Interface Two-Phase Stokes

For phases `k in {1,2}` in fixed subdomains `Ω_k`:

```math
-\nabla\cdot\left(2\mu_k D(u_k)\right) + \nabla p_k = f_k,
\qquad \nabla\cdot u_k = 0.
```

Current interface formulation in `StokesModelTwoPhase`:

- shared interface velocity trace unknowns `ugamma` (velocity continuity path),
- phase-wise pressure blocks `pomega1`, `pomega2`,
- traction-balance rows on `ugamma` with optional interface forcing
  `interface_force`,
- per-phase `mu1`, `mu2`, `rho1`, `rho2` supported.

Current scope is fixed-interface (no geometric interface evolution in two-phase
constructor path).

## 5. Pressure Gauges

A gauge is required to remove pressure nullspace.

- `PinPressureGauge(index=...)`: pin one pressure DOF.
- `MeanPressureGauge()`: enforce volume-weighted zero mean over active pressure
  cells.

Applied for mono/two-phase/moving models by row replacement in assembled systems.

## 6. Boundary and Interface Scope (Current)

Implemented:

- outer velocity BCs: `Dirichlet`, `Neumann`, `Periodic`, plus Stokes
  traction-family side BCs (`Traction`, `PressureOutlet`, `DoNothing`,
  `Symmetry`),
- optional outer pressure BC `bc_p` with `Dirichlet`/`Neumann`/`Periodic`
  handling,
- cut/interface velocity Dirichlet enforcement on `ugamma`.

Missing/unsupported in current code path:

- cut/interface velocity `Neumann` and `Periodic` (`ArgumentError`),
- moving two-phase interface model,
- deformable-body coupling,
- multi-body/contact models in FSI.

## 7. FSI Scope and Limits

Rigid-body FSI wrappers are implemented beyond translation-only:

- 2D/3D rigid states and parameters,
- rotational state updates (2D scalar spin, 3D angular velocity + orientation),
- split one-pass coupling (`step_fsi!`),
- strong fixed-point coupling with optional Aitken relaxation
  (`step_fsi_strong!`).

Current limits:

- single rigid body,
- no contact/collision handling,
- no deformable body model.
