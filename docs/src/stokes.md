**Stokes model and discretization**

This package assembles monophasic and fixed-interface two-phase Stokes systems on MAC-style staggered grids with cut-cell geometry.

Unknown ordering

For `N` dimensions and `nt = prod(grid.n)`:

`x = [uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]`

Two-phase fixed-interface ordering:

`x = [uomega1_1; ...; uomega1_N; uomega2_1; ...; uomega2_N; ugamma_1; ...; ugamma_N; pomega1; pomega2]`

Discrete structure

- Momentum block per component uses `G' * Winv * G` / `G' * Winv * H` diffusion-style operators on velocity grids.
- Pressure coupling is assembled via split gradient/divergence blocks built from pressure operators.
- Interface/cut rows enforce `ugamma = g_cut` (Dirichlet currently).
- Continuity rows enforce incompressibility with pressure gauge replacement for nullspace control.
- `PinPressureGauge(index=...)` now replaces the pressure equation at that same DOF (`pomega[index]`), rather than always overwriting the first pressure row.
- `MeanPressureGauge()` now enforces a volume-weighted zero-mean pressure on active pressure cells.

Two-phase interface rows

- `StokesModelTwoPhase` uses shared `ugamma` traces and two pressure blocks (`pomega1`, `pomega2`).
- Interface rows are assembled as traction-balance equations (pressure + viscous stress terms) with optional forcing `interface_force`.
- Row masking keeps only overlap-support interface rows where both pressure-interface geometry and velocity-trace support are present, avoiding singular uncoupled trace modes.

Wall closure on staggered grids

- On staggered velocity meshes, a box wall can be either:
- `collocated` with a velocity DOF (the DOF lies on the wall plane), or
- `non-collocated` (the nearest DOF is inside the domain).
- This distinction is geometric and depends on the component grid (`ux`, `uy`, ...), not only on the side label.
- For Dirichlet velocity walls, the package now applies:
- strong elimination (`u = u_wall`) only for collocated rows when `strong_wall_bc=true`,
- distance-based weak closure (`a = mu * Aface / dist`) for non-collocated rows, using the true DOF-to-wall distance.
- Using a fixed half-step distance on non-collocated rows can produce first-order wall truncation, especially on outer-box high-side rows.

Pressure wall handling (`bc_p`)

- `bc_p` is optional (`nothing` by default).
- When `bc_p === nothing`, pressure wall rows are not modified; the usual continuity + gauge system is assembled.
- When `bc_p` is provided, box pressure wall constraints (Dirichlet/Neumann) can be imposed explicitly.
- This is intended for cases where pressure-wall data is known/desired (for example MMS studies with prescribed normal pressure derivative).
- Pressure errors should generally be evaluated with a shift-invariant metric (`p - mean(p)` or equivalent), since raw pressure levels remain gauge-dependent.

Outer-box Stokes traction BCs (`bc_u`)

- In addition to scalar velocity BCs (`Dirichlet`, `Neumann`, `Periodic`), outer-box sides now support:
- `PressureOutlet(pout)` for `σn = -pout*n`,
- `DoNothing()` for homogeneous traction `σn = 0`,
- `Traction(t)` for prescribed full traction vector `σn = t`.
- `Symmetry()` for axis-aligned free-slip/symmetry walls:
  - normal velocity: `u⋅n = 0`,
  - tangential traction: `(I - n⊗n)σn = 0`.
- Traction BCs are enforced with side-based row overwrite of boundary-adjacent momentum rows, including:
- pressure coupling in normal traction rows,
- symmetric-gradient cross coupling in tangential traction rows (`∂ₙu_t + ∂_t u_n`).
- Symmetry sides are also assembled with side-based row overwrite:
  - the normal component row is set strongly to zero velocity,
  - tangential component rows enforce homogeneous tangential traction.
- A traction side must be declared on all velocity components for that side.
- `bc_p` is not allowed on traction sides (pressure is already part of the traction law there).
- A symmetry side must also be declared on all velocity components for that side.
- `bc_p` is not allowed on symmetry sides.

Embedded-boundary force and stress post-processing

- `embedded_boundary_quantities(model, x; ...)` computes stress tensors, traction vectors, and integrated force density on pressure-grid cut cells.
- `integrated_embedded_force(model, x; ...)` returns total force, split into pressure and viscous parts, plus torque.
- Pressure traction uses `p` trace reconstruction at the interface (`:none` or `:linear`).
- Viscous traction uses `mu * (grad(u) + grad(u)')` from the assembled staggered operators.

Unsteady formulation

`assemble_unsteady!` supports theta schemes (`:BE`, `:CN`, numeric `theta`) by adding mass `(rho/dt) * V` to momentum omega blocks and corresponding RHS history terms.

For `StokesModelTwoPhase`, unsteady assembly applies the same theta update independently on each phase momentum block (`uomega1`, `uomega2`) while keeping interface traction rows algebraic at `t_{n+1}`.

Moving embedded boundary (prescribed velocity)

- `MovingStokesModelMono` supports one-phase unsteady Stokes with time-dependent level-set geometry `body(x..., t)` (or static `body(x...)`) and prescribed interface motion through `bc_cut_u`.
- `assemble_unsteady_moving!` builds slab-integrated operators between `t_n` and `t_{n+1}` using `SpaceTimeCartesianGrid` reduction, then enforces cut-trace rows strongly at `t_{n+1}`.
- Outer wall BC (`bc_u`, optional `bc_p`) and pressure gauge are applied on end-time capacities, and row-activity masking is computed from end-time active cells/interface support.

Rigid-body FSI wrapper (v0)

- `StokesFSIProblem` adds a translation-only rigid-body ODE loop around `MovingStokesModelMono`.
- `step_fsi!` performs one coupled step:
  - predicts rigid translation over the slab,
  - solves `solve_unsteady_moving!`,
  - evaluates end-time force/torque with `integrated_embedded_force(endtime_static_model(...), ...)`,
  - advances rigid-body state with a first-order ODE scheme (`:symplectic_euler` or `:forward_euler`).
- `simulate_fsi!` runs repeated coupled steps and records diagnostics.
- Current scope: single rigid body, translation only, no contact/collision handling.
