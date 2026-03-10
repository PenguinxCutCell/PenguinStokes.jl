# PenguinStokes.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/PenguinStokes.jl/dev)
![CI](https://github.com/PenguinxCutCell/PenguinStokes.jl/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PenguinxCutCell/PenguinStokes.jl/branch/main/graph/badge.svg)

`PenguinStokes.jl` assembles monophasic cut-cell Stokes systems on MAC-style staggered grids using `CartesianGeometry.jl`, `CartesianOperators.jl`, and `PenguinSolverCore.jl`.

## Feature Status

| Area | Item | Status | Notes |
|---|---|---|---|
| Models | Steady monophasic Stokes | Implemented | `StokesModelMono` + `assemble_steady!` |
| Models | Unsteady monophasic Stokes | Implemented | Theta-form assembly via `assemble_unsteady!` |
| Models | Unsteady monophasic moving-boundary Stokes | Implemented | `MovingStokesModelMono` + `assemble_unsteady_moving!` |
| Models | Rigid-body FSI (2D/3D split + strong coupling) | Implemented | Generic `StokesFSIProblem` + `step_fsi!` / `step_fsi_strong!` |
| Models | Steady two-phase fixed-interface Stokes | Implemented | `StokesModelTwoPhase` with shared `u_γ` and traction rows |
| Grids | MAC staggered layout | Implemented | `staggered_velocity_grids` + per-component operators |
| BCs (velocity box) | Dirichlet | Implemented | Applied on momentum rows |
| BCs (velocity box) | Neumann | Implemented | Applied as flux terms |
| BCs (velocity box) | Periodic | Implemented | Through periodic stencil/operator construction |
| BCs (cut/interface) | Dirichlet on `u_γ` | Implemented | `bc_cut=Dirichlet(...)` |
| BCs (cut/interface) | Neumann / Periodic | Missing | Explicitly rejected in current cut BC applier |
| Pressure | Pin gauge | Implemented | `PinPressureGauge` |
| Pressure | Mean gauge | Implemented | `MeanPressureGauge` |

## Unknown ordering

For `N` dimensions and `nt = prod(grid.n)`:

`x = [uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]`

Total size: `(2N+1)*nt`.

Two-phase fixed-interface ordering:

`x = [uomega1_1; ...; uomega1_N; uomega2_1; ...; uomega2_N; ugamma_1; ...; ugamma_N; pomega1; pomega2]`

Total size: `(3N+2)*nt`.

## Quick start

```julia
using CartesianGrids, PenguinBCs, PenguinStokes

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
body(x...) = -1.0

bc = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

model = StokesModelMono(
    grid,
    body,
    1.0,
    1.0;
    bc_u=(bc, bc),
    bc_cut=Dirichlet(0.0),
    force=(0.0, 0.0),
)

sys = solve_steady!(model)
```

See `examples/` for complete scripts including MMS/convergence checks.

Key verification scripts:
- `examples/04_mms_convergence.jl`: pressure-coupled streamfunction MMS (prints error orders and exact-state momentum residual split `interior` vs `boundary` to track order loss sources).
- `examples/05_mms_convergence_zero_pressure.jl`: no-body zero-pressure MMS, near second-order velocity convergence.
- `examples/06_mms_convergence_embedded_outside_circle.jl`: embedded-interface MMS for outside-circle fluid (`ϕ = R - sqrt((x-xc)^2 + (y-yc)^2)`), with no-slip on box and cut interface.
- `examples/08_two_phase_mms_fixed_interface.jl`: two-phase fixed-interface equilibrium with viscosity ratio and prescribed interface traction.
- `examples/09_two_phase_planar_couette.jl`: two-layer planar Couette validation (periodic streamwise, fixed flat interface) with exact piecewise-linear profile.
- `examples/10_two_phase_planar_poiseuille.jl`: two-layer planar Poiseuille validation (body-force equivalent, periodic streamwise) with exact piecewise-parabolic profile.
- `examples/11_two_phase_oscillatory_couette.jl`: unsteady oscillatory two-layer Couette validation with harmonic amplitude/phase probe checks.
- `examples/12_two_phase_viscous_drop_drag.jl`: 3D fixed spherical drop run with numerical drag compared against Hadamard–Rybczynski drag scaling.
- `examples/13_unsteady_moving_body_translation.jl`: one-phase prescribed moving embedded boundary with oscillatory rigid translation and trace-row checks.
- `examples/14_unsteady_oscillating_cylinder.jl`: one-phase oscillating embedded cylinder with force/torque history output.
- `examples/17_fsi_free_falling_circle.jl`: translation-only rigid-body FSI free-fall demo (ODE-coupled moving-boundary Stokes).
- `examples/18_fsi_prescribed_rotating_cylinder.jl`: prescribed oscillatory cylinder rotation with force/torque history.
- `examples/19_fsi_spin_decay_calibrated.jl`: calibrated rotational-drag spin decay compared against exponential ODE prediction.
- `examples/20_fsi_falling_rotating_ellipse.jl`: translation + rotation FSI demo for a falling ellipse.
- `examples/21_3d_rigid_sphere_drag.jl`: 3D imposed-velocity rigid-sphere drag benchmark compared to `6πμRU`.
- `examples/22_3d_falling_rigid_sphere_split_vs_strong.jl`: 3D free-fall rigid sphere, split vs strong coupling, compared against linear-drag ODE trend.
- `examples/23_fsi_neutral_buoyancy_decay.jl`: neutral-buoyancy translational decay benchmark using calibrated drag.

## FSI (v0)

`PenguinStokes.jl` now includes rigid-body FSI wrappers for one moving embedded body:

- `StokesFSIProblem`: generic wrapper for 2D/3D rigid-body states and parameters.
- Rigid boundary velocity is unified as `uΓ = V + Ω × (x-X)` (2D scalar-spin specialization included).
- `step_fsi!`: one-pass split coupling.
- `step_fsi_strong!`: fixed-point strong coupling with optional Aitken relaxation.
- `simulate_fsi!` runs repeated split steps and returns history diagnostics.

Current scope/limits:

- single rigid body,
- 3D translational free fall is supported; 3D rotational dynamics currently target isotropic inertia use-cases,
- no contact/collision model (stop before wall contact in free-fall runs).
