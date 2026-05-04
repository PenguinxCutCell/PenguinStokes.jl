# PenguinStokes.jl

[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://PenguinxCutCell.github.io/PenguinStokes.jl/dev)
![CI](https://github.com/PenguinxCutCell/PenguinStokes.jl/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/PenguinxCutCell/PenguinStokes.jl/branch/main/graph/badge.svg)

`PenguinStokes.jl` assembles cut-cell Stokes systems on MAC-style staggered
Cartesian grids for monophasic, moving-boundary, fixed-interface two-phase, and
rigid-body FSI workflows.

## Documentation

- Home: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/>
- Models and equations: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/stokes/>
- Algorithms: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/algorithms/>
- Boundary conditions and gauges: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/boundary_conditions/>
- FSI: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/fsi/>
- Postprocessing: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/postprocessing/>
- API: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/api/>
- Examples and verification: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/examples/>
- Feature matrix: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/feature_matrix/>
- Developer notes: <https://PenguinxCutCell.github.io/PenguinStokes.jl/dev/developer_notes/>

## Audited Feature Status

Status labels: `Implemented`, `Partial`, `Missing`.

| Area | Feature | Status | Notes |
|---|---|---|---|
| Models | Steady monophasic Stokes | Implemented | `StokesModelMono`, `assemble_steady!` |
| Models | Unsteady monophasic Stokes | Implemented | `assemble_unsteady!` (`:BE`, `:CN`, numeric `theta`) |
| Models | Moving-boundary monophasic Stokes | Implemented | `MovingStokesModelMono`, `assemble_unsteady_moving!` |
| Models | Fixed-interface two-phase Stokes | Implemented | `StokesModelTwoPhase` with shared `ugamma` traction rows |
| Models | Moving-interface two-phase Stokes | Missing | no moving-interface two-phase constructor/path |
| FSI | Split rigid-body coupling | Implemented | `step_fsi!`, `simulate_fsi!` |
| FSI | Strong rigid-body coupling | Implemented | `step_fsi_strong!` |
| FSI | Rotational rigid-body coupling | Partial | strong 2D coverage; narrower 3D validation |
| FSI | Multi-body FSI | Implemented | `MultiBodyFSIProblem`, `step_multi_fsi!` — N bodies, shared fluid solve, per-body force partition |
| BCs (outer velocity) | Dirichlet / Neumann / Periodic | Implemented | component-wise BC path |
| BCs (outer velocity) | Traction / PressureOutlet / DoNothing | Implemented | side-level Stokes traction overwrite |
| BCs (outer velocity) | Symmetry | Implemented | side-level free-slip/symmetry overwrite |
| BCs (cut/interface velocity) | Dirichlet on `ugamma` | Implemented | `bc_cut=Dirichlet(...)` |
| BCs (cut/interface velocity) | Neumann / Periodic | Missing | explicitly rejected (`ArgumentError`) |
| Pressure | Pin gauge / mean gauge | Implemented | `PinPressureGauge`, `MeanPressureGauge` |
| Pressure | Optional side pressure BC (`bc_p`) | Implemented | with compatibility restrictions |
| Postprocessing | Stress / traction / force / torque | Implemented | `embedded_boundary_quantities` family (mono model scope) |

## Model Families

- `StokesModelMono`: steady and unsteady monophasic Stokes.
- `MovingStokesModelMono`: moving embedded boundary with prescribed cut velocity.
- `StokesModelTwoPhase`: fixed-interface two-phase Stokes.
- `StokesFSIProblem`: single rigid-body FSI wrappers (split and strong coupling).
- `MultiBodyFSIProblem`: N rigid bodies sharing one fluid solve; forces partitioned by interface-centroid proximity.

## Boundary-Condition Families

Outer velocity BCs:

- scalar: `Dirichlet`, `Neumann`, `Periodic`
- side-level Stokes laws: `Traction`, `PressureOutlet`, `DoNothing`, `Symmetry`

Cut/interface velocity BCs:

- supported: Dirichlet
- unsupported: Neumann, periodic (currently rejected)

Consistency rules (enforced):

- traction-type BC must be set on all velocity components for a side,
- symmetry must be set on all velocity components for a side,
- `bc_p` cannot be imposed on traction/symmetry sides.

## Gauges

A pressure gauge is required:

- `PinPressureGauge(; index=nothing)`
- `MeanPressureGauge()`

## Postprocessing

Exported helpers for `StokesModelMono`:

- `embedded_boundary_quantities`
- `embedded_boundary_traction`
- `embedded_boundary_stress`
- `integrated_embedded_force`

Outputs include pressure/viscous force split and torque.

## FSI Scope

Implemented rigid-body FSI scope:

- 2D and 3D rigid states/params,
- rotational state updates,
- split coupling (`step_fsi!`) and strong coupling (`step_fsi_strong!`),
- force/torque from end-time static-model postprocessing,
- **multi-body FSI** (`MultiBodyFSIProblem`, `step_multi_fsi!`): N bodies with shared fluid solve and per-body force extraction by interface-centroid proximity.

## Quick Start

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

## Validation Examples

Representative scripts:

- `examples/04_mms_convergence.jl` — mono MMS, 2nd-order convergence
- `examples/06_mms_convergence_embedded_outside_circle.jl` — embedded-interface MMS
- `examples/08_two_phase_mms_fixed_interface.jl` — two-phase traction-jump equilibrium
- `examples/27_two_phase_static_circle_spurious_current.jl` — static drop spurious-current benchmark
- `examples/09_two_phase_planar_couette.jl`, `10_two_phase_planar_poiseuille.jl`, `10_bis_two_phase_planar_couette_poiseuille.jl`
- `examples/13_unsteady_moving_body_translation.jl`, `25_moving_mms_time_schemes.jl`
- `examples/22_3d_falling_rigid_sphere_split_vs_strong.jl` — FSI split vs strong coupling

Physical benchmarks (Basilisk-inspired):

- `examples/31_periodic_cylinder_array.jl` — 2D periodic cylinder array; permeability vs Hasimoto (1959); 1.5–6% error at n=65 (φ=0.1–0.3)
- `examples/32_periodic_sphere_array.jl` — 3D periodic sphere array; permeability vs Hasimoto; 4–10% error at n=17
- `examples/33_couette_rotating_cylinders.jl` — Couette flow between rotating cylinders; 2nd-order convergence to analytic u_θ; L2=5×10⁻³ at n=128
- `examples/34_moving_cylinder_uniform_flow.jl` — cylinder moving at uniform flow speed; velocity error at machine precision (~7×10⁻¹⁵); validates moving-boundary implementation
- `examples/35_sphere_torque_couette.jl` — torque on fixed sphere in 3D Couette shear; vs Stokes formula T=8πμR³(γ̇/2); 3.6% error at n=33 (4 pts/d)
- `examples/36_sphere_towards_wall.jl` — sphere approaching a plane wall; vs Brenner (1961) / Cooley & O'Neill (1969); error halves with each grid doubling, approaching 2nd order
- `examples/37_fsi_cylinder_shear_flow_rotation.jl` — cylinder freely rotating in 2D shear (FSI); ω_∞ = −γ̇/2 exact for Stokes; 2.2% error at n=65 (Re_d=0.4)
- `examples/38_fsi_ballistic_sphere.jl` — 3D ballistic sphere (initial transverse velocity + gravity); Stokes limit (Re_d≈0.012); exponential velocity decay vs analytic
- `examples/39_two_spheres_couette.jl` — two spheres in 3D Couette shear (multi-body FSI); hydrodynamic approach orbit; exercises `MultiBodyFSIProblem`

Developer probe scripts (retained intentionally):

- `test/dev_pressure_outlet_mms.jl` (manual probe; not in `runtests.jl`)
- `test/dev_outlet_row_equivalence.jl` (included in `runtests.jl`)
- `test/dev_traction_box_probe.jl` (included in `runtests.jl`)

## Benchmarks

| Benchmark | Description | Status |
|---|---|---|
| No Interface MMS | 2nd order convergence | ok |
| Embedded Interface MMS | 2nd order convergence | ok |
| Two-Phase Fixed Interface MMS | 2nd order convergence | ok |
| Static Drop Spurious Current | Laplace jump test | ok |
| No Interface moving MMS | 2nd order convergence | ok |
| Moving Mono | 2nd order convergence | ok |
| Moving Two-Phase | 2nd order convergence | (BE ok / CN ok but pressure cell deactivation threshold must be adjusted) |

## Current Limitations

- cut/interface velocity Neumann and periodic BCs are not implemented,
- no rigid-body contact/collision model (lubrication correction not implemented; requires sufficient mesh resolution to resolve gaps),
- no deformable-body coupling model.
