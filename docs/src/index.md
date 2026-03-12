# PenguinStokes.jl

`PenguinStokes.jl` assembles and solves cut-cell Stokes systems on MAC-style
staggered Cartesian grids, with support for:

- monophasic steady and unsteady Stokes,
- prescribed moving embedded boundaries,
- fixed-interface two-phase Stokes,
- rigid-body FSI wrappers (split and strong coupling).

Within PenguinxCutCell, it sits on:

- `CartesianGeometry.jl` for geometric moments,
- `CartesianOperators.jl` for cut-cell operators,
- `PenguinBCs.jl` for BC types,
- `PenguinSolverCore.jl` for linear solves.

## Implemented Model Families

- `StokesModelMono`: steady/unsteady monophasic Stokes.
- `MovingStokesModelMono`: unsteady monophasic moving-boundary Stokes
  (`body(x..., t)` + `bc_cut_u`).
- `StokesModelTwoPhase`: fixed-interface two-phase Stokes with shared interface
  velocity trace and traction/interface-force rows.
- `StokesFSIProblem`: rigid-body coupling layer on top of moving-boundary
  monophasic Stokes (2D/3D states, split and strong coupling paths).

## Compact Feature Summary

| Area | Status |
|---|---|
| Mono steady/unsteady | Implemented |
| Moving mono (prescribed interface velocity) | Implemented |
| Two-phase fixed interface | Implemented |
| Rigid-body FSI split coupling | Implemented |
| Rigid-body FSI strong coupling | Implemented |
| Cut/interface Neumann or periodic velocity BC | Missing |
| Two-phase moving interface | Missing |

See [Feature Matrix](feature_matrix.md) for a detailed implemented/partial/missing
breakdown.

## Quick Start (Steady Monophasic)

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

## Navigation

- [Models and Equations](stokes.md)
- [Algorithms](algorithms.md)
- [Boundary Conditions and Gauges](boundary_conditions.md)
- [FSI](fsi.md)
- [Postprocessing](postprocessing.md)
- [API](api.md)
- [Examples and Verification](examples.md)
- [Feature Matrix](feature_matrix.md)
- [Developer Notes](developer_notes.md)

## Validation Surface

Package tests and the `examples/` scripts are the primary validation surface.
The examples page maps each family (MMS, Couette/Poiseuille, moving-boundary,
traction/outlet/symmetry, FSI, 3D runs) to what it verifies.
