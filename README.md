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
