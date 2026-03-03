**API and types**

Primary exported symbols

- `AbstractPressureGauge`, `PinPressureGauge`, `MeanPressureGauge`
- `StokesLayout`, `StokesModelMono`, `staggered_velocity_grids`
- `assemble_steady!`, `assemble_unsteady!`, `solve_steady!`, `solve_unsteady!`

Typical construction

- Create a pressure grid `CartesianGrid`.
- Build `StokesModelMono(grid, body, mu, rho; bc_u=..., bc_cut=..., force=..., gauge=...)`.
- Solve with `solve_steady!(model)` or `solve_unsteady!(model, u0, tspan; dt=..., scheme=...)`.

Notes

- Velocity boundary conditions are passed component-wise through `bc_u`.
- Pressure does not get direct Dirichlet/Neumann BC; use a gauge.
- Cut/interface BC currently supports Dirichlet values for `ugamma`.
