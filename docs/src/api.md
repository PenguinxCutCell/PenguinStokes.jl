**API and types**

Primary exported symbols

- `AbstractPressureGauge`, `PinPressureGauge`, `MeanPressureGauge`
- `StokesLayout`, `StokesModelMono`, `StokesLayoutTwoPhase`, `StokesModelTwoPhase`, `staggered_velocity_grids`
- `assemble_steady!`, `assemble_unsteady!`, `solve_steady!`, `solve_unsteady!`
- `embedded_boundary_quantities`, `embedded_boundary_traction`, `embedded_boundary_stress`, `integrated_embedded_force`

Typical construction

- Create a pressure grid `CartesianGrid`.
- Build `StokesModelMono(grid, body, mu, rho; bc_u=..., bc_cut=..., force=..., gauge=...)`.
- For fixed-interface two-phase runs, build `StokesModelTwoPhase(grid, body, mu1, mu2; rho1=..., rho2=..., force1=..., force2=..., interface_force=..., bc_u=..., gauge=...)`.
- Solve with `solve_steady!(model)` or `solve_unsteady!(model, x_prev; t=..., dt=..., scheme=...)`.

Notes

- Velocity boundary conditions are passed component-wise through `bc_u`.
- Pressure gauge remains required; optional wall constraints can be passed through `bc_p`.
- Cut/interface BC currently supports Dirichlet values for `ugamma`.
- `StokesModelTwoPhase` reuses `ugamma` rows as traction-balance equations (no separate `bc_cut`).
- Embedded-boundary utilities return pressure-grid cut-cell stress/traction fields and integrated force splits (pressure/viscous).
