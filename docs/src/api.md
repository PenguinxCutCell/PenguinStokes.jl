**API and types**

Primary exported symbols

- `AbstractPressureGauge`, `PinPressureGauge`, `MeanPressureGauge`
- `StokesLayout`, `StokesModelMono`, `MovingStokesModelMono`, `StokesLayoutTwoPhase`, `StokesModelTwoPhase`, `staggered_velocity_grids`
- `assemble_steady!`, `assemble_unsteady!`, `solve_steady!`, `solve_unsteady!`
- `assemble_unsteady_moving!`, `solve_unsteady_moving!`
- `embedded_boundary_quantities`, `embedded_boundary_traction`, `embedded_boundary_stress`, `integrated_embedded_force`

Typical construction

- Create a pressure grid `CartesianGrid`.
- Build `StokesModelMono(grid, body, mu, rho; bc_u=..., bc_cut=..., force=..., gauge=...)`.
- For prescribed moving cut boundaries, build `MovingStokesModelMono(grid, body, mu, rho; bc_u=..., bc_cut_u=..., force=..., gauge=...)`.
- For fixed-interface two-phase runs, build `StokesModelTwoPhase(grid, body, mu1, mu2; rho1=..., rho2=..., force1=..., force2=..., interface_force=..., bc_u=..., gauge=...)`.
- Solve with `solve_steady!(model)`, `solve_unsteady!(model, x_prev; t=..., dt=..., scheme=...)`, or `solve_unsteady_moving!(model, x_prev; t=..., dt=..., scheme=...)`.

Notes

- Velocity boundary conditions are passed component-wise through `bc_u`.
- Outer-box Stokes traction laws are available through `PenguinBCs` side entries in `bc_u`:
- `PressureOutlet(pout)`, `DoNothing()`, and `Traction(t)`.
- Outer-box Stokes symmetry/free-slip walls are available through `Symmetry()`.
- Traction-type BCs must be set on all velocity components for a given side.
- Symmetry must also be set on all velocity components for a given side.
- `bc_p` cannot be imposed on traction or symmetry sides.
- Pressure gauge remains required; optional wall constraints can be passed through `bc_p`.
- `PinPressureGauge(index=...)` pins that same pressure DOF row/column.
- `MeanPressureGauge()` uses active-cell-volume weights for the zero-mean constraint.
- Cut/interface BC currently supports Dirichlet values for `ugamma`.
- `MovingStokesModelMono` uses per-component cut Dirichlet data through `bc_cut_u`.
- `StokesModelTwoPhase` reuses `ugamma` rows as traction-balance equations (no separate `bc_cut`).
- Embedded-boundary utilities return pressure-grid cut-cell stress/traction fields and integrated force splits (pressure/viscous).
