**API and types**

Primary exported symbols

- `AbstractPressureGauge`, `PinPressureGauge`, `MeanPressureGauge`
- `StokesLayout`, `StokesModelMono`, `MovingStokesModelMono`, `StokesLayoutTwoPhase`, `StokesModelTwoPhase`, `staggered_velocity_grids`
- `assemble_steady!`, `assemble_unsteady!`, `solve_steady!`, `solve_unsteady!`
- `assemble_unsteady_moving!`, `solve_unsteady_moving!`
- `embedded_boundary_quantities`, `embedded_boundary_traction`, `embedded_boundary_stress`, `integrated_embedded_force`
- `RigidBodyState`, `RigidBodyParams`, `RigidBodyState2D`, `RigidBodyState3D`
- `RigidBodyParams2D`, `RigidBodyParams3D`
- `Circle`, `Sphere`, `Ellipse`, `Ellipsoid`, `StokesFSIProblem`
- `endtime_static_model`, `step_fsi!`, `step_fsi_rotation!`, `step_fsi_strong!`, `simulate_fsi!`, `simulate_fsi_rotation!`
- `rigid_boundary_velocity`, `rigid_velocity_2d`, `rigid_velocity`, `rigid_cut_bc_tuple`, `rigid_cut_bc_tuple_2d`, `rigid_body_levelset`

Typical construction

- Create a pressure grid `CartesianGrid`.
- Build `StokesModelMono(grid, body, mu, rho; bc_u=..., bc_cut=..., force=..., gauge=...)`.
- For prescribed moving cut boundaries, build `MovingStokesModelMono(grid, body, mu, rho; bc_u=..., bc_cut_u=..., force=..., gauge=...)`.
- For fixed-interface two-phase runs, build `StokesModelTwoPhase(grid, body, mu1, mu2; rho1=..., rho2=..., force1=..., force2=..., interface_force=..., bc_u=..., gauge=...)`.
- For rigid-body translation FSI, wrap a moving model with `StokesFSIProblem(...)` and step with `step_fsi!(...)`.
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

**Canonical docstrings**

```@docs
PenguinStokes.Circle
PenguinStokes.Sphere
PenguinStokes.Ellipse
PenguinStokes.Ellipsoid
PenguinStokes.Orientation2D
PenguinStokes.Orientation3D
PenguinStokes.volume
PenguinStokes.sdf
PenguinStokes.body_inertia
PenguinStokes.RigidBodyState
PenguinStokes.RigidBodyParams
PenguinStokes.RigidBodyState2D
PenguinStokes.RigidBodyState3D
PenguinStokes.RigidBodyParams2D
PenguinStokes.RigidBodyParams3D
PenguinStokes.external_force
PenguinStokes.external_torque
PenguinStokes.StokesFSIProblem
PenguinStokes.step_fsi!
PenguinStokes.step_fsi_strong!
PenguinStokes.simulate_fsi!
PenguinStokes.endtime_static_model
PenguinStokes.embedded_boundary_quantities
PenguinStokes.embedded_boundary_traction
PenguinStokes.embedded_boundary_stress
PenguinStokes.integrated_embedded_force
```
