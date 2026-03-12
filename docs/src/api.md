# API Reference

This page documents the primary exported API and practical constructor notes.

## 1. Pressure Gauges

- `PinPressureGauge(; index=nothing)`: pin one pressure DOF.
- `MeanPressureGauge()`: enforce active-volume weighted zero mean pressure.

Gauge is required for all model families.

## 2. Layout Types

- `StokesLayout{N}` for monophasic/moving mono systems.
- `StokesLayoutTwoPhase{N}` for fixed-interface two-phase systems.

## 3. Model Constructors

### `StokesModelMono`

Required:

- `gridp`, `body`, `mu`, `rho`

Key keywords:

- `bc_u`, `bc_p`, `bc_cut`, `force`, `gauge`, `strong_wall_bc`, `geom_method`

### `MovingStokesModelMono`

Required:

- `gridp`, `body`, `mu`, `rho`

Key keywords:

- `bc_u`, `bc_p`, `bc_cut_u`, `force`, `gauge`, `strong_wall_bc`, `geom_method`

### `StokesModelTwoPhase`

Required:

- `gridp`, `body`, `mu1`, `mu2`

Key keywords:

- `rho1`, `rho2`, `force1`, `force2`, `interface_force`,
  `bc_u`, `bc_p`, `gauge`, `strong_wall_bc`, `geom_method`, `check_interface`

## 4. Assembly and Solve Entry Points

- `assemble_steady!`, `assemble_unsteady!`
- `assemble_unsteady_moving!`
- `solve_steady!`, `solve_unsteady!`, `solve_unsteady_moving!`

All mutate/return `LinearSystem` objects with solution in `.x` after `solve!`.

## 5. Postprocessing

- `embedded_boundary_quantities`
- `embedded_boundary_traction`
- `embedded_boundary_stress`
- `integrated_embedded_force`

Current scope: `StokesModelMono`.

## 6. Rigid-Body and FSI APIs

Rigid-body primitives:

- states: `RigidBodyState`, `RigidBodyState2D`, `RigidBodyState3D`
- parameters: `RigidBodyParams`, `RigidBodyParams2D`, `RigidBodyParams3D`
- shapes: `Circle`, `Sphere`, `Ellipse`, `Ellipsoid`

FSI wrappers/steppers:

- `StokesFSIProblem`, `StokesFSIProblem2D`
- `endtime_static_model`
- `step_fsi!`, `step_fsi_rotation!`, `step_fsi_strong!`
- `simulate_fsi!`, `simulate_fsi_rotation!`

Rigid-motion helper callbacks:

- `rigid_boundary_velocity`, `rigid_velocity_2d`, `rigid_velocity`
- `rigid_body_levelset`, `rigid_cut_bc_tuple`, `rigid_cut_bc_tuple_2d`

## 7. Canonical Docstrings

```@docs
PenguinStokes.AbstractPressureGauge
PenguinStokes.PinPressureGauge
PenguinStokes.MeanPressureGauge
PenguinStokes.StokesLayout
PenguinStokes.StokesLayoutTwoPhase
PenguinStokes.StokesModelMono
PenguinStokes.MovingStokesModelMono
PenguinStokes.StokesModelTwoPhase
PenguinStokes.staggered_velocity_grids
PenguinStokes.assemble_steady!
PenguinStokes.assemble_unsteady!
PenguinStokes.assemble_unsteady_moving!
PenguinStokes.solve_steady!
PenguinStokes.solve_unsteady!
PenguinStokes.solve_unsteady_moving!
PenguinStokes.embedded_boundary_quantities
PenguinStokes.embedded_boundary_traction
PenguinStokes.embedded_boundary_stress
PenguinStokes.integrated_embedded_force
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
PenguinStokes.RigidBodyState2D
PenguinStokes.RigidBodyState3D
PenguinStokes.RigidBodyParams
PenguinStokes.RigidBodyParams2D
PenguinStokes.RigidBodyParams3D
PenguinStokes.external_force
PenguinStokes.external_torque
PenguinStokes.rigid_boundary_velocity
PenguinStokes.rigid_velocity_2d
PenguinStokes.rigid_velocity
PenguinStokes.rigid_body_levelset
PenguinStokes.rigid_cut_bc_tuple
PenguinStokes.rigid_cut_bc_tuple_2d
PenguinStokes.StokesFSIProblem
PenguinStokes.endtime_static_model
PenguinStokes.step_fsi!
PenguinStokes.step_fsi_rotation!
PenguinStokes.step_fsi_strong!
PenguinStokes.simulate_fsi!
PenguinStokes.simulate_fsi_rotation!
```
