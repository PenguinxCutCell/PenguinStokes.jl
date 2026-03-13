# Rigid-Body FSI

`PenguinStokes.jl` provides rigid-body coupling wrappers around
`MovingStokesModelMono`.

## 1. Core Types and Helpers

Main exported FSI-related symbols:

- `StokesFSIProblem`, `StokesFSIProblem2D`
- `RigidBodyState`, `RigidBodyState2D`, `RigidBodyState3D`
- `RigidBodyParams`, `RigidBodyParams2D`, `RigidBodyParams3D`
- shapes: `Circle`, `Sphere`, `Ellipse`, `Ellipsoid`
- motion helpers: `rigid_boundary_velocity`, `rigid_cut_bc_tuple`,
  `rigid_body_levelset`

## 2. Governing ODEs

Translation:

```math
m \frac{dV}{dt} = F_{hyd} + F_{ext}, \qquad \frac{dX}{dt} = V.
```

Rotation:

```math
I \frac{d\omega}{dt} = T_{hyd} + T_{ext}
```

(with scalar `I, omega` in 2D and scalar/tensor inertia with vector angular
velocity in 3D).

Rigid boundary velocity:

- 2D: `u_b(x) = V + omega * k × (x - X)`
- 3D: `u_b(x) = V + Omega × (x - X)`

## 3. Split Coupling (`step_fsi!`)

`step_fsi!` performs one partitioned loose-coupling update:

1. predict rigid state over slab,
2. solve moving-boundary fluid step,
3. evaluate hydrodynamic force/torque from end-time static model,
4. advance rigid ODE state (`:symplectic_euler` or `:forward_euler`).

`simulate_fsi!` repeats this loop and stores compact diagnostics.

## 4. Rotation Aliases

- `step_fsi_rotation!`
- `simulate_fsi_rotation!`

These are 2D aliases specialized for `RigidBodyState2D` workflows.

## 5. Strong Coupling (`step_fsi_strong!`)

Strong partitioned coupling is implemented via fixed-point iterations:

- iterate between fluid solve and rigid update at `t_{n+1}`,
- optional relaxation: `:none`, `:constant`, `:aitken`,
- convergence monitored by state-update residual,
- optional non-converged return path (`allow_nonconverged=true`).

## 6. Force/Torque Evaluation Path

Hydrodynamic loads are computed from:

1. `solve_unsteady_moving!` output,
2. `endtime_static_model(model)` reconstruction,
3. `integrated_embedded_force(...)` with selected pressure reconstruction.

This path is used by both split and strong coupling variants.

## 7. Scope and Limitations

Implemented:

- single rigid body,
- 2D and 3D rigid states,
- rotational state evolution (2D and 3D),
- split and strong coupling variants.

Current limits:

- no contact/collision model,
- no multi-body coupling,
- no deformable-body model.

## 8. Minimal Usage Sketch

```julia
model = MovingStokesModelMono(grid, body, mu, rho; bc_u=(bc, bc), bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)))
state = RigidBodyState2D((0.5, 0.5), (0.0, 0.0); theta=0.0, omega=0.0)
params = RigidBodyParams2D(1.0, 1.0, Circle(0.1), SVector(0.0, -9.81))
fsi = StokesFSIProblem(model, state, params)
out = step_fsi!(fsi; t=0.0, dt=0.01, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
```
