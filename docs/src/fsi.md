# Rigid-Body FSI

`PenguinStokes.jl` provides rigid-body coupling wrappers around
`MovingStokesModelMono`.

## 1. Core Types and Helpers

Main exported FSI-related symbols:

### Single-Body FSI
- `StokesFSIProblem`, `StokesFSIProblem2D`
- `RigidBodyState`, `RigidBodyState2D`, `RigidBodyState3D`
- `RigidBodyParams`, `RigidBodyParams2D`, `RigidBodyParams3D`
- shapes: `Circle`, `Sphere`, `Ellipse`, `Ellipsoid`
- motion helpers: `rigid_boundary_velocity`, `rigid_cut_bc_tuple`,
  `rigid_body_levelset`

### Multi-Body FSI
- `MultiBodyFSIProblem`
- helper: `multi_body_levelset`, `multi_body_cut_bc_tuple`

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

## 7. Multi-Body FSI (`MultiBodyFSIProblem`)

Multi-body FSI allows N rigid bodies to share a single fluid solve.  Each body
retains its own state and parameters; the fluid domain sees a union level-set
(max of all body SDFs) and a BC dispatcher that assigns interface points to the
nearest body.  After the fluid solve, forces are partitioned among bodies by
looping interface cells and assigning each to the body with the largest (least
negative) SDF at the interface centroid.

### Architecture

1. **Shared Level-Set**: `multi_body_levelset(shapes, statefuns)` returns a
   closure for the union of N body SDFs; this becomes the body function in
   `MovingStokesModelMono`.

2. **BC Dispatcher**: `multi_body_cut_bc_tuple(shapes, statefuns, Val(N))`
   returns per-component Dirichlet callbacks that query each body's SDF at the
   interface point and assign the velocity of the nearest body.

3. **Per-Body Force Extraction**: After fluid solve, `_per_body_forces(...)` loops
   all interface cells, assigns each to the body with the largest SDF at the
   cell's centroid, and accumulates force and torque per body independently.

4. **Body Stepping**: Each body's ODE is advanced separately using its extracted
   hydrodynamic load, so bodies evolve independently even though the fluid
   coupling is shared.

### Stepping Function

```julia
step_multi_fsi!(fsi; t, dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
```

1. Predict each body's state over slab [t, t+dt],
2. set combined level-set and BC dispatcher,
3. solve shared fluid problem,
4. partition forces among bodies by interface-centroid proximity,
5. advance each body ODE independently.

Returns `(sys=..., t=..., states=..., forces=...)`, one result per body.

### Key Features

- **No contact model**: bodies must be sufficiently resolved by the mesh to avoid
  overlap.  No lubrication correction is implemented; mesh resolution handles
  near-contact hydrodynamic forces.
- **Arbitrary N bodies**: works with 2, 3, or more bodies in 2D or 3D.
- **Symmetric / asymmetric**: each body can have different shape, mass, inertia.

## 8. Scope and Limitations

Implemented:

- single rigid body,
- multiple rigid bodies (shared fluid, per-body force partition),
- 2D and 3D rigid states,
- rotational state evolution (2D and 3D),
- split and strong coupling variants (single-body only),
- multi-body split coupling.

Current limits:

- no contact/collision model (mesh resolution required),
- no lubrication correction,
- no deformable-body model.

## 9. Minimal Usage Sketches

### Single-Body FSI

```julia
model = MovingStokesModelMono(grid, body, mu, rho; bc_u=(bc, bc), bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)))
state = RigidBodyState2D((0.5, 0.5), (0.0, 0.0); theta=0.0, omega=0.0)
params = RigidBodyParams2D(1.0, 1.0, Circle(0.1), SVector(0.0, -9.81))
fsi = StokesFSIProblem(model, state, params)
out = step_fsi!(fsi; t=0.0, dt=0.01, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
```

### Multi-Body FSI

```julia
shapes = [Sphere(0.1), Sphere(0.1)]
states = [
  RigidBodyState3D(SVector(0.5, 0.4, 0.5), SVector(0.0, 0.0, 0.0)),
  RigidBodyState3D(SVector(0.5, 0.6, 0.5), SVector(0.0, 0.0, 0.0)),
]
params_each = RigidBodyParams3D(mass, inertia, density, shapes[1], gravity_vec; rho_fluid=rho_f)
params = [params_each, params_each]

statefuns0 = [s -> states[1], s -> states[2]]
model = MovingStokesModelMono(
  grid,
  multi_body_levelset(shapes, statefuns0),
  mu, rho;
  bc_u=bcs,
  bc_cut_u=multi_body_cut_bc_tuple(shapes, statefuns0, Val(3)),
)

fsi = MultiBodyFSIProblem(model, states, params, shapes; pressure_reconstruction=:linear)
out = step_multi_fsi!(fsi; t=0.0, dt=0.01, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
# out.states contains per-body ODE results; out.forces contains per-body force/torque
```
