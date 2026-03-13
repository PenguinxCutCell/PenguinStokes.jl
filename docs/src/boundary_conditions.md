# Boundary Conditions and Gauges

This page documents boundary-condition semantics and consistency rules in the
current implementation.

## 1. Outer Velocity BCs (`bc_u`)

`bc_u` is provided component-wise as `NTuple{N,BorderConditions}`.

### Dirichlet

- Physical meaning: prescribed velocity component.
- Row effect: momentum-row elimination or distance-based weak closure depending
  on staggered wall collocation and `strong_wall_bc`.

### Neumann

- Physical meaning: prescribed normal derivative/flux of velocity component.
- Row effect: RHS flux contribution on boundary-adjacent momentum rows.

### Periodic

- Physical meaning: periodic domain in that axis.
- Row effect: periodic operator construction; no explicit wall overwrite row.

### Traction / PressureOutlet / DoNothing

- Physical meaning: side-level Stokes traction law.
- Row effect: side-based overwrite of boundary momentum rows including pressure
  coupling and symmetric-gradient cross terms.
- Must be declared on **all** velocity components on that side.

### Symmetry

- Physical meaning: axis-aligned free-slip/symmetry wall.
- Row effect: normal velocity constrained to zero, tangential traction rows set
  to homogeneous condition.
- Must be declared on **all** velocity components on that side.

## 2. Pressure Handling (`bc_p` + Gauge)

- `bc_p` is optional (`nothing` by default).
- When present, pressure wall rows can enforce `Dirichlet`/`Neumann`/`Periodic`
  wall constraints.
- Gauge is still required (`PinPressureGauge` or `MeanPressureGauge`).

Compatibility restrictions (enforced):

- if a side uses traction-family velocity BC (`Traction`, `PressureOutlet`,
  `DoNothing`) or `Symmetry`, `bc_p` on that side is not allowed,
- side-level vector BCs are incompatible with periodicity in that axis.

## 3. Cut/Interface Velocity BCs

### Monophasic cut BC (`bc_cut`)

- Current support: `Dirichlet` only.
- `Neumann` and `Periodic` are explicitly rejected (`ArgumentError`).

### Moving model cut BC (`bc_cut_u`)

- Per-component tuple of cut Dirichlet BCs applied at end-time in
  `assemble_unsteady_moving!`.
- Used directly by rigid-body FSI wrappers via `rigid_cut_bc_tuple`.

### Two-phase interface

- `StokesModelTwoPhase` does not use a separate `bc_cut` argument.
- Interface behavior is encoded through shared `ugamma` and traction/interface
  force rows.

## 4. Consistency Rules Summary

Implemented checks enforce:

- traction-type BC must be set on all components of a side,
- symmetry must be set on all components of a side,
- no mixing symmetry with traction-type BC on same side,
- pressure BC on a traction/symmetry side is disallowed,
- gauge remains required.

## 5. Compact Examples

### No-slip mono with pin gauge

```julia
bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0), bottom=Dirichlet(0.0), top=Dirichlet(0.0))
model = StokesModelMono(grid, body, 1.0, 1.0; bc_u=(bc, bc), bc_cut=Dirichlet(0.0), gauge=PinPressureGauge())
```

### Pressure outlet side law

```julia
bcx = BorderConditions(; left=Dirichlet(1.0), right=PressureOutlet(0.0), bottom=Dirichlet(0.0), top=Dirichlet(0.0))
bcy = BorderConditions(; left=Dirichlet(0.0), right=PressureOutlet(0.0), bottom=Dirichlet(0.0), top=Dirichlet(0.0))
model = StokesModelMono(grid, body, mu, rho; bc_u=(bcx, bcy), bc_cut=Dirichlet(0.0))
```

### Unsupported cut Neumann (expected to throw)

```julia
model = StokesModelMono(grid, body, 1.0, 1.0; bc_u=(bc, bc), bc_cut=Neumann(0.0))
solve_steady!(model)  # throws ArgumentError
```
