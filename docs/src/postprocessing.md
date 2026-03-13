# Postprocessing

Embedded-boundary diagnostics are available for `StokesModelMono` states/solutions.

## 1. `embedded_boundary_quantities`

```julia
embedded_boundary_quantities(model, x; mu=model.mu, pressure_reconstruction=:linear, x0=nothing)
```

Returns named tuple fields:

- `stress`: stress tensor at pressure-grid interface cells,
- `traction`: traction vector at interface cells,
- `force_density`: `traction * Gamma` per interface cell,
- `interface_indices`: active interface cell indices,
- `force`: integrated total force,
- `force_pressure`: integrated pressure contribution,
- `force_viscous`: integrated viscous contribution,
- `torque`: 2D scalar or 3D vector about `x0`.

`pressure_reconstruction` options:

- `:none`: use cell pressure value,
- `:linear`: linear trace reconstruction with pressure gradient.

## 2. Convenience Wrappers

- `embedded_boundary_traction(model, x; kwargs...)`
- `embedded_boundary_stress(model, x; kwargs...)`
- `integrated_embedded_force(model, x; kwargs...)`

Each also accepts `sys::LinearSystem` and uses `sys.x`.

## 3. Sign/Interpretation Notes

- Traction is computed as `sigma * n` with interface normal from capacity data.
- `force` is the integrated traction force on the embedded boundary.
- Split outputs satisfy `force ≈ force_pressure + force_viscous`.
- Torque is evaluated about `x0` (defaults to origin).

## 4. Usage Snippet

```julia
sys = solve_steady!(model)
q = embedded_boundary_quantities(model, sys; pressure_reconstruction=:linear, x0=(0.5, 0.5))
println("F_total = ", q.force)
println("F_pressure = ", q.force_pressure)
println("F_viscous = ", q.force_viscous)
println("Torque = ", q.torque)
```

## 5. Current Scope

Implemented for `StokesModelMono` only in current API.
