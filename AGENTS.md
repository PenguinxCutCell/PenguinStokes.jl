# PenguinStokes.jl — Implementation Reference for AI Assistants

This document describes the architecture, key data structures, assembly algorithms, and known gotchas in PenguinStokes.jl. It is intended for AI coding assistants working on this codebase.

---

## Package Overview

PenguinStokes.jl implements cut-cell finite-volume Stokes solvers on staggered MAC grids, supporting:
- Steady and unsteady monophasic flow around embedded boundaries
- Fixed-interface two-phase flow
- Moving embedded-boundary flow (prescribed motion, unsteady)
- Moving two-phase flow (prescribed interface motion)
- Fluid-structure interaction (FSI) with rigid bodies

Source is split across `src/`:
- `src/PenguinStokes.jl` — module header, `using`, `export`, and `include` calls only
- `src/types.jl` — pressure gauge types, layout structs (`StokesLayout`, `StokesLayoutTwoPhase`), all model structs
- `src/validation.jl` — BC normalization helpers, `_validate_stokes_*`, `staggered_velocity_grids`, `_side_pairs`
- `src/force.jl` — `_force_values`, `_interface_force_component/vector` overloads
- `src/activity.jl` — `_cell_activity_masks`, `_pressure_activity`, `_stokes_row_activity`, `_prune_uncoupled_active!`, sparse-insert helpers
- `src/bc_velocity.jl` — `_apply_row_identity_constraints!`, `_enforce_dirichlet!`, all traction/symmetry/velocity box BC application
- `src/bc_pressure.jl` — `_apply_pressure_box_bc!`, `_apply_pressure_gauge!`, `_apply_pin_pressure_gauge!`, `_apply_mean_pressure_gauge!`
- `src/moving_geometry.jl` — `_theta_from_scheme`, `_psi_functions`, `_build_moving_slab!`, `reduce_slab_to_space`, `_expand_prev_state`
- `src/assembly.jl` — `_stokes_blocks`, `_assemble_core!`, `_assemble_interface_traction_rows!`, `_apply_auxiliary_trace_rows!`, `assemble_steady!`, `assemble_unsteady!`, `assemble_unsteady_moving!`, all `solve_*` functions
- `src/analysis.jl` — `build_static_circle_equilibrium_state`, residual/pressure diagnostics, `embedded_boundary_*`, `integrated_embedded_force`
- `src/constructors.jl` — `StokesModelMono`, `MovingStokesModelMono`, `StokesModelTwoPhase`, `MovingStokesModelTwoPhase` constructors
- `src/rigidbody.jl` — rigid body state/params/shapes for FSI
- `src/fsi.jl` — FSI problem wrapper and split coupling
- `src/fsi_strong_coupling.jl` — strong (monolithic) FSI coupling
- `src/fsi_multibody.jl` — multi-body FSI (`MultiBodyFSIProblem`, `step_multi_fsi!`)
- `src/orientation.jl` — rotation/quaternion utilities

---

## External Dependencies

From `CartesianOperators.jl` (the most important dependency):
- `AssembledCapacity{N,T}` — cut-cell geometric data (volumes, face areas, interface areas, cell centers). Key fields: `.buf.V` (volumes as Vector), `.buf.A[d]` (face areas), `.ntotal`, `.nnodes`.
- `DiffusionOps{N,T}` — discrete differential operators on cut cells. Key fields: `.G` (gradient/flux matrix), `.H` (interface correction matrix), `.Winv` (inverse face-weight matrix). Both `G` and `H` map omega DOFs to faces.
- `assembled_capacity(grid, body; ...)` — constructs `AssembledCapacity`
- `DiffusionOps(cap; ...)` — constructs `DiffusionOps` from a capacity

From `CartesianGrids.jl`:
- `CartesianGrid{N,T}` — N-D grid with fields `lc` (lower corner), `hc` (upper corner), `n` (node counts per dim)
- `meshsize(grid)` → NTuple of cell sizes `h`
- `grid1d(lc, hc, n)` — 1D grid constructor

From `PenguinBCs.jl`:
- `BorderConditions` — BC tuple keyed by side (`:left`, `:right`, `:bottom`, `:top`, `:front`, `:back`)
- `InterfaceConditions` — holds `ScalarJump` (velocity jump) and `FluxJump` (traction jump)
- BC types: `Dirichlet`, `Neumann`, `Periodic`, `Traction`, `PressureOutlet`, `DoNothing`, `Symmetry`

From `PenguinSolverCore.jl`:
- `LinearSystem{T}` — holds `A` (sparse matrix), `b` (rhs vector), `x` (solution), and solver cache
- `solve!(sys)` — factorizes and solves in-place

---

## Staggered MAC Grid

**`staggered_velocity_grids(gridp)` (line 496)**

Creates N velocity grids from the pressure grid, each shifted by `-h[d]/2` in dimension `d`:

```julia
gridu[d] = CartesianGrid(lc .- h/2 * e_d, hc .- h/2 * e_d, gridp.n)
```

Both grids have **identical `ntotal = prod(n)`**. At linear index `i`, the pressure cell center is at `x_p[i]` and the velocity-d cell center is at `x_p[i] - h[d]/2`. This offset is critical when comparing pressure and velocity indices geometrically.

---

## Model Structs

### `StokesLayout{N}` (line 61)

DOF index ranges for monophasic systems. All ranges have length `nt = prod(n)`.

```
Unknown vector: [uomega_1 | ugamma_1 | ... | uomega_N | ugamma_N | pomega]
```

- `layout.uomega[d]` — bulk velocity component d
- `layout.ugamma[d]` — interface trace velocity component d  
- `layout.pomega` — pressure

### `StokesLayoutTwoPhase{N}` (line 74)

DOF index ranges for two-phase systems.

```
Unknown vector: [uomega1_1..N | uomega2_1..N | ugamma1_1..N | ugamma2_1..N | pomega1 | pomega2]
```

- `layout.ugamma` is an alias for `layout.ugamma1` (via `getproperty` override at line 84)

### `StokesModelMono{N,T,FT,BT}` (line 149)

Monophasic model. Key fields:
- `gridp, gridu[N]` — pressure and staggered velocity grids
- `cap_p, cap_u[N]` — geometric capacities
- `op_p, op_u[N]` — diffusion operators
- `mu, rho, force` — physics
- `bc_u[N], bc_p, bc_cut[N]` — boundary conditions
- `gauge::AbstractPressureGauge` — `PinPressureGauge` or `MeanPressureGauge`
- `body` — level-set function `body(x...) -> signed_distance`

### `StokesModelTwoPhase{N,T,...}` (line 180)

Fixed-interface two-phase model. Parallel fields for phase 1 (`cap_p1, op_p1, cap_u1, op_u1`) and phase 2. Additional:
- `interface_force` — traction jump callback
- `interface_jump` — velocity jump callback
- `bc_interface[N]` — `InterfaceConditions` per direction

### `MovingStokesModelMono{N,T,...}` (line 219)

Unsteady monophasic with prescribed body motion. Adds time-dependent caches (all `Union{Nothing,...}`, populated by `_build_moving_slab!`):
- `cap_p_slab, op_p_slab` — space-time slab capacity/operators for pressure
- `cap_p_end, op_p_end` — end-time (`t+dt`) capacity/operators for pressure
- `cap_u_slab[N], op_u_slab[N]` — slab velocity operators
- `cap_u_end[N]` — end-time velocity capacities
- `Vun[N], Vun1[N]` — velocity volumes at `t^n` and `t^{n+1}`
- `bc_cut_u[N]` — moving-boundary Dirichlet BC (wall velocity)

### `MovingStokesModelTwoPhase{N,T,...}` (line 251)

Unsteady two-phase with prescribed interface motion. Parallel slab/end-time caches for both phases:
- `cap_p1_slab/end, op_p1_slab/end, cap_u1_slab/end[N], op_u1_slab/end[N], Vu1n[N], Vu1n1[N]`
- Same for phase 2 with `p2`, `u2`, `Vu2`
- `check_interface::Bool` — enable interface consistency checks in slab build

---

## Slab vs End-Time Distinction

**This is the core of the space-time formulation.** Every moving-model time step maintains two geometric snapshots:

| Snapshot | When | Used for |
|----------|------|----------|
| **Slab** | `∫_{t}^{t+dt}` (time-integrated) | Momentum equation, divergence constraint |
| **End-time** | `t + dt` (point value) | Boundary conditions, pressure gauge, traction rows |

Slab volumes: `cap_p_slab.buf.V[i] ≈ h^N * dt` for full interior cells (since `V_slab = ∫ V(s) ds`).  
End-time volumes: `cap_p_end.buf.V[i] ≈ h^N` for full interior cells.

**Consequence**: any threshold on slab volumes must scale with `dt`, not just `h^N`.

---

## Key Algorithms

### `_build_moving_slab!(model, t, dt)` (line ~2542)

Called at the start of each time step. Evaluates the level-set body at `t` and `t+dt`, constructs space-time slab moments via `SpaceTimeCartesianGrid`, and populates all `_slab` and `_end` caches in the model.

### `assemble_unsteady_moving!(sys, model, x_prev, t, dt; scheme)` (line ~3475)

Main assembly for moving models. Steps:
1. `_build_moving_slab!(model, t, dt)` — populate geometric caches
2. Extract `op_p_slab`, `op_p_end`, `cap_u_slab`, etc.
3. Assemble gradient: `grad[d] = -(op_p_slab.G + op_p_slab.H)[active_rows, :]`
4. Assemble divergence (transpose): `div = -(op_p_slab.G' + op_p_slab.H')`
5. Assemble momentum: stiffness `K = mu*(G'*Winv*G)`, mass `M = diag(rho*V)`, temporal weights `Ψ`
6. `ugamma` rows: identity tie for mono (Dirichlet wall velocity); traction balance for two-phase
7. Apply box BCs, pressure gauge
8. `active = _stokes_row_activity(model, A)` — mark coupled DOFs
9. `_apply_row_identity_constraints!(A, b, active)` — deactivate uncoupled rows

**Critical**: grad and div must use **the same operator** (`op_p_slab`) for saddle-point consistency (`div = -grad^T`). Using `op_p_end` for div while using `op_p_slab` for grad breaks this.

### `_stokes_row_activity(model, A)` (line ~883 mono, ~1040 two-phase)

Returns `BitVector` of which rows are "active" (coupled). Inactive rows (no fluid volume, halo cells, isolated DOFs) get replaced by identity equations.

**Pressure activity for moving models** (line ~996 mono, ~1069 two-phase):
```julia
h = minimum(meshsize(model.gridp))
eps_cut = 1e-3
min_vol = eps_cut * h^N          # ← compared against slab volumes ~ h^N * dt
v = cap_p_slab.buf.V[i]
pactive[i] = isfinite(v) && v >= min_vol
```

**Known bug**: `min_vol = eps_cut * h^N` but slab volumes scale as `h^N * dt`. With `dt = cdt * h^2`, full interior cells have slab volume `≈ h^(N+2)`, which is smaller than `h^N` for fine meshes (`h < sqrt(eps_cut / cdt)`). This causes full interior pressure cells to be incorrectly deactivated at fine meshes, producing blow-ups and negative convergence orders. The correct threshold is `eps_cut * h^N * dt` or `eps_cut * h^(N+2)` (for `cdt ~ 1`).

**Two-phase `ugamma` activity** (line ~1063):
```julia
has_gamma = (agamma1[i] || agamma2[i]) && agamma_p[i]
```
`agamma_p` checks if the **pressure** cell at index `i` has cut interface area, while `agamma1/2` check **velocity** cells at index `i` (which are at a different physical position due to h/2 staggering). This can miss interface DOFs when the interface crosses the velocity cell but not the pressure cell at the same linear index.

### `_prune_uncoupled_active!(active, A)` (line ~858)

Iterates: if `active[j]` has no nonzero coupling to other active rows, deactivate `j`. Converges to the minimal coupled subsystem. Prevents isolated pressure islands from causing singular systems.

### `_apply_row_identity_constraints!(A, b, active)` (line ~1132)

Replaces all rows where `active[i] = false` with `A[i,:] = 0; A[i,i] = 1; b[i] = 0`. Keeps the system square with a well-defined solution (0) for inactive DOFs.

### `_assemble_interface_traction_rows!` (line ~3114 two-phase)

Assembles the interface traction balance into `ugamma1` rows and velocity jump into `ugamma2` rows:
- `ugamma1[d]` row: stress balance `σ1·n - σ2·n = interface_force`
- `ugamma2[d]` row: jump condition `α1*ugamma1[d] - α2*ugamma2[d] = interface_jump`

Uses **end-time** operators (`cap_p_end`, `op_p_end`) for the traction assembly, not slab operators.

### `_apply_pressure_gauge!` (line ~2383)

- `PinPressureGauge(index=i)`: replaces row `pomega[i]` with `p[i] = 0`
- `MeanPressureGauge()`: replaces one row with `(1/vol) * sum(V_i * p_i) = 0`
- For two-phase moving: only gauges `pomega1`; `pomega2` is constrained through traction rows

---

## DOF Extraction Pattern

```julia
# After solve
u1 = sys.x[model.layout.uomega[1]]   # x-velocity (mono)
u2 = sys.x[model.layout.uomega[2]]   # y-velocity (mono)
p  = sys.x[model.layout.pomega]       # pressure (mono)

# Two-phase
u1_phase1 = sys.x[model.layout.uomega1[1]]
p_phase2  = sys.x[model.layout.pomega2]
```

---

## Pressure Gauge Types

| Type | Description |
|------|-------------|
| `PinPressureGauge(; index=nothing)` | Sets `p[index] = 0` (or first active cell if index=nothing) |
| `MeanPressureGauge()` | Enforces `∫ p dV = 0` over active cells |

---

## Examples

| File | Description |
|------|-------------|
| `01_mms_box.jl` | Steady MMS, box domain (no embedded boundary) |
| `04_mms_convergence.jl` | Steady MMS convergence with circle obstacle |
| `08_two_phase_mms_fixed_interface.jl` | Fixed-interface two-phase MMS |
| `13_unsteady_moving_body_translation.jl` | Moving body translation |
| `25_moving_mms_time_schemes.jl` | Moving MMS with BE/CN scheme comparison |
| `29_moving_two_phase_mms_time_schemes.jl` | Moving two-phase MMS, time-scheme comparison |
| `30_moving_two_phase_mms_mesh_convergence.jl` | Moving two-phase MMS mesh convergence study |
| `17_fsi_free_falling_circle.jl` | FSI: free-falling circle |
| `20_fsi_falling_rotating_ellipse.jl` | FSI: falling + rotating ellipse |

---

## MMS Workflow

1. Define exact `u(x,t)`, `p(x,t)`, `f = rho*Du/Dt - mu*Δu + ∇p` analytically
2. Construct model with `force=f` and Dirichlet/periodic BCs matching `u`
3. Time-step loop: `solve_unsteady_moving!(model, x_prev; t, dt, scheme)`
4. Compute L2 error on active cells: `sqrt(sum(V .* (u_num - u_exact)^2) / sum(V))`
5. Check convergence rate over mesh levels

---

## Known Issues / Gotchas

1. **Slab volume threshold scales incorrectly** (`_stokes_row_activity` for moving models, line ~1001 and ~1072): `min_vol = 1e-3 * h^N` is compared against slab volumes `~ h^N * dt`. At fine meshes with `dt ~ h^2`, full cells are deactivated. Fix: use `min_vol = eps_cut * h^N * dt` or `eps_cut * h^(N+2)`.

2. **`agamma_p` index mismatch** (two-phase `_stokes_row_activity`, line ~1063): velocity grids are shifted by `h/2` from pressure grid. At the same linear index, they correspond to different physical locations. The check `agamma_p[i]` tests the pressure cell at index `i`, not the velocity cell. Can miss interface DOFs when staggering places them at different indices.

3. **Saddle-point consistency**: `grad` and `div` operators must derive from the same underlying operator (`op_p_slab`) or the system is not symmetric/consistent. Do not mix `op_p_slab` for grad with `op_p_end` for div.

4. **Two-phase pressure gauge**: only `pomega1` is explicitly gauged. `pomega2` is determined only through the traction coupling. If the interface traction rows are degenerate (zero `gamma`), `pomega2` is undetermined.

5. **Moving-interface two-phase is incomplete**: `MovingStokesModelTwoPhase` exists but convergence with prescribed motion is not fully fixed. The examples `29_*.jl` and `30_*.jl` are test cases for ongoing work.

6. **Halo nodes are always inactive**: both pressure and velocity grids have one layer of halo nodes (last index in each dimension). These are always set to `pactive[i] = false`.

7. **`bc_cut` must be Dirichlet**: the assembly assumes no-slip or prescribed-velocity BCs on the embedded boundary. Neumann cut BCs are not implemented.

---

## Typical Solve Flow

```julia
# Steady monophasic
model = StokesModelMono(gridp, body, mu, rho; bc_u=..., bc_cut=Dirichlet(0), force=f)
sys = solve_steady!(model)

# Unsteady moving
model = MovingStokesModelMono(gridp, body, mu, rho; bc_u=..., bc_cut_u=wall_vel, force=f)
x = zeros(T, nunknowns(model.layout))
for (t, dt) in timesteps
    sys = solve_unsteady_moving!(model, x; t=t, dt=dt, scheme=:CN)
    x = sys.x
end
```

---

## Investigation Note: Moving Two-Phase MMS Trace-Activity Bug

Date: 2026-04-17

Context: `examples/30_moving_two_phase_mms_mesh_convergence.jl` had erratic BE convergence and CN blow-ups for the moving two-phase MMS with
`u(x,t) = (sin(2πt), 0)`, constant pressure, and a translating circular interface with interface speed `x_c'(t) = sin(2πt)`.

Main result: the analytical MMS is consistent. The bug was not a traction sign flip, not missing `mu` in the jump, and not primarily the pressure gauge. For constant velocity and zero pressure, the raw end-time interface derivative blocks satisfy

```julia
deriv_omega * ones + deriv_gamma * ones ≈ 0
```

to roundoff. However, after `_apply_row_identity_constraints!`, active traction rows lost stencil columns for neighboring trace DOFs. This made a constant velocity produce artificial interface traction of order `1e-1` in the `ugamma1` rows. The linear system then amplified that artificial traction through a pressure-dominated near-null mode, especially for CN.

Root cause: `_stokes_row_activity(model::MovingStokesModelTwoPhase, A)` used

```julia
has_gamma = (agamma1[i] || agamma2[i]) && agamma_p[i]
```

where `agamma1/agamma2` are velocity-grid interface masks, but `agamma_p` is the pressure-grid interface mask at the same linear index. Because MAC velocity grids are staggered by `h/2`, a velocity trace DOF can be required by an active traction stencil even when the pressure cell at the same index has no interface. Those trace columns were zeroed, breaking constant preservation.

Implemented change in `src/activity.jl` (row activity) and `src/assembly.jl` (auxiliary trace rows):

- Moving two-phase `ugamma` activity now keeps velocity-grid trace DOFs active when the velocity cut mask is active and there is a local bulk reference.
- Added `_apply_auxiliary_trace_rows!` for moving two-phase. For velocity-grid trace DOFs that are required by velocity stencils but do not have a pressure-grid traction row, it replaces the `ugamma1` row with a volume-weighted trace-extension equation tying `ugamma1[d][i]` to local active bulk velocities. The existing `ugamma2` jump row still enforces `ugamma1 = ugamma2`.
- Called `_apply_auxiliary_trace_rows!` immediately after `_assemble_interface_traction_rows!` in moving two-phase assembly.

Diagnostic evidence:

- Before the fix, plugging the exact constant velocity into the pruned `n=13` CN matrix gave `exact_nonid_res ≈ 5.7e-2`, with the residual concentrated in `ugamma1` traction rows and a smallest singular value around `1.6e-8`.
- The raw traction derivative on constants was roundoff-zero; applying the active mask alone produced artificial traction `O(1)`, proving the error came from activity pruning, not from the derivative operator itself.
- After the fix, the same `n=13` CN exact residual dropped to `O(1e-7)` to `O(1e-6)` over steps, pressure magnitudes stayed small, and the solution norm remained physical.

Verification after the fix using `examples/30_moving_two_phase_mms_mesh_convergence.jl`:

- BE is stable on `(9, 13, 17, 25, 33, 49)`. Ignoring the coarse `n=9` geometry outlier, observed L2 orders are roughly `0.83, 1.97, 1.52, 1.94`.
- CN no longer has the original catastrophic blow-ups. Results are stable at `n=13`, `n=17`, `n=33`, and `n=49`, with `n=49` L2 error about `8.5e-7` and final `33->49` L2 order about `1.99`.
- There remains a CN-only geometry outlier at `n=25` (`l2_err ≈ 2.5e-2`) that grows over time but recovers at finer meshes. This appears to be a remaining near-null algebraic mode, not the original constant-traction preservation bug.
