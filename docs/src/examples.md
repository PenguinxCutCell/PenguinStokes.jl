# Examples and Verification

The repository ships runnable scripts in `examples/` plus regression tests in
`test/`. This page groups them by validation purpose.

## 1. How to Run

From repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Representative examples:

```bash
julia --project=. examples/04_mms_convergence.jl
julia --project=. examples/08_two_phase_mms_fixed_interface.jl
julia --project=. examples/27_two_phase_static_circle_spurious_current.jl
julia --project=. examples/14_unsteady_oscillating_cylinder.jl
julia --project=. examples/22_3d_falling_rigid_sphere_split_vs_strong.jl
julia --project=. examples/31_periodic_cylinder_array.jl
julia --project=. examples/37_fsi_cylinder_shear_flow_rotation.jl
julia --project=. examples/39_two_spheres_couette.jl
```

Developer probes retained intentionally:

```bash
julia --project=. test/dev_pressure_outlet_mms.jl
julia --project=. test/dev_outlet_row_equivalence.jl
julia --project=. test/dev_traction_box_probe.jl
```

## 2. Monophasic MMS and Convergence

### Scripts

- `examples/01_mms_box.jl`
- `examples/04_mms_convergence.jl`
- `examples/05_mms_convergence_zero_pressure.jl`
- `examples/24_mms_convergence_suite.jl` (multi-case suite)

### What this validates

- steady mono assembly consistency,
- pressure-gradient/divergence coupling,
- convergence behavior under different BC scenarios,
- gauge-aware pressure error handling.

## 3. Embedded-Boundary MMS and Obstacle Cases

### Scripts

- `examples/02_mms_circle_obstacle.jl`
- `examples/03_flow_past_circle.jl`
- `examples/06_mms_convergence_embedded_outside_circle.jl`

### What this validates

- cut-cell geometry coupling,
- cut Dirichlet trace enforcement,
- obstacle-style embedded-boundary behavior.

## 4. Two-Phase Fixed-Interface Validations

### Scripts

- `examples/08_two_phase_mms_fixed_interface.jl`
- `examples/27_two_phase_static_circle_spurious_current.jl`
- `examples/09_two_phase_planar_couette.jl`
- `examples/10_two_phase_planar_poiseuille.jl`
- `examples/10_bis_two_phase_planar_couette_poiseuille.jl`
- `examples/11_two_phase_oscillatory_couette.jl`
- `examples/12_two_phase_viscous_drop_drag.jl`

### What this validates

- traction/interface-force row path,
- shared interface trace behavior,
- pressure-jump handling,
- static-circle snapshot plotting (velocity arrows + pressure map) and Laplace sweep diagnostics (`La = rho*D*sigma/mu^2`, `Ca = ||u||_inf*mu/sigma`),
- layered profile reproduction (Couette/Poiseuille),
- oscillatory response, fixed-drop drag trends.

## 5. Unsteady and Moving-Boundary Monophasic

### Scripts

- `examples/07_unsteady_sphere_drag.jl`
- `examples/13_unsteady_moving_body_translation.jl`
- `examples/14_unsteady_oscillating_cylinder.jl`
- `examples/25_moving_mms_time_schemes.jl`
- `examples/34_moving_cylinder_uniform_flow.jl`

### What this validates

- BE/CN/theta unsteady path,
- moving slab assembly behavior,
- prescribed moving interface velocity path,
- force/torque diagnostics in moving settings,
- moving boundary with zero disturbance (validation of kinematic condition).

## 6. Outlet / Traction / Symmetry Families

### Scripts/tests

- `examples/15_channel_pressure_outlet_traction.jl`
- `examples/16_channel_poiseuille_pressure_outlet.jl`
- `test/dev_outlet_row_equivalence.jl`
- `test/dev_traction_box_probe.jl`
- targeted BC testsets in `test/runtests.jl`

### What this validates

- side-level traction overwrite algebra,
- pressure-outlet and do-nothing specializations,
- symmetry side rules and cross-coupling terms.

## 7. Rigid-Body FSI Families

### Scripts

- `examples/17_fsi_free_falling_circle.jl`
- `examples/18_fsi_prescribed_rotating_cylinder.jl`
- `examples/19_fsi_spin_decay_calibrated.jl`
- `examples/20_fsi_falling_rotating_ellipse.jl`
- `examples/22_3d_falling_rigid_sphere_split_vs_strong.jl`
- `examples/23_fsi_neutral_buoyancy_decay.jl`
- `examples/35_sphere_torque_couette.jl`
- `examples/36_sphere_towards_wall.jl`
- `examples/37_fsi_cylinder_shear_flow_rotation.jl`
- `examples/38_fsi_ballistic_sphere.jl`

### What this validates

- split-coupled rigid-body updates,
- rotational coupling and torque response,
- strong-coupling fixed-point iterations,
- 3D rigid-sphere coupling behavior and split-vs-strong comparison,
- torque on fixed/moving bodies under shear (validation vs Stokes formula),
- lubrication-dominated sphere-wall interaction (κ correction factor),
- free-rotation dynamics in shear flow (Stokes limit),
- ballistic trajectory in gravity field with viscous drag.

## 8. 3D Coverage

### Scripts

- `examples/07_unsteady_sphere_drag.jl`
- `examples/12_two_phase_viscous_drop_drag.jl`
- `examples/21_3d_rigid_sphere_drag.jl`
- `examples/22_3d_falling_rigid_sphere_split_vs_strong.jl`
- `examples/32_periodic_sphere_array.jl`
- `examples/35_sphere_torque_couette.jl`
- `examples/36_sphere_towards_wall.jl`
- `examples/38_fsi_ballistic_sphere.jl`
- `examples/39_two_spheres_couette.jl`

### What this validates

- 3D mono and moving runs,
- 3D fixed-interface two-phase run,
- 3D rigid-body/FSI helper path,
- 3D hydrodynamic interaction (periodic sphere arrays),
- 3D torque dynamics and lubrication effects,
- 3D multi-body FSI with shared fluid solve.

## 9. Verification Map

| Example / test family | Model family | What it checks | Expected outcome | Status |
|---|---|---|---|---|
| `runtests.jl` mono MMS blocks | Mono steady/unsteady | assembly consistency + convergence trends | residuals small, errors decrease with refinement | Implemented |
| `06_mms_convergence_embedded_outside_circle.jl` | Mono cut-cell | embedded-interface MMS behavior | convergence trend and stable solve | Implemented |
| `08_two_phase_mms_fixed_interface.jl` | Two-phase fixed interface | traction jump path | near-zero velocity with imposed jump | Implemented |
| `27_two_phase_static_circle_spurious_current.jl` + test | Two-phase fixed interface | spurious-current level + Laplace jump | `u` near zero and pressure jump close to theory | Implemented |
| `09/10/11` layered examples | Two-phase fixed interface | Couette/Poiseuille/oscillatory profiles | profile and phase behavior match expectations | Implemented |
| `13/25` moving examples + `moving_boundary_stokes_tests.jl` | Moving mono | slab/end-time moving assembly behavior | stable stepping and expected temporal trends | Implemented |
| outlet/symmetry testsets + `dev_*` probes | Mono BC algebra | traction/outlet/symmetry row semantics | row-equivalence and sign rules hold | Implemented |
| `fsi_tests.jl` + `17..23` examples | Rigid-body FSI | split/rotation/strong/3D helper paths | force/torque trends and convergence sanity checks | Implemented |
| `31_periodic_cylinder_array.jl` | Mono periodic | permeability vs Hasimoto (1959) | 1.5–6% error at n=65 (φ=0.1–0.3) | Implemented |
| `32_periodic_sphere_array.jl` | Mono periodic 3D | permeability vs Hasimoto | 4–10% error at n=17 | Implemented |
| `33_couette_rotating_cylinders.jl` | Mono Couette | analytic u_θ validation | L2=5×10⁻³ at n=128 (2nd-order convergence) | Implemented |
| `34_moving_cylinder_uniform_flow.jl` | Moving mono | kinematic condition (zero disturbance) | velocity error ≈ 7×10⁻¹⁵ (machine precision) | Implemented |
| `35_sphere_torque_couette.jl` | Mono Couette with torque | Stokes formula T=8πμR³(γ̇/2) | 1.5–6% error across n=17,25,33 | Implemented |
| `36_sphere_towards_wall.jl` | Mono lubrication | κ(gap) vs Brenner/Cooley & O'Neill | 2nd-order convergence once gap/h > 1 | Implemented |
| `37_fsi_cylinder_shear_flow_rotation.jl` | FSI 2D | ω_∞ = −γ̇/2 exact for Stokes | 2.2% error at n=65 (Re_d=0.4) | Implemented |
| `38_fsi_ballistic_sphere.jl` | FSI 3D ballistic | Vx exponential decay + Vy terminal-velocity approach | 14–31% error at n=17 (L/d confinement) | Implemented |
| `39_two_spheres_couette.jl` | Multi-body FSI 3D | hydrodynamic interaction / repulsion | gap evolution + reversible orbit (coarse n=25 approach) | Implemented |
| moving two-phase interface | Two-phase moving interface | geometric interface evolution | n/a | Implemented |
| multi-body/contact FSI | Rigid-body FSI | collisions/contact constraints | resolved by mesh, no lubrication correction | Designed |

## 10. Notes on Test vs Script Roles

- `test/runtests.jl` and included files are CI regression coverage.
- `test/dev_pressure_outlet_mms.jl` is a manual developer probe (not included by
  `runtests.jl`).
- `test/dev_outlet_row_equivalence.jl` and `test/dev_traction_box_probe.jl` are
  retained developer probes currently included in `runtests.jl`.
