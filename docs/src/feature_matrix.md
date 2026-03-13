# Feature Matrix

Status labels used here:

- **Implemented**: present in current source and covered by tests/examples.
- **Partial**: available with notable scope limits.
- **Missing**: not implemented in current package.
- **Experimental**: available but intended as exploratory/developer path.

## 1. Models

| Feature | Status | Notes / evidence |
|---|---|---|
| Steady monophasic Stokes | Implemented | `StokesModelMono`, `assemble_steady!`, `solve_steady!`; core test coverage in `test/runtests.jl` |
| Unsteady monophasic Stokes | Implemented | `assemble_unsteady!`, `solve_unsteady!`; BE/CN/theta paths tested |
| Moving-boundary monophasic Stokes | Implemented | `MovingStokesModelMono`, `assemble_unsteady_moving!`; tests in `test/moving_boundary_stokes_tests.jl` |
| Fixed-interface two-phase Stokes | Implemented | `StokesModelTwoPhase`; geometry/traction/profile tests in `test/runtests.jl` and `examples/08..12` |
| Moving-interface two-phase Stokes | Missing | no two-phase moving-geometry constructor/path in source |
| Rigid-body FSI split coupling | Implemented | `step_fsi!`, `simulate_fsi!`; `test/fsi_tests.jl`, `examples/17..23` |
| Rigid-body FSI strong coupling | Implemented | `step_fsi_strong!`; strong-coupling testset + `examples/22` |
| Rotational rigid-body FSI | Partial | 2D rotational path and 3D rotational state updates are present; practical 3D rotation validation is narrower than 2D |
| Multi-body FSI | Missing | single-body wrapper design |
| Deformable-body FSI | Missing | rigid-body-only scope |

## 2. Dimensionality

| Capability | Status | Notes / evidence |
|---|---|---|
| 2D monophasic | Implemented | extensive tests + examples |
| 3D monophasic | Implemented | examples `07`, `21` |
| 2D two-phase fixed interface | Implemented | tests + examples `08..11` |
| 3D two-phase fixed interface | Partial | example `12` demonstrates path; broad regression coverage is lighter than 2D |
| 2D moving mono | Implemented | tests in `moving_boundary_stokes_tests.jl`, examples `13`, `14`, `25` |
| 3D moving mono | Partial | `07` exercises 3D moving path; narrower validation set than 2D |
| 2D FSI | Implemented | multiple tests/examples |
| 3D FSI | Partial | split/strong sphere examples exist (`21`, `22`), but coverage is less broad than 2D |

## 3. Boundary Conditions

| BC feature | Status | Notes / evidence |
|---|---|---|
| Velocity Dirichlet | Implemented | core assembly + tests |
| Velocity Neumann | Implemented | assembly path + tests |
| Velocity periodic | Implemented | operator periodic flags + examples/tests |
| Traction (vector) | Implemented | row overwrite path + BC regression tests |
| PressureOutlet | Implemented | special traction form + dedicated tests/examples |
| DoNothing | Implemented | zero-traction specialization + tests |
| Symmetry | Implemented | side-level free-slip/symmetry row path + tests |
| Cut/interface Dirichlet velocity | Implemented | `_cut_values` Dirichlet path |
| Cut/interface Neumann velocity | Missing | explicitly rejected (`ArgumentError`) |
| Cut/interface periodic velocity | Missing | explicitly rejected (`ArgumentError`) |

## 4. Pressure / Nullspace Control

| Feature | Status | Notes / evidence |
|---|---|---|
| Pin gauge | Implemented | `PinPressureGauge`, tested row replacement |
| Mean gauge | Implemented | `MeanPressureGauge`, active-volume weighted test |
| Optional side pressure BC (`bc_p`) | Implemented | supported with compatibility checks |
| Pressure BC compatibility rules | Implemented | side-level conflicts with traction/symmetry enforced |

## 5. Postprocessing

| Feature | Status | Notes / evidence |
|---|---|---|
| Embedded boundary stress | Implemented | `embedded_boundary_stress` |
| Embedded boundary traction | Implemented | `embedded_boundary_traction` |
| Integrated force | Implemented | `integrated_embedded_force` |
| Pressure/viscous split | Implemented | returned by `embedded_boundary_quantities` |
| Torque | Implemented | scalar 2D / vector 3D path in postprocessing |
| Postprocessing on two-phase model | Missing | current exported implementation is for `StokesModelMono` |

## 6. Validation Coverage

| Validation family | Status | Notes / evidence |
|---|---|---|
| MMS (mono) | Implemented | `examples/01/04/05/24`, multiple testsets |
| Planar Couette (two-phase) | Implemented | `examples/09`, testset in `runtests.jl` |
| Planar Poiseuille (two-phase) | Implemented | `examples/10`, testset in `runtests.jl` |
| Oscillatory Couette (two-phase) | Implemented | `examples/11` |
| Moving-body verification | Implemented | `examples/13/14/25` + moving tests |
| Drag/force verification | Implemented | postprocessing tests + `examples/12/21` |
| FSI split vs strong comparison | Implemented | `examples/22`, strong-coupling testset |
| 3D sphere runs | Implemented | `examples/07/21/22` |

## 7. Documentation Coverage (This PR)

| Area | Status | Notes |
|---|---|---|
| Theory/models page | Implemented | `stokes.md` |
| Algorithms page | Implemented | `algorithms.md` |
| BC/gauge semantics page | Implemented | `boundary_conditions.md` |
| FSI page | Implemented | `fsi.md` |
| Postprocessing page | Implemented | `postprocessing.md` |
| API reference page | Implemented | `api.md` with expanded symbol coverage |
| Examples/verification map | Implemented | `examples.md` |
| Developer notes | Implemented | `developer_notes.md` |

## Known Gaps / Next Steps

1. Add broader 3D regression coverage for two-phase and rotational FSI paths.
2. Add dedicated multi-body/contact/deformable-body roadmap docs when relevant.
3. If two-phase moving interface is added, expand matrix and examples accordingly.
4. Add optional postprocessing utilities for two-phase models if needed.
