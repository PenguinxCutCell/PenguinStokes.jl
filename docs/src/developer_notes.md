# Developer Notes

This page is maintainer-oriented and summarizes key implementation conventions.

## 1. Layout and Block Conventions

Monophasic layout (`StokesLayout`):

```text
[uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]
```

Two-phase layout (`StokesLayoutTwoPhase`):

```text
[uomega1_1; ...; uomega1_N; uomega2_1; ...; uomega2_N; ugamma_1; ...; ugamma_N; pomega1; pomega2]
```

These global index ranges are intentionally stable across assembly variants.

## 2. Why Identity Regularization Exists

Inactive/halo/uncoupled rows are set to identity with zero RHS to avoid singular
systems while preserving fixed unknown numbering.

This simplifies:

- shared assembly utilities,
- moving-geometry support,
- postprocessing index assumptions.

## 3. Active/Inactive Row Logic

Activity is derived from capacity support (`V`, `Gamma`) and coupling checks.
For moving cases, activity is evaluated on end-time capacities/operators.

Row masks are then applied before solving.

## 4. Pressure Gauge Replacement

Gauge replacement is done by explicit pressure-row overwrite.

- `PinPressureGauge`: one pressure row fixed to a point constraint.
- `MeanPressureGauge`: one pressure row replaced by active-volume weighted mean.

Gauge logic is shared across mono, two-phase, and moving model paths.

## 5. Adding a New Outer BC

Typical steps:

1. Extend BC validation (`_validate_stokes_box_bcs!`).
2. Implement row overwrite/update in velocity or pressure BC appliers.
3. Add testset(s) for algebra/sign/coupling behavior.
4. Update `boundary_conditions.md`, `feature_matrix.md`, and README scope table.

## 6. Adding a New Cut/Interface BC

Typical steps:

1. Extend `_cut_values` and related assembly branch.
2. Ensure compatibility with moving path (`bc_cut_u`) when relevant.
3. Add mono + moving regression coverage.
4. Document support level and restrictions clearly.

## 7. Adding a Postprocessing Utility

Typical steps:

1. Reuse/extend gradient/trace/stress helpers in postprocessing section.
2. Define sign convention and returned tuple fields explicitly.
3. Add unit-style regression tests (force split, symmetry checks, etc.).
4. Document intended model scope (mono/two-phase/moving).

## 8. Moving Geometry Assembly Notes

`assemble_unsteady_moving!` uses slab-reduced geometry between `t_n` and
`t_{n+1}` then applies:

- end-time trace constraints (`bc_cut_u`),
- end-time box BCs,
- end-time gauge,
- end-time activity masking.

## 9. FSI Wrappers and Reuse

FSI wrappers reuse moving-boundary mono solves:

- update `model.body` via `rigid_body_levelset`,
- update `model.bc_cut_u` via `rigid_cut_bc_tuple`,
- solve moving Stokes,
- evaluate end-time hydrodynamic loads through
  `endtime_static_model` + `integrated_embedded_force`.

Split and strong coupling differ only in state-iteration logic around this core
fluid solve/load evaluation path.
