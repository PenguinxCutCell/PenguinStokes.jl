# PenguinStokes.jl

`PenguinStokes.jl` assembles a monophasic cut-cell Stokes system on a MAC-style set of shifted grids:

- pressure unknowns on `gridp`
- velocity-component unknowns on staggered grids `gridu[d]`
- per scalar/component unknown split `(uomega, ugamma)` using `CartesianOperators` (`G`, `H`, `Winv`)

Supported dimensions: 1D, 2D, 3D.

## Unknown ordering

For `N` dimensions and `nt = prod(grid.n)`:

`x = [uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]`

with total size `(2N+1)*nt`.

## Discrete operators and signs

For each velocity component `d`, using `op_u[d]` and pressure operator `op_p`:

- viscous blocks

`L_oo[d] = mu * G_u[d]' * Winv_u[d] * G_u[d]`

`L_og[d] = mu * G_u[d]' * Winv_u[d] * H_u[d]`

- pressure gradient block (split from pressure operator rows)

`Grad[d] = -(G_p + H_p)[rows_d, :]`

- divergence/continuity blocks

`Div_oo[d] = -(G_p[rows_d,:]' + H_p[rows_d,:]')`

`Div_og[d] =  H_p[rows_d,:]'`

Steady assembled equations:

- momentum rows (`uomega_d`):

`L_oo[d]*uomega_d + L_og[d]*ugamma_d + Grad[d]*pomega = V_d * f_d`

- cut/interface tie rows (`ugamma_d`):

`ugamma_d = g_cut,d`

- continuity rows (`pomega`):

`sum_d (Div_oo[d]*uomega_d + Div_og[d]*ugamma_d) = 0`

Pressure gauge replaces one continuity row (pin or mean gauge).

## Unsteady theta-scheme

For `theta in [0,1]` (`:BE => 1`, `:CN => 0.5`):

`(rho/dt)*V_d*uomega_d^{n+1} + theta*(L_oo[d]*uomega_d^{n+1} + L_og[d]*ugamma_d^{n+1}) + Grad[d]*pomega^{n+1}`

`= (rho/dt)*V_d*uomega_d^n - (1-theta)*(L_oo[d]*uomega_d^n + L_og[d]*ugamma_d^n) + V_d*f_d^{n+theta}`

Continuity and tie equations are assembled at `n+1`.

## Boundary handling

- Velocity box BCs are applied on **momentum (`uomega`) rows** via boundary-face flux terms (ghost elimination style), consistent with face-bounded MAC control volumes:
  - `Dirichlet`: adds boundary conductance to diagonal + RHS contribution.
  - `Neumann`: adds prescribed flux contribution to RHS.
  - `Periodic`: handled through periodic stencils in `DiffusionOps`.
- Cut/interface rows still enforce `ugamma = g_cut`.
- Pressure BC is not imposed directly; use a pressure gauge (`PinPressureGauge` or `MeanPressureGauge`).

## Quick start

```julia
using CartesianGrids, PenguinBCs, PenguinStokes

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
body(x...) = -1.0

bc = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

model = StokesModelMono(
    grid,
    body,
    1.0,
    1.0;
    bc_u=(bc, bc),
    bc_cut=Dirichlet(0.0),
    force=(0.0, 0.0),
)

sys = solve_steady!(model)
```

See `examples/` for full scripts.

`examples/04_mms_convergence.jl` runs a mesh refinement study (`n=17,33,65`) for a divergence-free MMS and prints:

- velocity L2/L∞ error convergence (about first-order with current MAC operator stack),
- gauge-corrected pressure L2 convergence and pressure range diagnostics (`min(p), max(p), max|p|`),
- divergence norm (`L2`) on active pressure cells (near machine precision).

`examples/05_mms_convergence_zero_pressure.jl` runs a zero-pressure polynomial MMS on a full box and reports near second-order velocity convergence in L2.
