**Stokes model and discretization**

This package assembles a monophasic Stokes system on MAC-style staggered grids with cut-cell geometry.

Unknown ordering

For `N` dimensions and `nt = prod(grid.n)`:

`x = [uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]`

Discrete structure

- Momentum block per component uses `G' * Winv * G` / `G' * Winv * H` diffusion-style operators on velocity grids.
- Pressure coupling is assembled via split gradient/divergence blocks built from pressure operators.
- Interface/cut rows enforce `ugamma = g_cut` (Dirichlet currently).
- Continuity rows enforce incompressibility with pressure gauge replacement for nullspace control.

Unsteady formulation

`assemble_unsteady!` supports theta schemes (`:BE`, `:CN`, numeric `theta`) by adding mass `(rho/dt) * V` to momentum omega blocks and corresponding RHS history terms.
