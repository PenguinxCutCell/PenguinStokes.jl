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

Wall closure on staggered grids

- On staggered velocity meshes, a box wall can be either:
- `collocated` with a velocity DOF (the DOF lies on the wall plane), or
- `non-collocated` (the nearest DOF is inside the domain).
- This distinction is geometric and depends on the component grid (`ux`, `uy`, ...), not only on the side label.
- For Dirichlet velocity walls, the package now applies:
- strong elimination (`u = u_wall`) only for collocated rows when `strong_wall_bc=true`,
- distance-based weak closure (`a = mu * Aface / dist`) for non-collocated rows, using the true DOF-to-wall distance.
- Using a fixed half-step distance on non-collocated rows can produce first-order wall truncation, especially on outer-box high-side rows.

Pressure wall handling (`bc_p`)

- `bc_p` is optional (`nothing` by default).
- When `bc_p === nothing`, pressure wall rows are not modified; the usual continuity + gauge system is assembled.
- When `bc_p` is provided, box pressure wall constraints (Dirichlet/Neumann) can be imposed explicitly.
- This is intended for cases where pressure-wall data is known/desired (for example MMS studies with prescribed normal pressure derivative).

Embedded-boundary force and stress post-processing

- `embedded_boundary_quantities(model, x; ...)` computes stress tensors, traction vectors, and integrated force density on pressure-grid cut cells.
- `integrated_embedded_force(model, x; ...)` returns total force, split into pressure and viscous parts, plus torque.
- Pressure traction uses `p` trace reconstruction at the interface (`:none` or `:linear`).
- Viscous traction uses `mu * (grad(u) + grad(u)')` from the assembled staggered operators.

Unsteady formulation

`assemble_unsteady!` supports theta schemes (`:BE`, `:CN`, numeric `theta`) by adding mass `(rho/dt) * V` to momentum omega blocks and corresponding RHS history terms.
