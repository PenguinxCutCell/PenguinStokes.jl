#
# Schematic for Figure `fig:val-lub` of the staggered cut-cell Stokes article
# (Section ``Sphere approaching a plane wall: lubrication regime'').
#
# Pure geometric diagram — no solver run. The schematic shows the cubic
# domain [-L/2, L/2]³ in a 2-D side view (the (x, y) plane through the
# sphere centre), with:
#   - the no-slip plane wall on the left (x = -L/2);
#   - the sphere of radius R held at a fixed gap δ above the wall;
#   - the prescribed sphere velocity u_sphere = (-u_ref, 0);
#   - annotations for δ, R, L, and the resistance coefficient
#     F_x = κ · 6 π μ u_ref R.

using CairoMakie

const L      = 8.0           # box side
const R_SPH  = 0.5           # sphere radius
const DELTA  = 0.4 * R_SPH   # gap (matches Part 1 of example 36)

# Sphere centre on the (x, y) symmetry plane.
const X_C = -L / 2 + DELTA + R_SPH
const Y_C = 0.0

fig = Figure(; size = (820, 740), fontsize = 16)
ax = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    xlabel = "x", ylabel = "y",
    title = "Sphere approaching a plane wall: lubrication geometry",
    limits = (-L / 2 - 0.4, L / 2 + 0.4, -L / 2 - 0.4, L / 2 + 0.4),
)
hidedecorations!(ax; label = false, ticklabels = false, ticks = false,
                 grid = true)

# Box outline ----------------------------------------------------------------
poly!(ax, Point2f[
    (-L / 2, -L / 2), ( L / 2, -L / 2),
    ( L / 2,  L / 2), (-L / 2,  L / 2),
];
      color = (:gray95, 0.6), strokecolor = :black, strokewidth = 1.5)

# Left wall: highlight with a thick hatched-style strip --------------------
wall_xs = [-L / 2, -L / 2 + 0.18]
wall_ys = [-L / 2,  L / 2]
poly!(ax, Point2f[
    (wall_xs[1], wall_ys[1]), (wall_xs[2], wall_ys[1]),
    (wall_xs[2], wall_ys[2]), (wall_xs[1], wall_ys[2]),
];
      color = (:steelblue, 0.6), strokewidth = 0)
# Hatching marks for the wall.
for ys0 in range(-L / 2 + 0.1, L / 2 - 0.1; length = 30)
    lines!(ax, [-L / 2, -L / 2 + 0.18],
              [ys0,      ys0 - 0.18];
              color = (:black, 0.65), linewidth = 0.9)
end
text!(ax, -L / 2 - 0.30, 0.0;
      text = "no-slip wall  u = 0",
      align = (:center, :center), fontsize = 14, rotation = π / 2)

# Sphere ---------------------------------------------------------------------
θ = range(0, 2π; length = 400)
sphere_xs = X_C .+ R_SPH .* cos.(θ)
sphere_ys = Y_C .+ R_SPH .* sin.(θ)
poly!(ax, Point2f.(sphere_xs, sphere_ys);
      color = (:crimson, 0.18), strokecolor = :crimson, strokewidth = 2.5)

# Sphere centre marker.
scatter!(ax, [X_C], [Y_C]; color = :crimson, markersize = 9)
text!(ax, X_C + 0.05, Y_C + 0.05;
      text = L"\mathbf{X}_c", align = (:left, :bottom), fontsize = 15)

# Gap δ — horizontal arrow between wall and sphere left edge ----------------
y_gap = Y_C - R_SPH - 0.5
lines!(ax, [-L / 2, X_C - R_SPH], [y_gap, y_gap];
       color = :black, linewidth = 1.2)
scatter!(ax, [-L / 2, X_C - R_SPH], [y_gap, y_gap];
         color = :black, marker = :vline, markersize = 14)
text!(ax, (-L / 2 + X_C - R_SPH) / 2, y_gap - 0.15;
      text = L"\delta", align = (:center, :top), fontsize = 18)
# Vertical tick lines connecting the gap arrow to its endpoints.
lines!(ax, [-L / 2, -L / 2], [y_gap - 0.08, y_gap + 0.08];
       color = :black, linewidth = 1.0)
lines!(ax, [X_C - R_SPH, X_C - R_SPH], [y_gap - 0.08, y_gap + 0.08];
       color = :black, linewidth = 1.0)

# Sphere radius R — line from centre to surface, top-right ------------------
α = π / 4
xr = X_C + R_SPH * cos(α)
yr = Y_C + R_SPH * sin(α)
lines!(ax, [X_C, xr], [Y_C, yr]; color = :black, linewidth = 1.0)
text!(ax, X_C + 0.55 * R_SPH * cos(α) + 0.04,
          Y_C + 0.55 * R_SPH * sin(α) + 0.04;
      text = L"R", align = (:left, :bottom), fontsize = 17)

# Sphere velocity arrow ------------------------------------------------------
arrows2d!(
    ax,
    [X_C], [Y_C + R_SPH + 0.35],
    [-1.0], [0.0];
    lengthscale = 1.0,
    shaftwidth = 4.0,
    tiplength  = 16,
    tipwidth   = 16,
    color = :crimson,
)
text!(ax, X_C + 0.6, Y_C + R_SPH + 0.55;
      text = L"\mathbf{u}_{\mathrm{sphere}} = -u_{\mathrm{ref}}\,\mathbf{e}_x",
      align = (:left, :bottom), fontsize = 15)

# Box side L annotation -----------------------------------------------------
ybot = -L / 2 - 0.25
lines!(ax, [-L / 2, L / 2], [ybot, ybot]; color = :black, linewidth = 1.0)
scatter!(ax, [-L / 2, L / 2], [ybot, ybot];
         color = :black, marker = :vline, markersize = 14)
text!(ax, 0.0, ybot - 0.20; text = L"L", align = (:center, :top), fontsize = 17)

# Periodicity / far-field annotations (right wall is periodic / no-slip,
# matches example 36's box BCs).
text!(ax, L / 2 + 0.15, 0.0;
      text = "no-slip / far field",
      align = (:left, :center), fontsize = 13, rotation = π / 2)

# Resistance formula at the bottom of the panel -----------------------------
text!(ax, 0.0, -L / 2 - 0.95;
      text = L"F_x = \kappa(\delta/R)\,\cdot\,6\pi\mu\,u_{\mathrm{ref}}\,R",
      align = (:center, :top), fontsize = 18)

# Save -----------------------------------------------------------------------
outpath = joinpath(@__DIR__, "fig_val_lub_schema.png")
save(outpath, fig; px_per_unit = 2)
println("saved figure: ", outpath)
