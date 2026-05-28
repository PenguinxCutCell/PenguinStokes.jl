#
# Schematic for Figure `fig:val-torque` of the staggered cut-cell Stokes
# article (Section ``Torque on a sphere in Couette shear'').
#
# Pure geometric diagram — no solver run. The schematic shows the cubic
# domain [-L/2, L/2]³ in a 2-D side view (the (x, y) plane through the
# sphere centre), with:
#   - top/bottom walls translating in opposite x-directions to impose
#     a Couette shear with γ̇;
#   - x and z directions periodic;
#   - the sphere of radius R fixed at the centre;
#   - the analytic Stokes torque T_z = 8 π μ R³ (γ̇ / 2).

using CairoMakie

const L     = 8.0
const R_SPH = 0.5
const GDOT  = 1.0
const U_WALL = GDOT * L / 2

fig = Figure(; size = (820, 740), fontsize = 16)
ax = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    xlabel = "x", ylabel = "y",
    title = "Sphere in Couette shear: torque benchmark",
    limits = (-L / 2 - 0.6, L / 2 + 0.6, -L / 2 - 0.6, L / 2 + 0.6),
)
hidedecorations!(ax; label = false, ticklabels = false, ticks = false,
                 grid = true)

# Box outline ----------------------------------------------------------------
poly!(ax, Point2f[
    (-L / 2, -L / 2), ( L / 2, -L / 2),
    ( L / 2,  L / 2), (-L / 2,  L / 2),
];
      color = (:gray95, 0.6), strokecolor = :black, strokewidth = 1.5)

# Top and bottom walls highlighted as shear walls --------------------------
for (ywall, label_y, dir) in (
    ( L / 2,  L / 2 + 0.22,  1.0),     # top wall: +x
    (-L / 2, -L / 2 - 0.22, -1.0),     # bottom wall: -x
)
    lines!(ax, [-L / 2, L / 2], [ywall, ywall];
           color = :steelblue, linewidth = 3.5)
    # Shear-arrow row.
    nseg = 9
    for x0 in range(-L / 2 + 0.4, L / 2 - 0.4; length = nseg)
        arrows2d!(
            ax, [x0], [ywall], [dir * 0.55], [0.0];
            lengthscale = 1.0,
            shaftwidth = 2.0,
            tiplength  = 9,
            tipwidth   = 9,
            color = :steelblue,
        )
    end
    lbl = dir > 0 ? L"u_x = +\dot\gamma\,L/2" : L"u_x = -\dot\gamma\,L/2"
    text!(ax, 0.0, label_y;
          text = lbl,
          align = (:center, dir > 0 ? :bottom : :top), fontsize = 15)
end

# x and z periodic (left/right faces) -------------------------------------
for (xwall, halign) in ((-L / 2, :right), (L / 2, :left))
    lines!(ax, [xwall, xwall], [-L / 2, L / 2];
           color = (:black, 0.6), linewidth = 2, linestyle = :dash)
end
text!(ax, -L / 2 - 0.05, 0.0;
      text = "periodic",
      align = (:right, :center), fontsize = 13, rotation = π / 2)
text!(ax,  L / 2 + 0.05, 0.0;
      text = "periodic",
      align = (:left,  :center), fontsize = 13, rotation = π / 2)

# Linear shear profile u_x(y) = γ̇ · y, overlaid as a graph -----------------
ys = range(-L / 2, L / 2; length = 100)
us = GDOT .* ys                                       # u_x(y)
# Plot the profile as a curve in (u_x_scaled, y) — anchored at x = 0
# horizontally with a small visual scaling for readability.
profile_scale = 0.7 * L / 2 / U_WALL                  # so |profile| ≤ 0.7·L/2
prof_xs = profile_scale .* us
lines!(ax, prof_xs, ys; color = (:orange, 0.9), linewidth = 2.5)
# Tick arrows along the profile at a few y-stations.
for y0 in (-3.0, -1.5, 1.5, 3.0)
    u0 = GDOT * y0
    arrows2d!(
        ax, [0.0], [y0], [profile_scale * u0], [0.0];
        lengthscale = 1.0,
        shaftwidth = 1.6,
        tiplength  = 7,
        tipwidth   = 7,
        color = (:orange, 0.85),
    )
end
text!(ax, profile_scale * GDOT * (L / 2) + 0.15, L / 2 - 0.2;
      text = L"\mathbf{u}_\infty(y) = \dot\gamma\,y\,\mathbf{e}_x",
      align = (:left, :top), fontsize = 14)

# Sphere centred at the origin ---------------------------------------------
θ = range(0, 2π; length = 400)
sphere_xs = R_SPH .* cos.(θ)
sphere_ys = R_SPH .* sin.(θ)
poly!(ax, Point2f.(sphere_xs, sphere_ys);
      color = (:crimson, 0.18), strokecolor = :crimson, strokewidth = 2.5)
scatter!(ax, [0.0], [0.0]; color = :crimson, markersize = 8)

# Radius R line ------------------------------------------------------------
α = π / 3
xr = R_SPH * cos(α)
yr = R_SPH * sin(α)
lines!(ax, [0.0, xr], [0.0, yr]; color = :black, linewidth = 1.0)
text!(ax, 0.55 * R_SPH * cos(α) + 0.05,
          0.55 * R_SPH * sin(α);
      text = L"R", align = (:left, :center), fontsize = 16)

# Curved arrow indicating induced rotation about z ------------------------
nrot = 80
rrot = R_SPH + 0.55
θrot = range(-0.65π, 0.65π; length = nrot)
rot_xs = rrot .* cos.(θrot)
rot_ys = rrot .* sin.(θrot)
lines!(ax, rot_xs, rot_ys; color = :black, linewidth = 2)
# Arrow tip at the end of the curved trajectory.
tipx = rot_xs[end]; tipy = rot_ys[end]
prevx = rot_xs[end - 4]; prevy = rot_ys[end - 4]
arrows2d!(
    ax,
    [prevx], [prevy], [tipx - prevx], [tipy - prevy];
    lengthscale = 1.0,
    shaftwidth = 0.001,
    tiplength  = 14,
    tipwidth   = 14,
    color = :black,
)
text!(ax, 0.0, rrot + 0.05;
      text = L"T_z = 8\pi\mu R^{3}\,\dot\gamma/2",
      align = (:center, :bottom), fontsize = 16)

# Box side L annotation ----------------------------------------------------
ybot = -L / 2 - 0.85
lines!(ax, [-L / 2, L / 2], [ybot, ybot]; color = :black, linewidth = 1.0)
scatter!(ax, [-L / 2, L / 2], [ybot, ybot];
         color = :black, marker = :vline, markersize = 14)
text!(ax, 0.0, ybot - 0.18; text = L"L", align = (:center, :top), fontsize = 17)

# Save ---------------------------------------------------------------------
outpath = joinpath(@__DIR__, "fig_val_torque_schema.png")
save(outpath, fig; px_per_unit = 2)
println("saved figure: ", outpath)
