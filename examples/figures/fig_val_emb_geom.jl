#
# Generates Figure `fig:val-emb-geom` of the staggered cut-cell Stokes article:
#   (left)  cut-cell mesh on the unit square with a circular embedded boundary
#   (right) contours of the manufactured streamfunction
#           ψ = (r² - R²)² · P(x) · Q(y),  P(ξ) = ξ²(1-ξ)²,  Q = P
#
# Geometry matches examples/06_mms_convergence_embedded_outside_circle.jl:
#   Ω = [0,1]²,  circular hole centred at (0.5, 0.5) of radius R = 0.18.
#
# Output: PNG written next to this script.

using CairoMakie

const XC = 0.5
const YC = 0.5
const R0 = 0.18

P(x) = x^2 * (1 - x)^2
Q(y) = y^2 * (1 - y)^2
psi(x, y) = ((x - XC)^2 + (y - YC)^2 - R0^2)^2 * P(x) * Q(y)

# ── Mesh used for the left panel (kept coarse for visibility) ───────────────
const NMESH = 25                          # nodes per side
const HMESH = 1.0 / (NMESH - 1)
xs = range(0.0, 1.0; length = NMESH)
ys = range(0.0, 1.0; length = NMESH)

# Classify each cell against the circle: fluid (outside), solid (inside), or cut.
in_solid(x, y) = (x - XC)^2 + (y - YC)^2 < R0^2

function cell_class(i, j)
    # Cell with lower-left corner (xs[i], ys[j]); examine its 4 vertices.
    v = (
        in_solid(xs[i],     ys[j]),
        in_solid(xs[i + 1], ys[j]),
        in_solid(xs[i],     ys[j + 1]),
        in_solid(xs[i + 1], ys[j + 1]),
    )
    s = count(identity, v)
    return s == 0 ? :fluid : s == 4 ? :solid : :cut
end

# ── Build the figure ────────────────────────────────────────────────────────
fig = Figure(; size = (1100, 520), fontsize = 16)

# Left panel: cut-cell mesh ---------------------------------------------------
ax1 = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    xlabel = "x",
    ylabel = "y",
    title  = "Cut-cell mesh on Ω = [0,1]² with circular obstacle",
    limits = (0, 1, 0, 1),
)

# Shade cells by class for visibility.
for i in 1:(NMESH - 1), j in 1:(NMESH - 1)
    cls = cell_class(i, j)
    col = cls === :fluid ? (:white, 0.0) :
          cls === :solid ? (:gray85, 1.0) :
                           (:orange, 0.45)
    poly!(
        ax1,
        Point2f[
            (xs[i],     ys[j]),
            (xs[i + 1], ys[j]),
            (xs[i + 1], ys[j + 1]),
            (xs[i],     ys[j + 1]),
        ];
        color = col,
        strokewidth = 0,
    )
end

# Grid lines.
for x in xs
    lines!(ax1, [x, x], [0.0, 1.0]; color = (:black, 0.25), linewidth = 0.6)
end
for y in ys
    lines!(ax1, [0.0, 1.0], [y, y]; color = (:black, 0.25), linewidth = 0.6)
end

# Embedded boundary Γ.
θ = range(0, 2π; length = 400)
lines!(
    ax1,
    XC .+ R0 .* cos.(θ),
    YC .+ R0 .* sin.(θ);
    color = :crimson,
    linewidth = 2.5,
    label = "Γ",
)

# Legend swatches via invisible markers.
scatter!(ax1, [NaN], [NaN]; color = (:orange, 0.45), marker = :rect,
         markersize = 14, label = "cut cells")
scatter!(ax1, [NaN], [NaN]; color = :gray85, marker = :rect,
         markersize = 14, label = "Ω⁺ (solid)")
axislegend(ax1; position = :rt, framevisible = true, labelsize = 12)

# Right panel: streamfunction contours ---------------------------------------
ax2 = Axis(
    fig[1, 2];
    aspect = DataAspect(),
    xlabel = "x",
    ylabel = "y",
    title  = "Streamfunction",
    limits = (0, 1, 0, 1),
)

# Sample ψ on a fine grid; mask the interior of the disk.
nfine = 400
xf = range(0.0, 1.0; length = nfine)
yf = range(0.0, 1.0; length = nfine)
ψ  = [in_solid(x, y) ? NaN : psi(x, y) for x in xf, y in yf]

hm = heatmap!(ax2, xf, yf, ψ; colormap = :viridis, nan_color = :gray85)
contour!(
    ax2, xf, yf, ψ;
    levels = 15,
    color = :black,
    linewidth = 0.8,
)
lines!(ax2, XC .+ R0 .* cos.(θ), YC .+ R0 .* sin.(θ);
       color = :crimson, linewidth = 2.5)

Colorbar(fig[1, 3], hm; label = "ψ", width = 14)

# Save ----------------------------------------------------------------------
outdir = @__DIR__
outpath = joinpath(outdir, "fig_val_emb_geom.png")
save(outpath, fig; px_per_unit = 2)
println("saved figure: ", outpath)
