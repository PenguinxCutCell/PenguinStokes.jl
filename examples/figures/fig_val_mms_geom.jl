#
# Generates Figure `fig:val-mms-geom` of the staggered cut-cell Stokes article.
#
# Three snapshots of the manufactured moving two-phase configuration of
# Section ``Two-phase manufactured solution with a prescribed moving interface''
# (Sec. val-moving-mms in the article).
#
# Exact fields (matching examples/30_moving_two_phase_mms_mesh_convergence.jl):
#   u_ex(x, t) = (sin(ω t), 0),    p_ex^±  = const,
#   X_c(t)     = (x_c^0 + (1 − cos(ω t))/ω,  y_c),
#   ω = 2π,  x_c^0 = 0.5,  y_c = 0.5,  R = 0.18.
#
# Output: PNG written next to this script.

using CairoMakie
using Printf

const ω    = 2π
const X0   = 0.5
const YC   = 0.5
const R0   = 0.18
const Tper = 2π / ω                # = 1

xc(t) = X0 + (1 - cos(ω * t)) / ω
ux(t) = sin(ω * t)

# Three time instants spaced over the first half-period.
const TS = (Tper / 8, Tper / 4, 3Tper / 8)

# ── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(; size = (1300, 460), fontsize = 16)

# Shared colour range tied to u_x ∈ [−1, 1] so the three panels are comparable.
crange = (-1.0, 1.0)
cmap   = :balance

θ = range(0, 2π; length = 400)

for (k, t) in enumerate(TS)
    Uk = ux(t)
    Xk = xc(t)

    ax = Axis(
        fig[1, k];
        aspect = DataAspect(),
        xlabel = "x", ylabel = (k == 1 ? "y" : ""),
        title  = @sprintf("t / T = %.3f,   u_x = %.3f", t / Tper, Uk),
        limits = (0, 1, 0, 1),
    )

    # Background heatmap: u_x(x, t) is spatially uniform in Ω⁻, so a single
    # filled rectangle suffices. We use a 2×2 array on the corners so the
    # colour scale is meaningful when the panels are compared.
    heatmap!(
        ax,
        [0.0, 1.0], [0.0, 1.0],
        [Uk Uk; Uk Uk];
        colormap   = cmap,
        colorrange = crange,
        interpolate = false,
    )

    # Solid body Ω⁺(t): drawn as a filled disk, plus the two periodic images
    # to make the wrap-around visible when X_c is near the boundary.
    for shift in (-1.0, 0.0, 1.0)
        xs = Xk + shift .+ R0 .* cos.(θ)
        ys = YC          .+ R0 .* sin.(θ)
        poly!(ax, Point2f.(xs, ys);
              color = (:gray70, 0.95), strokecolor = :crimson, strokewidth = 2.0)
    end

    # Body-velocity arrow at the centre of Ω⁺(t): Ẋ_c(t) = (sin ω t, 0).
    arrows2d!(
        ax,
        [Xk], [YC], [Uk], [0.0];
        lengthscale = 0.18,
        shaftwidth  = 3.0,
        tiplength   = 12,
        tipwidth    = 12,
        color       = :black,
    )

    # Label the body.
    text!(ax, Xk, YC - R0 - 0.04; text = L"\Omega^{+}(t)",
          align = (:center, :top), fontsize = 14)
end

Colorbar(
    fig[1, length(TS) + 1];
    colormap   = cmap,
    colorrange = crange,
    label      = L"u_x(t) = \sin(\omega t)",
    width      = 14,
)

Label(
    fig[0, 1:length(TS) + 1],
    "Moving two-phase MMS: three snapshots of the manufactured solution";
    fontsize = 18, halign = :center,
)

outpath = joinpath(@__DIR__, "fig_val_mms_geom.png")
save(outpath, fig; px_per_unit = 2)
println("saved figure: ", outpath)
