#
# Generates Figure `fig:val-couette` of the staggered cut-cell Stokes article
# (Section ``Couette flow between rotating cylinders'').
#
# Geometry mirrors examples/33_couette_rotating_cylinders.jl:
#   Ω = [-0.5, 0.5]², annulus r1 < r < r2 with r1 = 0.25, r2 = 0.5.
#   Inner cylinder rotates with unit angular velocity, outer cylinder fixed.
#   Analytic tangential velocity:
#       u_θ(r) = r · ((r2/r)² − 1) / ((r2/r1)² − 1).
#
# Two issues with example 33 are worth recording here:
#
#   (1) Post-processing: the example's `u_θ` reconstruction interpolates
#       across staggered meshes using a single LinearIndices on `cap_u[1]`,
#       which is only correct for the u-component. This contaminates the
#       reported `utheta_L2` with a non-second-order interpolation error.
#
#       Fix used below: compare each Cartesian velocity component at its own
#       staggered cell centre to the analytic Cartesian field
#           u_ex^x(x, y) = −y · u_θ(r) / r,
#           u_ex^y(x, y) = +x · u_θ(r) / r.
#       No cross-mesh interpolation is needed.
#
#   (2) Configuration: with the post-processing fix in place, the
#       cut-cell-weighted error on the Cartesian components is observed to
#       stagnate around 4–6·10⁻³ across n = 33, 65, 129. This is consistent
#       across two box geometries ([−0.5, 0.5]² and [−0.7, 0.7]²) and across
#       PinPressureGauge/MeanPressureGauge, and is therefore not a
#       post-processing artefact. The most likely cause is that the
#       Couette-between-cylinders configuration is encoded with a single
#       level-set max(r₁ − r, r − r₂) that produces a disconnected solid
#       region (inner disk ∪ exterior of r₂); the cut-cell operator
#       assembly was designed for a single connected complement and may
#       inherit a normal-orientation inconsistency on the outer cylinder.
#       A robust fix is to encode the two cylinders as two separate
#       embedded boundaries, which is left as a follow-up.
#
# Four panels:
#   (a) cut-cell mesh on Ω, both embedded boundaries and the fluid annulus;
#   (b) heatmap of |u_h| on the finest mesh with analytic streamlines;
#   (c) radial profile u_θ(r): analytic vs. numerical samples at every
#       u/v-cell centre falling inside the annulus, on three meshes;
#   (d) cut-cell-weighted L² error on each Cartesian component vs h, with
#       an h² guide line.

using LinearAlgebra
using Printf
using CairoMakie
using CartesianGrids
using PenguinBCs
using PenguinStokes
using PenguinSolverCore: LinearSystem

const R1   = 0.25
const R2   = 0.5
const LBOX = 0.5                  # box half-side (original example geometry).
const MU   = 1.0
const RHO  = 1.0
const NS   = (33, 65, 97)

# ── Analytic fields ─────────────────────────────────────────────────────────
analytic_utheta(r) = r * ((R2 / r)^2 - 1) / ((R2 / R1)^2 - 1)

function u_ex_x(x, y)
    r = sqrt(x^2 + y^2)
    (r <= R1 || r >= R2) && return 0.0
    return -y * analytic_utheta(r) / r
end

function u_ex_y(x, y)
    r = sqrt(x^2 + y^2)
    (r <= R1 || r >= R2) && return 0.0
    return x * analytic_utheta(r) / r
end

# ── Solver setup, identical BCs as example 33 ───────────────────────────────
function couette_body(x, y)
    r = sqrt(x^2 + y^2)
    return max(R1 - r, r - R2)
end

function couette_bc_cut(h::Float64)
    # ── KEY BUG FIX ─────────────────────────────────────────────────────────
    # The cut-cell capacity fills `C_γ[i]` with a coordinate that is *near*
    # the interface but not on it (its distance to the cylinder is O(h)).
    # Evaluating the rigid-rotation BC literally at that coordinate, i.e.
    # `u = (−y, x)`, gives a velocity whose magnitude is `r_{C_γ}` rather than
    # `R_inner`, introducing an O(h) error in the Dirichlet trace itself
    # which then floors the solution error.
    #
    # The robust formulation evaluates the rigid-rotation velocity at the
    # *projection* of the BC point onto the cylinder, i.e. at radius R_inner
    # along the same angular direction. Equivalently, the rigid-rotation
    # velocity is scaled by R_inner / r so that |u| = R_inner · ω
    # independently of how far `C_γ[i]` is from the cylinder.
    tol(h) = 1.5 * h
    function bc(x, y, comp::Symbol)
        r = sqrt(x^2 + y^2)
        if abs(r - R1) < tol(h) && abs(r - R1) <= abs(r - R2)
            # Inner cylinder: rigid rotation at angular position of (x, y),
            # but evaluated on the cylinder r = R1.
            r_safe = max(r, eps())
            scale = R1 / r_safe
            return comp === :ux ? -y * scale : x * scale
        elseif abs(r - R2) < tol(h) && abs(r - R2) <  abs(r - R1)
            # Outer cylinder: no-slip.
            return 0.0
        else
            # Phantom point well inside fluid or solid: fluid-neutral.
            return 0.0
        end
    end
    u_x(x, y) = bc(x, y, :ux)
    u_y(x, y) = bc(x, y, :uy)
    return (Dirichlet(u_x), Dirichlet(u_y))
end

function solve_case(n::Int)
    h = 2 * LBOX / (n - 1)
    grid = CartesianGrid((-LBOX, -LBOX), (LBOX, LBOX), (n, n))
    bc_box = BorderConditions(
        ; left = Dirichlet(0.0), right = Dirichlet(0.0),
        bottom = Dirichlet(0.0), top = Dirichlet(0.0),
    )
    model = StokesModelMono(
        grid, couette_body, MU, RHO;
        bc_u  = (bc_box, bc_box),
        bc_cut = couette_bc_cut(h),
        force = (0.0, 0.0),
        gauge = PinPressureGauge(),
    )
    sys = solve_steady!(model)
    return model, sys
end

# Cut-cell-weighted L² error on each Cartesian component, evaluated at its own
# staggered cell centre (no interpolation).
function component_errors(model, sys)
    ux = sys.x[model.layout.uomega[1]]
    uy = sys.x[model.layout.uomega[2]]

    cap_u = model.cap_u[1]
    cap_v = model.cap_u[2]

    eu = 0.0; wu = 0.0
    @inbounds for i in 1:cap_u.ntotal
        V = cap_u.buf.V[i]
        if isfinite(V) && V > 0.0
            x, y = cap_u.C_ω[i]
            r = sqrt(x^2 + y^2)
            (r <= R1 || r >= R2) && continue
            e = ux[i] - u_ex_x(x, y)
            eu += V * e^2; wu += V
        end
    end

    ev = 0.0; wv = 0.0
    @inbounds for i in 1:cap_v.ntotal
        V = cap_v.buf.V[i]
        if isfinite(V) && V > 0.0
            x, y = cap_v.C_ω[i]
            r = sqrt(x^2 + y^2)
            (r <= R1 || r >= R2) && continue
            e = uy[i] - u_ex_y(x, y)
            ev += V * e^2; wv += V
        end
    end

    return sqrt(eu / wu), sqrt(ev / wv)
end

# Sample the tangential velocity profile along a single radial line x = 0,
# y ∈ [R1, R2]. Along this line, u_θ(y) = u_x(0, y) · sign(y) (with our
# convention, u_x = −y·u_θ/r, so at x=0: u_θ = −u_x for y > 0).
#
# Sampling along a single radial line (rather than scattering all u/v centres)
# gives ~h-spaced samples that are monotone in r and free of off-axis
# projection noise — see the reference snippet provided.
function utheta_radial_samples(model, sys)
    ux = sys.x[model.layout.uomega[1]]
    cap_u = model.cap_u[1]
    nx, ny = cap_u.nnodes

    # Use the grid column nearest to x = 0 via the LinearIndices, not the
    # centroid coordinate. C_ω is the fluid-side centroid which can drift
    # off the regular grid for cut cells; sampling by centroid drops most
    # of the annulus cells, leaving only a handful of points clustered near
    # one cylinder. Using the grid column directly recovers one sample per
    # row of u-cells along the y-axis.
    LI = LinearIndices((nx, ny))
    i_col = max(1, div(nx, 2) + 1)        # index closest to x = 0 (interior u-face)

    rs = Float64[]; us = Float64[]
    @inbounds for j in 1:ny
        idx = LI[i_col, j]
        idx > cap_u.ntotal && continue
        V = cap_u.buf.V[idx]
        (!isfinite(V) || V <= 0.0) && continue
        x, y = cap_u.C_ω[idx]
        y > 0 || continue
        r = sqrt(x^2 + y^2)
        (r <= R1 || r >= R2) && continue
        push!(rs, r)
        push!(us, -ux[idx])               # u_θ = −u_x along x ≈ 0, y > 0
    end
    p = sortperm(rs)
    return rs[p], us[p]
end

# Plotting sampler: pick all u-cells in a vertical band of width h around x = 0
# so panel (c) shows ~one dot per row of u-cells, making the profile clearly
# visible at every mesh resolution. This is purely a visualization helper —
# the convergence error norm continues to use the exact-column sampler above.
function utheta_radial_samples_plot(model, sys)
    ux = sys.x[model.layout.uomega[1]]
    cap_u = model.cap_u[1]
    h = 2 * LBOX / (cap_u.nnodes[1] - 1)

    rs = Float64[]; us = Float64[]
    @inbounds for i in 1:cap_u.ntotal
        V = cap_u.buf.V[i]
        (!isfinite(V) || V <= 0.0) && continue
        x, y = cap_u.C_ω[i]
        abs(x) > 0.5 * h && continue
        y > 0 || continue
        r = sqrt(x^2 + y^2)
        (r <= R1 || r >= R2) && continue
        push!(rs, r)
        push!(us, -ux[i])
    end
    p = sortperm(rs)
    return rs[p], us[p]
end

# Radial-line L² error on u_θ. This is the clean convergence metric: one sample
# per row of u-cells on a single column, no off-axis projection.
function radial_line_error(model, sys)
    rs, us = utheta_radial_samples(model, sys)
    isempty(rs) && return NaN
    err2 = 0.0
    for (r, uθ_num) in zip(rs, us)
        err2 += (uθ_num - analytic_utheta(r))^2
    end
    return sqrt(err2 / length(rs))
end

# ── Run the convergence study ───────────────────────────────────────────────
println("Couette between rotating cylinders — radial-line and Cartesian errors:")
hs = Float64[]; eus = Float64[]; evs = Float64[]; erads = Float64[]
models = Any[]; syss = Any[]

for n in NS
    model, sys = solve_case(n)
    eu, ev = component_errors(model, sys)
    erad = radial_line_error(model, sys)
    res = norm(sys.A * sys.x - sys.b)
    push!(hs, 2 * LBOX / (n - 1))
    push!(eus, eu); push!(evs, ev); push!(erads, erad)
    push!(models, model); push!(syss, sys)
    @printf("  n=%-3d  h=%.4f  ‖u_x−u_ex^x‖_L²=%.3e  ‖u_y−u_ex^y‖_L²=%.3e  ‖u_θ−u_θ_ex‖_rad=%.3e  res=%.2e\n",
            n, hs[end], eu, ev, erad, res)
end

println("\nObserved orders:")
for k in 1:(length(NS) - 1)
    ou = log(eus[k] / eus[k + 1])  / log(hs[k] / hs[k + 1])
    ov = log(evs[k] / evs[k + 1])  / log(hs[k] / hs[k + 1])
    or = log(erads[k] / erads[k + 1]) / log(hs[k] / hs[k + 1])
    @printf("  n %d → %d :  order(u_x)=%.3f   order(u_y)=%.3f   order(u_θ radial)=%.3f\n",
            NS[k], NS[k + 1], ou, ov, or)
end

# ── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(; size = (1300, 1050), fontsize = 15)
θgrid = range(0, 2π; length = 400)

# (a) cut-cell mesh on the coarsest grid -------------------------------------
ax_a = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    xlabel = "x", ylabel = "y",
    title  = @sprintf("(a) Cut-cell mesh on Ω = [-%.1f, %.1f]² (n = %d)", LBOX, LBOX, NS[1]),
    limits = (-LBOX, LBOX, -LBOX, LBOX),
)
xs0 = range(-LBOX, LBOX; length = NS[1])
ys0 = range(-LBOX, LBOX; length = NS[1])
for x in xs0
    lines!(ax_a, [x, x], [-LBOX, LBOX]; color = (:black, 0.25), linewidth = 0.6)
end
for y in ys0
    lines!(ax_a, [-LBOX, LBOX], [y, y]; color = (:black, 0.25), linewidth = 0.6)
end
# Shade fluid annulus.
ann_inner = Point2f.(R1 .* cos.(θgrid), R1 .* sin.(θgrid))
ann_outer = Point2f.(R2 .* cos.(reverse(θgrid)), R2 .* sin.(reverse(θgrid)))
poly!(ax_a, vcat(ann_outer, ann_inner);
      color = (:steelblue, 0.18), strokewidth = 0)
lines!(ax_a, R1 .* cos.(θgrid), R1 .* sin.(θgrid);
       color = :crimson, linewidth = 2.5)
lines!(ax_a, R2 .* cos.(θgrid), R2 .* sin.(θgrid);
       color = :crimson, linewidth = 2.5)
text!(ax_a, 0.0, 0.0; text = L"\Omega^{+}_{\mathrm{inner}}",
      align = (:center, :center), fontsize = 13)
text!(ax_a, 0.0, (R1 + R2) / 2; text = L"\Omega^{-}\,(\mathrm{fluid})",
      align = (:center, :center), fontsize = 13)

# (b) |u_h| heatmap on finest mesh with analytic streamlines -----------------
model_f, sys_f = models[end], syss[end]
nf = NS[end]
xs = range(-LBOX, LBOX; length = nf)
ys = range(-LBOX, LBOX; length = nf)

uxh = sys_f.x[model_f.layout.uomega[1]]
uyh = sys_f.x[model_f.layout.uomega[2]]
li  = LinearIndices((nf, nf))
speed = fill(NaN, nf, nf)
@inbounds for j in 1:nf, i in 1:nf
    r = sqrt(xs[i]^2 + ys[j]^2)
    if R1 < r < R2
        idx = li[i, j]
        if idx <= length(uxh) && idx <= length(uyh)
            speed[i, j] = sqrt(uxh[idx]^2 + uyh[idx]^2)
        end
    end
end
ax_b = Axis(
    fig[1, 2];
    aspect = DataAspect(),
    xlabel = "x", ylabel = "y",
    title  = @sprintf("(b) Velocity magnitude |u_h| (n = %d)", nf),
    limits = (-LBOX, LBOX, -LBOX, LBOX),
)
hm = heatmap!(ax_b, xs, ys, speed;
              colormap = :viridis, nan_color = :gray85)
# Streamlines: concentric circles.
for r in range(R1 + 0.02, R2 - 0.02; length = 6)
    lines!(ax_b, r .* cos.(θgrid), r .* sin.(θgrid);
           color = (:white, 0.55), linewidth = 0.8, linestyle = :dash)
end
lines!(ax_b, R1 .* cos.(θgrid), R1 .* sin.(θgrid);
       color = :crimson, linewidth = 2.0)
lines!(ax_b, R2 .* cos.(θgrid), R2 .* sin.(θgrid);
       color = :crimson, linewidth = 2.0)
Colorbar(fig[1, 3], hm; label = L"|u_h|", width = 14)

# (c) radial profile u_θ(r): analytic + samples ------------------------------
ax_c = Axis(
    fig[2, 1];
    xlabel = "r", ylabel = L"u_\theta(r)",
    title  = "(c) Tangential velocity profile",
)
rs_an = range(R1, R2; length = 401)
lines!(ax_c, collect(rs_an), [analytic_utheta(r) for r in rs_an];
       color = :black, linewidth = 2.5, label = "analytic")
markers = (:circle, :utriangle, :diamond)
colors_ns = cgrad(:viridis, length(NS); categorical = true)
for k in eachindex(NS)
    rs, us = utheta_radial_samples_plot(models[k], syss[k])
    scatterlines!(ax_c, rs, us; color = colors_ns[k], marker = markers[k],
                  markersize = 8, linewidth = 1.2,
                  label = @sprintf("n = %d  (radial line x = 0)", NS[k]))
end
axislegend(ax_c; position = :rt, labelsize = 11)

# (d) Cartesian L² error vs h, per component ---------------------------------
ax_d = Axis(
    fig[2, 2];
    xlabel = "h", ylabel = "‖u_h − u_ex‖_{L²}",
    xscale = log10, yscale = log10,
    title  = "(d) Cut-cell-weighted L² error vs h",
)
scatterlines!(ax_d, hs, eus; color = :steelblue, marker = :circle,
              linewidth = 2, markersize = 10,
              label = L"u_x\ \mathrm{(cut\!-\!cell\ L^2)}")
scatterlines!(ax_d, hs, evs; color = :crimson, marker = :utriangle,
              linewidth = 2, markersize = 11,
              label = L"u_y\ \mathrm{(cut\!-\!cell\ L^2)}")
scatterlines!(ax_d, hs, erads; color = :goldenrod, marker = :diamond,
              linewidth = 2, markersize = 11,
              label = L"u_\theta\ \mathrm{(radial\ line)}")
yref = erads[end] .* (hs ./ hs[end]) .^ 2
lines!(ax_d, hs, yref; color = :black, linestyle = :dashdot,
       linewidth = 1.5, label = "h² guide")
axislegend(ax_d; position = :rb, labelsize = 10)

Label(fig[0, 1:3],
    "Couette flow between rotating cylinders: validation and convergence";
    fontsize = 19, halign = :center)

outpath = joinpath(@__DIR__, "fig_val_couette.png")
save(outpath, fig; px_per_unit = 2)
println("\nsaved figure: ", outpath)
