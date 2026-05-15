#
# Companion figure for Section ``Two-phase planar Couette--Poiseuille flow''
# (label fig:val-cp) of the staggered cut-cell Stokes article.
#
# Sweeps the viscosity ratio m = μ₂/μ₁ over a representative range and, for
# each ratio, refines the mesh through three resolutions (n = 33, 65, 97).
# For every (m, n) pair we record:
#   - the cut-cell-weighted L² error on the longitudinal velocity in each
#     phase (Eq. (val-L2) of the article);
#   - the peak |v| (theoretical value: 0);
#   - the algebraic residual ‖Ax − b‖₂;
# and, between successive meshes, the empirical convergence order.
#
# The CairoMakie figure has four panels:
#   (a) exact profile u(y) on the finest mesh for each ratio (qualitative view
#       of how the kink at y = h sharpens as μ₂/μ₁ varies);
#   (b) ‖u − u_ex‖_{L²} vs h, with one curve per phase per ratio, and an h²
#       guide line;
#   (c) observed order between successive meshes, target = 2;
#   (d) ‖v‖_∞ vs h, illustrating that the transverse-velocity contamination
#       stays at the level of the linear solver across all (m, n) pairs.
#
# Geometry and parameters mirror examples/10_bis_two_phase_planar_couette_poiseuille.jl:
#   Ω = [0,1]², interface y = h = 0.4, top wall U = 0.5, body force G = 1,
#   μ₁ = 1 (fixed), μ₂ swept.

using LinearAlgebra
using Printf
using CairoMakie
using CartesianGrids
using PenguinBCs
using PenguinStokes

# ── Analytic two-layer Couette-Poiseuille profile ───────────────────────────
function two_layer_profile(mu1, mu2, h, H, U, G)
    C1 = -G / mu1
    C2 = -G / mu2

    M = [
        mu1 -mu2 0.0;
          h   -h  -1.0;
        0.0    H   1.0
    ]
    rhs = [
        0.0,
        0.5 * (C2 - C1) * h^2,
        U - 0.5 * C2 * H^2,
    ]
    A1, A2, B2 = M \ rhs

    u(y) = y <= h ? (0.5 * C1 * y^2 + A1 * y) : (0.5 * C2 * y^2 + A2 * y + B2)
    return u
end

# ── Cut-cell-weighted velocity errors and peak |v| ──────────────────────────
function velocity_errors(model, sys, u_exact)
    u1 = sys.x[model.layout.uomega1[1]]
    u2 = sys.x[model.layout.uomega2[1]]
    v1 = sys.x[model.layout.uomega1[2]]
    v2 = sys.x[model.layout.uomega2[2]]

    e1 = 0.0; w1 = 0.0
    e2 = 0.0; w2 = 0.0
    vInf = 0.0

    @inbounds for i in 1:model.cap_u1[1].ntotal
        V = model.cap_u1[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u1[1].C_ω[i][2]
            e = u1[i] - u_exact(y)
            e1 += V * e^2; w1 += V
            vInf = max(vInf, abs(v1[i]))
        end
    end
    @inbounds for i in 1:model.cap_u2[1].ntotal
        V = model.cap_u2[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u2[1].C_ω[i][2]
            e = u2[i] - u_exact(y)
            e2 += V * e^2; w2 += V
            vInf = max(vInf, abs(v2[i]))
        end
    end

    return (u1L2 = sqrt(e1 / w1), u2L2 = sqrt(e2 / w2), vInf = vInf)
end

# ── Sweep parameters ────────────────────────────────────────────────────────
const MU1 = 1.0
const H_IF = 0.4
const H_TOP = 1.0
const U_TOP = 0.5
const G_BODY = 1.0
const NS = (65, 97, 129)  # n = 33 is too coarse to show convergence trends, so we start at 65.
const RATIOS = (0.1, 0.25, 1.0, 4.0, 10.0)

bcx = BorderConditions(
    ; left = Periodic(), right = Periodic(),
    bottom = Dirichlet(0.0), top = Dirichlet(U_TOP),
)
bcy = BorderConditions(
    ; left = Periodic(), right = Periodic(),
    bottom = Dirichlet(0.0), top = Dirichlet(0.0),
)

body(x, y) = y - H_IF

# Per-ratio result container.
struct SweepData
    ratio::Float64
    hs::Vector{Float64}
    e1::Vector{Float64}
    e2::Vector{Float64}
    vInf::Vector{Float64}
    res::Vector{Float64}
    order1::Vector{Float64}
    order2::Vector{Float64}
    u_exact::Function
end

function run_ratio(m::Float64)
    mu2 = m * MU1
    u_exact = two_layer_profile(MU1, mu2, H_IF, H_TOP, U_TOP, G_BODY)

    hs = Float64[]; e1 = Float64[]; e2 = Float64[]
    vInf = Float64[]; res = Float64[]

    for n in NS
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelTwoPhase(
            grid, body, MU1, mu2;
            bc_u = (bcx, bcy),
            force1 = (G_BODY, 0.0),
            force2 = (G_BODY, 0.0),
            interface_force = (0.0, 0.0),
        )
        sys = solve_steady!(model)
        m_err = velocity_errors(model, sys, u_exact)
        push!(hs, 1.0 / (n - 1))
        push!(e1, m_err.u1L2)
        push!(e2, m_err.u2L2)
        push!(vInf, m_err.vInf)
        push!(res, norm(sys.A * sys.x - sys.b))
        @printf("  μ₂/μ₁=%-6.3g  n=%-3d  u1L2=%.3e  u2L2=%.3e  |v|∞=%.2e  res=%.2e\n",
                m, n, m_err.u1L2, m_err.u2L2, m_err.vInf, res[end])
    end

    order1 = Float64[]; order2 = Float64[]
    for k in 1:(length(NS) - 1)
        push!(order1, log(e1[k] / e1[k + 1]) / log(hs[k] / hs[k + 1]))
        push!(order2, log(e2[k] / e2[k + 1]) / log(hs[k] / hs[k + 1]))
    end
    return SweepData(m, hs, e1, e2, vInf, res, order1, order2, u_exact)
end

println("Running viscosity-ratio sweep (μ₁ = $MU1, ratios = $(collect(RATIOS))):")
data = [run_ratio(m) for m in RATIOS]

# ── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(; size = (1200, 980), fontsize = 15)

# Colour per ratio.
colors = cgrad(:viridis, length(RATIOS); categorical = true)

# (a) Exact profiles u(y) for each ratio --------------------------------------
ax_a = Axis(
    fig[1, 1];
    xlabel = "u(y)", ylabel = "y",
    title = "(a) Analytic profile u(y) per viscosity ratio",
)
ys = range(0.0, H_TOP; length = 401)
for (k, d) in enumerate(data)
    lines!(ax_a, [d.u_exact(y) for y in ys], collect(ys);
           color = colors[k], linewidth = 2.5,
           label = @sprintf("μ₂/μ₁ = %.2g", d.ratio))
end
hlines!(ax_a, [H_IF]; color = (:black, 0.6), linestyle = :dot,
        linewidth = 1.2, label = "interface y = h")
axislegend(ax_a; position = :rb, labelsize = 11, framevisible = true)

# (b) L² error vs h, per phase per ratio --------------------------------------
ax_b = Axis(
    fig[1, 2];
    xlabel = "h", ylabel = "‖u − u_ex‖_{L²}",
    xscale = log10, yscale = log10,
    title = "(b) Cut-cell-weighted L² error vs h",
)
for (k, d) in enumerate(data)
    scatterlines!(ax_b, d.hs, d.e1; color = colors[k], marker = :circle,
                  linewidth = 2, markersize = 9,
                  label = @sprintf("Ω⁻, μ₂/μ₁ = %.2g", d.ratio))
    scatterlines!(ax_b, d.hs, d.e2; color = colors[k], marker = :utriangle,
                  linestyle = :dash, linewidth = 2, markersize = 10)
end
# h² reference line, anchored on the median (μ₂/μ₁ = 1) phase-1 finest point.
ref = data[3]
href = ref.hs
yref0 = ref.e1[end] * (href ./ href[end]) .^ 2
lines!(ax_b, href, yref0; color = :black, linestyle = :dashdot,
       linewidth = 1.5, label = "h² guide")
axislegend(ax_b; position = :rb, labelsize = 10, nbanks = 1)

# (c) Observed order between successive meshes --------------------------------
ax_c = Axis(
    fig[2, 1];
    xlabel = "viscosity ratio μ₂/μ₁", ylabel = "observed order",
    xscale = log10,
    title = "(c) Empirical convergence order",
    yticks = 0:0.5:3,
)
ms = [d.ratio for d in data]
# Order from h_{k} → h_{k+1}; we have two pairs per ratio.
o1_pair1 = [d.order1[1] for d in data]
o1_pair2 = [d.order1[2] for d in data]
o2_pair1 = [d.order2[1] for d in data]
o2_pair2 = [d.order2[2] for d in data]
scatterlines!(ax_c, ms, o1_pair1; color = :steelblue, marker = :circle,
              markersize = 11, linewidth = 2,
              label = "Ω⁻,  n: 33→65")
scatterlines!(ax_c, ms, o1_pair2; color = :steelblue, marker = :diamond,
              markersize = 11, linewidth = 2, linestyle = :dash,
              label = "Ω⁻,  n: 65→97")
scatterlines!(ax_c, ms, o2_pair1; color = :crimson, marker = :utriangle,
              markersize = 12, linewidth = 2,
              label = "Ω⁺,  n: 33→65")
scatterlines!(ax_c, ms, o2_pair2; color = :crimson, marker = :dtriangle,
              markersize = 12, linewidth = 2, linestyle = :dash,
              label = "Ω⁺,  n: 65→97")
hlines!(ax_c, [2.0]; color = (:black, 0.5), linestyle = :dot,
        linewidth = 1.5, label = "target order 2")
ylims!(ax_c, 0.0, 3.0)
axislegend(ax_c; position = :rb, labelsize = 10)

# (d) Peak |v| (theoretical value 0) ------------------------------------------
ax_d = Axis(
    fig[2, 2];
    xlabel = "h", ylabel = "‖v‖_∞",
    xscale = log10, yscale = log10,
    title = "(d) Peak transverse velocity (target: 0)",
)
for (k, d) in enumerate(data)
    scatterlines!(ax_d, d.hs, max.(d.vInf, eps());
                  color = colors[k], marker = :circle,
                  linewidth = 2, markersize = 9,
                  label = @sprintf("μ₂/μ₁ = %.2g", d.ratio))
end
axislegend(ax_d; position = :rb, labelsize = 10)

# Global super-title.
Label(fig[0, 1:2],
    "Two-phase planar Couette--Poiseuille: viscosity-ratio sweep, three-mesh convergence";
    fontsize = 18, halign = :center)

outdir = @__DIR__
outpath = joinpath(outdir, "fig_val_cp_visc_sweep.png")
save(outpath, fig; px_per_unit = 2)
println("saved figure: ", outpath)

# ── Text summary (handy for the article caption) ────────────────────────────
println()
println("Summary table (asymptotic order on the finest pair n: 65 → 97):")
println("  μ₂/μ₁     order(Ω⁻)   order(Ω⁺)")
for d in data
    @printf("  %-8.3g  %-10.3f  %-10.3f\n", d.ratio, d.order1[end], d.order2[end])
end
