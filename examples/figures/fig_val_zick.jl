#
# Generates Figure `fig:val-zick` of the staggered cut-cell Stokes article
# (Section ``Stokes flow past a periodic array of spheres'').
#
# The configuration follows examples/32_periodic_sphere_array.jl:
#   - periodic unit cube Ω = [0, 1]³ with a single sphere of radius
#     R = (3 φ /(4π))^{1/3} at the centre;
#   - uniform horizontal body force drives the flow;
#   - no-slip on the sphere;
#   - mean-pressure gauge.
#
# Drag coefficient K following the Zick & Homsy convention:
#   F = 6 π μ R K U_avg   ⇒   K = L² / (6 π μ R U_avg (1 − φ)).
#
# Three-panel layout:
#   (a) mid-plane slice (z = 0.5) of |u|, with the sphere outlined as a circle;
#   (b) K vs φ for three mesh resolutions vs the Zick-Homsy reference;
#   (c) relative error of K vs h at fixed φ = 0.125.

using LinearAlgebra
using Printf
using CairoMakie
using CartesianGrids
using PenguinBCs
using PenguinStokes
using PenguinSolverCore: LinearSystem

# ── Zick & Homsy reference table ────────────────────────────────────────────
const ZICK_REF = [
    (0.027,  2.008),
    (0.064,  2.810),
    (0.125,  4.292),
    (0.216,  7.442),
    (0.343, 15.40),
    (0.450, 28.10),
    (0.524, 42.10),
]

function periodic_3d_bc()
    return (
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic(),
                         backward=Periodic(), forward=Periodic()),
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic(),
                         backward=Periodic(), forward=Periodic()),
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic(),
                         backward=Periodic(), forward=Periodic()),
    )
end

function compute_average_velocity_3d(model, sys)
    u1 = sys.x[model.layout.uomega[1]]
    tot = 0.0; vol = 0.0
    @inbounds for i in 1:model.cap_u[1].ntotal
        V = model.cap_u[1].buf.V[i]
        if isfinite(V) && V > 0
            tot += V * u1[i]; vol += V
        end
    end
    return vol > 0 ? tot / vol : 1.0
end

function solve_sphere_case(n::Int, phi::Float64)
    L = 1.0
    grid = CartesianGrid((0.0, 0.0, 0.0), (L, L, L), (n, n, n))
    R = (3 * phi / (4π))^(1/3)
    body(x, y, z) = R - sqrt((x - L/2)^2 + (y - L/2)^2 + (z - L/2)^2)

    model = StokesModelMono(
        grid, body, 1.0, 1.0;
        bc_u = periodic_3d_bc(),
        bc_p = BorderConditions(; left=Periodic(), right=Periodic(),
                                bottom=Periodic(), top=Periodic(),
                                backward=Periodic(), forward=Periodic()),
        bc_cut = Dirichlet(0.0),
        force = (1.0, 0.0, 0.0),
        gauge = MeanPressureGauge(),
    )
    sys = solve_steady!(model)

    u_avg = compute_average_velocity_3d(model, sys)
    K = L^2 / (6π * R * u_avg * (1 - phi) + 1e-15)
    res = norm(sys.A * sys.x - sys.b)
    return (K = K, R = R, residual = res, model = model, sys = sys, u_avg = u_avg)
end

# ── Sweep parameters (kept small for 3D runtime) ────────────────────────────
const NS    = (7, 9, 11)
const PHIS  = [r[1] for r in ZICK_REF]
const PHI_CONV = 0.125
const K_REF_CONV = 4.292
const NS_CONV  = (7, 9, 11, 13)

println("Zick-Homsy sweep — drag coefficient K vs φ:")
results = Dict{Tuple{Int,Float64},Any}()
for n in NS, phi in PHIS
    r = solve_sphere_case(n, phi)
    results[(n, phi)] = r
    @printf("  n=%-3d  φ=%.3f  K=%.3f  res=%.2e\n", n, phi, r.K, r.residual)
end

println("\nConvergence at fixed φ = $PHI_CONV:")
conv = Any[]
for n in NS_CONV
    r = solve_sphere_case(n, PHI_CONV)
    push!(conv, (n = n, h = 1.0 / (n - 1), K = r.K, R = r.R, res = r.residual,
                 model = r.model, sys = r.sys))
    @printf("  n=%-2d  h=%.4f  K=%.3f  rel.err.=%.2e  res=%.2e\n",
            n, conv[end].h, r.K,
            abs(r.K - K_REF_CONV) / K_REF_CONV, r.residual)
end

# ── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(; size = (1200, 520), fontsize = 15)

# (a) K vs φ for three meshes vs Zick-Homsy ----------------------------------
ax_a = Axis(
    fig[1, 1];
    xlabel = L"\phi", ylabel = L"K",
    yscale = log10,
    title = "(a) Drag coefficient K vs volume fraction",
)
colors_ns = cgrad(:viridis, length(NS); categorical = true)
markers   = (:circle, :utriangle, :diamond)
for (k, n) in enumerate(NS)
    Ks = [results[(n, phi)].K for phi in PHIS]
    scatterlines!(ax_a, PHIS, Ks;
                  color = colors_ns[k], marker = markers[k],
                  linewidth = 1.8, markersize = 10,
                  label = @sprintf("n = %d", n))
end
phi_ref = [r[1] for r in ZICK_REF]
K_ref   = [r[2] for r in ZICK_REF]
scatterlines!(ax_a, phi_ref, K_ref;
              color = :black, marker = :xcross,
              linestyle = :dash, linewidth = 2, markersize = 12,
              label = "Zick-Homsy (1982)")
axislegend(ax_a; position = :lt, labelsize = 11)

# (b) Convergence at fixed φ = 0.125 -----------------------------------------
ax_b = Axis(
    fig[1, 2];
    xlabel = L"h", ylabel = L"|K_h - K_\mathrm{ref}|/K_\mathrm{ref}",
    xscale = log10, yscale = log10,
    title = @sprintf("(b) K convergence at φ = %.3f", PHI_CONV),
)
hs_c   = [c.h for c in conv]
errs_c = [abs(c.K - K_REF_CONV) / K_REF_CONV for c in conv]
scatterlines!(ax_b, hs_c, errs_c;
              color = :steelblue, marker = :circle,
              linewidth = 2, markersize = 11,
              label = "this work")
yref1 = errs_c[end] .* (hs_c ./ hs_c[end])
yref2 = errs_c[end] .* (hs_c ./ hs_c[end]) .^ 2
lines!(ax_b, hs_c, yref1; color = :black, linestyle = :dashdot,
       linewidth = 1.4, label = L"\mathcal{O}(h)")
lines!(ax_b, hs_c, yref2; color = :black, linestyle = :dash,
       linewidth = 1.4, label = L"\mathcal{O}(h^{2})")
axislegend(ax_b; position = :rb, labelsize = 11)

Label(fig[0, 1:2],
    "Stokes flow past a periodic array of spheres (Zick-Homsy benchmark)";
    fontsize = 18, halign = :center)

outpath = joinpath(@__DIR__, "fig_val_zick.png")
save(outpath, fig; px_per_unit = 2)
println("\nsaved figure: ", outpath)
