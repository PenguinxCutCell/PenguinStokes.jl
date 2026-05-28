#
# Generates Figure `fig:val-sangani` of the staggered cut-cell Stokes article
# (Section ``Stokes flow past a periodic array of cylinders'').
#
# The configuration follows examples/31_periodic_cylinder_array.jl:
#   - periodic unit square Ω = [0, 1]² with a single cylinder of radius
#     R = sqrt(Φ/π) at the centre;
#   - uniform horizontal body force drives the flow;
#   - no-slip on the cylinder;
#   - mean-pressure gauge.
#
# The non-dimensional drag F/(μ U_avg) is obtained from the integrated
# embedded traction divided by the spatial-average velocity, with the
# Basilisk/Sangani-Acrivos normalisation F/(μU) = L²/(1 − Φ) / (μ U_avg).
#
# Three-panel layout:
#   (a) flow snapshot at moderate Φ = 0.30 (|u| heatmap + streamlines);
#   (b) F/(μU) vs Φ for three mesh resolutions vs the Sangani-Acrivos data;
#   (c) relative drag error vs h at fixed Φ = 0.30, with an h¹ guide line.

using LinearAlgebra
using Printf
using CairoMakie
using CartesianGrids
using PenguinBCs
using PenguinStokes
using PenguinSolverCore: LinearSystem

# ── Sangani & Acrivos reference table ───────────────────────────────────────
const SANGANI_REF = [
    (0.05,  15.56),
    (0.10,  24.83),
    (0.20,  51.53),
    (0.30, 102.90),
    (0.40, 217.89),
    (0.50, 532.55),
    (0.60, 1763.0),
    (0.70, 13520.0),
]

# ── Solver call (lifted from example 31) ────────────────────────────────────
function periodic_2d_bc()
    return (
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic()),
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic()),
    )
end

function compute_average_velocity(model, sys)
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

function compute_volume_fraction(model)
    cap_p = model.cap_p
    fv = 0.0
    @inbounds for i in 1:cap_p.ntotal
        V = cap_p.buf.V[i]
        isfinite(V) && V > 0 && (fv += V)
    end
    Lx = model.gridp.hc[1] - model.gridp.lc[1]
    Ly = model.gridp.hc[2] - model.gridp.lc[2]
    dom = Lx * Ly
    return (phi = 1 - fv / dom, dom = dom)
end

function solve_cylinder_case(n::Int, phi::Float64)
    L = 1.0
    grid = CartesianGrid((0.0, 0.0), (L, L), (n, n))
    R = sqrt(phi * L^2 / π)
    body(x, y) = R - sqrt((x - L/2)^2 + (y - L/2)^2)

    model = StokesModelMono(
        grid, body, 1.0, 1.0;
        bc_u = periodic_2d_bc(),
        bc_p = BorderConditions(; left=Periodic(), right=Periodic(),
                                bottom=Periodic(), top=Periodic()),
        bc_cut = Dirichlet(0.0),
        force = (1.0, 0.0),
        gauge = MeanPressureGauge(),
    )
    sys = solve_steady!(model)

    u_avg = compute_average_velocity(model, sys)
    vol = compute_volume_fraction(model)
    drag_nondim = (vol.dom / (1 - vol.phi)) / (u_avg + 1e-15)
    res = norm(sys.A * sys.x - sys.b)
    return (drag = drag_nondim, phi_h = vol.phi, residual = res,
            model = model, sys = sys, u_avg = u_avg)
end

# ── Sweep parameters ────────────────────────────────────────────────────────
const NS    = (17, 33, 49)
const PHIS  = [r[1] for r in SANGANI_REF]
const PHI_CONV = 0.30                          # for convergence panel (c)
const NS_CONV  = (13, 17, 25, 33, 49, 65)

println("Sangani-Acrivos sweep — non-dimensional drag F/(μU) vs Φ:")
results = Dict{Tuple{Int,Float64},Any}()
for n in NS, phi in PHIS
    r = solve_cylinder_case(n, phi)
    results[(n, phi)] = r
    @printf("  n=%-3d  Φ=%.3f  F/(μU)=%.3f  Φ_h=%.3f  res=%.2e\n",
            n, phi, r.drag, r.phi_h, r.residual)
end

println("\nConvergence at fixed Φ = $PHI_CONV:")
conv = Any[]
for n in NS_CONV
    r = solve_cylinder_case(n, PHI_CONV)
    push!(conv, (n = n, h = 1.0 / (n - 1), drag = r.drag, res = r.residual,
                 model = r.model, sys = r.sys, u_avg = r.u_avg))
    @printf("  n=%-3d  h=%.4f  F/(μU)=%.3f  rel.err.=%.2e  res=%.2e\n",
            n, conv[end].h, r.drag,
            abs(r.drag - 102.90) / 102.90, r.residual)
end

# Pick the n=33 case at Φ = 0.30 for the snapshot.
snap = let pick = nothing
    for c in conv
        c.n == 33 && (pick = c; break)
    end
    pick === nothing ? conv[end] : pick
end

# ── Figure ──────────────────────────────────────────────────────────────────
fig = Figure(; size = (1500, 520), fontsize = 15)

# (a) Flow snapshot ----------------------------------------------------------
ax_a = Axis(
    fig[1, 1];
    aspect = DataAspect(),
    xlabel = "x", ylabel = "y",
    title = @sprintf("(a) |u| field at Φ = %.2f, n = %d", PHI_CONV, snap.n),
    limits = (0, 1, 0, 1),
)
nf = snap.n
xs = range(0.0, 1.0; length = nf)
ys = range(0.0, 1.0; length = nf)
uxh = snap.sys.x[snap.model.layout.uomega[1]]
uyh = snap.sys.x[snap.model.layout.uomega[2]]
LI = LinearIndices((nf, nf))

R_snap = sqrt(PHI_CONV / π)
speed = fill(NaN, nf, nf)
ux_grid = fill(NaN, nf, nf)
uy_grid = fill(NaN, nf, nf)
@inbounds for j in 1:nf, i in 1:nf
    if (xs[i] - 0.5)^2 + (ys[j] - 0.5)^2 > R_snap^2
        idx = LI[i, j]
        if idx <= length(uxh) && idx <= length(uyh)
            ux_grid[i, j] = uxh[idx]
            uy_grid[i, j] = uyh[idx]
            speed[i, j]   = sqrt(uxh[idx]^2 + uyh[idx]^2)
        end
    end
end

hm = heatmap!(ax_a, xs, ys, speed; colormap = :viridis, nan_color = :gray85)
# Streamlines via a simple seed sweep along the inflow column. Stop a half
# cell before the cylinder so streamlines never touch its surface.
nseed = 18
stop_margin = 1.0 / (nf - 1)              # one grid cell
for ys0 in range(0.05, 0.95; length = nseed)
    xs_path = Float64[0.02]
    ys_path = Float64[ys0]
    x = 0.02; y = ys0
    for _ in 1:600
        # Halt if the next step would enter the disk's safety band.
        if (x - 0.5)^2 + (y - 0.5)^2 < (R_snap + stop_margin)^2
            break
        end
        i = clamp(round(Int, x * (nf - 1)) + 1, 1, nf)
        j = clamp(round(Int, y * (nf - 1)) + 1, 1, nf)
        u = ux_grid[i, j]; v = uy_grid[i, j]
        (!isfinite(u) || !isfinite(v)) && break
        sp = sqrt(u^2 + v^2) + 1e-12
        ds = 0.002
        x += ds * u / sp
        y += ds * v / sp
        x > 1.0 && break
        (y < 0 || y > 1) && break
        push!(xs_path, x); push!(ys_path, y)
    end
    length(xs_path) > 5 && lines!(ax_a, xs_path, ys_path;
                                  color = (:white, 0.55), linewidth = 1.0)
end

θ = range(0, 2π; length = 400)
poly!(ax_a, Point2f.(0.5 .+ R_snap .* cos.(θ), 0.5 .+ R_snap .* sin.(θ));
      color = :white, strokecolor = :crimson, strokewidth = 2.0)
Colorbar(fig[1, 2], hm; label = L"|u|", width = 14)

# (b) F/(μU) vs Φ for three meshes vs Sangani-Acrivos ------------------------
ax_b = Axis(
    fig[1, 3];
    xlabel = L"\Phi", ylabel = L"F/(\mu U)",
    yscale = log10,
    title = "(b) Non-dimensional drag vs volume fraction",
)
colors_ns = cgrad(:viridis, length(NS); categorical = true)
markers   = (:circle, :utriangle, :diamond)
for (k, n) in enumerate(NS)
    drags = [results[(n, phi)].drag for phi in PHIS]
    scatterlines!(ax_b, PHIS, drags;
                  color = colors_ns[k], marker = markers[k],
                  linewidth = 1.8, markersize = 10,
                  label = @sprintf("n = %d", n))
end
phi_ref = [r[1] for r in SANGANI_REF]
f_ref   = [r[2] for r in SANGANI_REF]
scatterlines!(ax_b, phi_ref, f_ref;
              color = :black, marker = :xcross,
              linestyle = :dash, linewidth = 2, markersize = 12,
              label = "Sangani-Acrivos (1982)")
axislegend(ax_b; position = :lt, labelsize = 11)

# (c) Convergence at fixed Φ = 0.30 ------------------------------------------
ax_c = Axis(
    fig[1, 4];
    xlabel = L"h", ylabel = L"|F_h - F_\mathrm{ref}|/F_\mathrm{ref}",
    xscale = log10, yscale = log10,
    title = @sprintf("(c) Drag convergence at Φ = %.2f", PHI_CONV),
)
hs_c    = [c.h for c in conv]
errs_c  = [abs(c.drag - 102.90) / 102.90 for c in conv]
scatterlines!(ax_c, hs_c, errs_c;
              color = :steelblue, marker = :circle,
              linewidth = 2, markersize = 11,
              label = "this work")
yref1 = errs_c[end] .* (hs_c ./ hs_c[end])
yref2 = errs_c[end] .* (hs_c ./ hs_c[end]) .^ 2
lines!(ax_c, hs_c, yref1; color = :black, linestyle = :dashdot,
       linewidth = 1.4, label = L"\mathcal{O}(h)")
lines!(ax_c, hs_c, yref2; color = :black, linestyle = :dash,
       linewidth = 1.4, label = L"\mathcal{O}(h^{2})")
axislegend(ax_c; position = :rb, labelsize = 11)

Label(fig[0, 1:4],
    "Stokes flow past a periodic array of cylinders (Sangani-Acrivos benchmark)";
    fontsize = 18, halign = :center)

outpath = joinpath(@__DIR__, "fig_val_sangani.png")
save(outpath, fig; px_per_unit = 2)
println("\nsaved figure: ", outpath)
