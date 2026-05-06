using LinearAlgebra
using Printf

using CartesianGrids
using PenguinBCs
using PenguinStokes

# Poiseuille flow in an embedded periodic channel.
#
# This mirrors the Basilisk embedded-boundary Poiseuille test:
#   domain: [-1,1] x [-1,1]
#   fluid:  -H < y < H, with H = 0.5
#   x:      periodic
#   walls:  embedded no-slip boundaries at y = +/-H
#   drive:  constant body force G in the x direction
#
# For mu = 1 and G = 1, the exact velocity is
#   u(y) = G/(2mu) * (H^2 - y^2), v = 0, p = const.

const MU = 1.0
const G = 1.0
const H = 0.5
const EPSWALL = 1e-12

u_exact(y) = (G / (2 * MU)) * (H^2 - y^2)
v_exact(_x, _y) = 0.0

# PenguinStokes capacities use the negative side of the level set as fluid.
# The tiny inward offset avoids exact wall/grid alignment on powers-of-two meshes.
channel_levelset(_x, y) = abs(y) - (H - EPSWALL)

function channel_bcs()
    ux = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    uy = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    p = BorderConditions(
        ; left=Periodic(), right=Periodic(),
    )
    return ux, uy, p
end

function active_velocity_indices(cap)
    idx = Int[]
    li = LinearIndices(cap.nnodes)
    @inbounds for I in CartesianIndices(cap.nnodes)
        i = li[I]
        if I[1] == cap.nnodes[1] || I[2] == cap.nnodes[2]
            continue
        end
        V = cap.buf.V[i]
        if isfinite(V) && V > 0.0
            push!(idx, i)
        end
    end
    return idx
end

function velocity_error_metrics(model::StokesModelMono{2,T}, sys) where {T}
    ux = sys.x[model.layout.uomega[1]]
    uy = sys.x[model.layout.uomega[2]]

    h = minimum(meshsize(model.gridp))
    full_area = h^2
    cut_tol = 1e-10 * full_area

    sum_all = zero(T)
    sum_cut = zero(T)
    sum_full = zero(T)
    vol_all = zero(T)
    vol_cut = zero(T)
    vol_full = zero(T)
    max_all = zero(T)
    max_cut = zero(T)
    max_full = zero(T)

    for (d, ufield) in enumerate((ux, uy))
        cap = model.cap_u[d]
        for i in active_velocity_indices(cap)
            V = cap.buf.V[i]
            x = cap.C_ω[i]
            target = d == 1 ? u_exact(x[2]) : v_exact(x[1], x[2])
            e = abs(ufield[i] - target)
            e2 = e^2
            is_cut = cap.buf.Γ[i] > 0.0 || V < full_area - cut_tol

            sum_all += V * e2
            vol_all += V
            max_all = max(max_all, e)

            if is_cut
                sum_cut += V * e2
                vol_cut += V
                max_cut = max(max_cut, e)
            else
                sum_full += V * e2
                vol_full += V
                max_full = max(max_full, e)
            end
        end
    end

    return (
        l2_all=sqrt(sum_all / vol_all),
        linf_all=max_all,
        l2_cut=vol_cut > 0 ? sqrt(sum_cut / vol_cut) : zero(T),
        linf_cut=max_cut,
        l2_full=vol_full > 0 ? sqrt(sum_full / vol_full) : zero(T),
        linf_full=max_full,
    )
end

rate(eh, e2h, h, h2) = log(eh / e2h) / log(h / h2)

function solve_case(n::Int)
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    bcux, bcuy, bcp = channel_bcs()

    model = StokesModelMono(
        grid,
        channel_levelset,
        MU,
        1.0;
        bc_u=(bcux, bcuy),
        bc_p=bcp,
        bc_cut=Dirichlet(0.0),
        force=(G, 0.0),
        gauge=MeanPressureGauge(),
    )

    sys = solve_steady!(model)
    err = velocity_error_metrics(model, sys)
    h = 2.0 / (n - 1)
    residual = norm(sys.A * sys.x - sys.b)
    return (; n, h, model, sys, err, residual)
end

function main()
    ns = (33, 65, 129)
    results = [solve_case(n) for n in ns]

    println("Embedded periodic-channel Poiseuille flow")
    println("Domain=[-1,1]^2, fluid |y| < $H, periodic in x")
    println("Exact: u(y) = 0.5 * ($H^2 - y^2), v = 0")
    println()
    @printf "%6s %10s %12s %12s %12s %12s %12s %12s %12s\n" "n" "h" "L2 all" "Linf all" "L2 cut" "Linf cut" "L2 full" "Linf full" "residual"
    for r in results
        @printf "%6d %10.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n" r.n r.h r.err.l2_all r.err.linf_all r.err.l2_cut r.err.linf_cut r.err.l2_full r.err.linf_full r.residual
    end

    println()
    for k in 1:(length(results) - 1)
        a = results[k]
        b = results[k + 1]
        @printf "order %d->%d: L2 all %.3f, cut %.3f, full %.3f; Linf all %.3f, cut %.3f, full %.3f\n" a.n b.n (
            rate(a.err.l2_all, b.err.l2_all, a.h, b.h),
            rate(a.err.l2_cut, b.err.l2_cut, a.h, b.h),
            rate(a.err.l2_full, b.err.l2_full, a.h, b.h),
            rate(a.err.linf_all, b.err.linf_all, a.h, b.h),
            rate(a.err.linf_cut, b.err.linf_cut, a.h, b.h),
            rate(a.err.linf_full, b.err.linf_full, a.h, b.h),
        )...
    end
end

main()
