using LinearAlgebra
using StaticArrays: SVector
using CartesianGrids
using PenguinBCs
using PenguinStokes

const MU = 1.0

# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------

function active_velocity_indices(cap)
    li = LinearIndices(cap.nnodes)
    idx = Int[]
    @inbounds for I in CartesianIndices(cap.nnodes)
        i = li[I]
        if I[1] < cap.nnodes[1] && I[2] < cap.nnodes[2]
            V = cap.buf.V[i]
            if isfinite(V) && V > 0.0
                push!(idx, i)
            end
        end
    end
    return idx
end

function velocity_l2(cap, vals::AbstractVector{T}, exact::Function) where {T}
    e2 = zero(T)
    w = zero(T)
    idx = active_velocity_indices(cap)
    @inbounds for i in idx
        x = cap.C_ω[i]
        V = cap.buf.V[i]
        e = vals[i] - exact(x[1], x[2])
        e2 += V * e * e
        w += V
    end
    return sqrt(e2 / w)
end

rate(eh, e2h, h, h2) = log(eh / e2h) / log(h / h2)

function print_orders(name, ns, hs, errs)
    println("\n[$name]")
    for k in eachindex(ns)
        println("n=$(ns[k])  h=$(hs[k])  err=$(errs[k])")
    end
    for k in 1:(length(ns) - 1)
        if errs[k] == 0.0 || errs[k + 1] == 0.0
            println("order $(ns[k])->$(ns[k+1]) = NaN (zero error level)")
        else
            println("order $(ns[k])->$(ns[k+1]) = ", rate(errs[k], errs[k + 1], hs[k], hs[k + 1]))
        end
    end
end

# ------------------------------------------------------------
# Case A: Monophasic no-body MMS
# ------------------------------------------------------------

uA(x, y) = pi * cos(pi * y) * sin(pi * x)
vA(x, y) = -pi * cos(pi * x) * sin(pi * y)
pA(x, y) = cos(pi * x) * sin(pi * y)

lap_uA(x, y) = -2 * pi^3 * cos(pi * y) * sin(pi * x)
lap_vA(x, y) = 2 * pi^3 * cos(pi * x) * sin(pi * y)

pxA(x, y) = -pi * sin(pi * x) * sin(pi * y)
pyA(x, y) = pi * cos(pi * x) * cos(pi * y)

fxA(x, y) = -MU * lap_uA(x, y) - pxA(x, y)
fyA(x, y) = -MU * lap_vA(x, y) - pyA(x, y)

body_full(x...) = -1.0

function caseA_convergence(; ns=(17, 33, 65))
    hs = Float64[]
    errs = Float64[]

    bcx = BorderConditions(
        ; left=Dirichlet((x, y) -> uA(x, y)),
        right=Dirichlet((x, y) -> uA(x, y)),
        bottom=Dirichlet((x, y) -> uA(x, y)),
        top=Dirichlet((x, y) -> uA(x, y)),
    )
    bcy = BorderConditions(
        ; left=Dirichlet((x, y) -> vA(x, y)),
        right=Dirichlet((x, y) -> vA(x, y)),
        bottom=Dirichlet((x, y) -> vA(x, y)),
        top=Dirichlet((x, y) -> vA(x, y)),
    )

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            body_full,
            MU,
            1.0;
            bc_u=(bcx, bcy),
            bc_cut=Dirichlet(0.0),
            force=(fxA, fyA),
        )
        sys = solve_steady!(model)

        ux = sys.x[model.layout.uomega[1]]
        uy = sys.x[model.layout.uomega[2]]
        eu = velocity_l2(model.cap_u[1], ux, uA)
        ev = velocity_l2(model.cap_u[2], uy, vA)
        e = sqrt(0.5 * (eu^2 + ev^2))

        push!(hs, 1.0 / (n - 1))
        push!(errs, e)

        println("[mono/no-body] n=$n  h=$(hs[end])  e=$(errs[end])  ||Ax-b||=$(norm(sys.A * sys.x - sys.b))")
    end

    print_orders("Mono no-body", collect(ns), hs, errs)
    return hs, errs
end

# ------------------------------------------------------------
# Case B: Monophasic embedded-body MMS (p=0)
# ------------------------------------------------------------

const XC = 0.5
const YC = 0.5
const R0 = 0.18
const R02 = R0^2

P(x) = x^2 * (1 - x)^2
Px(x) = 2x - 6x^2 + 4x^3
Pxx(x) = 2 - 12x + 12x^2
Pxxx(x) = -12 + 24x

Q(y) = y^2 * (1 - y)^2
Qy(y) = 2y - 6y^2 + 4y^3
Qyy(y) = 2 - 12y + 12y^2
Qyyy(y) = -12 + 24y

@inline function geom_terms(x, y)
    dx = x - XC
    dy = y - YC
    D = dx * dx + dy * dy - R02
    F = D^2
    Fx = 4 * dx * D
    Fy = 4 * dy * D
    Fxx = 4 * D + 8 * dx * dx
    Fyy = 4 * D + 8 * dy * dy
    Fxy = 8 * dx * dy
    Fxxy = 8 * dy
    Fxyy = 8 * dx
    Fxxx = 24 * dx
    Fyyy = 24 * dy
    return (; F, Fx, Fy, Fxx, Fyy, Fxy, Fxxy, Fxyy, Fxxx, Fyyy)
end

function uB(x, y)
    g = geom_terms(x, y)
    return P(x) * (g.F * Qy(y) + g.Fy * Q(y))
end

function vB(x, y)
    g = geom_terms(x, y)
    return -Q(y) * (g.F * Px(x) + g.Fx * P(x))
end

function lap_uB(x, y)
    g = geom_terms(x, y)
    px = Px(x)
    pxx = Pxx(x)
    p = P(x)
    q = Q(y)
    qy = Qy(y)
    qyy = Qyy(y)
    qyyy = Qyyy(y)

    U1 = g.F * qy + g.Fy * q
    U1x = g.Fx * qy + g.Fxy * q
    U1xx = g.Fxx * qy + g.Fxxy * q
    U1yy = 3 * g.Fyy * qy + 3 * g.Fy * qyy + g.F * qyyy + g.Fyyy * q

    uxx = pxx * U1 + 2 * px * U1x + p * U1xx
    uyy = p * U1yy
    return uxx + uyy
end

function lap_vB(x, y)
    g = geom_terms(x, y)
    p = P(x)
    px = Px(x)
    pxx = Pxx(x)
    pxxx = Pxxx(x)
    q = Q(y)
    qy = Qy(y)
    qyy = Qyy(y)

    V1 = g.F * px + g.Fx * p
    V1y = g.Fy * px + g.Fxy * p
    V1yy = g.Fyy * px + g.Fxyy * p
    V1xx = 3 * g.Fxx * px + 3 * g.Fx * pxx + g.F * pxxx + g.Fxxx * p

    vxx = -q * V1xx
    vyy = -(qyy * V1 + 2 * qy * V1y + q * V1yy)
    return vxx + vyy
end

fxB(x, y) = -MU * lap_uB(x, y)
fyB(x, y) = -MU * lap_vB(x, y)

levelset_outside(x, y) = R0 - sqrt((x - XC)^2 + (y - YC)^2)

function caseB_convergence(; ns=(25, 33, 49))
    hs = Float64[]
    errs = Float64[]

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            levelset_outside,
            MU,
            1.0;
            bc_u=(bc0, bc0),
            bc_cut=Dirichlet(0.0),
            force=(fxB, fyB),
        )
        sys = solve_steady!(model)

        ux = sys.x[model.layout.uomega[1]]
        uy = sys.x[model.layout.uomega[2]]
        eu = velocity_l2(model.cap_u[1], ux, uB)
        ev = velocity_l2(model.cap_u[2], uy, vB)
        e = sqrt(0.5 * (eu^2 + ev^2))

        push!(hs, 1.0 / (n - 1))
        push!(errs, e)

        println("[mono/embedded] n=$n  h=$(hs[end])  e=$(errs[end])  ||Ax-b||=$(norm(sys.A * sys.x - sys.b))")
    end

    print_orders("Mono embedded", collect(ns), hs, errs)
    return hs, errs
end

# ------------------------------------------------------------
# Case C: Two-phase fixed-interface MMS (nontrivial)
# ------------------------------------------------------------
#
# Nonzero velocity + nonzero interface traction forcing:
#   - two-layer body-force Poiseuille profile (u(y), v=0)
#   - pressure jump p2-p1 = Δp across interface
#   - interface forcing gΓ = (0, Δp)
#
# This avoids equilibrium-zero error levels while keeping an analytic profile.

function two_layer_bodyforce_poiseuille_profile(mu1, mu2, h, H, G)
    # With PenguinStokes sign convention: μ Δu = -G for f=(G,0), p_x=0.
    C1 = -G / mu1
    C2 = -G / mu2
    M = [
        mu1 -mu2 0.0
        h -h -1.0
        0.0 H 1.0
    ]
    rhs = [
        0.0,
        0.5 * (C2 - C1) * h^2,
        -0.5 * C2 * H^2,
    ]
    A1, A2, B2 = M \ rhs
    u(y) = y <= h ? (0.5 * C1 * y^2 + A1 * y) : (0.5 * C2 * y^2 + A2 * y + B2)
    return u
end

function velocity_l2_two_phase_profile(cap_u, ux, uy, u_exact::Function)
    eu = zero(eltype(ux))
    ev = zero(eltype(ux))
    wu = zero(eltype(ux))
    wv = zero(eltype(ux))

    idxu = active_velocity_indices(cap_u[1])
    idxv = active_velocity_indices(cap_u[2])

    @inbounds for i in idxu
        x = cap_u[1].C_ω[i]
        V = cap_u[1].buf.V[i]
        e = ux[i] - u_exact(x[2])
        eu += V * e * e
        wu += V
    end
    @inbounds for i in idxv
        V = cap_u[2].buf.V[i]
        e = uy[i] # v_exact = 0
        ev += V * e * e
        wv += V
    end

    return sqrt(0.5 * (eu / wu + ev / wv))
end

function caseC_convergence(; ns=(65, 97, 129), mu1=1.0, mu2=4.0, hif=0.4, G=1.5, dp=0.75)
    hs = Float64[]
    err1 = Float64[]
    err2 = Float64[]

    body(x, y) = y - hif
    u_exact = two_layer_bodyforce_poiseuille_profile(mu1, mu2, hif, 1.0, G)

    bcx = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    iface(x, y) = SVector(0.0, dp)

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelTwoPhase(
            grid,
            body,
            mu1,
            mu2;
            bc_u=(bcx, bcy),
            force1=(G, 0.0),
            force2=(G, 0.0),
            interface_force=iface,
            gauge=MeanPressureGauge(),
        )
        sys = solve_steady!(model)

        u1x = sys.x[model.layout.uomega1[1]]
        u1y = sys.x[model.layout.uomega1[2]]
        u2x = sys.x[model.layout.uomega2[1]]
        u2y = sys.x[model.layout.uomega2[2]]

        e1 = velocity_l2_two_phase_profile(model.cap_u1, u1x, u1y, u_exact)
        e2 = velocity_l2_two_phase_profile(model.cap_u2, u2x, u2y, u_exact)

        idxv1 = active_velocity_indices(model.cap_u1[2])
        idxv2 = active_velocity_indices(model.cap_u2[2])
        v1inf = isempty(idxv1) ? 0.0 : maximum(abs.(u1y[idxv1]))
        v2inf = isempty(idxv2) ? 0.0 : maximum(abs.(u2y[idxv2]))

        push!(hs, 1.0 / (n - 1))
        push!(err1, e1)
        push!(err2, e2)

        println("[two-phase/fixed] n=$n  h=$(hs[end])  e1=$e1  e2=$e2  v1Inf=$v1inf  v2Inf=$v2inf  ||Ax-b||=$(norm(sys.A * sys.x - sys.b))")
    end

    print_orders("Two-phase fixed interface (phase 1 vel)", collect(ns), hs, err1)
    print_orders("Two-phase fixed interface (phase 2 vel)", collect(ns), hs, err2)
    if length(hs) >= 2
        r1 = rate(err1[end - 1], err1[end], hs[end - 1], hs[end])
        r2 = rate(err2[end - 1], err2[end], hs[end - 1], hs[end])
        println("[two-phase/fixed] fine-grid order check: phase1=$r1, phase2=$r2, target>1.3")
    end
    return hs, err1, err2
end

function main()
    println("MMS convergence suite (steady)")
    println("- monophasic no body")
    println("- monophasic embedded body")
    println("- two-phase fixed interface")

    caseA_convergence()
    caseB_convergence()
    caseC_convergence()
end

main()
