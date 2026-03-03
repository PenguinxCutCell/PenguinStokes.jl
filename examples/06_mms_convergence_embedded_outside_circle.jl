using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

const MU = 1.0
const XC = 0.5
const YC = 0.5
const R0 = 0.18
const R02 = R0^2

# Polynomial factors on [0,1]
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

# Streamfunction-based divergence-free velocity:
# ψ = (r²-R²)² * P(x) * Q(y)
function u_exact(x, y)
    g = geom_terms(x, y)
    return P(x) * (g.F * Qy(y) + g.Fy * Q(y))
end

function v_exact(x, y)
    g = geom_terms(x, y)
    return -Q(y) * (g.F * Px(x) + g.Fx * P(x))
end

function lap_u(x, y)
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

function lap_v(x, y)
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

fx(x, y) = -MU * lap_u(x, y)
fy(x, y) = -MU * lap_v(x, y)

# "outside circle" fluid convention (negative outside circle)
levelset_outside(x, y) = R0 - sqrt((x - XC)^2 + (y - YC)^2)

bc0 = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

function physical_active_indices(cap)
    li = LinearIndices(cap.nnodes)
    idx = Int[]
    for I in CartesianIndices(cap.nnodes)
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

function interface_active_indices(cap)
    li = LinearIndices(cap.nnodes)
    idx = Int[]
    for I in CartesianIndices(cap.nnodes)
        i = li[I]
        if I[1] < cap.nnodes[1] && I[2] < cap.nnodes[2]
            Γ = cap.buf.Γ[i]
            if isfinite(Γ) && Γ > 0.0
                push!(idx, i)
            end
        end
    end
    return idx
end

function velocity_errors(model, sys)
    u = sys.x[model.layout.uomega[1]]
    v = sys.x[model.layout.uomega[2]]
    li_u = LinearIndices(model.cap_u[1].nnodes)
    li_v = LinearIndices(model.cap_u[2].nnodes)

    eu2 = 0.0
    ev2 = 0.0
    wu = 0.0
    wv = 0.0

    for i in physical_active_indices(model.cap_u[1])
        pt = model.cap_u[1].C_ω[i]
        V = model.cap_u[1].buf.V[i]
        eu2 += V * (u[i] - u_exact(pt[1], pt[2]))^2
        wu += V
    end
    for i in physical_active_indices(model.cap_u[2])
        pt = model.cap_u[2].C_ω[i]
        V = model.cap_u[2].buf.V[i]
        ev2 += V * (v[i] - v_exact(pt[1], pt[2]))^2
        wv += V
    end

    iu = interface_active_indices(model.cap_u[1])
    iv = interface_active_indices(model.cap_u[2])
    uγ = sys.x[model.layout.ugamma[1]]
    vγ = sys.x[model.layout.ugamma[2]]
    eγu = isempty(iu) ? 0.0 : maximum(abs, uγ[iu])
    eγv = isempty(iv) ? 0.0 : maximum(abs, vγ[iv])

    return (uL2=sqrt(eu2 / wu), vL2=sqrt(ev2 / wv), uγInf=eγu, vγInf=eγv)
end

function main()
    ns = (25, 33, 49)
    hs = Float64[]
    uL2 = Float64[]
    vL2 = Float64[]

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            levelset_outside,
            MU,
            1.0;
            bc_u=(bc0, bc0),
            bc_cut=Dirichlet(0.0),
            force=(fx, fy),
        )
        sys = solve_steady!(model)
        m = velocity_errors(model, sys)

        h = 1.0 / (n - 1)
        push!(hs, h)
        push!(uL2, m.uL2)
        push!(vL2, m.vL2)

        println("n=$n  h=$h  uL2=$(m.uL2)  vL2=$(m.vL2)  uγInf=$(m.uγInf)  vγInf=$(m.vγInf)  ||Ax-b||=$(norm(sys.A * sys.x - sys.b))")
    end

    for k in 1:(length(ns) - 1)
        ou = log(uL2[k] / uL2[k + 1]) / log(hs[k] / hs[k + 1])
        ov = log(vL2[k] / vL2[k + 1]) / log(hs[k] / hs[k + 1])
        println("order $(ns[k])->$(ns[k+1]): u=$ou  v=$ov")
    end
end

main()
