using LinearAlgebra
using SparseArrays: nzrange
using CartesianGrids
using PenguinBCs
using PenguinStokes

mu = 1.0

# Streamfunction-based MMS on [0,1]^2
u_exact(x, y) = pi * cos(pi * y) * sin(pi * x)
v_exact(x, y) = -pi * cos(pi * x) * sin(pi * y)
p_exact(x, y) = cos(pi * x) * sin(pi * y)

lap_u(x, y) = -2 * pi^3 * cos(pi * y) * sin(pi * x)
lap_v(x, y) = 2 * pi^3 * cos(pi * x) * sin(pi * y)

dpx(x, y) = -pi * sin(pi * x) * sin(pi * y)
dpy(x, y) = pi * cos(pi * x) * cos(pi * y)

# Sign matches current PenguinStokes momentum operator convention.
fx(x, y) = -mu * lap_u(x, y) - dpx(x, y)
fy(x, y) = -mu * lap_v(x, y) - dpy(x, y)

body(x...) = -1.0

bcx = BorderConditions(
    ; left=Dirichlet((x, y) -> u_exact(x, y)),
    right=Dirichlet((x, y) -> u_exact(x, y)),
    bottom=Dirichlet((x, y) -> u_exact(x, y)),
    top=Dirichlet((x, y) -> u_exact(x, y)),
)

bcy = BorderConditions(
    ; left=Dirichlet((x, y) -> v_exact(x, y)),
    right=Dirichlet((x, y) -> v_exact(x, y)),
    bottom=Dirichlet((x, y) -> v_exact(x, y)),
    top=Dirichlet((x, y) -> v_exact(x, y)),
)

function pressure_active_indices(model, A)
    nn = model.cap_p.nnodes
    li = LinearIndices(nn)
    pfirst = first(model.layout.pomega)
    plast = last(model.layout.pomega)
    idx = Int[]

    for I in CartesianIndices(nn)
        if !(1 < I[1] < nn[1] && 1 < I[2] < nn[2])
            continue
        end
        i = li[I]
        V = model.cap_p.buf.V[i]
        if !(isfinite(V) && V > 0.0)
            continue
        end

        col = model.layout.pomega[i]
        coupled = false
        for ptr in nzrange(A, col)
            r = A.rowval[ptr]
            if (r < pfirst || r > plast) && A.nzval[ptr] != 0.0
                coupled = true
                break
            end
        end
        coupled && push!(idx, i)
    end

    return idx
end

function metrics(model, sys)
    u = sys.x[model.layout.uomega[1]]
    v = sys.x[model.layout.uomega[2]]
    p = sys.x[model.layout.pomega]

    li_u = LinearIndices(model.cap_u[1].nnodes)
    li_v = LinearIndices(model.cap_u[2].nnodes)
    eu2 = 0.0
    ev2 = 0.0
    wu = 0.0
    wv = 0.0
    euinf = 0.0
    evinf = 0.0

    for I in CartesianIndices(model.cap_u[1].nnodes)
        i = li_u[I]
        if I[1] < model.cap_u[1].nnodes[1] && I[2] < model.cap_u[1].nnodes[2]
            V = model.cap_u[1].buf.V[i]
            if V > 0.0
                x = model.cap_u[1].C_ω[i]
                e = abs(u[i] - u_exact(x[1], x[2]))
                eu2 += V * e^2
                wu += V
                euinf = max(euinf, e)
            end
        end
    end

    for I in CartesianIndices(model.cap_u[2].nnodes)
        i = li_v[I]
        if I[1] < model.cap_u[2].nnodes[1] && I[2] < model.cap_u[2].nnodes[2]
            V = model.cap_u[2].buf.V[i]
            if V > 0.0
                x = model.cap_u[2].C_ω[i]
                e = abs(v[i] - v_exact(x[1], x[2]))
                ev2 += V * e^2
                wv += V
                evinf = max(evinf, e)
            end
        end
    end

    pidx = pressure_active_indices(model, sys.A)
    pn = [p[i] for i in pidx]
    pe = [begin
        x = model.cap_p.C_ω[i]
        p_exact(x[1], x[2])
    end for i in pidx]
    pshift = sum(pn .- pe) / length(pn)

    ep2 = 0.0
    wp = 0.0
    for (k, i) in enumerate(pidx)
        V = model.cap_p.buf.V[i]
        e = (pn[k] - pshift) - pe[k]
        ep2 += V * e^2
        wp += V
    end

    nt = model.cap_p.ntotal
    div = zeros(Float64, nt)
    for d in 1:2
        rows = ((d - 1) * nt + 1):(d * nt)
        Gd = model.op_p.G[rows, :]
        Hd = model.op_p.H[rows, :]
        div .+= -((Gd' + Hd') * sys.x[model.layout.uomega[d]]) + (Hd' * sys.x[model.layout.ugamma[d]])
    end

    d2 = 0.0
    for i in pidx
        V = model.cap_p.buf.V[i]
        d2 += V * div[i]^2
    end

    return (
        uL2=sqrt(eu2 / wu),
        vL2=sqrt(ev2 / wv),
        uInf=euinf,
        vInf=evinf,
        pL2=sqrt(ep2 / wp),
        divL2=sqrt(d2 / wp),
        pmax=maximum(abs.(pn)),
    )
end

function exact_state(model)
    x = zeros(Float64, last(model.layout.pomega))
    for d in 1:2
        cap = model.cap_u[d]
        li = LinearIndices(cap.nnodes)
        rows = model.layout.uomega[d]
        for I in CartesianIndices(cap.nnodes)
            i = li[I]
            if I[1] < cap.nnodes[1] && I[2] < cap.nnodes[2]
                pt = cap.C_ω[i]
                x[rows[i]] = d == 1 ? u_exact(pt[1], pt[2]) : v_exact(pt[1], pt[2])
            end
        end
    end

    cap = model.cap_p
    li = LinearIndices(cap.nnodes)
    rows = model.layout.pomega
    for I in CartesianIndices(cap.nnodes)
        i = li[I]
        if I[1] < cap.nnodes[1] && I[2] < cap.nnodes[2]
            pt = cap.C_ω[i]
            x[rows[i]] = p_exact(pt[1], pt[2])
        end
    end
    return x
end

function momentum_residual_split(model, sys)
    xe = exact_state(model)
    r = sys.A * xe - sys.b
    interior = Int[]
    boundary = Int[]

    for d in 1:2
        cap = model.cap_u[d]
        rows = model.layout.uomega[d]
        li = LinearIndices(cap.nnodes)
        for I in CartesianIndices(cap.nnodes)
            i = li[I]
            if !(I[1] < cap.nnodes[1] && I[2] < cap.nnodes[2])
                continue
            end
            V = cap.buf.V[i]
            if !(isfinite(V) && V > 0.0)
                continue
            end
            strict_interior = (2 <= I[1] <= cap.nnodes[1] - 2) && (2 <= I[2] <= cap.nnodes[2] - 2)
            if strict_interior
                push!(interior, rows[i])
            else
                push!(boundary, rows[i])
            end
        end
    end

    r_int = isempty(interior) ? 0.0 : norm(r[interior], Inf)
    r_bnd = isempty(boundary) ? 0.0 : norm(r[boundary], Inf)
    return r_int, r_bnd
end

ns = (17, 33, 65)
hs = Float64[]
uL2 = Float64[]
vL2 = Float64[]
uInf = Float64[]
vInf = Float64[]
pL2 = Float64[]
divL2 = Float64[]
pmax = Float64[]
rInt = Float64[]
rBnd = Float64[]

for n in ns
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    model = StokesModelMono(grid, body, mu, 1.0; bc_u=(bcx, bcy), bc_cut=Dirichlet(0.0), force=(fx, fy))
    sys = solve_steady!(model)

    m = metrics(model, sys)
    push!(hs, 1.0 / (n - 1))
    push!(uL2, m.uL2)
    push!(vL2, m.vL2)
    push!(uInf, m.uInf)
    push!(vInf, m.vInf)
    push!(pL2, m.pL2)
    push!(divL2, m.divL2)
    push!(pmax, m.pmax)
    ri, rb = momentum_residual_split(model, sys)
    push!(rInt, ri)
    push!(rBnd, rb)

    println("n=$n  h=$(hs[end])  uL2=$(uL2[end])  vL2=$(vL2[end])  uInf=$(uInf[end])  vInf=$(vInf[end])  pL2=$(pL2[end])  divL2=$(divL2[end])  max|p|=$(pmax[end])  rIntInf=$(ri)  rBndInf=$(rb)")
end

for k in 1:(length(ns) - 1)
    ord_u = log(uL2[k] / uL2[k + 1]) / log(hs[k] / hs[k + 1])
    ord_v = log(vL2[k] / vL2[k + 1]) / log(hs[k] / hs[k + 1])
    ord_p = log(pL2[k] / pL2[k + 1]) / log(hs[k] / hs[k + 1])
    ord_ri = log(rInt[k] / rInt[k + 1]) / log(hs[k] / hs[k + 1])
    ord_rb = log(rBnd[k] / rBnd[k + 1]) / log(hs[k] / hs[k + 1])
    println("order $(ns[k])->$(ns[k+1]): u=$ord_u  v=$ord_v  p=$ord_p  rInt=$ord_ri  rBnd=$ord_rb")
end
