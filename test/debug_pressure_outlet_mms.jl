using LinearAlgebra
using SparseArrays
using Printf
using StaticArrays: SVector

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, DoNothing, PressureOutlet, Traction
using PenguinSolverCore: LinearSystem
using PenguinStokes

full_body_dbg(args...) = -1.0
const MU = 1.0

# Matched full-traction MMS (right side uses exact traction vector).
mms_u(x, y) = pi * cos(pi * y) * sin(pi * x)
mms_v(x, y) = -pi * cos(pi * x) * sin(pi * y)
mms_p(x, y) = cos(pi * x) * sin(pi * y)
mms_lap_u(x, y) = -2 * pi^3 * cos(pi * y) * sin(pi * x)
mms_lap_v(x, y) = 2 * pi^3 * cos(pi * x) * sin(pi * y)
mms_dpx(x, y) = -pi * sin(pi * x) * sin(pi * y)
mms_dpy(x, y) = pi * cos(pi * x) * cos(pi * y)
mms_dux(x, y) = pi^2 * cos(pi * y) * cos(pi * x)
mms_duy(x, y) = -pi^2 * sin(pi * y) * sin(pi * x)
mms_dvx(x, y) = pi^2 * sin(pi * x) * sin(pi * y)
mms_fx(x, y) = -MU * mms_lap_u(x, y) - mms_dpx(x, y)
mms_fy(x, y) = -MU * mms_lap_v(x, y) - mms_dpy(x, y)

# Matched PressureOutlet / DoNothing MMS:
# ψ = (1-x)^3 g(y), p = p_out + (1-x)^2 s(y)
po_g(y) = y^2 * (1 - y)^2
po_gp(y) = 2y - 6y^2 + 4y^3
po_gpp(y) = 2 - 12y + 12y^2
po_g3(y) = -12 + 24y
po_s(y) = sin(pi * y)
po_sp(y) = pi * cos(pi * y)
po_u(x, y) = (1 - x)^3 * po_gp(y)
po_v(x, y) = 3 * (1 - x)^2 * po_g(y)
po_p(x, y; pout=0.0) = pout + (1 - x)^2 * po_s(y)
po_lap_u(x, y) = 6 * (1 - x) * po_gp(y) + (1 - x)^3 * po_g3(y)
po_lap_v(x, y) = 6 * po_g(y) + 3 * (1 - x)^2 * po_gpp(y)
po_px(x, y) = -2 * (1 - x) * po_s(y)
po_py(x, y) = (1 - x)^2 * po_sp(y)
po_fx(x, y) = -MU * po_lap_u(x, y) - po_px(x, y)
po_fy(x, y) = -MU * po_lap_v(x, y) - po_py(x, y)

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

function velocity_pressure_metrics(model, sys, ufun, vfun, pfun)
    u = sys.x[model.layout.uomega[1]]
    v = sys.x[model.layout.uomega[2]]
    p = sys.x[model.layout.pomega]
    eu2 = 0.0
    ev2 = 0.0
    wu = 0.0
    wv = 0.0
    for i in 1:model.cap_u[1].ntotal
        V = model.cap_u[1].buf.V[i]
        if isfinite(V) && V > 0.0
            x = model.cap_u[1].C_ω[i]
            eu2 += V * (u[i] - ufun(x[1], x[2]))^2
            wu += V
        end
    end
    for i in 1:model.cap_u[2].ntotal
        V = model.cap_u[2].buf.V[i]
        if isfinite(V) && V > 0.0
            x = model.cap_u[2].C_ω[i]
            ev2 += V * (v[i] - vfun(x[1], x[2]))^2
            wv += V
        end
    end
    pidx = pressure_active_indices(model, sys.A)
    pn = [p[i] for i in pidx]
    pe = [begin
        x = model.cap_p.C_ω[i]
        pfun(x[1], x[2])
    end for i in pidx]
    shift = sum(pn .- pe) / length(pn)
    ep2 = 0.0
    wp = 0.0
    for (k, i) in enumerate(pidx)
        V = model.cap_p.buf.V[i]
        ep2 += V * ((pn[k] - shift) - pe[k])^2
        wp += V
    end
    return (uL2=sqrt(eu2 / wu), vL2=sqrt(ev2 / wv), pL2=sqrt(ep2 / wp))
end

function build_exact_state(model, ufun, vfun, pfun)
    x = zeros(Float64, last(model.layout.pomega))
    for i in 1:model.cap_u[1].ntotal
        pt = model.cap_u[1].C_ω[i]
        x[model.layout.uomega[1][i]] = ufun(pt[1], pt[2])
    end
    for i in 1:model.cap_u[2].ntotal
        pt = model.cap_u[2].C_ω[i]
        x[model.layout.uomega[2][i]] = vfun(pt[1], pt[2])
    end
    for i in 1:model.cap_p.ntotal
        pt = model.cap_p.C_ω[i]
        if isfinite(pt[1]) && isfinite(pt[2])
            x[model.layout.pomega[i]] = pfun(pt[1], pt[2])
        else
            x[model.layout.pomega[i]] = 0.0
        end
    end
    return x
end

function outlet_row_residuals(model, sys, xexact)
    r = sys.A * xexact - sys.b
    row_u = Int[]
    row_v = Int[]
    capu = model.cap_u[1]
    capv = model.cap_u[2]
    liu = LinearIndices(capu.nnodes)
    liv = LinearIndices(capv.nnodes)
    iwall_u = capu.nnodes[1] - 1
    iwall_v = capv.nnodes[1] - 1
    for j in 1:(capu.nnodes[2] - 1)
        i = liu[CartesianIndex(iwall_u, j)]
        V = capu.buf.V[i]
        Aface = capu.buf.A[1][i]
        if isfinite(V) && V > 0.0 && isfinite(Aface) && Aface > 0.0
            push!(row_u, model.layout.uomega[1][i])
        end
    end
    for j in 1:(capv.nnodes[2] - 1)
        i = liv[CartesianIndex(iwall_v, j)]
        V = capv.buf.V[i]
        Aface = capv.buf.A[1][i]
        if isfinite(V) && V > 0.0 && isfinite(Aface) && Aface > 0.0
            push!(row_v, model.layout.uomega[2][i])
        end
    end
    rows = vcat(row_u, row_v)
    l2 = norm(r[rows]) / sqrt(length(rows))
    return (
        uInf=maximum(abs, r[row_u]),
        vInf=maximum(abs, r[row_v]),
        allL2=l2,
    )
end

function gauge_row_diagnostics(model, sys, xexact)
    prow = model.layout.pomega
    up_to = first(prow) - 1
    best_nz = 0
    gauge_row = 0
    for i in 1:model.cap_p.ntotal
        row = prow[i]
        if up_to > 0 && maximum(abs, Array(sys.A[row, 1:up_to])) > 0.0
            continue
        end
        nz = count(!iszero, Array(sys.A[row, prow]))
        if nz > best_nz
            best_nz = nz
            gauge_row = row
        end
    end
    α = Array(sys.A[gauge_row, prow])
    pexact = xexact[prow]
    c = dot(α, pexact) # best constant shift for this row (sum α = 1)
    xshift = copy(xexact)
    xshift[prow] .-= c
    r_raw = (sys.A * xexact - sys.b)[gauge_row]
    r_shift = (sys.A * xshift - sys.b)[gauge_row]
    return (row=gauge_row, residual_raw=r_raw, residual_shift=r_shift)
end

function print_orders(label, hs, eu, ev, ep)
    println("\n=== ", label, " ===")
    println("n-grid   h          L2(u)         L2(v)         L2(p-shift)")
    for k in eachindex(hs)
        n = round(Int, 1 / hs[k] + 1)
        println(@sprintf("%-8d %.5e   %.6e   %.6e   %.6e", n, hs[k], eu[k], ev[k], ep[k]))
    end
    if length(hs) > 1
        ou = Float64[]
        ov = Float64[]
        op = Float64[]
        for k in 1:(length(hs) - 1)
            den = log(hs[k] / hs[k + 1])
            push!(ou, log(eu[k] / eu[k + 1]) / den)
            push!(ov, log(ev[k] / ev[k + 1]) / den)
            push!(op, log(ep[k] / ep[k + 1]) / den)
        end
        println("orders u: ", join(map(x -> @sprintf("%.3f", x), ou), ", "))
        println("orders v: ", join(map(x -> @sprintf("%.3f", x), ov), ", "))
        println("orders p: ", join(map(x -> @sprintf("%.3f", x), op), ", "))
    end
end

function run_case(label, bcx, bcy, force, ufun, vfun, pfun)
    ns = (17, 33, 65)
    hs = Float64[]
    eu = Float64[]
    ev = Float64[]
    ep = Float64[]
    println("\n-------------------------------")
    println(label)
    println("-------------------------------")
    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            full_body_dbg,
            MU,
            1.0;
            bc_u=(bcx, bcy),
            bc_p=nothing,
            bc_cut=Dirichlet(0.0),
            force=force,
            gauge=MeanPressureGauge(),
        )
        sys = solve_steady!(model)
        m = velocity_pressure_metrics(model, sys, ufun, vfun, pfun)
        xexact = build_exact_state(model, ufun, vfun, pfun)
        ro = outlet_row_residuals(model, sys, xexact)
        gr = gauge_row_diagnostics(model, sys, xexact)

        println(@sprintf("n=%-4d  L2(u)=%.3e  L2(v)=%.3e  L2(p)=%.3e  outlet[Linf u,v]=[%.2e, %.2e]  outlet[L2]=%.2e  gauge_row=%d  gauge_res(raw,shift)=(%.2e, %.2e)  ||Ax-b||=%.2e",
            n, m.uL2, m.vL2, m.pL2, ro.uInf, ro.vInf, ro.allL2, gr.row, gr.residual_raw, gr.residual_shift, norm(sys.A * sys.x - sys.b)))

        push!(hs, 1.0 / (n - 1))
        push!(eu, m.uL2)
        push!(ev, m.vL2)
        push!(ep, m.pL2)
    end
    print_orders(label, hs, eu, ev, ep)
end

# Case A: matched full traction on the right.
tx(x, y) = -mms_p(x, y) + 2 * MU * mms_dux(x, y)
ty(x, y) = MU * (mms_dvx(x, y) + mms_duy(x, y))
tvec(x, y) = SVector(tx(x, y), ty(x, y))
bcx_t = BorderConditions(
    ; left=Dirichlet((x, y) -> mms_u(x, y)),
    right=Traction((x, y) -> tvec(x, y)),
    bottom=Dirichlet((x, y) -> mms_u(x, y)),
    top=Dirichlet((x, y) -> mms_u(x, y)),
)
bcy_t = BorderConditions(
    ; left=Dirichlet((x, y) -> mms_v(x, y)),
    right=Traction((x, y) -> tvec(x, y)),
    bottom=Dirichlet((x, y) -> mms_v(x, y)),
    top=Dirichlet((x, y) -> mms_v(x, y)),
)
run_case("Case A: exact Traction(t) on right", bcx_t, bcy_t, (mms_fx, mms_fy), mms_u, mms_v, mms_p)

# Case B: matched PressureOutlet(p_out) on the right.
pout = 0.7
bcx_po = BorderConditions(
    ; left=Dirichlet((x, y) -> po_u(x, y)),
    right=PressureOutlet(pout),
    bottom=Dirichlet((x, y) -> po_u(x, y)),
    top=Dirichlet((x, y) -> po_u(x, y)),
)
bcy_po = BorderConditions(
    ; left=Dirichlet((x, y) -> po_v(x, y)),
    right=PressureOutlet(pout),
    bottom=Dirichlet((x, y) -> po_v(x, y)),
    top=Dirichlet((x, y) -> po_v(x, y)),
)
run_case("Case B: matched PressureOutlet(p_out) on right", bcx_po, bcy_po, (po_fx, po_fy), po_u, po_v, (x, y) -> po_p(x, y; pout=pout))

# Case C: matched DoNothing() on the right.
bcx_dn = BorderConditions(
    ; left=Dirichlet((x, y) -> po_u(x, y)),
    right=DoNothing(),
    bottom=Dirichlet((x, y) -> po_u(x, y)),
    top=Dirichlet((x, y) -> po_u(x, y)),
)
bcy_dn = BorderConditions(
    ; left=Dirichlet((x, y) -> po_v(x, y)),
    right=DoNothing(),
    bottom=Dirichlet((x, y) -> po_v(x, y)),
    top=Dirichlet((x, y) -> po_v(x, y)),
)
run_case("Case C: matched DoNothing() on right", bcx_dn, bcy_dn, (po_fx, po_fy), po_u, po_v, (x, y) -> po_p(x, y; pout=0.0))
