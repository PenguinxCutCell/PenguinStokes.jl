using Test
using LinearAlgebra
using SparseArrays

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet
using PenguinSolverCore: LinearSystem
using PenguinStokes

full_body(args...) = -1.0

const MMS_MU = 1.0

# Full-box streamfunction MMS (divergence-free velocity).
mms_u(x, y) = pi * cos(pi * y) * sin(pi * x)
mms_v(x, y) = -pi * cos(pi * x) * sin(pi * y)
mms_p(x, y) = cos(pi * x) * sin(pi * y)
mms_lap_u(x, y) = -2 * pi^3 * cos(pi * y) * sin(pi * x)
mms_lap_v(x, y) = 2 * pi^3 * cos(pi * x) * sin(pi * y)
mms_dpx(x, y) = -pi * sin(pi * x) * sin(pi * y)
mms_dpy(x, y) = pi * cos(pi * x) * cos(pi * y)

# Sign follows current operator convention in PenguinStokes assembly.
mms_fx(x, y) = -MMS_MU * mms_lap_u(x, y) - mms_dpx(x, y)
mms_fy(x, y) = -MMS_MU * mms_lap_v(x, y) - mms_dpy(x, y)

# Zero-pressure polynomial MMS (exact homogeneous Dirichlet on box walls).
poly_h(x) = x^2 * (1 - x)^2
poly_hp(x) = 2x - 6x^2 + 4x^3
poly_hpp(x) = 2 - 12x + 12x^2
poly_h3(x) = -12 + 24x
poly_g(y) = y^2 * (1 - y)^2
poly_gp(y) = 2y - 6y^2 + 4y^3
poly_gpp(y) = 2 - 12y + 12y^2
poly_g3(y) = -12 + 24y

poly_u(x, y) = poly_h(x) * poly_gp(y)
poly_v(x, y) = -poly_hp(x) * poly_g(y)
poly_lap_u(x, y) = poly_hpp(x) * poly_gp(y) + poly_h(x) * poly_g3(y)
poly_lap_v(x, y) = -(poly_h3(x) * poly_g(y) + poly_hp(x) * poly_gpp(y))
poly_fx(x, y) = -MMS_MU * poly_lap_u(x, y)
poly_fy(x, y) = -MMS_MU * poly_lap_v(x, y)

function all_dirichlet_bc(::Val{1})
    return BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0))
end

function all_dirichlet_bc(::Val{2})
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
end

function all_dirichlet_bc(::Val{3})
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
end

function physical_active_indices(cap)
    li = LinearIndices(cap.nnodes)
    idx = Int[]
    for I in CartesianIndices(cap.nnodes)
        i = li[I]
        physical = true
        for d in 1:length(cap.nnodes)
            if I[d] == cap.nnodes[d]
                physical = false
                break
            end
        end
        if physical && isfinite(cap.buf.V[i]) && cap.buf.V[i] > 0.0
            push!(idx, i)
        end
    end
    return idx
end

function interface_active_indices(cap)
    li = LinearIndices(cap.nnodes)
    idx = Int[]
    for I in CartesianIndices(cap.nnodes)
        i = li[I]
        physical = true
        for d in 1:length(cap.nnodes)
            if I[d] == cap.nnodes[d]
                physical = false
                break
            end
        end
        if physical && isfinite(cap.buf.Γ[i]) && cap.buf.Γ[i] > 0.0
            push!(idx, i)
        end
    end
    return idx
end

function mms_box_bcs()
    bx = BorderConditions(
        ; left=Dirichlet((x, y) -> mms_u(x, y)),
        right=Dirichlet((x, y) -> mms_u(x, y)),
        bottom=Dirichlet((x, y) -> mms_u(x, y)),
        top=Dirichlet((x, y) -> mms_u(x, y)),
    )
    by = BorderConditions(
        ; left=Dirichlet((x, y) -> mms_v(x, y)),
        right=Dirichlet((x, y) -> mms_v(x, y)),
        bottom=Dirichlet((x, y) -> mms_v(x, y)),
        top=Dirichlet((x, y) -> mms_v(x, y)),
    )
    return bx, by
end

function poly_box_bcs()
    bx = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    by = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    return bx, by
end

function pressure_active_indices(model, A)
    nn = model.cap_p.nnodes
    li = LinearIndices(nn)
    pfirst = first(model.layout.pomega)
    plast = last(model.layout.pomega)
    idx = Int[]

    for I in CartesianIndices(nn)
        # Mirror pressure-row activity used by the solver.
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

function mms_box_metrics(model, sys)
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
                e = abs(u[i] - mms_u(x[1], x[2]))
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
                e = abs(v[i] - mms_v(x[1], x[2]))
                ev2 += V * e^2
                wv += V
                evinf = max(evinf, e)
            end
        end
    end

    pidx = pressure_active_indices(model, sys.A)
    pn = [p[i] for i in pidx]
    pexact = [begin
        x = model.cap_p.C_ω[i]
        mms_p(x[1], x[2])
    end for i in pidx]
    # Gauge correction by mean shift.
    pshift = sum(pn .- pexact) / length(pn)
    ep2 = 0.0
    wp = 0.0
    for (k, i) in enumerate(pidx)
        V = model.cap_p.buf.V[i]
        e = (pn[k] - pshift) - pexact[k]
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
        pmin=minimum(pn),
        pmax=maximum(pn),
    )
end

function poly_box_velocity_metrics(model, sys)
    u = sys.x[model.layout.uomega[1]]
    v = sys.x[model.layout.uomega[2]]

    li_u = LinearIndices(model.cap_u[1].nnodes)
    li_v = LinearIndices(model.cap_u[2].nnodes)
    eu2 = 0.0
    ev2 = 0.0
    wu = 0.0
    wv = 0.0

    for I in CartesianIndices(model.cap_u[1].nnodes)
        i = li_u[I]
        if I[1] < model.cap_u[1].nnodes[1] && I[2] < model.cap_u[1].nnodes[2]
            V = model.cap_u[1].buf.V[i]
            if V > 0.0
                x = model.cap_u[1].C_ω[i]
                e = u[i] - poly_u(x[1], x[2])
                eu2 += V * e^2
                wu += V
            end
        end
    end

    for I in CartesianIndices(model.cap_u[2].nnodes)
        i = li_v[I]
        if I[1] < model.cap_u[2].nnodes[1] && I[2] < model.cap_u[2].nnodes[2]
            V = model.cap_u[2].buf.V[i]
            if V > 0.0
                x = model.cap_u[2].C_ω[i]
                e = v[i] - poly_v(x[1], x[2])
                ev2 += V * e^2
                wv += V
            end
        end
    end

    return (uL2=sqrt(eu2 / wu), vL2=sqrt(ev2 / wv))
end

@testset "Stokes layout and assembly dimensions" begin
    cases = (
        CartesianGrid((0.0,), (1.0,), (11,)),
        CartesianGrid((0.0, 0.0), (1.0, 1.0), (7, 6)),
        CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (5, 4, 4)),
    )

    for grid in cases
        N = length(grid.n)
        bc_comp = all_dirichlet_bc(Val(N))
        bcu = ntuple(_ -> bc_comp, N)
        model = StokesModelMono(grid, full_body, 1.0, 1.0; bc_u=bcu, bc_cut=Dirichlet(0.0))
        nt = prod(grid.n)
        @test model.layout.nt == nt
        @test last(model.layout.pomega) == (2 * N + 1) * nt

        nsys = last(model.layout.pomega)
        sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
        assemble_steady!(sys, model, 0.0)

        @test size(sys.A) == (nsys, nsys)
        @test length(sys.b) == nsys
        @test issparse(sys.A)
    end
end

@testset "Steady assembly zero-state residual (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (9, 9))
    bc = all_dirichlet_bc(Val(2))
    model = StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
    )

    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)
    x0 = zeros(Float64, nsys)
    @test norm(sys.b) <= 1e-10
    @test norm(sys.A * x0 - sys.b) <= 1e-10
end

@testset "Unsteady assembly zero-state residual (2D, BE/CN)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (8, 8))
    bc = all_dirichlet_bc(Val(2))
    model = StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
    )

    x0 = zeros(Float64, last(model.layout.pomega))

    nsys = last(model.layout.pomega)
    sys_be = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    sys_cn = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_unsteady!(sys_be, model, x0, 0.0, 0.05; scheme=:BE)
    assemble_unsteady!(sys_cn, model, x0, 0.0, 0.05; scheme=:CN)

    @test norm(sys_be.b) <= 1e-10
    @test norm(sys_cn.b) <= 1e-10
    @test norm(sys_be.A * x0 - sys_be.b) <= 1e-10
    @test norm(sys_cn.A * x0 - sys_cn.b) <= 1e-10
end

@testset "Unsteady BE mass shift on momentum rows" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (8, 7))
    model = StokesModelMono(grid, full_body, 1.2, 0.7; bc_cut=Dirichlet(0.0), force=(0.0, 0.0))

    nsys = last(model.layout.pomega)
    sys_s = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    sys_u = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))

    assemble_steady!(sys_s, model, 0.3)
    assemble_unsteady!(sys_u, model, zeros(nsys), 0.3, 0.2; scheme=:BE)

    for d in 1:2
        idx = physical_active_indices(model.cap_u[d])
        rows = model.layout.uomega[d]
        @test !isempty(idx)
        for i in idx
            expected = model.rho * model.cap_u[d].buf.V[i] / 0.2
            got = sys_u.A[rows[i], rows[i]] - sys_s.A[rows[i], rows[i]]
            @test isapprox(got, expected; atol=1e-10, rtol=1e-10)
        end
    end
end

@testset "MMS box convergence + pressure/divergence diagnostics (2D)" begin
    bcx, bcy = mms_box_bcs()
    ns = (17, 33, 65)
    hs = Float64[]
    uL2 = Float64[]
    vL2 = Float64[]
    pL2 = Float64[]
    divL2 = Float64[]
    uInf = Float64[]
    vInf = Float64[]
    pAbs = Float64[]

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            full_body,
            MMS_MU,
            1.0;
            bc_u=(bcx, bcy),
            bc_cut=Dirichlet(0.0),
            force=(mms_fx, mms_fy),
        )
        sys = solve_steady!(model)
        m = mms_box_metrics(model, sys)

        push!(hs, 1.0 / (n - 1))
        push!(uL2, m.uL2)
        push!(vL2, m.vL2)
        push!(uInf, m.uInf)
        push!(vInf, m.vInf)
        push!(pL2, m.pL2)
        push!(divL2, m.divL2)
        push!(pAbs, max(abs(m.pmin), abs(m.pmax)))
    end

    for k in 1:(length(ns) - 1)
        ord_u = log(uL2[k] / uL2[k + 1]) / log(hs[k] / hs[k + 1])
        ord_v = log(vL2[k] / vL2[k + 1]) / log(hs[k] / hs[k + 1])
        ord_p = log(pL2[k] / pL2[k + 1]) / log(hs[k] / hs[k + 1])
        @test ord_u > 0.75
        @test ord_v > 0.75
        @test ord_p > 0.3
    end

    @test uInf[2] < uInf[1]
    @test uInf[3] < uInf[2]
    @test vInf[2] < vInf[1]
    @test vInf[3] < vInf[2]
    @test pL2[2] < pL2[1]
    @test pL2[3] < pL2[2]
    @test divL2[1] < 1e-10
    @test divL2[2] < 1e-10
    @test divL2[3] < 1e-10
    @test isfinite(pAbs[1])
    @test isfinite(pAbs[2])
    @test isfinite(pAbs[3])
end

@testset "MMS box convergence (zero-pressure, no body) near second order (2D)" begin
    bcx, bcy = poly_box_bcs()
    ns = (17, 33, 65)
    hs = Float64[]
    uL2 = Float64[]
    vL2 = Float64[]

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            full_body,
            MMS_MU,
            1.0;
            bc_u=(bcx, bcy),
            bc_cut=Dirichlet(0.0),
            force=(poly_fx, poly_fy),
        )
        sys = solve_steady!(model)
        m = poly_box_velocity_metrics(model, sys)
        push!(hs, 1.0 / (n - 1))
        push!(uL2, m.uL2)
        push!(vL2, m.vL2)
    end

    for k in 1:(length(ns) - 1)
        ord_u = log(uL2[k] / uL2[k + 1]) / log(hs[k] / hs[k + 1])
        ord_v = log(vL2[k] / vL2[k + 1]) / log(hs[k] / hs[k + 1])
        @test ord_u > 1.7
        @test ord_v > 1.7
    end
end

@testset "Cut-cell no-slip zero solution (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
    xc, yc, r = 0.5, 0.5, 0.2
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r

    bc = all_dirichlet_bc(Val(2))
    model = StokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
    )

    sys = solve_steady!(model)
    @test norm(sys.A * sys.x - sys.b) < 1e-10

    for d in 1:2
        uω = sys.x[model.layout.uomega[d]]
        uγ = sys.x[model.layout.ugamma[d]]
        phys = physical_active_indices(model.cap_u[d])
        iface = interface_active_indices(model.cap_u[d])
        @test maximum(abs, uω[phys]) < 1e-10
        if !isempty(iface)
            @test maximum(abs, uγ[iface]) < 1e-10
        end
    end

    p = sys.x[model.layout.pomega]
    pidx = pressure_active_indices(model, sys.A)
    if !isempty(pidx)
        @test maximum(abs, p[pidx]) < 1e-9
    end
end
