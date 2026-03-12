using Test
using LinearAlgebra
using SparseArrays
using StaticArrays: SMatrix, SVector

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, DoNothing, Neumann, Periodic, PressureOutlet, Symmetry, Traction
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
mms_dux(x, y) = pi^2 * cos(pi * y) * cos(pi * x)
mms_duy(x, y) = -pi^2 * sin(pi * y) * sin(pi * x)
mms_dvx(x, y) = pi^2 * sin(pi * x) * sin(pi * y)
mms_dvy(x, y) = -pi^2 * cos(pi * x) * cos(pi * y)

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

# Tangential-traction MMS (polynomial streamfunction, p=0):
# ψ = x^2 * y * (1-y), u = ∂y ψ, v = -∂x ψ.
tan_u(x, y) = x^2 * (1 - 2y)
tan_v(x, y) = -2x * y * (1 - y)
tan_ux(x, y) = 2x * (1 - 2y)
tan_uy(x, y) = -2x^2
tan_vx(x, y) = -2y * (1 - y)
tan_vy(x, y) = -2x * (1 - 2y)
tan_lap_u(x, y) = 2 * (1 - 2y)
tan_lap_v(x, y) = 4x
tan_fx(x, y) = -MMS_MU * tan_lap_u(x, y)
tan_fy(x, y) = -MMS_MU * tan_lap_v(x, y)

# Matched outlet MMS for PressureOutlet/DoNothing:
# streamfunction ψ = (1-x)^3 g(y), p = p_out + (1-x)^2 s(y)
# so on x=1: u=v=ux=0 and (vx + uy)=0, giving exact outlet traction (-p_out, 0).
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
po_fx(x, y) = -MMS_MU * po_lap_u(x, y) - po_px(x, y)
po_fy(x, y) = -MMS_MU * po_lap_v(x, y) - po_py(x, y)

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

function two_layer_couette_profile(mu1, mu2, h, H, U)
    τ = U / (h / mu1 + (H - h) / mu2)
    a1 = τ / mu1
    a2 = τ / mu2
    b2 = τ * h * (inv(mu1) - inv(mu2))
    u(y) = y <= h ? (a1 * y) : (a2 * y + b2)
    return u
end

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

function two_phase_velocity_error_metrics(model, sys, u_exact)
    u1 = sys.x[model.layout.uomega1[1]]
    u2 = sys.x[model.layout.uomega2[1]]
    v1 = sys.x[model.layout.uomega1[2]]
    v2 = sys.x[model.layout.uomega2[2]]

    e1 = 0.0
    e2 = 0.0
    w1 = 0.0
    w2 = 0.0
    vInf = 0.0

    for i in 1:model.cap_u1[1].ntotal
        V = model.cap_u1[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u1[1].C_ω[i][2]
            e = u1[i] - u_exact(y)
            e1 += V * e^2
            w1 += V
            vInf = max(vInf, abs(v1[i]))
        end
    end

    for i in 1:model.cap_u2[1].ntotal
        V = model.cap_u2[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u2[1].C_ω[i][2]
            e = u2[i] - u_exact(y)
            e2 += V * e^2
            w2 += V
            vInf = max(vInf, abs(v2[i]))
        end
    end

    return (u1L2=sqrt(e1 / w1), u2L2=sqrt(e2 / w2), vInf=vInf)
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
    pexact = [begin
        x = model.cap_p.C_ω[i]
        pfun(x[1], x[2])
    end for i in pidx]
    shift = sum(pn .- pexact) / length(pn)

    ep2 = 0.0
    wp = 0.0
    for (k, i) in enumerate(pidx)
        V = model.cap_p.buf.V[i]
        ep2 += V * ((pn[k] - shift) - pexact[k])^2
        wp += V
    end

    return (uL2=sqrt(eu2 / wu), vL2=sqrt(ev2 / wv), pL2=sqrt(ep2 / wp))
end

function _is_identity_row(A, b, row; atol=1e-14)
    r = Array(A[row, :])
    nz = findall(x -> abs(x) > atol, r)
    return length(nz) == 1 && nz[1] == row && isapprox(r[row], 1.0; atol=atol, rtol=0.0) &&
           isapprox(b[row], 0.0; atol=atol, rtol=0.0)
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

@testset "Pressure-gradient consistency on full box walls (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
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

    blocks = PenguinStokes._stokes_blocks(model)
    nt = model.cap_p.ntotal
    pconst = ones(Float64, nt)
    px = zeros(Float64, nt)
    py = zeros(Float64, nt)

    for i in 1:nt
        x = model.cap_p.C_ω[i]
        if isfinite(x[1])
            px[i] = x[1]
            py[i] = x[2]
        end
    end

    for d in 1:2
        gconst = blocks.grad[d] * pconst
        gx = blocks.grad[d] * px
        gy = blocks.grad[d] * py
        idx = physical_active_indices(model.cap_u[d])
        for i in idx
            V = model.cap_u[d].buf.V[i]
            tx = d == 1 ? -V : 0.0
            ty = d == 2 ? -V : 0.0
            @test isapprox(gconst[i], 0.0; atol=1e-12, rtol=0.0)
            @test isapprox(gx[i], tx; atol=1e-12, rtol=0.0)
            @test isapprox(gy[i], ty; atol=1e-12, rtol=0.0)
        end
    end
end

@testset "Outer-box traction BC validation rules (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))

    bcx_mix = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_mix = BorderConditions(
        ; left=Dirichlet(0.0), right=Neumann(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    @test_throws ArgumentError StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bcx_mix, bcy_mix),
    )

    bcx = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcp_conflict = BorderConditions(; right=Neumann(0.0))
    @test_throws ArgumentError StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bcx, bcy),
        bc_p=bcp_conflict,
    )

    # Pure traction-side setup should be accepted.
    bcx_ok = BorderConditions(
        ; left=Dirichlet(0.0), right=DoNothing(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_ok = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction(SVector(0.0, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    model_ok = StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bcx_ok, bcy_ok),
    )
    @test model_ok isa StokesModelMono
end

@testset "Outer-box pressure-outlet traction rows couple p and cross-shear (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bcx = BorderConditions(
        ; left=Dirichlet(1.0), right=PressureOutlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    model = StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bcx, bcy),
        bc_p=nothing,
        force=(0.0, 0.0),
    )

    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)

    liux = LinearIndices(model.cap_u[1].nnodes)
    liuy = LinearIndices(model.cap_u[2].nnodes)
    lip = LinearIndices(model.cap_p.nnodes)
    I = CartesianIndex(model.cap_u[1].nnodes[1] - 1, 4)
    iux = liux[I]
    iuy = liuy[I]
    ip = lip[I]

    row_ux = model.layout.uomega[1][iux]
    row_uy = model.layout.uomega[2][iuy]
    row_p = model.layout.pomega[ip]

    # Normal traction equation includes pressure on outlet rows.
    @test any(abs.(Array(sys.A[row_ux, model.layout.pomega])) .> 0.0)
    # Tangential traction equation includes ∂_t u_n cross-component coupling.
    @test any(abs.(Array(sys.A[row_uy, model.layout.uomega[1]])) .> 0.0)
    # Pressure rows on traction sides are not overwritten as scalar pressure BC rows.
    @test nnz(sys.A[row_p, :]) > 1
end

@testset "Outer-box do-nothing equals zero-traction rows (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (15, 15))
    bcx_dn = BorderConditions(
        ; left=Dirichlet(0.0), right=DoNothing(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_dn = BorderConditions(
        ; left=Dirichlet(0.0), right=DoNothing(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcx_t0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction(SVector(0.0, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_t0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction(SVector(0.0, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    m_dn = StokesModelMono(grid, full_body, 1.0, 1.0; bc_u=(bcx_dn, bcy_dn), force=(0.0, 0.0))
    m_t0 = StokesModelMono(grid, full_body, 1.0, 1.0; bc_u=(bcx_t0, bcy_t0), force=(0.0, 0.0))
    nsys = last(m_dn.layout.pomega)
    s_dn = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    s_t0 = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(s_dn, m_dn, 0.0)
    assemble_steady!(s_t0, m_t0, 0.0)

    li = LinearIndices(m_dn.cap_u[1].nnodes)
    I = CartesianIndex(m_dn.cap_u[1].nnodes[1] - 1, 5)
    i = li[I]
    rux_dn = m_dn.layout.uomega[1][i]
    ruy_dn = m_dn.layout.uomega[2][i]
    rux_t0 = m_t0.layout.uomega[1][i]
    ruy_t0 = m_t0.layout.uomega[2][i]

    @test Array(s_dn.A[rux_dn, :]) ≈ Array(s_t0.A[rux_t0, :]) atol=1e-14 rtol=0.0
    @test Array(s_dn.A[ruy_dn, :]) ≈ Array(s_t0.A[ruy_t0, :]) atol=1e-14 rtol=0.0
    @test s_dn.b[rux_dn] == s_t0.b[rux_t0] == 0.0
    @test s_dn.b[ruy_dn] == s_t0.b[ruy_t0] == 0.0
end

@testset "Outer-box pressure-outlet row RHS/sign (2D)" begin
    pout = 2.3
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bcx = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(pout),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(pout),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    model = StokesModelMono(grid, full_body, 1.0, 1.0; bc_u=(bcx, bcy), force=(0.0, 0.0))
    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)

    li = LinearIndices(model.cap_u[1].nnodes)
    I = CartesianIndex(model.cap_u[1].nnodes[1] - 1, 4)
    i = li[I]
    rux = model.layout.uomega[1][i]
    ruy = model.layout.uomega[2][i]
    @test isapprox(sys.b[rux], -pout; atol=1e-12, rtol=0.0)
    @test isapprox(sys.b[ruy], 0.0; atol=1e-12, rtol=0.0)
end

@testset "Outer-box symmetry BC validation rules (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))

    # Symmetry must be set on all velocity components of the side.
    bcx_partial = BorderConditions(
        ; left=Dirichlet(0.0), right=Symmetry(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_partial = BorderConditions(
        ; left=Dirichlet(0.0), right=Neumann(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    @test_throws ArgumentError StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bcx_partial, bcy_partial),
    )

    # Symmetry and traction cannot be mixed on the same side.
    bcx_mix = BorderConditions(
        ; left=Dirichlet(0.0), right=Symmetry(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_mix = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    @test_throws ArgumentError StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bcx_mix, bcy_mix),
    )

    # Pressure BC cannot be imposed on a symmetry side.
    bcx_sym = BorderConditions(
        ; left=Dirichlet(0.0), right=Symmetry(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_sym = BorderConditions(
        ; left=Dirichlet(0.0), right=Symmetry(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcp_bad = BorderConditions(; right=Neumann(0.0))
    @test_throws ArgumentError StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bcx_sym, bcy_sym),
        bc_p=bcp_bad,
    )
end

@testset "Outer-box right symmetry rows enforce mixed free-slip form (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bcx = BorderConditions(
        ; left=Dirichlet(0.0), right=Symmetry(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy = BorderConditions(
        ; left=Dirichlet(0.0), right=Symmetry(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    model = StokesModelMono(grid, full_body, 1.0, 1.0; bc_u=(bcx, bcy), force=(0.0, 0.0))
    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)

    liux = LinearIndices(model.cap_u[1].nnodes)
    liuy = LinearIndices(model.cap_u[2].nnodes)
    I = CartesianIndex(model.cap_u[1].nnodes[1] - 1, 4)
    iux = liux[I]
    iuy = liuy[I]
    row_ux = model.layout.uomega[1][iux]
    row_uy = model.layout.uomega[2][iuy]

    # Normal component (u on right side) is strongly set to zero.
    @test _is_identity_row(sys.A, sys.b, row_ux)

    # Tangential symmetry row has no direct pressure term and includes cross-coupling.
    prow = Array(sys.A[row_uy, model.layout.pomega])
    @test maximum(abs, prow) == 0.0
    @test any(abs.(Array(sys.A[row_uy, model.layout.uomega[1]])) .> 0.0)
    @test any(abs.(Array(sys.A[row_uy, model.layout.uomega[2]])) .> 0.0)
    @test isapprox(sys.b[row_uy], 0.0; atol=1e-12, rtol=0.0)
end

@testset "Outer-box top symmetry rows enforce mixed free-slip form (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bcx = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Symmetry(),
    )
    bcy = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Symmetry(),
    )
    model = StokesModelMono(grid, full_body, 1.0, 1.0; bc_u=(bcx, bcy), force=(0.0, 0.0))
    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)

    liux = LinearIndices(model.cap_u[1].nnodes)
    liuy = LinearIndices(model.cap_u[2].nnodes)
    I = CartesianIndex(4, model.cap_u[2].nnodes[2] - 1)
    iux = liux[I]
    iuy = liuy[I]
    row_ux = model.layout.uomega[1][iux]
    row_uy = model.layout.uomega[2][iuy]

    # Normal component (v on top side) is strongly set to zero.
    @test _is_identity_row(sys.A, sys.b, row_uy)

    # Tangential symmetry row has no direct pressure term and includes cross-coupling.
    prow = Array(sys.A[row_ux, model.layout.pomega])
    @test maximum(abs, prow) == 0.0
    @test any(abs.(Array(sys.A[row_ux, model.layout.uomega[1]])) .> 0.0)
    @test any(abs.(Array(sys.A[row_ux, model.layout.uomega[2]])) .> 0.0)
    @test isapprox(sys.b[row_ux], 0.0; atol=1e-12, rtol=0.0)
end

function _find_gauge_row(model, sys)
    prow = model.layout.pomega
    row = 0
    local_idx = 0
    best_nz = 0
    up_to = first(prow) - 1
    for i in 1:model.cap_p.ntotal
        r = prow[i]
        if up_to > 0 && maximum(abs, Array(sys.A[r, 1:up_to])) > 0.0
            continue
        end
        pvals = Array(sys.A[r, prow])
        nz = count(!iszero, pvals)
        if nz > best_nz
            best_nz = nz
            row = r
            local_idx = i
        end
    end
    return local_idx, row, best_nz
end

@testset "Pressure gauge pin row uses selected DOF (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = all_dirichlet_bc(Val(2))
    m_ref = StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
        gauge=MeanPressureGauge(),
    )
    nsys = last(m_ref.layout.pomega)
    s_ref = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(s_ref, m_ref, 0.0)
    pidx = pressure_active_indices(m_ref, s_ref.A)
    @test length(pidx) >= 2

    i1 = pidx[1]
    i2 = pidx[2]

    m1 = StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
        gauge=PinPressureGauge(index=i1),
    )
    m2 = StokesModelMono(
        grid,
        full_body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
        gauge=PinPressureGauge(index=i2),
    )

    s1 = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    s2 = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(s1, m1, 0.0)
    assemble_steady!(s2, m2, 0.0)

    row1 = m1.layout.pomega[i1]
    row2 = m2.layout.pomega[i2]

    @test _is_identity_row(s1.A, s1.b, row1)
    @test !_is_identity_row(s2.A, s2.b, row1)
    @test _is_identity_row(s2.A, s2.b, row2)
    @test !_is_identity_row(s1.A, s1.b, row2)
end

@testset "Mean pressure gauge uses active-volume weights (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
    circle_body(x, y) = sqrt((x - 0.5)^2 + (y - 0.5)^2) - 0.2
    bc = all_dirichlet_bc(Val(2))
    model = StokesModelMono(
        grid,
        circle_body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
        gauge=MeanPressureGauge(),
    )

    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)

    local_row, row, nnz_p = _find_gauge_row(model, sys)
    @test local_row > 0
    @test nnz_p > 1

    pactive = PenguinStokes._pressure_activity(model.cap_p)
    active_idx = findall(pactive)
    @test !isempty(active_idx)

    prow = model.layout.pomega
    coeff_all = Array(sys.A[row, prow])
    coeff = [coeff_all[i] for i in active_idx]
    vols = [model.cap_p.buf.V[i] for i in active_idx]
    s_coeff = sum(coeff)
    s_vol = sum(vols)

    @test isapprox(s_coeff, 1.0; atol=1e-12, rtol=0.0)
    @test isfinite(s_vol) && s_vol > 0.0

    coeffn = coeff ./ s_coeff
    voln = vols ./ s_vol
    @test maximum(abs.(coeffn .- voln)) < 1e-10
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

@testset "MMS convergence with exact right traction (2D)" begin
    ns = (17, 33, 65)
    hs = Float64[]
    eu = Float64[]
    ev = Float64[]
    ep = Float64[]

    tx(x, y) = -mms_p(x, y) + 2 * MMS_MU * mms_dux(x, y)
    ty(x, y) = MMS_MU * (mms_dvx(x, y) + mms_duy(x, y))
    tvec(x, y) = SVector(tx(x, y), ty(x, y))

    bcx = BorderConditions(
        ; left=Dirichlet((x, y) -> mms_u(x, y)),
        right=Traction((x, y) -> tvec(x, y)),
        bottom=Dirichlet((x, y) -> mms_u(x, y)),
        top=Dirichlet((x, y) -> mms_u(x, y)),
    )
    bcy = BorderConditions(
        ; left=Dirichlet((x, y) -> mms_v(x, y)),
        right=Traction((x, y) -> tvec(x, y)),
        bottom=Dirichlet((x, y) -> mms_v(x, y)),
        top=Dirichlet((x, y) -> mms_v(x, y)),
    )

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            full_body,
            MMS_MU,
            1.0;
            bc_u=(bcx, bcy),
            bc_p=nothing,
            bc_cut=Dirichlet(0.0),
            force=(mms_fx, mms_fy),
            gauge=MeanPressureGauge(),
        )
        sys = solve_steady!(model)
        m = mms_box_metrics(model, sys)
        @test norm(sys.A * sys.x - sys.b) < 1e-8
        push!(hs, 1.0 / (n - 1))
        push!(eu, m.uL2)
        push!(ev, m.vL2)
        push!(ep, m.pL2)
    end

    @test eu[2] < eu[1]
    @test eu[3] < eu[2]
    @test ev[2] < ev[1]
    @test ev[3] < ev[2]
    @test ep[2] < ep[1]
    @test ep[3] < ep[2]

    for k in 1:2
        ord_u = log(eu[k] / eu[k + 1]) / log(hs[k] / hs[k + 1])
        ord_v = log(ev[k] / ev[k + 1]) / log(hs[k] / hs[k + 1])
        @test ord_u > 0.5
        @test ord_v > 0.5
    end
end

@testset "MMS convergence with matched right PressureOutlet (2D)" begin
    pout = 0.7
    ns = (17, 33, 65)
    hs = Float64[]
    eu = Float64[]
    ev = Float64[]
    ep = Float64[]

    bcx = BorderConditions(
        ; left=Dirichlet((x, y) -> po_u(x, y)),
        right=PressureOutlet(pout),
        bottom=Dirichlet((x, y) -> po_u(x, y)),
        top=Dirichlet((x, y) -> po_u(x, y)),
    )
    bcy = BorderConditions(
        ; left=Dirichlet((x, y) -> po_v(x, y)),
        right=PressureOutlet(pout),
        bottom=Dirichlet((x, y) -> po_v(x, y)),
        top=Dirichlet((x, y) -> po_v(x, y)),
    )

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            full_body,
            MMS_MU,
            1.0;
            bc_u=(bcx, bcy),
            bc_p=nothing,
            bc_cut=Dirichlet(0.0),
            force=(po_fx, po_fy),
            gauge=MeanPressureGauge(),
        )
        sys = solve_steady!(model)
        m = velocity_pressure_metrics(model, sys, po_u, po_v, (x, y) -> po_p(x, y; pout=pout))
        @test norm(sys.A * sys.x - sys.b) < 1e-8
        push!(hs, 1.0 / (n - 1))
        push!(eu, m.uL2)
        push!(ev, m.vL2)
        push!(ep, m.pL2)
    end

    @test eu[2] < eu[1]
    @test eu[3] < eu[2]
    @test ev[2] < ev[1]
    @test ev[3] < ev[2]
    @test ep[2] < ep[1]
    @test ep[3] < ep[2]

    for k in 1:2
        ord_u = log(eu[k] / eu[k + 1]) / log(hs[k] / hs[k + 1])
        ord_v = log(ev[k] / ev[k + 1]) / log(hs[k] / hs[k + 1])
        ord_p = log(ep[k] / ep[k + 1]) / log(hs[k] / hs[k + 1])
        @test ord_u > 1.7
        @test ord_v > 1.7
        @test ord_p > 0.45
    end
end

@testset "MMS convergence with matched right DoNothing (2D)" begin
    ns = (17, 33, 65)
    hs = Float64[]
    eu = Float64[]
    ev = Float64[]
    ep = Float64[]

    bcx = BorderConditions(
        ; left=Dirichlet((x, y) -> po_u(x, y)),
        right=DoNothing(),
        bottom=Dirichlet((x, y) -> po_u(x, y)),
        top=Dirichlet((x, y) -> po_u(x, y)),
    )
    bcy = BorderConditions(
        ; left=Dirichlet((x, y) -> po_v(x, y)),
        right=DoNothing(),
        bottom=Dirichlet((x, y) -> po_v(x, y)),
        top=Dirichlet((x, y) -> po_v(x, y)),
    )

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            full_body,
            MMS_MU,
            1.0;
            bc_u=(bcx, bcy),
            bc_p=nothing,
            bc_cut=Dirichlet(0.0),
            force=(po_fx, po_fy),
            gauge=MeanPressureGauge(),
        )
        sys = solve_steady!(model)
        m = velocity_pressure_metrics(model, sys, po_u, po_v, (x, y) -> po_p(x, y; pout=0.0))
        @test norm(sys.A * sys.x - sys.b) < 1e-8
        push!(hs, 1.0 / (n - 1))
        push!(eu, m.uL2)
        push!(ev, m.vL2)
        push!(ep, m.pL2)
    end

    @test eu[2] < eu[1]
    @test eu[3] < eu[2]
    @test ev[2] < ev[1]
    @test ev[3] < ev[2]
    @test ep[2] < ep[1]
    @test ep[3] < ep[2]

    for k in 1:2
        ord_u = log(eu[k] / eu[k + 1]) / log(hs[k] / hs[k + 1])
        ord_v = log(ev[k] / ev[k + 1]) / log(hs[k] / hs[k + 1])
        ord_p = log(ep[k] / ep[k + 1]) / log(hs[k] / hs[k + 1])
        @test ord_u > 1.7
        @test ord_v > 1.7
        @test ord_p > 0.45
    end
end

@testset "MMS convergence with top Symmetry (2D)" begin
    μ = 1.0
    ns = (17, 33, 65)
    hs = Float64[]
    eu = Float64[]
    ev = Float64[]

    g(y) = y^2 - 3y^3 + 3y^4 - y^5
    gp(y) = 2y - 9y^2 + 12y^3 - 5y^4
    gpp(y) = 2 - 18y + 36y^2 - 20y^3
    gppp(y) = -18 + 72y - 60y^2

    uex(x, y) = sin(pi * x) * gp(y)
    vex(x, y) = -pi * cos(pi * x) * g(y)

    lapu(x, y) = sin(pi * x) * (gppp(y) - pi^2 * gp(y))
    lapv(x, y) = pi * cos(pi * x) * (pi^2 * g(y) - gpp(y))
    fx(x, y) = -μ * lapu(x, y)
    fy(x, y) = -μ * lapv(x, y)

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        bcx = BorderConditions(
            ; left=Dirichlet((x, y) -> uex(x, y)),
            right=Dirichlet((x, y) -> uex(x, y)),
            bottom=Dirichlet((x, y) -> uex(x, y)),
            top=Symmetry(),
        )
        bcy = BorderConditions(
            ; left=Dirichlet((x, y) -> vex(x, y)),
            right=Dirichlet((x, y) -> vex(x, y)),
            bottom=Dirichlet((x, y) -> vex(x, y)),
            top=Symmetry(),
        )
        model = StokesModelMono(
            grid,
            full_body,
            μ,
            1.0;
            bc_u=(bcx, bcy),
            bc_p=nothing,
            bc_cut=Dirichlet(0.0),
            force=(fx, fy),
            gauge=MeanPressureGauge(),
        )
        sys = solve_steady!(model)
        @test norm(sys.A * sys.x - sys.b) < 1e-8
        m = velocity_pressure_metrics(model, sys, uex, vex, (x, y) -> 0.0)
        push!(hs, 1.0 / (n - 1))
        push!(eu, m.uL2)
        push!(ev, m.vL2)
    end

    @test eu[2] < eu[1]
    @test eu[3] < eu[2]
    @test ev[2] < ev[1]
    @test ev[3] < ev[2]
    for k in 1:2
        ord_u = log(eu[k] / eu[k + 1]) / log(hs[k] / hs[k + 1])
        ord_v = log(ev[k] / ev[k + 1]) / log(hs[k] / hs[k + 1])
        @test ord_u > 1.7
        @test ord_v > 1.6
    end
end

@testset "MMS tangential traction rows and DoNothing contrast (2D)" begin
    tx(x, y) = 2 * MMS_MU * tan_ux(x, y)
    ty(x, y) = MMS_MU * (tan_vx(x, y) + tan_uy(x, y))
    tvec(x, y) = SVector(tx(x, y), ty(x, y))

    bcx_t = BorderConditions(
        ; left=Dirichlet((x, y) -> tan_u(x, y)),
        right=Traction((x, y) -> tvec(x, y)),
        bottom=Dirichlet((x, y) -> tan_u(x, y)),
        top=Dirichlet((x, y) -> tan_u(x, y)),
    )
    bcy_t = BorderConditions(
        ; left=Dirichlet((x, y) -> tan_v(x, y)),
        right=Traction((x, y) -> tvec(x, y)),
        bottom=Dirichlet((x, y) -> tan_v(x, y)),
        top=Dirichlet((x, y) -> tan_v(x, y)),
    )
    n = 33
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bcx_dn = BorderConditions(
        ; left=Dirichlet((x, y) -> tan_u(x, y)),
        right=DoNothing(),
        bottom=Dirichlet((x, y) -> tan_u(x, y)),
        top=Dirichlet((x, y) -> tan_u(x, y)),
    )
    bcy_dn = BorderConditions(
        ; left=Dirichlet((x, y) -> tan_v(x, y)),
        right=DoNothing(),
        bottom=Dirichlet((x, y) -> tan_v(x, y)),
        top=Dirichlet((x, y) -> tan_v(x, y)),
    )
    model_t = StokesModelMono(
        grid,
        full_body,
        MMS_MU,
        1.0;
        bc_u=(bcx_t, bcy_t),
        bc_p=nothing,
        bc_cut=Dirichlet(0.0),
        force=(tan_fx, tan_fy),
        gauge=MeanPressureGauge(),
    )
    model_dn = StokesModelMono(
        grid,
        full_body,
        MMS_MU,
        1.0;
        bc_u=(bcx_dn, bcy_dn),
        bc_p=nothing,
        bc_cut=Dirichlet(0.0),
        force=(tan_fx, tan_fy),
        gauge=MeanPressureGauge(),
    )
    nsys = last(model_t.layout.pomega)
    asm_t = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    asm_dn = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(asm_t, model_t, 0.0)
    assemble_steady!(asm_dn, model_dn, 0.0)

    li = LinearIndices(model_t.cap_u[2].nnodes)
    I = CartesianIndex(model_t.cap_u[2].nnodes[1] - 1, 4)
    i = li[I]
    row_t = model_t.layout.uomega[2][i]
    row_dn = model_dn.layout.uomega[2][i]

    # Tangential traction RHS is nonzero, while DoNothing is homogeneous.
    @test abs(asm_t.b[row_t]) > 1e-6
    @test isapprox(asm_dn.b[row_dn], 0.0; atol=1e-12, rtol=0.0)
    # Cross-component coupling (∂_t u_n term) must appear.
    @test any(abs.(Array(asm_t.A[row_t, model_t.layout.uomega[1]])) .> 0.0)

    sys_t = solve_steady!(model_t)
    sys_dn = solve_steady!(model_dn)
    @test norm(sys_t.A * sys_t.x - sys_t.b) < 1e-8
    @test norm(sys_dn.A * sys_dn.x - sys_dn.b) < 1e-8

    function _uv_l2(model, x)
        u = x[model.layout.uomega[1]]
        v = x[model.layout.uomega[2]]
        e2 = 0.0
        w = 0.0
        for i in 1:model.cap_u[1].ntotal
            V = model.cap_u[1].buf.V[i]
            if isfinite(V) && V > 0.0
                xw = model.cap_u[1].C_ω[i]
                e2 += V * (u[i] - tan_u(xw[1], xw[2]))^2
                e2 += V * (v[i] - tan_v(xw[1], xw[2]))^2
                w += 2V
            end
        end
        return sqrt(e2 / w)
    end

    err_t = _uv_l2(model_t, sys_t.x)
    err_dn = _uv_l2(model_dn, sys_dn.x)
    @test abs(err_t - err_dn) > 1e-4
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

@testset "Embedded boundary force/stress utilities (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (41, 41))
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

    nsys = last(model.layout.pomega)
    x = zeros(Float64, nsys)
    p = view(x, model.layout.pomega)
    p .= 1.0

    q = embedded_boundary_quantities(model, x; pressure_reconstruction=:none, x0=(xc, yc))
    fint = integrated_embedded_force(model, x; pressure_reconstruction=:none, x0=(xc, yc))

    @test !isempty(q.interface_indices)

    @test isapprox(norm(q.force_viscous), 0.0; atol=1e-12, rtol=0.0)
    @test isapprox(norm(q.force), 0.0; atol=5e-3, rtol=0.0)
    @test isapprox(norm(q.force - (q.force_pressure + q.force_viscous)), 0.0; atol=1e-12, rtol=0.0)

    @test q.force == fint.force
    @test q.force_pressure == fint.force_pressure
    @test q.force_viscous == fint.force_viscous

    for i in q.interface_indices
        @test all(isfinite, q.traction[i])
        @test all(isfinite, q.force_density[i])
    end
    @test isfinite(q.torque)
end

@testset "Embedded boundary pressure/viscous consistency (2D periodic)" begin
    xc, yc, r = 0.5, 0.5, 0.2
    solid_area = pi * r^2
    ns = (33, 65, 129)
    err_none = Float64[]
    err_lin = Float64[]

    bcper = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Periodic(), top=Periodic(),
    )
    body(x, y) = r - sqrt((x - xc)^2 + (y - yc)^2)

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            body,
            1.0,
            1.0;
            bc_u=(bcper, bcper),
            bc_cut=Dirichlet(0.0),
            force=(0.0, 0.0),
        )

        nt = model.cap_p.ntotal
        nsys = last(model.layout.pomega)

        # Sanity: H*1 has zero global resultant on closed embedded boundary.
        h1 = model.op_p.H * ones(Float64, nt)
        @test abs(sum(h1[1:nt])) < 1e-12
        @test abs(sum(h1[(nt + 1):(2 * nt)])) < 1e-12

        # Sanity: geometric closure Σ Γ n ≈ 0 on closed interface.
        gsum = zeros(2)
        for i in 1:nt
            V = model.cap_p.buf.V[i]
            Γ = model.cap_p.buf.Γ[i]
            if isfinite(V) && V > 0.0 && isfinite(Γ) && Γ > 0.0
                gsum .+= Γ .* collect(model.cap_p.n_γ[i])
            end
        end
        @test norm(gsum) < 1e-12

        # A1: constant pressure -> zero net pressure force.
        x_const = zeros(Float64, nsys)
        p_const = view(x_const, model.layout.pomega)
        p_const .= 1.0
        q_const = embedded_boundary_quantities(model, x_const; pressure_reconstruction=:none)
        @test norm(q_const.force_pressure) < 1e-10
        @test norm(q_const.force_viscous) < 1e-12
        for x0 in ((0.5, 0.5), (0.15, 0.8), (1.1, -0.2))
            q0 = embedded_boundary_quantities(model, x_const; pressure_reconstruction=:none, x0=x0)
            @test norm(q0.force_pressure) < 1e-10
            @test abs(q0.torque) < 1e-10
        end

        # A2: linear pressure p=x (periodic outer box) -> |F_x| ≈ area(solid), F_y ≈ 0.
        x_lin = zeros(Float64, nsys)
        p_lin = view(x_lin, model.layout.pomega)
        for i in 1:nt
            xω = model.cap_p.C_ω[i]
            if isfinite(xω[1])
                p_lin[i] = xω[1]
            end
        end
        q_none = embedded_boundary_quantities(model, x_lin; pressure_reconstruction=:none)
        q_lin = embedded_boundary_quantities(model, x_lin; pressure_reconstruction=:linear)
        @test abs(q_none.force_pressure[2]) < 1e-8
        @test abs(q_lin.force_pressure[2]) < 1e-8
        push!(err_none, abs(abs(q_none.force_pressure[1]) - solid_area))
        push!(err_lin, abs(abs(q_lin.force_pressure[1]) - solid_area))
        @test err_lin[end] <= err_none[end] + 5e-3

        # B (robust): constant velocity field -> zero viscous force.
        x_vel = zeros(Float64, nsys)
        view(x_vel, model.layout.uomega[1]) .= 1.23
        view(x_vel, model.layout.uomega[2]) .= -0.7
        view(x_vel, model.layout.ugamma[1]) .= 1.23
        view(x_vel, model.layout.ugamma[2]) .= -0.7
        q_vel = embedded_boundary_quantities(model, x_vel; pressure_reconstruction=:none)
        @test norm(q_vel.force_viscous) < 1e-10
    end

    @test err_none[2] < err_none[1]
    @test err_none[3] < err_none[2]
    @test err_lin[2] < err_lin[1]
    @test err_lin[3] < err_lin[2]

    for k in 1:2
        ord_lin = log(err_lin[k] / err_lin[k + 1]) / log(2)
        @test ord_lin > 0.7
    end
end

@testset "Stress/traction rigid-rotation kernel (symmetric-gradient killer)" begin
    Grot = SMatrix{2,2,Float64}(0.0, -1.0, 1.0, 0.0) # skew-symmetric: rigid rotation
    normals = (
        SVector{2,Float64}(1.0, 0.0),
        SVector{2,Float64}(0.0, 1.0),
        normalize(SVector{2,Float64}(1.0, 1.0)),
        normalize(SVector{2,Float64}(-2.0, 3.0)),
    )

    σrot = PenguinStokes._stress_tensor(1.0, Grot, 0.0)
    @test norm(Matrix(σrot)) < 1e-14
    for n in normals
        t = PenguinStokes._traction_from_stress(σrot, n)
        @test norm(t) < 1e-14
    end

    # Nonzero pressure should yield pure -p*n traction when symmetric part vanishes.
    p0 = 2.5
    σp = PenguinStokes._stress_tensor(1.0, Grot, p0)
    for n in normals
        t = PenguinStokes._traction_from_stress(σp, n)
        @test isapprox(t, -p0 .* n; atol=1e-14, rtol=0.0)
    end
end

@testset "Two-phase geometry sanity (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (41, 41))
    xc, yc, r = 0.5, 0.5, 0.2
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r
    bc = all_dirichlet_bc(Val(2))

    model = StokesModelTwoPhase(
        grid,
        body,
        1.0,
        8.0;
        bc_u=(bc, bc),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=(0.0, 0.0),
    )

    nt = model.cap_p1.ntotal
    iface_count = 0
    gsum = zeros(2)
    for i in 1:nt
        Γ1 = model.cap_p1.buf.Γ[i]
        Γ2 = model.cap_p2.buf.Γ[i]
        has1 = isfinite(Γ1) && Γ1 > 0.0
        has2 = isfinite(Γ2) && Γ2 > 0.0
        @test has1 == has2
        has1 || continue
        iface_count += 1
        @test isapprox(Γ1, Γ2; atol=1e-10, rtol=1e-6)
        n1 = model.cap_p1.n_γ[i]
        n2 = model.cap_p2.n_γ[i]
        @test norm(n1 + n2) < 2e-5
        gsum .+= Γ1 .* collect(n1)
    end

    @test iface_count > 0
    @test norm(gsum) < 2e-4
end

@testset "Two-phase pressure-jump traction balance (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
    xc, yc, r = 0.5, 0.5, 0.2
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r
    bc = all_dirichlet_bc(Val(2))

    interface_force(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = sqrt(dx^2 + dy^2)
        rr == 0.0 ? SVector(0.0, 0.0) : SVector(dx / rr, dy / rr)
    end

    model = StokesModelTwoPhase(
        grid,
        body,
        1.0,
        3.0;
        bc_u=(bc, bc),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=interface_force,
    )

    sys = solve_steady!(model)
    rsys = sys.A * sys.x - sys.b
    @test norm(rsys) < 1e-8

    for d in 1:2
        u1 = sys.x[model.layout.uomega1[d]]
        u2 = sys.x[model.layout.uomega2[d]]
        idx1 = physical_active_indices(model.cap_u1[d])
        idx2 = physical_active_indices(model.cap_u2[d])
        @test maximum(abs, u1[idx1]) < 2e-4
        @test maximum(abs, u2[idx2]) < 2e-4
    end

    iface = interface_active_indices(model.cap_p1)
    @test !isempty(iface)
    for α in 1:2
        rows = model.layout.ugamma[α]
        rowvals = rsys[rows]
        @test maximum(abs, rowvals[iface]) < 5e-7
    end

    p1 = sys.x[model.layout.pomega1]
    p2 = sys.x[model.layout.pomega2]
    idxp1 = findall(PenguinStokes._pressure_activity(model.cap_p1))
    idxp2 = findall(PenguinStokes._pressure_activity(model.cap_p2))
    p1mean = sum(p1[idxp1]) / length(idxp1)
    p2mean = sum(p2[idxp2]) / length(idxp2)
    @test abs(abs(p2mean - p1mean) - 1.0) < 0.2
end

@testset "Two-phase unsteady zero-state residual (2D, BE/CN)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (21, 21))
    xc, yc, r = 0.5, 0.5, 0.2
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r
    bc = all_dirichlet_bc(Val(2))
    model = StokesModelTwoPhase(
        grid,
        body,
        1.0,
        3.0;
        rho1=1.2,
        rho2=0.7,
        bc_u=(bc, bc),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=(0.0, 0.0),
    )

    nsys = last(model.layout.pomega2)
    x0 = zeros(Float64, nsys)
    sys_be = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    sys_cn = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))

    assemble_unsteady!(sys_be, model, x0, 0.0, 0.05; scheme=:BE)
    assemble_unsteady!(sys_cn, model, x0, 0.0, 0.05; scheme=:CN)

    @test norm(sys_be.b) < 1e-10
    @test norm(sys_cn.b) < 1e-10
    @test norm(sys_be.A * x0 - sys_be.b) < 1e-10
    @test norm(sys_cn.A * x0 - sys_cn.b) < 1e-10
end

@testset "Two-phase planar Couette validation (steady)" begin
    mu1 = 1.0
    mu2 = 5.0
    h = 0.45
    U = 1.0
    H = 1.0
    ns = (33, 65)
    u_exact = two_layer_couette_profile(mu1, mu2, h, H, U)
    body(x, y) = y - h

    bcx = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(U),
    )
    bcy = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    e1 = Float64[]
    e2 = Float64[]
    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelTwoPhase(
            grid,
            body,
            mu1,
            mu2;
            bc_u=(bcx, bcy),
            force1=(0.0, 0.0),
            force2=(0.0, 0.0),
            interface_force=(0.0, 0.0),
        )
        sys = solve_steady!(model)
        m = two_phase_velocity_error_metrics(model, sys, u_exact)
        @test norm(sys.A * sys.x - sys.b) < 1e-8
        @test m.vInf < 1e-6
        push!(e1, m.u1L2)
        push!(e2, m.u2L2)
    end

    # On this benchmark the error can be close to machine precision already on the coarser grid.
    # In that regime strict monotonicity is not robust, so use a floating floor.
    floor_tol = 1e-10
    @test e1[2] < max(e1[1], floor_tol)
    @test e2[2] < max(e2[1], floor_tol)
    @test e1[2] < 5e-5
    @test e2[2] < 5e-5
end

@testset "Two-phase planar Poiseuille validation (steady body-force equivalent)" begin
    mu1 = 1.0
    mu2 = 4.0
    h = 0.4
    G = 1.5
    H = 1.0
    ns = (33, 65)
    u_exact = two_layer_bodyforce_poiseuille_profile(mu1, mu2, h, H, G)
    body(x, y) = y - h

    bcx = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    e1 = Float64[]
    e2 = Float64[]
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
            interface_force=(0.0, 0.0),
        )
        sys = solve_steady!(model)
        m = two_phase_velocity_error_metrics(model, sys, u_exact)
        @test norm(sys.A * sys.x - sys.b) < 1e-8
        @test m.vInf < 1e-6
        push!(e1, m.u1L2)
        push!(e2, m.u2L2)
    end

    @test e1[2] < e1[1]
    @test e2[2] < e2[1]
    @test e1[2] < 2e-3
    @test e2[2] < 2e-3
end

include("moving_boundary_stokes_tests.jl")
include("fsi_tests.jl")
include("test_outlet_row_equivalence.jl")
include("test_traction_box_debug.jl")
include("two_phase_static_circle_spurious_current_tests.jl")
