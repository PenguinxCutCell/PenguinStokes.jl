using Test
using LinearAlgebra: norm
using SparseArrays: nnz
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, InterfaceConditions, ScalarJump, FluxJump
using PenguinStokes

function _active_velocity_indices(cap)
    return findall(isfinite.(cap.buf.V) .& (cap.buf.V .> 0.0))
end

function _avg(vals, idx)
    s = 0.0
    @inbounds for i in idx
        s += vals[i]
    end
    return s / length(idx)
end

function _maxabs(vals, idx)
    return isempty(idx) ? 0.0 : maximum(abs, vals[idx])
end

function _interface_indices(cap)
    return findall(isfinite.(cap.buf.Γ) .& (cap.buf.Γ .> 0.0))
end

function _active_jump_indices(sys, rows)
    return [i for i in eachindex(rows) if nnz(sys.A[rows[i], :]) > 1]
end

@testset "Step 0 - null Stokes with spherical interface (2D)" begin
    n = 36
    R = 0.35
    dp = 1.0
    xc, yc = 0.0, 0.0

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - R

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    interface_force(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? SVector(0.0, 0.0) : -(dp) * SVector(dx / rr, dy / rr)
    end

    model = StokesModelTwoPhase(
        grid,
        body,
        1.0,
        1.0;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc0, bc0),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=interface_force,
    )

    sys = solve_steady!(model)
    @test norm(sys.A * sys.x - sys.b) < 1e-8

    u1x = sys.x[model.layout.uomega1[1]]
    u1y = sys.x[model.layout.uomega1[2]]
    u2x = sys.x[model.layout.uomega2[1]]
    u2y = sys.x[model.layout.uomega2[2]]
    ug1x = sys.x[model.layout.ugamma1[1]]
    ug1y = sys.x[model.layout.ugamma1[2]]
    ug2x = sys.x[model.layout.ugamma2[1]]
    ug2y = sys.x[model.layout.ugamma2[2]]

    idx_u1x = _active_velocity_indices(model.cap_u1[1])
    idx_u1y = _active_velocity_indices(model.cap_u1[2])
    idx_u2x = _active_velocity_indices(model.cap_u2[1])
    idx_u2y = _active_velocity_indices(model.cap_u2[2])
    idx_ug = _interface_indices(model.cap_p1)

    u1_inf = max(_maxabs(u1x, idx_u1x), _maxabs(u1y, idx_u1y))
    u2_inf = max(_maxabs(u2x, idx_u2x), _maxabs(u2y, idx_u2y))
    ug1_inf = max(_maxabs(ug1x, idx_ug), _maxabs(ug1y, idx_ug))
    ug2_inf = max(_maxabs(ug2x, idx_ug), _maxabs(ug2y, idx_ug))

    @test u1_inf < 3e-2
    @test u2_inf < 3e-2
    @test ug1_inf < 3e-2
    @test ug2_inf < 3e-2
end

@testset "Two-phase interface BC API: [u]=f and [traction]=g (2D)" begin
    n = 24
    R = 0.25
    xc, yc = 0.0, 0.0

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - R

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    gtrac(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? SVector(0.0, 0.0) : -SVector(dx / rr, dy / rr)
    end

    icx = InterfaceConditions(
        scalar=ScalarJump(1.0, 1.0, 0.0),
        flux=FluxJump(1.0, 1.0, (x, y) -> gtrac(x, y)[1]),
    )
    icy = InterfaceConditions(
        scalar=ScalarJump(1.0, 1.0, 0.0),
        flux=FluxJump(1.0, 1.0, (x, y) -> gtrac(x, y)[2]),
    )

    model = StokesModelTwoPhase(
        grid,
        body,
        1.0,
        1.0;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc0, bc0),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        bc_interface=(icx, icy),
    )

    sys = solve_steady!(model)
    rsys = sys.A * sys.x - sys.b
    @test norm(rsys) < 1e-8

    iface = _interface_indices(model.cap_p1)
    @test !isempty(iface)

    ug1x = sys.x[model.layout.ugamma1[1]]
    ug1y = sys.x[model.layout.ugamma1[2]]
    ug2x = sys.x[model.layout.ugamma2[1]]
    ug2y = sys.x[model.layout.ugamma2[2]]

    @test maximum(abs, (ug1x .- ug2x)[iface]) < 1e-10
    @test maximum(abs, (ug1y .- ug2y)[iface]) < 1e-10
end

@testset "Two-phase radial flow from interface velocity jump (2D)" begin
    n = 24
    R = 0.25
    Uj = 0.05
    xc, yc = 0.0, 0.0

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - R

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    jumpx(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? 0.0 : Uj * dx / rr
    end
    jumpy(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? 0.0 : Uj * dy / rr
    end

    icx = InterfaceConditions(
        scalar=ScalarJump(1.0, 1.0, jumpx),
        flux=FluxJump(1.0, 1.0, 0.0),
    )
    icy = InterfaceConditions(
        scalar=ScalarJump(1.0, 1.0, jumpy),
        flux=FluxJump(1.0, 1.0, 0.0),
    )

    model = StokesModelTwoPhase(
        grid,
        body,
        1.0,
        1.0;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc0, bc0),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        bc_interface=(icx, icy),
    )

    sys = solve_steady!(model)
    @test norm(sys.A * sys.x - sys.b) < 1e-8

    ug1x = sys.x[model.layout.ugamma1[1]]
    ug1y = sys.x[model.layout.ugamma1[2]]
    ug2x = sys.x[model.layout.ugamma2[1]]
    ug2y = sys.x[model.layout.ugamma2[2]]

    idx_jump_x = _active_jump_indices(sys, model.layout.ugamma2[1])
    idx_jump_y = _active_jump_indices(sys, model.layout.ugamma2[2])
    @test !isempty(idx_jump_x)
    @test !isempty(idx_jump_y)

    ex = zeros(length(idx_jump_x))
    ey = zeros(length(idx_jump_y))
    @inbounds for (k, i) in enumerate(idx_jump_x)
        xg = model.cap_p1.C_γ[i]
        ex[k] = (ug1x[i] - ug2x[i]) - jumpx(xg[1], xg[2])
    end
    @inbounds for (k, i) in enumerate(idx_jump_y)
        xg = model.cap_p1.C_γ[i]
        ey[k] = (ug1y[i] - ug2y[i]) - jumpy(xg[1], xg[2])
    end

    @test maximum(abs, ex) < 1e-10
    @test maximum(abs, ey) < 1e-10

    idx_jump = intersect(idx_jump_x, idx_jump_y)
    @test !isempty(idx_jump)
    er = zeros(length(idx_jump))
    et = zeros(length(idx_jump))
    @inbounds for (k, i) in enumerate(idx_jump)
        xg = model.cap_p1.C_γ[i]
        rr = hypot(xg[1] - xc, xg[2] - yc)
        nx = (xg[1] - xc) / rr
        ny = (xg[2] - yc) / rr
        tx = -ny
        ty = nx
        djx = ug1x[i] - ug2x[i]
        djy = ug1y[i] - ug2y[i]
        er[k] = djx * nx + djy * ny - Uj
        et[k] = djx * tx + djy * ty
    end
    @test maximum(abs, er) < 1e-10
    @test maximum(abs, et) < 1e-10

    idx_u1x = _active_velocity_indices(model.cap_u1[1])
    idx_u1y = _active_velocity_indices(model.cap_u1[2])
    u1x = sys.x[model.layout.uomega1[1]]
    u1y = sys.x[model.layout.uomega1[2]]
    u1_inf = max(_maxabs(u1x, idx_u1x), _maxabs(u1y, idx_u1y))
    @test u1_inf > 1e-3
end

@testset "One-phase radial permeable circle (2D): interface velocity and flux" begin
    n = 49
    R = 0.35
    UΓ = 0.05
    xc, yc = 0.0, 0.0

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    body(x, y) = R - sqrt((x - xc)^2 + (y - yc)^2)

    uexact(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        if rr == 0.0
            return (0.0, 0.0)
        end
        ur = UΓ * R / rr
        return (ur * dx / rr, ur * dy / rr)
    end

    bcx = BorderConditions(
        ; left=Dirichlet((x, y) -> uexact(x, y)[1]),
        right=Dirichlet((x, y) -> uexact(x, y)[1]),
        bottom=Dirichlet((x, y) -> uexact(x, y)[1]),
        top=Dirichlet((x, y) -> uexact(x, y)[1]),
    )
    bcy = BorderConditions(
        ; left=Dirichlet((x, y) -> uexact(x, y)[2]),
        right=Dirichlet((x, y) -> uexact(x, y)[2]),
        bottom=Dirichlet((x, y) -> uexact(x, y)[2]),
        top=Dirichlet((x, y) -> uexact(x, y)[2]),
    )

    ugx_bc = Dirichlet((x, y) -> begin
        rr = hypot(x - xc, y - yc)
        rr == 0.0 ? 0.0 : UΓ * (x - xc) / rr
    end)
    ugy_bc = Dirichlet((x, y) -> begin
        rr = hypot(x - xc, y - yc)
        rr == 0.0 ? 0.0 : UΓ * (y - yc) / rr
    end)

    model = StokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bcx, bcy),
        bc_cut=(ugx_bc, ugy_bc),
        force=(0.0, 0.0),
    )

    sys = solve_steady!(model)
    @test norm(sys.A * sys.x - sys.b) < 1e-8

    ugx = sys.x[model.layout.ugamma[1]]
    ugy = sys.x[model.layout.ugamma[2]]
    idx_ugx = _interface_indices(model.cap_u[1])
    idx_ugy = _interface_indices(model.cap_u[2])
    @test !isempty(idx_ugx)
    @test !isempty(idx_ugy)

    println("uwx", sys.x[model.layout.uomega[1]])
    println("uwy", sys.x[model.layout.uomega[2]])
    ex = zeros(length(idx_ugx))
    ey = zeros(length(idx_ugy))
    @inbounds for (k, i) in enumerate(idx_ugx)
        xg = model.cap_u[1].C_γ[i]
        rr = hypot(xg[1] - xc, xg[2] - yc)
        ex[k] = ugx[i] - UΓ * (xg[1] - xc) / rr
    end
    @inbounds for (k, i) in enumerate(idx_ugy)
        xg = model.cap_u[2].C_γ[i]
        rr = hypot(xg[1] - xc, xg[2] - yc)
        ey[k] = ugy[i] - UΓ * (xg[2] - yc) / rr
    end

    @test maximum(abs, ex) < 1e-12
    @test maximum(abs, ey) < 1e-12

    # Flux through the permeable circle: |Q| = 2πRUΓ (sign depends on normal orientation).
    Qh = 0.0
    @inbounds for i in idx_ugx
        Qh += model.cap_u[1].buf.Γ[i] * ugx[i] * model.cap_u[1].n_γ[i][1]
    end
    @inbounds for i in idx_ugy
        Qh += model.cap_u[2].buf.Γ[i] * ugy[i] * model.cap_u[2].n_γ[i][2]
    end
    Qth = 2π * R * UΓ
    @test abs(abs(Qh) - Qth) / Qth < 1e-2
end

@testset "Two-phase static spurious-current benchmark (2D, Dcell=6.4)" begin
    # Literature-style static cylinder benchmark parameters.
    D = 0.4
    R = D / 2
    mu = 0.1
    rho = 300.0
    sigma = 1.0
    dp_th = sigma / R
    La = rho * sigma * D / mu^2
    n = 16 # Dcell = D / (1/n) = 6.4
    xc, yc = 0.5, 0.5

    @test isapprox(La, 1.2e4; atol=1e-9, rtol=0.0)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - R

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    interface_force(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? SVector(0.0, 0.0) : -(dp_th) * SVector(dx / rr, dy / rr)
    end

    model = StokesModelTwoPhase(
        grid,
        body,
        mu,
        mu;
        rho1=rho,
        rho2=rho,
        bc_u=(bc0, bc0),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=interface_force,
    )

    sys = solve_steady!(model)

    u1x = sys.x[model.layout.uomega1[1]]
    u1y = sys.x[model.layout.uomega1[2]]
    u2x = sys.x[model.layout.uomega2[1]]
    u2y = sys.x[model.layout.uomega2[2]]

    idx_u1x = _active_velocity_indices(model.cap_u1[1])
    idx_u1y = _active_velocity_indices(model.cap_u1[2])
    idx_u2x = _active_velocity_indices(model.cap_u2[1])
    idx_u2y = _active_velocity_indices(model.cap_u2[2])

    umax = max(maximum(abs, u1x[idx_u1x]), maximum(abs, u2x[idx_u2x]))
    vmax = max(maximum(abs, u1y[idx_u1y]), maximum(abs, u2y[idx_u2y]))
    uabsmax = max(umax, vmax)
    camax = mu * uabsmax / sigma

    p1 = sys.x[model.layout.pomega1]
    p2 = sys.x[model.layout.pomega2]
    idx_p1 = findall(PenguinStokes._pressure_activity(model.cap_p1))
    idx_p2 = findall(PenguinStokes._pressure_activity(model.cap_p2))
    dp_num = _avg(p1, idx_p1) - _avg(p2, idx_p2)
    jump_relerr = abs(dp_num - dp_th) / abs(dp_th)

    @test camax < 5e-3
    @test jump_relerr < 0.05
end
