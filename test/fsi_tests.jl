using Test
using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

fsi_periodic_2d() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

function fsi_single_slab_force(
    U::SVector{2,T};
    n::Int=17,
    dt::T=T(0.03),
    R::T=T(0.12),
    center::SVector{2,T}=SVector{2,T}(T(0.5), T(0.5)),
    rho::T=one(T),
) where {T}
    grid = CartesianGrid((zero(T), zero(T)), (one(T), one(T)), (n, n))
    bc = fsi_periodic_2d()
    shape = Circle(R)
    body(x, y, t) = shape.R - hypot(x - center[1], y - center[2])

    model = MovingStokesModelMono(
        grid,
        body,
        one(T),
        rho;
        bc_u=(bc, bc),
        force=(zero(T), zero(T)),
        bc_cut_u=(Dirichlet(U[1]), Dirichlet(U[2])),
    )

    xprev = zeros(T, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=zero(T), dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(center))
    return SVector{2,T}(Tuple(q.force)), q.torque, sys
end

function fsi_single_slab_force_torque_rotation(
    shape;
    X0::SVector{2,Float64}=SVector(0.5, 0.5),
    V::SVector{2,Float64}=SVector(0.0, 0.0),
    theta::Float64=0.0,
    omega::Float64=0.0,
    n::Int=17,
    dt::Float64=0.03,
)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = fsi_periodic_2d()
    s = RigidBodyState2D(X0, V; theta=theta, omega=omega)
    statefun = (_t) -> s

    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple_2d(statefun),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=0.0, dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(X0))
    return SVector{2,Float64}(Tuple(q.force)), q.torque, sys, model
end

@testset "FSI translation wrapper: force linearity and odd symmetry (2D)" begin
    F1, _, _ = fsi_single_slab_force(SVector(1.0, 0.0))
    F2, _, _ = fsi_single_slab_force(SVector(2.0, 0.0))
    Fm, _, _ = fsi_single_slab_force(SVector(-1.0, 0.0))

    @test isapprox(F2[1], 2 * F1[1]; rtol=1e-2, atol=1e-6)
    @test isapprox(F2[2], 2 * F1[2]; rtol=1e-2, atol=1e-6)
    @test isapprox(Fm[1], -F1[1]; rtol=1e-2, atol=1e-6)
    @test isapprox(Fm[2], -F1[2]; rtol=1e-2, atol=1e-6)
end

@testset "FSI rotation helpers: rigid velocity and cut BCs" begin
    x = SVector(4.0, 5.0)
    X = SVector(0.0, 0.0)
    V = SVector(1.0, 2.0)
    omega = 3.0

    u = rigid_velocity_2d(x, X, V, omega)
    @test u == SVector(-14.0, 14.0)

    s0 = (_t) -> RigidBodyState2D(X, V; theta=0.0, omega=0.0)
    bct = rigid_cut_bc_tuple_2d(s0)
    @test isapprox(eval_bc(bct[1].value, SVector(0.1, 0.2), 0.3), V[1]; atol=0, rtol=0)
    @test isapprox(eval_bc(bct[2].value, SVector(0.1, 0.2), 0.3), V[2]; atol=0, rtol=0)

    Xc = SVector(0.25, 0.5)
    w0 = 0.7
    srot = RigidBodyState2D(Xc, SVector(0.0, 0.0); theta=0.0, omega=w0)

    ux = rigid_velocity(SVector(Xc[1] + 0.2, Xc[2]), srot)
    uy = rigid_velocity(SVector(Xc[1], Xc[2] + 0.2), srot)
    @test isapprox(ux[1], 0.0; atol=1e-12)
    @test isapprox(ux[2], w0 * 0.2; atol=1e-12)
    @test isapprox(uy[1], -w0 * 0.2; atol=1e-12)
    @test isapprox(uy[2], 0.0; atol=1e-12)
end

@testset "FSI rotation wrapper: torque odd symmetry and linear scaling (2D)" begin
    shape = Circle(0.12)
    X0 = SVector(0.5, 0.5)
    w0 = 0.6

    _, tau_p, _, _ = fsi_single_slab_force_torque_rotation(shape; X0=X0, omega=w0)
    _, tau_m, _, _ = fsi_single_slab_force_torque_rotation(shape; X0=X0, omega=-w0)
    _, tau_2, _, _ = fsi_single_slab_force_torque_rotation(shape; X0=X0, omega=2w0)

    @test isfinite(tau_p)
    @test abs(tau_p) > 1e-8
    @test isapprox(tau_m, -tau_p; rtol=2e-2, atol=1e-6)
    @test isapprox(tau_2, 2 * tau_p; rtol=2e-2, atol=1e-6)
end

@testset "FSI rotation wrapper: calibrated spin decay (2D)" begin
    dt = 0.03
    nsteps = 10
    tf = nsteps * dt

    shape = Circle(0.12)
    X0 = SVector(0.5, 0.5)

    _, tau_ref, _, _ = fsi_single_slab_force_torque_rotation(shape; X0=X0, omega=1.0, n=17, dt=dt)
    torque_sign = tau_ref <= 0 ? 1.0 : -1.0
    kappa = -(torque_sign * tau_ref)

    @test kappa > 0

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = fsi_periodic_2d()

    state = RigidBodyState2D(X0, SVector(0.0, 0.0); theta=0.0, omega=0.3)
    statefun = (_t) -> state
    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple_2d(statefun),
    )

    params = RigidBodyParams2D(
        1.0,
        1.0,
        1.0,
        shape,
        SVector(0.0, 0.0);
        rho_fluid=0.0,
        buoyancy=false,
    )

    fsi = StokesFSIProblem2D(
        model,
        state,
        params;
        pressure_reconstruction=:linear,
        force_sign=1.0,
        torque_sign=torque_sign,
    )

    omega0 = fsi.state.omega
    t = 0.0
    for _ in 1:nsteps
        step_fsi_rotation!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t += dt
    end

    omega_exact = omega0 * exp(-(kappa / params.I) * tf)
    rel = abs(fsi.state.omega - omega_exact) / max(abs(omega_exact), 1e-8)

    @test sign(fsi.state.omega) == sign(omega0)
    @test rel < 0.5
end

@testset "FSI rotation wrapper: ellipse geometry rotation integration" begin
    dt = 0.03
    shape = Ellipse(0.18, 0.12)
    X0 = SVector(0.5, 0.5)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = fsi_periodic_2d()

    state = RigidBodyState2D(X0, SVector(0.0, 0.0); theta=0.0, omega=0.4)
    statefun = (_t) -> state
    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple_2d(statefun),
    )

    params = RigidBodyParams2D(
        1.0,
        PenguinStokes.body_inertia(shape, 1.0),
        1.0,
        shape,
        SVector(0.0, 0.0);
        rho_fluid=0.0,
        buoyancy=false,
    )

    fsi = StokesFSIProblem2D(
        model,
        state,
        params;
        pressure_reconstruction=:linear,
        force_sign=1.0,
        torque_sign=1.0,
    )

    out = step_fsi_rotation!(fsi; t=0.0, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
    @test norm(out.sys.A * out.sys.x - out.sys.b) < 1e-8
    @test isfinite(out.force.torque)

    # At t=dt, lab-frame point on initial major-axis endpoint should not remain on boundary.
    phi_end = fsi.model.body(X0[1] + shape.a, X0[2], dt)
    @test abs(phi_end) > 1e-5
end
