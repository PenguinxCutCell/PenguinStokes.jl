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

fsi_periodic_3d() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
    backward=Periodic(), forward=Periodic(),
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
    q = integrated_embedded_force(sm, sys; x0=Tuple(center))
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
    q = integrated_embedded_force(sm, sys; x0=Tuple(X0))
    return SVector{2,Float64}(Tuple(q.force)), q.torque, sys, model
end

function fsi_single_slab_force_3d(
    U::SVector{3,T};
    n::Int=11,
    dt::T=T(0.03),
    R::T=T(0.16),
    center::SVector{3,T}=SVector{3,T}(T(0.5), T(0.5), T(0.5)),
) where {T}
    grid = CartesianGrid((zero(T), zero(T), zero(T)), (one(T), one(T), one(T)), (n, n, n))
    bc = fsi_periodic_3d()
    shape = Sphere(R)
    body(x, y, z, t) = shape.R - sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2)

    model = MovingStokesModelMono(
        grid,
        body,
        one(T),
        one(T);
        bc_u=(bc, bc, bc),
        force=(zero(T), zero(T), zero(T)),
        bc_cut_u=(Dirichlet(U[1]), Dirichlet(U[2]), Dirichlet(U[3])),
    )

    xprev = zeros(T, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=zero(T), dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; x0=Tuple(center))
    return SVector{3,T}(Tuple(q.force)), q.torque, sys
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

@testset "FSI rigid velocity helpers: 3D + orientation orthonormality" begin
    x = SVector(2.0, -1.0, 4.0)
    X = SVector(0.5, 1.5, -0.5)
    V = SVector(1.0, -2.0, 0.75)
    Omega = SVector(0.2, -0.3, 0.4)

    u = rigid_boundary_velocity(x, X, V, Omega)
    uref = V + SVector(
        Omega[2] * (x[3] - X[3]) - Omega[3] * (x[2] - X[2]),
        Omega[3] * (x[1] - X[1]) - Omega[1] * (x[3] - X[3]),
        Omega[1] * (x[2] - X[2]) - Omega[2] * (x[1] - X[1]),
    )
    @test isapprox(norm(u - uref), 0.0; atol=1e-12)

    ori = PenguinStokes.identity_orientation(Val(3), Float64)
    for _ in 1:100
        ori = PenguinStokes.advance_orientation(ori, Omega, 0.01)
    end
    I3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @test norm(Matrix(PenguinStokes.rotation_matrix(ori)' * PenguinStokes.rotation_matrix(ori)) - I3) < 1e-10
end

@testset "FSI sphere/circle geometry invariance under rotation" begin
    cshape = Circle(0.2)
    s2a = (_t) -> RigidBodyState2D(SVector(0.5, 0.5), SVector(0.0, 0.0); theta=0.0, omega=0.0)
    s2b = (_t) -> RigidBodyState2D(SVector(0.5, 0.5), SVector(0.0, 0.0); theta=1.37, omega=0.0)
    phi2a = rigid_body_levelset(cshape, s2a)
    phi2b = rigid_body_levelset(cshape, s2b)
    @test isapprox(phi2a(0.6, 0.4, 0.0), phi2b(0.6, 0.4, 0.0); atol=1e-12)

    sshape = Sphere(0.2)
    Qrot = PenguinStokes.advance_orientation(
        PenguinStokes.identity_orientation(Val(3), Float64),
        SVector(0.3, -0.2, 0.1),
        1.0,
    ).Q
    s3a = (_t) -> RigidBodyState3D(SVector(0.5, 0.5, 0.5), SVector(0.0, 0.0, 0.0))
    s3b = (_t) -> RigidBodyState3D(SVector(0.5, 0.5, 0.5), SVector(0.0, 0.0, 0.0); Q=Qrot, Omega=SVector(0.0, 0.0, 0.0))
    phi3a = rigid_body_levelset(sshape, s3a)
    phi3b = rigid_body_levelset(sshape, s3b)
    @test isapprox(phi3a(0.58, 0.44, 0.61, 0.0), phi3b(0.58, 0.44, 0.61, 0.0); atol=1e-12)
end

@testset "FSI 3D prescribed sphere drag: linearity and odd symmetry" begin
    F1, _, _ = fsi_single_slab_force_3d(SVector(0.0, 0.0, 1.0))
    F2, _, _ = fsi_single_slab_force_3d(SVector(0.0, 0.0, 2.0))
    Fm, _, _ = fsi_single_slab_force_3d(SVector(0.0, 0.0, -1.0))

    @test isapprox(F2[1], 2 * F1[1]; rtol=4e-2, atol=2e-6)
    @test isapprox(F2[2], 2 * F1[2]; rtol=4e-2, atol=2e-6)
    @test isapprox(F2[3], 2 * F1[3]; rtol=4e-2, atol=2e-6)
    @test isapprox(Fm[3], -F1[3]; rtol=4e-2, atol=2e-6)
end

@testset "FSI strong coupling: convergence and split consistency at small dt" begin
    dt = 0.01
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = fsi_periodic_2d()
    shape = Circle(0.12)
    state0 = RigidBodyState((0.5, 0.5), (0.08, 0.0))
    body0(x, y, t) = shape.R - hypot(x - state0.X[1], y - state0.X[2])

    function build_problem(state)
        model = MovingStokesModelMono(
            grid,
            body0,
            1.0,
            1.0;
            bc_u=(bc, bc),
            force=(0.0, 0.0),
            bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)),
        )
        params = RigidBodyParams(
            1.0,
            1.0,
            shape,
            SVector(0.0, 0.0);
            rho_fluid=0.0,
            buoyancy=false,
        )
        return StokesFSIProblem(
            model,
            state,
            params;
            force_sign=1.0,
            torque_sign=1.0,
        )
    end

    fsi_split = build_problem(RigidBodyState{2,Float64}(state0.X, state0.V))
    out_split = step_fsi!(fsi_split; t=0.0, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)

    fsi_strong = build_problem(RigidBodyState{2,Float64}(state0.X, state0.V))
    out_strong = step_fsi_strong!(
        fsi_strong;
        t=0.0,
        dt=dt,
        fluid_scheme=:CN,
        ode_scheme=:symplectic_euler,
        maxiter=8,
        atol=1e-9,
        rtol=1e-7,
        relaxation=:aitken,
        omega_relax=0.8,
    )

    @test out_strong.converged
    @test out_strong.iterations <= 8
    @test norm(out_split.X - out_strong.X) < 2e-2
    @test norm(out_split.V - out_strong.V) < 2e-2
end
