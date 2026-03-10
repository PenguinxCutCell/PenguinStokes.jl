using Test
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
    return SVector{2,T}(Tuple(q.force)), sys
end

function calibrated_drag(
    ; n::Int=17,
    dt::Float64=0.03,
    R::Float64=0.12,
    center::SVector{2,Float64}=SVector(0.5, 0.5),
)
    Funit, _ = fsi_single_slab_force(SVector(0.0, 1.0); n=n, dt=dt, R=R, center=center)
    force_sign = Funit[2] <= 0 ? 1.0 : -1.0
    zeta = -(force_sign * Funit[2])
    return (zeta=zeta, force_sign=force_sign)
end

function build_fsi_problem(
    ; n::Int=17,
    R::Float64=0.12,
    X0::SVector{2,Float64}=SVector(0.5, 0.5),
    V0::SVector{2,Float64}=SVector(0.0, 0.0),
    m::Float64=1.0,
    g::SVector{2,Float64}=SVector(0.0, 0.0),
    rho_fluid::Float64=0.0,
    buoyancy::Bool=false,
    force_sign::Float64=-1.0,
)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = fsi_periodic_2d()
    shape = Circle(R)
    body0(x, y, t) = shape.R - hypot(x - X0[1], y - X0[2])

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
        m,
        m / PenguinStokes.volume(shape),
        shape,
        g;
        rho_fluid=rho_fluid,
        buoyancy=buoyancy,
    )
    state = RigidBodyState{2,Float64}(X0, V0)
    fsi = StokesFSIProblem(
        model,
        state,
        params;
        pressure_reconstruction=:linear,
        force_sign=force_sign,
    )
    return fsi, params
end

@testset "FSI rigid-body integrator: exact linear-drag ODE (no fluid solve)" begin
    m = 2.0
    zeta = 5.0
    Fg = -1.2
    V0 = 0.7
    X0 = -0.1
    tf = 1.2

    function integrate(dt)
        state = RigidBodyState{2,Float64}(SVector(0.0, X0), SVector(0.0, V0))
        params = RigidBodyParams(
            m,
            1.0,
            Circle(0.05),
            SVector(0.0, Fg / m);
            rho_fluid=0.0,
            buoyancy=false,
        )
        t = 0.0
        while t < tf - 1e-12
            Fhydro = SVector(0.0, -zeta * state.V[2])
            PenguinStokes._advance_rigid_translation!(
                state,
                params,
                Fhydro,
                dt;
                ode_scheme=:symplectic_euler,
            )
            t += dt
        end
        return state
    end

    Vinf = Fg / zeta
    τ = m / zeta
    Vex = Vinf + (V0 - Vinf) * exp(-tf / τ)
    Xex = X0 + Vinf * tf + (V0 - Vinf) * τ * (1 - exp(-tf / τ))

    s1 = integrate(0.02)
    s2 = integrate(0.01)

    eV1 = abs(s1.V[2] - Vex)
    eV2 = abs(s2.V[2] - Vex)
    eX1 = abs(s1.X[2] - Xex)
    eX2 = abs(s2.X[2] - Xex)

    @test eV2 < eV1
    @test eX2 < eX1
    @test eV1 / eV2 > 1.7
    @test eX1 / eX2 > 1.7
end

@testset "FSI rigid-body wrapper: force linearity and odd symmetry (2D)" begin
    F1, _ = fsi_single_slab_force(SVector(1.0, 0.0))
    F2, _ = fsi_single_slab_force(SVector(2.0, 0.0))
    Fm, _ = fsi_single_slab_force(SVector(-1.0, 0.0))

    @test isapprox(F2[1], 2 * F1[1]; rtol=1e-2, atol=1e-6)
    @test isapprox(F2[2], 2 * F1[2]; rtol=1e-2, atol=1e-6)
    @test isapprox(Fm[1], -F1[1]; rtol=1e-2, atol=1e-6)
    @test isapprox(Fm[2], -F1[2]; rtol=1e-2, atol=1e-6)
end

@testset "FSI rigid-body wrapper: neutral-buoyancy velocity decay (2D)" begin
    dt = 0.03
    nsteps = 12
    tf = nsteps * dt

    cal = calibrated_drag(; n=17, dt=dt, R=0.12)
    @test cal.zeta > 0

    V0 = SVector(0.0, 0.05)
    fsi, _ = build_fsi_problem(
        ; n=17,
        R=0.12,
        X0=SVector(0.5, 0.5),
        V0=V0,
        m=1.0,
        g=SVector(0.0, 0.0),
        rho_fluid=0.0,
        buoyancy=false,
        force_sign=cal.force_sign,
    )

    t = 0.0
    for _ in 1:nsteps
        step_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t += dt
    end

    Vy_end = fsi.state.V[2]
    Vy_exact = V0[2] * exp(-(cal.zeta / 1.0) * tf)
    rel = abs(Vy_end - Vy_exact) / max(abs(Vy_exact), 1e-8)

    @test sign(Vy_end) == sign(V0[2])
    @test rel < 0.45
end

@testset "FSI rigid-body wrapper: calibrated free-fall transient (2D)" begin
    dt = 0.03
    nsteps = 16
    tf = nsteps * dt

    cal = calibrated_drag(; n=17, dt=dt, R=0.12)
    @test cal.zeta > 0

    fsi, params = build_fsi_problem(
        ; n=17,
        R=0.12,
        X0=SVector(0.5, 0.5),
        V0=SVector(0.0, 0.0),
        m=1.0,
        g=SVector(0.0, -0.15),
        rho_fluid=0.0,
        buoyancy=false,
        force_sign=cal.force_sign,
    )

    Fg = PenguinStokes.external_force(params)[2]
    Vinf = Fg / cal.zeta
    Vy_exact = Vinf + (0.0 - Vinf) * exp(-(cal.zeta / params.m) * tf)

    t = 0.0
    for _ in 1:nsteps
        step_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t += dt
    end

    Vy_end = fsi.state.V[2]
    rel = abs(Vy_end - Vy_exact) / max(abs(Vy_exact), 1e-8)

    @test sign(Vy_end) == sign(Vinf)
    @test rel < 0.25
end
