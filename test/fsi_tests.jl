using Test
using LinearAlgebra
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

fsi_periodic_2d() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

function fsi_single_slab_force(U::SVector{2,T}; n::Int=17, dt::T=T(0.03), R::T=T(0.12)) where {T}
    grid = CartesianGrid((zero(T), zero(T)), (one(T), one(T)), (n, n))
    bc = fsi_periodic_2d()
    X0 = SVector{2,T}(convert(T, 0.5), convert(T, 0.5))
    shape = Circle(R)
    body(x, y, t) = shape.R - hypot(x - X0[1], y - X0[2])

    model = MovingStokesModelMono(
        grid,
        body,
        one(T),
        one(T);
        bc_u=(bc, bc),
        force=(zero(T), zero(T)),
        bc_cut_u=(Dirichlet(U[1]), Dirichlet(U[2])),
    )

    xprev = zeros(T, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=zero(T), dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(X0))
    return SVector{2,T}(Tuple(q.force)), sys
end

@testset "FSI rigid-body wrapper: force linearity in cut velocity (2D)" begin
    F1, _ = fsi_single_slab_force(SVector(1.0, 0.0))
    F2, _ = fsi_single_slab_force(SVector(2.0, 0.0))

    @test isapprox(F2[1], 2 * F1[1]; rtol=1e-2, atol=1e-6)
    @test isapprox(F2[2], 2 * F1[2]; rtol=1e-2, atol=1e-6)
end

@testset "FSI rigid-body wrapper: terminal speed from measured drag (2D)" begin
    T = Float64
    dt = 0.03
    n = 17

    Funit, _ = fsi_single_slab_force(SVector{2,T}(0.0, 1.0); n=n, dt=dt, R=0.12)
    rawFy = Funit[2]
    force_sign = rawFy <= 0 ? one(T) : -one(T)
    drag_coeff = -(force_sign * rawFy)

    @test drag_coeff > 0

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = fsi_periodic_2d()
    shape = Circle(0.12)
    X0 = SVector(0.5, 0.5)
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

    m = 1.0
    gvec = SVector(0.0, -0.15)
    params = RigidBodyParams(m, m / PenguinStokes.volume(shape), shape, gvec; rho_fluid=0.0, buoyancy=false)
    state = RigidBodyState{2,Float64}(X0, SVector(0.0, 0.0))

    fsi = StokesFSIProblem(
        model,
        state,
        params;
        pressure_reconstruction=:linear,
        force_sign=force_sign,
    )

    Fext = PenguinStokes.external_force(params)
    Vinf_pred = Fext[2] / drag_coeff

    t = 0.0
    nsteps = 20
    for _ in 1:nsteps
        step_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t += dt
    end

    Vy_end = fsi.state.V[2]
    rel = abs(Vy_end - Vinf_pred) / max(abs(Vinf_pred), 1e-8)

    @test sign(Vy_end) == sign(Vinf_pred)
    @test rel < 0.2
end
