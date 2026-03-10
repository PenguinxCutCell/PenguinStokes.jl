using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

periodic_2d_bc() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

function calibrated_drag(; n=41, dt=0.02, R=0.12)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = periodic_2d_bc()
    X0 = SVector(0.5, 0.5)
    shape = Circle(R)
    body(x, y, t) = shape.R - hypot(x - X0[1], y - X0[2])

    model = MovingStokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(1.0)),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=0.0, dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(X0))
    Funit = SVector{2,Float64}(Tuple(q.force))
    force_sign = Funit[2] <= 0 ? 1.0 : -1.0
    zeta = -(force_sign * Funit[2])
    return (zeta=zeta, force_sign=force_sign)
end

function main()
    n = 41
    dt = 0.02
    nsteps = 40

    R = 0.12
    X0 = SVector(0.5, 0.5)
    V0 = SVector(0.0, 0.05)

    cal = calibrated_drag(; n=n, dt=dt, R=R)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = periodic_2d_bc()
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
        1.0,
        1.0 / PenguinStokes.volume(shape),
        shape,
        SVector(0.0, 0.0);
        rho_fluid=1.0,
        buoyancy=true,
    )
    state = RigidBodyState{2,Float64}(X0, V0)
    fsi = StokesFSIProblem(
        model,
        state,
        params;
        pressure_reconstruction=:linear,
        force_sign=cal.force_sign,
    )

    τ = params.m / cal.zeta
    println("Neutral-buoyancy velocity decay (calibrated drag)")
    println("zeta=", cal.zeta, ", tau=", τ)
    println("Columns: step, t, Vy_num, Vy_exact")

    t = 0.0
    for step in 1:nsteps
        step_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t += dt
        vy_exact = V0[2] * exp(-t / τ)
        println(step, ", ", t, ", ", fsi.state.V[2], ", ", vy_exact)
    end
end

main()
