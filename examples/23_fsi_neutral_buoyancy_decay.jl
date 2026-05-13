using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

# This example demonstrates the neutral-buoyancy velocity decay of a circular body in a 2D periodic domain.
# CN split update has a visible two-step oscillation around the plateau. BE is smoother so that oscillation looks like a time-coupling artifact 

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
    q = integrated_embedded_force(sm, sys; x0=Tuple(X0))
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
        force_sign=cal.force_sign,
    )

    fluid_mass = 1.0 - PenguinStokes.volume(shape)
    vcm = params.m * V0[2] / (params.m + fluid_mass)
    τ = 1.0 / (cal.zeta * (1.0 / params.m + 1.0 / fluid_mass))
    println("Neutral-buoyancy velocity decay (calibrated drag)")
    println("zeta=", cal.zeta, ", tau_rel=", τ, ", Vy_inf=", vcm)
    println("Columns: step, t, Vy_num, Vy_momentum_model")

    t = 0.0
    for step in 1:nsteps
        step_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t += dt
        vy_exact = vcm + (V0[2] - vcm) * exp(-t / τ)
        println(step, ", ", t, ", ", fsi.state.V[2], ", ", vy_exact)
    end
end

main()
