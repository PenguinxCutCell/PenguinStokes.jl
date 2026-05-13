using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

box_noslip_2d() = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

function calibrate_rotational_drag(shape::Circle{Float64}, X0::SVector{2,Float64}; n::Int=41, dt::Float64=0.02)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = box_noslip_2d()

    probe_state = RigidBodyState2D(X0, SVector(0.0, 0.0); theta=0.0, omega=1.0)
    statefun(_t) = probe_state

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

    torque_sign = q.torque <= 0 ? 1.0 : -1.0
    kappa = -(torque_sign * q.torque)

    return (kappa=kappa, torque_sign=torque_sign)
end

function main()
    n = 41
    dt = 0.02
    nsteps = 12

    shape = Circle(0.12)
    X0 = SVector(0.5, 0.5)

    cal = calibrate_rotational_drag(shape, X0; n=n, dt=dt)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = box_noslip_2d()

    state0 = RigidBodyState2D(X0, SVector(0.0, 0.0); theta=0.0, omega=0.6)
    statefun(_t) = state0

    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple_2d(statefun),
    )

    # Use a moderate inertia to keep the explicit rigid-body step stable at this dt.
    Ieff = 1.0
    params = RigidBodyParams2D(
        1.0,
        Ieff,
        1.0,
        shape,
        SVector(0.0, 0.0);
        rho_fluid=0.0,
        buoyancy=false,
    )

    fsi = StokesFSIProblem2D(
        model,
        state0,
        params;
        force_sign=1.0,
        torque_sign=cal.torque_sign,
    )

    tau = params.I / cal.kappa
    println("FSI spin decay with calibrated rotational drag")
    println("kappa=", cal.kappa, ", tau=", tau, ", torque_sign=", cal.torque_sign)
    println("Columns: step, t, omega_num, omega_exact, theta_num, theta_exact, ||Ax-b||")

    omega0 = fsi.state.omega
    theta0 = fsi.state.theta

    t = 0.0
    for step in 1:nsteps
        out = step_fsi_rotation!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t = out.t
        res = norm(out.sys.A * out.sys.x - out.sys.b)

        omega_exact = omega0 * exp(-t / tau)
        theta_exact = theta0 + omega0 * tau * (1 - exp(-t / tau))

        println(step, ", ", t, ", ", out.omega, ", ", omega_exact, ", ", out.theta, ", ", theta_exact, ", ", res)
    end
end

main()
