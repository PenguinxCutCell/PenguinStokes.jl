using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid, meshsize
using PenguinBCs
using PenguinStokes

function box_noslip_2d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
end

function one_slab_force_torque(
    grid::CartesianGrid{2,Float64},
    bc,
    shape,
    state::RigidBodyState2D{Float64};
    dt::Float64,
)
    statefun(_t) = state
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
    q = integrated_embedded_force(sm, sys; x0=Tuple(state.X))
    return q
end

function main()
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (51, 51))
    bc = box_noslip_2d()

    shape = Ellipse(0.14, 0.09)
    X0 = SVector(0.5, 0.70)
    V0 = SVector(0.0, 0.0)
    theta0 = 0.0
    omega0 = 0.10

    dt = 0.01
    nsteps = 80

    q_force = one_slab_force_torque(
        grid,
        bc,
        shape,
        RigidBodyState2D(X0, SVector(0.0, 1.0); theta=theta0, omega=0.0);
        dt=dt,
    )
    force_sign = q_force.force[2] <= 0 ? 1.0 : -1.0

    q_torque = one_slab_force_torque(
        grid,
        bc,
        shape,
        RigidBodyState2D(X0, SVector(0.0, 0.0); theta=theta0, omega=1.0);
        dt=dt,
    )
    torque_sign = q_torque.torque <= 0 ? 1.0 : -1.0

    rho_body = 1.6
    rho_fluid = 1.0
    m = 20.0
    I = 2.0

    params = RigidBodyParams2D(
        m,
        I,
        rho_body,
        shape,
        SVector(0.0, -0.02);
        rho_fluid=rho_fluid,
        buoyancy=true,
    )

    state0 = RigidBodyState2D(X0, V0; theta=theta0, omega=omega0)
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

    fsi = StokesFSIProblem2D(
        model,
        state0,
        params;
        force_sign=force_sign,
        torque_sign=torque_sign,
    )

    hmin = minimum(meshsize(grid))
    stop_margin = max(shape.a, shape.b) + 2hmin

    println("Free-falling rotating ellipse (2D FSI)")
    println("force_sign=", force_sign, ", torque_sign=", torque_sign)
    println("Columns: step, t, X, V, theta, omega, Fhydro, tau_hydro, force, torque, ||Ax-b||")

    t = 0.0
    for step in 1:nsteps
        out = step_fsi_rotation!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t = out.t
        res = norm(out.sys.A * out.sys.x - out.sys.b)

        println(
            step, ", ", t,
            ", ", Tuple(out.X),
            ", ", Tuple(out.V),
            ", ", out.theta,
            ", ", out.omega,
            ", ", Tuple(out.Fhydro),
            ", ", out.tau_hydro,
            ", ", Tuple(out.force.force),
            ", ", out.force.torque,
            ", ", res,
        )

        x = out.X
        dmin = min(x[1] - grid.lc[1], grid.hc[1] - x[1], x[2] - grid.lc[2], grid.hc[2] - x[2])
        if dmin <= stop_margin
            println("Stopping before contact: min wall distance=$dmin, threshold=$stop_margin")
            break
        end
    end
end

main()
