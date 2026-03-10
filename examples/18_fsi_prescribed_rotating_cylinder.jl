using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

function box_noslip_2d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
end

function main()
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (51, 51))
    bc = box_noslip_2d()

    shape = Circle(0.12)
    X0 = SVector(0.5, 0.5)

    amp = 0.35
    Omega = 2pi
    dt = 0.02
    nsteps = 40

    theta_fun(t) = amp * sin(Omega * t)
    omega_fun(t) = amp * Omega * cos(Omega * t)

    statefun(t) = RigidBodyState2D(
        X0,
        SVector(0.0, 0.0);
        theta=theta_fun(t),
        omega=omega_fun(t),
    )

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

    println("Prescribed rotating cylinder (2D)")
    println("Columns: step, t, theta, omega, torque, force, ||Ax-b||")

    t = 0.0
    for step in 1:nsteps
        sys = solve_unsteady_moving!(model, xprev; t=t, dt=dt, scheme=:CN)
        tnext = t + dt

        sm = endtime_static_model(model)
        q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(X0))
        res = norm(sys.A * sys.x - sys.b)

        println(
            step, ", ", tnext,
            ", ", theta_fun(tnext),
            ", ", omega_fun(tnext),
            ", ", q.torque,
            ", ", Tuple(q.force),
            ", ", res,
        )

        xprev .= sys.x
        t = tnext
    end
end

main()
