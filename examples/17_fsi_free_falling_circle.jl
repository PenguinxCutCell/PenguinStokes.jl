using LinearAlgebra
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

function main()
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (51, 51))

    μ = 1.0
    ρ = 1.0

    shape = Circle(0.10)
    X0 = SVector(0.5, 0.75)
    V0 = SVector(0.0, 0.0)

    rho_body = 1.0
    rho_fluid = 1.0
    m = 1.0
    g = SVector(0.0, -0.15)

    body0(x, y, t) = shape.R - hypot(x - X0[1], y - X0[2])

    bc = box_noslip_2d()
    model = MovingStokesModelMono(
        grid,
        body0,
        μ,
        ρ;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)),
    )

    # Calibrate force sign so +Vy gives opposite (drag) hydrodynamic Fy on the body.
    model_drag = MovingStokesModelMono(
        grid,
        body0,
        μ,
        ρ;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(1.0)),
    )
    xcal = zeros(Float64, last(model_drag.layout.pomega))
    sys_cal = solve_unsteady_moving!(model_drag, xcal; t=0.0, dt=0.02, scheme=:CN)
    sm_cal = endtime_static_model(model_drag)
    q_cal = integrated_embedded_force(sm_cal, sys_cal; x0=Tuple(X0))
    force_sign = q_cal.force[2] <= 0 ? 1.0 : -1.0

    state = RigidBodyState{2,Float64}(X0, V0)
    params = RigidBodyParams(
        m,
        rho_body,
        shape,
        g;
        rho_fluid=rho_fluid,
        buoyancy=true,
    )

    fsi = StokesFSIProblem(
        model,
        state,
        params;
        force_sign=force_sign,
    )

    dt = 0.02
    nsteps = 60
    t = 0.0

    hmin = minimum(meshsize(grid))
    stop_margin = shape.R + 2hmin

    println("Free-falling rigid circle FSI (translation-only)")
    println("grid=$(grid.n), dt=$dt, steps=$nsteps")
    println("force_sign=$force_sign")
    println("Columns: step, t, X, V, Fhydro, force, torque, ||Ax-b||")

    for step in 1:nsteps
        out = step_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t = out.t
        res = norm(out.sys.A * out.sys.x - out.sys.b)

        println(
            step, ", ", t,
            ", ", Tuple(out.X),
            ", ", Tuple(out.V),
            ", ", Tuple(out.Fhydro),
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
