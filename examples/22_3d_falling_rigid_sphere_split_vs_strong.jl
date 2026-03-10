using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid, meshsize
using PenguinBCs
using PenguinStokes

function box_noslip_3d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
end

function calibrate_drag_sign_and_kappa(
    grid,
    bc,
    shape,
    center::SVector{3,Float64};
    mu::Float64=1.0,
    rho::Float64=1.0,
    dt::Float64=0.03,
    Uprobe::Float64=1.0,
)
    state_probe = RigidBodyState3D(center, SVector(0.0, 0.0, Uprobe))
    statefun(_t) = state_probe

    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        mu,
        rho;
        bc_u=(bc, bc, bc),
        force=(0.0, 0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple(statefun, Val(3)),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=0.0, dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(center))

    force_sign = q.force[3] <= 0 ? 1.0 : -1.0
    Fhydro_probe = force_sign * q.force[3]
    kappa = -Fhydro_probe / Uprobe
    return (force_sign=force_sign, kappa=kappa, q=q)
end

function build_problem(
    grid,
    bc,
    shape,
    state0::RigidBodyState3D{Float64},
    params::RigidBodyParams3D{Float64,Sphere{Float64}};
    force_sign::Float64,
) 
    statefun(_t) = state0
    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        1.0,
        1.0;
        bc_u=(bc, bc, bc),
        force=(0.0, 0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple(statefun, Val(3)),
    )
    return StokesFSIProblem(
        model,
        state0,
        params;
        pressure_reconstruction=:linear,
        force_sign=force_sign,
        torque_sign=1.0,
    )
end

function main()
    mu = 1.0
    rho = 1.0
    rho_s = 1.4
    rho_f = 1.0
    g0 = 0.2

    R = 0.12
    center0 = SVector(0.5, 0.5, 0.5)
    V0 = SVector(0.0, 0.0, 0.0)

    dt = 0.02
    nsteps = 3

    grid = CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (9, 9, 9))
    bc = box_noslip_3d()
    shape = Sphere(R)

    Iiso = 1.0
    m = rho_s * PenguinStokes.volume(shape)
    params = RigidBodyParams3D(
        m,
        Iiso,
        rho_s,
        shape,
        SVector(0.0, 0.0, -g0);
        rho_fluid=rho_f,
        buoyancy=true,
    )

    cal = calibrate_drag_sign_and_kappa(grid, bc, shape, center0; mu=mu, rho=rho, dt=dt)

    split = build_problem(
        grid,
        bc,
        shape,
        RigidBodyState3D(center0, V0),
        params;
        force_sign=cal.force_sign,
    )

    strong = build_problem(
        grid,
        bc,
        shape,
        RigidBodyState3D(center0, V0),
        params;
        force_sign=cal.force_sign,
    )

    println("3D falling rigid sphere: split vs strong coupling")
    println("kappa_num=", cal.kappa, ", force_sign=", cal.force_sign)
    println("Columns: step, t, Vz_split, Vz_strong, Vz_exact, res_split, res_strong, it_strong")

    t = 0.0
    Fg = PenguinStokes.external_force(params)[3]
    kappa = cal.kappa
    tau = m / kappa
    Vinf = Fg / kappa

    hmin = minimum(meshsize(grid))
    stop_margin = R + hmin

    for step in 1:nsteps
        out_split = step_fsi!(split; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        out_strong = step_fsi_strong!(
            strong;
            t=t,
            dt=dt,
            fluid_scheme=:CN,
            ode_scheme=:symplectic_euler,
            maxiter=3,
            atol=1e-7,
            rtol=1e-5,
            relaxation=:aitken,
            omega_relax=0.8,
            allow_nonconverged=true,
        )

        t += dt
        vz_exact = Vinf + (V0[3] - Vinf) * exp(-t / tau)
        res_split = norm(out_split.sys.A * out_split.sys.x - out_split.sys.b)
        res_strong = norm(out_strong.sys.A * out_strong.sys.x - out_strong.sys.b)

        println(
            step, ", ", t,
            ", ", out_split.V[3],
            ", ", out_strong.V[3],
            ", ", vz_exact,
            ", ", res_split,
            ", ", res_strong,
            ", ", out_strong.iterations,
        )

        z = out_strong.X[3]
        dmin = min(z - grid.lc[3], grid.hc[3] - z)
        if dmin <= stop_margin
            println("Stopping before contact: min wall distance=$dmin, threshold=$stop_margin")
            break
        end
    end

    Uinf_ref = (2.0 / 9.0) * ((rho_s - rho_f) * g0 * R^2 / mu)
    println("Classical Stokes terminal speed magnitude (unbounded): ", Uinf_ref)
    println("Numerical split Vz_end: ", split.state.V[3])
    println("Numerical strong Vz_end: ", strong.state.V[3])
end

main()
