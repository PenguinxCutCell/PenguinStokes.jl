using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

function periodic_x_wall_y_bc()
    return BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
end

function phase_volume(model)
    cap = something(model.cap_p1_end)
    return sum(v for v in cap.buf.V if isfinite(v))
end

function run_static_flat_ghf(; n=33, h0=0.5, dt=0.01, nsteps=3)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    phi0(x, y) = y - h0
    rep = GlobalHFRep(grid, phi0; axis=2, periodic_transverse=true)

    bc = periodic_x_wall_y_bc()
    model = MovingStokesModelTwoPhase(
        grid,
        rep.body,
        1.0,
        1.0;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc, bc),
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
        check_interface=false,
    )
    opts = FreeSurfaceStokesOptions(
        ; max_iter=4,
        tol=1e-12,
        reltol=1e-12,
        damping=0.8,
        scheme=:CN,
        drive_phase=1,
    )
    prob = FreeSurfaceStokesProblem(model, rep; options=opts)

    x = zeros(Float64, last(model.layout.pomega2))
    t = 0.0
    volume0 = NaN
    max_height_drift = 0.0
    max_volume_drift = 0.0
    max_velocity = 0.0
    max_gcl = 0.0

    println("GHF static flat two-phase equilibrium")
    println("No interface velocity is prescribed; the interface is held by zero solved velocity.")

    for step in 1:nsteps
        out = step_free_surface_stokes!(prob, x; t=t, dt=dt)
        out.converged || error("GHF equilibrium step did not converge at step $step")
        x .= out.x
        t += dt

        vol = phase_volume(model)
        if step == 1
            volume0 = vol
        end

        height_drift = maximum(abs.(rep.xf .- h0))
        volume_drift = abs(vol - volume0)
        velocity = maximum(abs, x)
        gcl = out.history.gcl_norm[end]

        max_height_drift = max(max_height_drift, height_drift)
        max_volume_drift = max(max_volume_drift, volume_drift)
        max_velocity = max(max_velocity, velocity)
        max_gcl = max(max_gcl, gcl)

        println(
            "step=$step  t=$t  iter=$(out.history.iter[end])  ",
            "height_drift=$height_drift  volume_drift=$volume_drift  ",
            "max|x|=$velocity  gcl=$gcl",
        )
    end

    return (; max_height_drift, max_volume_drift, max_velocity, max_gcl, rep, model)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_static_flat_ghf()
    result.max_height_drift <= 1e-12 || error("flat GHF interface drifted")
    result.max_volume_drift <= 1e-12 || error("phase volume drifted")
    result.max_velocity <= 1e-12 || error("static equilibrium generated velocity")
    result.max_gcl <= 1e-12 || error("GCL residual is not near zero")
end
