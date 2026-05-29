using LinearAlgebra
using CartesianGrids
using FrontTrackingMethods
using PenguinBCs
using PenguinStokes
using StaticArrays

function run_ft_redistribute_rising_bubble(;
    n=41,
    nmarkers=60,
    dt=0.1,
    nsteps=10,
    R=0.2,
    mu=0.1,
    buoyancy=1.0,
)
    grid = CartesianGrid((-1.0, -1.5), (1.0, 1.5), (n, Int(round(1.5*(n-1)))+1))
    mesh = FrontTrackingMethods.make_circle_benchmark_curve(
        ; center=SVector(0.0, -0.9),
        R=R,
        N=nmarkers,
    )
    rep = FrontTrackingRep(grid, mesh; coupling=:ft_redistribute)

    wall = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    # Buoyancy is encoded as an effective body force on the lighter phase only
    # (phase 1 = bubble interior). With equal densities, this represents
    # the residual after subtracting the uniform gravity-pressure balance.
    model = MovingStokesModelTwoPhase(
        grid,
        rep.body,
        mu,
        mu;
        rho1=1.0,
        rho2=1.0,
        bc_u=(wall, wall),
        force1=(0.0, buoyancy),
        force2=(0.0, 0.0),
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
        check_interface=false,
    )

    opts = CoupledFrontTrackingOptions(
        ; max_iter=6,
        tol=1e-10,
        reltol=1e-6,
        damping=1.0,
        scheme=:BE,
        drive_phase=1,
        step_clip=0.35,
        smooth=2,
        max_backtracks=3,
        velocity_predictor=true,
    )
    prob = CoupledMovingStokesProblem(model, rep; options=opts)

    x = zeros(Float64, last(model.layout.pomega2))
    t = 0.0
    c0 = FrontTrackingMethods.front_centroid(rep.state)
    area0 = FrontTrackingMethods.front_enclosed_measure(rep.state)

    println("Front-tracking ft_redistribute rising bubble")
    println("The interface velocity is obtained from the two-phase Stokes solve; no marker motion is prescribed.")

    last_out = nothing
    for step in 1:nsteps
        out = step_coupled_fronttracking!(prob, x; t=t, dt=dt)
        x .= out.x
        t += dt
        last_out = out

        c = FrontTrackingMethods.front_centroid(rep.state)
        area = FrontTrackingMethods.front_enclosed_measure(rep.state)
        gcl = isempty(out.history.gcl_norm) ? norm(out.residual, Inf) : out.history.gcl_norm[end]
        println(
            "step=$step  t=$t  centroid=($(c[1]), $(c[2]))  ",
            "dy=$(c[2] - c0[2])  area_drift=$(area - area0)  ",
            "gcl=$gcl  converged=$(out.converged)",
        )
    end

    c1 = FrontTrackingMethods.front_centroid(rep.state)
    area1 = FrontTrackingMethods.front_enclosed_measure(rep.state)
    return (; centroid0=c0, centroid=c1, dy=c1[2] - c0[2],
        area0, area=area1, area_drift=area1 - area0,
        x, rep, model, last_out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_ft_redistribute_rising_bubble()
    result.dy > 0 || error("bubble centroid did not rise")
end
