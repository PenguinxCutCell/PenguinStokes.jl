using Test
using LinearAlgebra
using CartesianGrids
using FrontTrackingMethods
using PenguinBCs
using PenguinStokes
using StaticArrays

@testset "Front-tracking ft_redistribute static circle" begin
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (9, 9))
    mesh = FrontTrackingMethods.make_circle_benchmark_curve(; center=SVector(0.0, 0.0), R=0.24, N=12)
    rep = FrontTrackingRep(grid, mesh; coupling=:ft_redistribute)

    bc = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    model = MovingStokesModelTwoPhase(
        grid,
        rep.body,
        1.0,
        1.0;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc, bc),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
        check_interface=false,
    )
    opts = CoupledFrontTrackingOptions(
        ; max_iter=2,
        tol=1e-12,
        reltol=1e-12,
        damping=0.7,
        scheme=:BE,
        step_clip=0.12,
        smooth=1,
        max_backtracks=0,
    )
    prob = CoupledMovingStokesProblem(model, rep; options=opts)

    x0 = zeros(Float64, last(model.layout.pomega2))
    c0 = FrontTrackingMethods.front_centroid(rep.state)
    a0 = FrontTrackingMethods.front_enclosed_measure(rep.state)
    out = step_coupled_fronttracking!(prob, x0; t=0.0, dt=0.005)
    c1 = FrontTrackingMethods.front_centroid(rep.state)
    a1 = FrontTrackingMethods.front_enclosed_measure(rep.state)

    @test out.converged
    @test norm(c1 - c0) <= 1e-12
    @test abs(a1 - a0) <= 1e-12
    @test maximum(abs, out.x) <= 1e-12
end
