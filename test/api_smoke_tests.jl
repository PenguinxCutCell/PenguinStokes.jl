using Test
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, Neumann, Periodic
using PenguinStokes

full_body_smoke(args...) = -1.0

function noslip_bc_2d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
end

@testset "Public API constructor smoke (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (9, 9))
    bc = noslip_bc_2d()

    model_mono = StokesModelMono(
        grid,
        full_body_smoke,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
    )
    @test model_mono isa StokesModelMono
    @test last(model_mono.layout.pomega) > 0

    body_if(x, y) = y - 0.5
    model_two = StokesModelTwoPhase(
        grid,
        body_if,
        1.0,
        2.0;
        bc_u=(bc, bc),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=(0.0, 0.0),
    )
    @test model_two isa StokesModelTwoPhase
    @test last(model_two.layout.pomega2) > last(model_two.layout.pomega1)

    model_moving = MovingStokesModelMono(
        grid,
        (x, y, t) -> -1.0,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)),
    )
    @test model_moving isa MovingStokesModelMono
    @test last(model_moving.layout.pomega) > 0

    model_moving_two = MovingStokesModelTwoPhase(
        grid,
        (x, y, t) -> y - (0.5 + 0.05 * sin(t)),
        1.0,
        2.0;
        bc_u=(bc, bc),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
    )
    @test model_moving_two isa MovingStokesModelTwoPhase
    @test last(model_moving_two.layout.pomega2) > last(model_moving_two.layout.pomega1)

    shape = Circle(0.12)
    state = RigidBodyState2D(SVector(0.5, 0.5), SVector(0.0, 0.0); theta=0.0, omega=0.0)
    params = RigidBodyParams2D(1.0, 1.0, shape, SVector(0.0, 0.0); rho_fluid=0.0, buoyancy=false)
    fsi = StokesFSIProblem(model_moving, state, params)
    @test fsi isa StokesFSIProblem
    @test length(fsi.xprev) == last(model_moving.layout.pomega)
end

@testset "Cut BC rejects Neumann/Periodic on ugamma rows" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (7, 7))
    bc = noslip_bc_2d()

    model_neu = StokesModelMono(
        grid,
        full_body_smoke,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Neumann(0.0),
        force=(0.0, 0.0),
    )
    @test_throws ArgumentError solve_steady!(model_neu)

    model_per = StokesModelMono(
        grid,
        full_body_smoke,
        1.0,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Periodic(),
        force=(0.0, 0.0),
    )
    @test_throws ArgumentError solve_steady!(model_per)
end
