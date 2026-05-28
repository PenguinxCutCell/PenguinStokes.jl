using Test
using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

free_surface_periodic_2d() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

@testset "Free-surface graph API smoke" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (9, 9))
    rep = GlobalHFRep(grid, 2, fill(0.5, 9))
    bc = free_surface_periodic_2d()
    model = MovingStokesModelMono(
        grid,
        rep.body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)),
    )
    prob = FreeSurfaceStokesProblem(model, rep)
    @test prob.rep === rep
    @test model.body === rep.body
    @test rep.body(0.25, 0.4, 0.0) < 0.0
    @test size(rep.phi) == grid.n
end

@testset "GHF static flat two-phase equilibrium" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    h0 = 0.5
    rep = GlobalHFRep(grid, (x, y) -> y - h0; axis=2, periodic_transverse=true)
    bc = BorderConditions(
        ; left=Periodic(), right=Periodic(),
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
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
        check_interface=false,
    )
    prob = FreeSurfaceStokesProblem(
        model,
        rep;
        options=FreeSurfaceStokesOptions(; max_iter=4, tol=1e-12, reltol=1e-12, scheme=:CN, drive_phase=1),
    )
    x0 = zeros(Float64, last(model.layout.pomega2))
    out = step_free_surface_stokes!(prob, x0; t=0.0, dt=0.01)

    @test out.converged
    @test maximum(abs.(rep.xf .- h0)) <= 1e-12
    @test maximum(abs, out.x) <= 1e-12
    @test out.history.gcl_norm[end] <= 1e-12
end

@testset "Moving GCL residual identity: mono" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = free_surface_periodic_2d()
    R = 0.18
    xc0 = 0.5
    yc = 0.5
    U = 0.2
    body(x, y, t) = R - sqrt((x - (xc0 + U * t))^2 + (y - yc)^2)
    model = MovingStokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet((x, y, t) -> U), Dirichlet(0.0)),
    )

    x0 = zeros(Float64, last(model.layout.pomega))
    Vn = stokes_pressure_volume(model, 0.0)
    sys = solve_unsteady_moving!(model, x0; t=0.0, dt=0.04, scheme=:CN)
    terms = stokes_gcl_terms_mono(model, sys.x, Vn)

    @test norm(terms.R_gcl .- (terms.R_kin .+ terms.R_div), Inf) < 1e-12
    @test norm((sys.A * sys.x - sys.b)[model.layout.pomega], Inf) < 1e-10
end

@testset "Moving GCL residual identity: two phase" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = free_surface_periodic_2d()
    R = 0.18
    xc0 = 0.5
    yc = 0.5
    U = 0.15
    body(x, y, t) = R - sqrt((x - (xc0 + U * t))^2 + (y - yc)^2)
    model = MovingStokesModelTwoPhase(
        grid,
        body,
        1.0,
        2.0;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc, bc),
        bc_p=bc,
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
    )

    x0 = zeros(Float64, last(model.layout.pomega2))
    V1n, V2n = stokes_pressure_volume(model, 0.0)
    sys = solve_unsteady_moving!(model, x0; t=0.0, dt=0.04, scheme=:CN)
    terms = stokes_gcl_terms_diph(model, sys.x, V1n, V2n)

    @test norm(terms.phase1.R_gcl .- (terms.phase1.R_kin .+ terms.phase1.R_div), Inf) < 1e-12
    @test norm(terms.phase2.R_gcl .- (terms.phase2.R_kin .+ terms.phase2.R_div), Inf) < 1e-12
    div_rows = vcat(collect(model.layout.pomega1), collect(model.layout.pomega2))
    @test norm((sys.A * sys.x - sys.b)[div_rows], Inf) < 1e-9
end

@testset "Planar material interface residual convergence" begin
    U = 0.1
    H0 = 0.48123
    ns = (17, 25, 33)
    hs = Float64[]
    errs = Float64[]

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        h = 1.0 / (n - 1)
        dt = 0.5 * h^2
        body(x, y, t) = y - (H0 + U * t)
        bc = free_surface_periodic_2d()
        model = MovingStokesModelMono(
            grid,
            body,
            1.0,
            1.0;
            bc_u=(bc, bc),
            force=(0.0, 0.0),
            bc_cut_u=(Dirichlet(0.0), Dirichlet((x, y, t) -> U)),
        )

        x0 = zeros(Float64, last(model.layout.pomega))
        Vn = stokes_pressure_volume(model, 0.0)
        sys = solve_unsteady_moving!(model, x0; t=0.0, dt=dt, scheme=:CN)
        terms = stokes_gcl_terms_mono(model, sys.x, Vn)

        push!(hs, h)
        push!(errs, norm(terms.R_kin, Inf))
        @test norm(mask_inactive_pressure_cells(model, terms.R_gcl, sys.A), Inf) < 1e-12
    end

    orders = [log(errs[k] / errs[k + 1]) / log(hs[k] / hs[k + 1]) for k in 1:(length(errs) - 1)]
    @test minimum(orders) > 1.4
end
