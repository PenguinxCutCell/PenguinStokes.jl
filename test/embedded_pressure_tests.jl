using Test
using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, Periodic
using PenguinStokes

function _noslip_bc_2d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
end

@testset "Embedded boundary pressure helper consistency (2D)" begin
    grid = CartesianGrid((-2.0, -1.0), (6.0, 1.0), (65, 33))
    xc, yc, r = 0.0, 0.0, 0.2
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r

    bcx = BorderConditions(
        ; left=Dirichlet(1.0), right=Dirichlet(1.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy = _noslip_bc_2d()

    model = StokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bcx, bcy),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
    )

    sys = solve_steady!(model)
    pdata = embedded_boundary_pressure(model, sys; pressure_reconstruction=:linear)

    @test !isempty(pdata.interface_indices)
    @test all(i -> isfinite(pdata.pressure[i]), pdata.interface_indices)
    @test all(isfinite, pdata.force)
end

@testset "Embedded boundary pressure linear trace exactness (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (129, 129))
    xc, yc, r = 0.5, 0.5, 0.2
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r

    bcper = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Periodic(), top=Periodic(),
    )

    model = StokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bcper, bcper),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
    )

    nsys = last(model.layout.pomega)
    x = zeros(Float64, nsys)
    pω = view(x, model.layout.pomega)

    a, b, c = 0.7, -1.3, 2.1
    for i in 1:model.cap_p.ntotal
        xω = model.cap_p.C_ω[i]
        if isfinite(xω[1]) && isfinite(xω[2])
            pω[i] = a + b * xω[1] + c * xω[2]
        end
    end

    pdata = embedded_boundary_pressure(model, x; pressure_reconstruction=:linear)
    @test !isempty(pdata.interface_indices)

    pω_vec = Vector{Float64}(pω)
    grad_p = PenguinStokes._pressure_gradient_reconstruction(model, pω_vec, :linear)

    max_err_linear = 0.0
    for k in eachindex(pdata.interface_indices)
        i = pdata.interface_indices[k]
        xω = model.cap_p.C_ω[i]
        xγ = pdata.centers[k]
        pdisc = pω_vec[i]
        for d in 1:2
            pdisc += grad_p[d][i] * (xγ[d] - xω[d])
        end
        @test isapprox(pdata.pressure[i], pdisc; atol=1e-12, rtol=1e-12)

        pexact = a + b * xγ[1] + c * xγ[2]
        max_err_linear = max(max_err_linear, abs(pdata.pressure[i] - pexact))
    end
    @test max_err_linear < 7e-3
end
