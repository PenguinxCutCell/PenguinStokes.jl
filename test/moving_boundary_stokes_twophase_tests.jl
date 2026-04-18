using Test
using LinearAlgebra
using SparseArrays
using CartesianGrids
using PenguinBCs
using PenguinSolverCore
using PenguinStokes

moving_periodic_2d() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

function moving2_divergence_inf(model::MovingStokesModelTwoPhase{2,T}, x::AbstractVector{T}) where {T}
    op_p1_end = something(model.op_p1_end)
    cap_p1_end = something(model.cap_p1_end)
    op_p2_end = something(model.op_p2_end)
    cap_p2_end = something(model.cap_p2_end)
    nt = model.layout.nt

    div1 = zeros(T, nt)
    div2 = zeros(T, nt)
    @inbounds for d in 1:2
        rows = ((d - 1) * nt + 1):(d * nt)

        G1 = op_p1_end.G[rows, :]
        H1 = op_p1_end.H[rows, :]
        uω1 = view(x, model.layout.uomega1[d])
        uγ1 = view(x, model.layout.ugamma1[d])
        div1 .+= -((transpose(G1) + transpose(H1)) * uω1) + (transpose(H1) * uγ1)

        G2 = op_p2_end.G[rows, :]
        H2 = op_p2_end.H[rows, :]
        uω2 = view(x, model.layout.uomega2[d])
        uγ2 = view(x, model.layout.ugamma2[d])
        div2 .+= -((transpose(G2) + transpose(H2)) * uω2) + (transpose(H2) * uγ2)
    end

    dmax = zero(T)
    @inbounds for i in 1:nt
        V1 = cap_p1_end.buf.V[i]
        if isfinite(V1) && V1 > zero(T)
            dmax = max(dmax, abs(div1[i]))
        end
        V2 = cap_p2_end.buf.V[i]
        if isfinite(V2) && V2 > zero(T)
            dmax = max(dmax, abs(div2[i]))
        end
    end
    return dmax
end

@testset "Moving two-phase: prescribed interface jump rows (2D)" begin
    R = 0.18
    xc0 = 0.5
    yc = 0.5
    U = 0.2
    Uj = 0.07
    body(x, y, t) = R - sqrt((x - (xc0 + U * t))^2 + (y - yc)^2)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (21, 21))
    bc = moving_periodic_2d()
    model = MovingStokesModelTwoPhase(
        grid,
        body,
        1.0,
        3.0;
        rho1=1.0,
        rho2=2.0,
        bc_u=(bc, bc),
        bc_p=bc,
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_jump=(Uj, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
    )

    x0 = zeros(Float64, last(model.layout.pomega2))
    sys = solve_unsteady_moving!(model, x0; t=0.0, dt=0.05, scheme=:CN)

    @test norm(sys.A * sys.x - sys.b) < 1e-8

    cap_u1_end = something(model.cap_u1_end)
    cap_u2_end = something(model.cap_u2_end)
    ug1x = sys.x[model.layout.ugamma1[1]]
    ug1y = sys.x[model.layout.ugamma1[2]]
    ug2x = sys.x[model.layout.ugamma2[1]]
    ug2y = sys.x[model.layout.ugamma2[2]]

    iface = Int[]
    @inbounds for i in 1:cap_u1_end[1].ntotal
        Γ1 = cap_u1_end[1].buf.Γ[i]
        Γ2 = cap_u2_end[1].buf.Γ[i]
        has1 = isfinite(Γ1) && Γ1 > 0.0
        has2 = isfinite(Γ2) && Γ2 > 0.0
        if has1 || has2
            push!(iface, i)
        end
    end

    active = PenguinStokes._stokes_row_activity(model, sys.A)
    iface_active = Int[]
    @inbounds for i in iface
        ax = active[model.layout.ugamma1[1][i]] && active[model.layout.ugamma2[1][i]]
        ay = active[model.layout.ugamma1[2][i]] && active[model.layout.ugamma2[2][i]]
        if ax && ay
            push!(iface_active, i)
        end
    end

    @test !isempty(iface)
    @test !isempty(iface_active)
    @test maximum(abs.((ug1x[iface_active] .- ug2x[iface_active]) .- Uj)) < 1e-11
    @test maximum(abs.(ug1y[iface_active] .- ug2y[iface_active])) < 1e-11
    div_rows = vcat(collect(model.layout.pomega1), collect(model.layout.pomega2))
    @test norm((sys.A * sys.x - sys.b)[div_rows], Inf) < 1e-10
end

@testset "Moving two-phase: no-interface rows masked (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = moving_periodic_2d()
    model = MovingStokesModelTwoPhase(
        grid,
        (x, y, t) -> -1.0,
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

    nsys = last(model.layout.pomega2)
    x0 = zeros(Float64, nsys)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_unsteady_moving!(sys, model, x0, 0.0, 0.1; scheme=:CN)

    @test norm(sys.b) < 1e-12
    @test norm(sys.A * x0 - sys.b) < 1e-12

    active = PenguinStokes._stokes_row_activity(model, sys.A)
    ug = vcat(
        collect(model.layout.ugamma1[1]),
        collect(model.layout.ugamma1[2]),
        collect(model.layout.ugamma2[1]),
        collect(model.layout.ugamma2[2]),
    )
    @test !any(active[ug])
    @test !any(active[model.layout.pomega2])

    sys2 = solve_unsteady_moving!(model, x0; t=0.0, dt=0.1, scheme=:CN)
    @test norm(sys2.A * sys2.x - sys2.b) < 1e-10
end
