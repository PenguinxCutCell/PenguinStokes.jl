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

moving_full_body(args...) = -1.0

function moving_mean_ux(model::MovingStokesModelMono{2,T}, x::AbstractVector{T}) where {T}
    cap_u_end = something(model.cap_u_end)
    cap = cap_u_end[1]
    ux = x[model.layout.uomega[1]]
    acc = zero(T)
    vol = zero(T)
    @inbounds for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if isfinite(V) && V > zero(T)
            acc += V * ux[i]
            vol += V
        end
    end
    return acc / vol
end

function moving_divergence_inf(model::MovingStokesModelMono{2,T}, x::AbstractVector{T}) where {T}
    op_p_end = something(model.op_p_end)
    cap_p_end = something(model.cap_p_end)
    nt = model.layout.nt
    div = zeros(T, nt)
    @inbounds for d in 1:2
        rows = ((d - 1) * nt + 1):(d * nt)
        Gd = op_p_end.G[rows, :]
        Hd = op_p_end.H[rows, :]
        uω = view(x, model.layout.uomega[d])
        uγ = view(x, model.layout.ugamma[d])
        div .+= -((transpose(Gd) + transpose(Hd)) * uω) + (transpose(Hd) * uγ)
    end
    dmax = zero(T)
    @inbounds for i in 1:nt
        V = cap_p_end.buf.V[i]
        if isfinite(V) && V > zero(T)
            dmax = max(dmax, abs(div[i]))
        end
    end
    return dmax
end

function moving_solve_mean(tf, dt, scheme)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (9, 9))
    bc = moving_periodic_2d()
    fx(x, y, t) = cos(t)
    model = MovingStokesModelMono(
        grid,
        moving_full_body,
        0.8,
        1.0;
        bc_u=(bc, bc),
        force=(fx, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)),
    )
    nsys = last(model.layout.pomega)
    x = zeros(Float64, nsys)
    t = 0.0
    while t < tf - 1e-12
        sys = solve_unsteady_moving!(model, x; t=t, dt=dt, scheme=scheme)
        x .= sys.x
        t += dt
    end
    return moving_mean_ux(model, x)
end

@testset "Moving one-phase: prescribed interface velocity rows (2D)" begin
    R = 0.18
    xc0 = 0.5
    yc = 0.5
    U = 0.25
    body(x, y, t) = R - sqrt((x - (xc0 + U * t))^2 + (y - yc)^2)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (21, 21))
    bc = moving_periodic_2d()
    model = MovingStokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet((x, y, t) -> U), Dirichlet(0.0)),
    )

    nsys = last(model.layout.pomega)
    x0 = zeros(Float64, nsys)
    sys = solve_unsteady_moving!(model, x0; t=0.0, dt=0.05, scheme=:CN)

    @test norm(sys.A * sys.x - sys.b) < 1e-9

    cap_u_end = something(model.cap_u_end)
    ugx = sys.x[model.layout.ugamma[1]]
    ugy = sys.x[model.layout.ugamma[2]]

    iface = Int[]
    @inbounds for i in 1:cap_u_end[1].ntotal
        Γ = cap_u_end[1].buf.Γ[i]
        if isfinite(Γ) && Γ > 0.0
            push!(iface, i)
        end
    end
    @test !isempty(iface)
    @test maximum(abs.(ugx[iface] .- U)) < 1e-12
    @test maximum(abs.(ugy[iface])) < 1e-12
    @test moving_divergence_inf(model, sys.x) < 5e-2
end

@testset "Moving one-phase: no-interface rows masked (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    bc = moving_periodic_2d()
    model = MovingStokesModelMono(
        grid,
        (x, y, t) -> -1.0,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)),
    )

    nsys = last(model.layout.pomega)
    x0 = zeros(Float64, nsys)

    sys_be = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    sys_cn = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_unsteady_moving!(sys_be, model, x0, 0.0, 0.1; scheme=:BE)
    assemble_unsteady_moving!(sys_cn, model, x0, 0.0, 0.1; scheme=:CN)

    @test norm(sys_be.b) < 1e-12
    @test norm(sys_cn.b) < 1e-12
    @test norm(sys_be.A * x0 - sys_be.b) < 1e-12
    @test norm(sys_cn.A * x0 - sys_cn.b) < 1e-12

    active = PenguinStokes._stokes_row_activity(model, sys_be.A)
    ug = vcat(collect(model.layout.ugamma[1]), collect(model.layout.ugamma[2]))
    @test !any(active[ug])
end

@testset "Moving one-phase: temporal order sanity on uniform forcing" begin
    tf = 0.4
    exact = sin(tf)

    u_be_1 = moving_solve_mean(tf, 0.1, :BE)
    u_be_2 = moving_solve_mean(tf, 0.05, :BE)
    e_be_1 = abs(u_be_1 - exact)
    e_be_2 = abs(u_be_2 - exact)
    @test e_be_2 < e_be_1
    @test e_be_1 / e_be_2 > 1.8

    u_cn_1 = moving_solve_mean(tf, 0.1, :CN)
    u_cn_2 = moving_solve_mean(tf, 0.05, :CN)
    e_cn_1 = abs(u_cn_1 - exact)
    e_cn_2 = abs(u_cn_2 - exact)
    @test e_cn_2 < e_cn_1
    @test e_cn_1 / e_cn_2 > 3.2
end
