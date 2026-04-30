"""
    33_couette_rotating_cylinders.jl

Couette flow between two concentric rotating cylinders (steady Stokes).

The fluid occupies the annulus r1 < r < r2 with r1=0.25 and r2=0.5.
The outer cylinder is fixed and the inner cylinder rotates with unit
angular velocity. The analytic tangential velocity is

    u_theta(r) = r * ((r2 / r)^2 - 1) / ((r2 / r1)^2 - 1)

We solve the steady Stokes system on a sequence of grids and report
L2 errors for u_x and u_y against the analytic profile.
"""

using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes
using PenguinSolverCore: LinearSystem
using Printf

function analytic_utheta(r, r1, r2)
    return r * ((r2 / r)^2 - 1) / ((r2 / r1)^2 - 1)
end

function couette_body(x, y, r1, r2)
    r = sqrt(x^2 + y^2)
    # Negative in fluid (annulus), positive in solid (inner/outer)
    return max(r1 - r, r - r2)
end

function couette_bc_cut(r1, r2)
    r_mid2 = ((r1 + r2) / 2)^2
    u_x(x, y) = (x^2 + y^2) < r_mid2 ? -y : 0.0
    u_y(x, y) = (x^2 + y^2) < r_mid2 ?  x : 0.0
    return (Dirichlet(u_x), Dirichlet(u_y))
end

function velocity_error(model::StokesModelMono{2,T}, sys::LinearSystem{T}, r1::T, r2::T) where {T}
    u1 = sys.x[model.layout.uomega[1]]
    u2 = sys.x[model.layout.uomega[2]]

    cap_u1 = model.cap_u[1]
    cap_u2 = model.cap_u[2]
    nnodes = cap_u1.nnodes
    li = LinearIndices(nnodes)

    e = zero(T)
    w = zero(T)

    # Evaluate u_theta at u-centers using linear interpolation of v (second-order).
    @inbounds for I in CartesianIndices(ntuple(d -> 1:(nnodes[d] - 1), 2))
        i = li[I]
        V = cap_u1.buf.V[i]
        if !isfinite(V) || V <= zero(T)
            continue
        end

        x = cap_u1.C_ω[i][1]
        y = cap_u1.C_ω[i][2]
        r = sqrt(x^2 + y^2)
        if r <= r1 || r >= r2
            continue
        end

        Ix = CartesianIndex(I[1] + 1, I[2])
        if Ix[1] > nnodes[1]
            continue
        end

        i_x = li[Ix]
        v_interp = 0.5 * (u2[i] + u2[i_x])

        theta = atan(y, x)
        utheta = analytic_utheta(r, r1, r2)
        utheta_num = -sin(theta) * u1[i] + cos(theta) * v_interp
        e += V * (utheta_num - utheta)^2
        w += V
    end

    # Evaluate u_theta at v-centers using linear interpolation of u (second-order).
    @inbounds for I in CartesianIndices(ntuple(d -> 1:(nnodes[d] - 1), 2))
        i = li[I]
        V = cap_u2.buf.V[i]
        if !isfinite(V) || V <= zero(T)
            continue
        end

        x = cap_u2.C_ω[i][1]
        y = cap_u2.C_ω[i][2]
        r = sqrt(x^2 + y^2)
        if r <= r1 || r >= r2
            continue
        end

        Iy = CartesianIndex(I[1], I[2] + 1)
        if Iy[2] > nnodes[2]
            continue
        end

        i_y = li[Iy]
        u_interp = 0.5 * (u1[i] + u1[i_y])

        theta = atan(y, x)
        utheta = analytic_utheta(r, r1, r2)
        utheta_num = -sin(theta) * u_interp + cos(theta) * u2[i]
        e += V * (utheta_num - utheta)^2
        w += V
    end

    return (utheta_L2=sqrt(e / w),)
end

function solve_couette_case(n::Int)
    r1 = 0.25
    r2 = 0.5

    grid = CartesianGrid((-0.5, -0.5), (0.5, 0.5), (n, n))
    body(x, y) = couette_body(x, y, r1, r2)

    bc_box = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    model = StokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bc_box, bc_box),
        bc_cut=couette_bc_cut(r1, r2),
        force=(0.0, 0.0),
        gauge=PinPressureGauge(),
    )

    sys = solve_steady!(model)
    err = velocity_error(model, sys, r1, r2)
    res = norm(sys.A * sys.x - sys.b)
    return (utheta_L2=err.utheta_L2, residual=res)
end

function run_convergence_study()
    println("\n" * "="^80)
    println("COUETTE FLOW BETWEEN ROTATING CYLINDERS")
    println("Steady Stokes, embedded boundaries")
    println("="^80)
    println()

    println(@sprintf "%8s %10s %12s %12s" "n" "h" "utheta_L2" "residual")
    println(repeat("-", 72))

    for n in (16, 32, 64, 128)
        result = solve_couette_case(n)
        h = 1.0 / (n - 1)
        @printf "%8d %10.4f %12.3e %12.3e\n" n h result.utheta_L2 result.residual
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_study()
end
