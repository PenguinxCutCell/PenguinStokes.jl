"""
    32_periodic_sphere_array.jl

Stokes flow past a periodic array of spheres (3D), following Zick and Homsy (1982).
The domain is a periodic unit cube with a single sphere at the center.

Reference data (adapted from Zick & Homsy, 1982):
  Volume fraction phi | Drag coefficient K
  0.027               | 2.008
  0.064               | 2.810
  0.125               | 4.292
  0.216               | 7.442
  0.343               | 15.4
  0.45                | 28.1
  0.5236              | 42.1

Zick & Homsy define K such that the force on each sphere is:
  F = 6*pi*mu*a*K*U
where a is sphere radius, mu is viscosity, and U is the average fluid velocity.

The Basilisk example computes:
  K = dp*L^2 / (6*pi*mu*a*U*(1 - phi))
with dp=1, L=1.
"""

using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes
using PenguinSolverCore: LinearSystem
using Printf

const zick_ref = [
    (0.027, 2.008),
    (0.064, 2.810),
    (0.125, 4.292),
    (0.216, 7.442),
    (0.343, 15.4),
    (0.45, 28.1),
    (0.5236, 42.1),
]

function periodic_3d_bc()
    return (
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic(),
                         backward=Periodic(), forward=Periodic()),
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic(),
                         backward=Periodic(), forward=Periodic()),
        BorderConditions(; left=Periodic(), right=Periodic(),
                         bottom=Periodic(), top=Periodic(),
                         backward=Periodic(), forward=Periodic()),
    )
end

function compute_average_velocity(model::StokesModelMono{3,T}, sys::LinearSystem{T}) where {T}
    u1 = sys.x[model.layout.uomega[1]]
    total_flux = zero(T)
    total_vol = zero(T)

    for i in 1:model.cap_u[1].ntotal
        V = model.cap_u[1].buf.V[i]
        if isfinite(V) && V > 0
            total_flux += V * u1[i]
            total_vol += V
        end
    end

    return total_vol > 0 ? total_flux / total_vol : one(T)
end

function solve_sphere_case(n::Int, phi::T, f_magnitude::T=1.0) where {T}
    L = one(T)
    grid = CartesianGrid((zero(T), zero(T), zero(T)), (L, L, L), (n, n, n))

    # Sphere radius from volume fraction: phi = 4/3*pi*a^3 / L^3
    radius = (3 * phi / (4 * pi))^(one(T) / 3)

    xc = L / 2
    yc = L / 2
    zc = L / 2

    # Level-set function: negative in fluid, positive in solid
    body(x, y, z) = radius - sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)

    force = (f_magnitude, zero(T), zero(T))

    bc_u = periodic_3d_bc()
    bc_p = BorderConditions(; left=Periodic(), right=Periodic(),
                            bottom=Periodic(), top=Periodic(),
                            backward=Periodic(), forward=Periodic())

    mu = one(T)
    rho = one(T)

    model = StokesModelMono(
        grid,
        body,
        mu,
        rho;
        bc_u=bc_u,
        bc_p=bc_p,
        bc_cut=Dirichlet(zero(T)),
        force=force,
        gauge=MeanPressureGauge(),
    )

    sys = solve_steady!(model)

    u_avg = compute_average_velocity(model, sys)

    # Basilisk/Zick-Homsy coefficient: K = dp*L^2 / (6*pi*mu*a*U*(1 - phi))
    dp = one(T)
    K = dp * L^2 / (6 * pi * mu * radius * u_avg * (1 - phi) + 1e-15)

    residual = norm(sys.A * sys.x - sys.b)

    return (K=K, u_avg=u_avg, residual=residual, radius=radius, model=model, sys=sys)
end

function run_convergence_study()
    println("\n" * "="^80)
    println("PERIODIC SPHERE ARRAY - STOKES FLOW")
    println("Zick & Homsy (1982) Validation Study")
    println("="^80)
    println()

    # Smaller meshes for 3D: keep runtime reasonable
    mesh_sizes = (7, 9, 11)

    println("Reference data (Zick & Homsy 1982, Table 2):")
    println("-" * "="^79)
    @printf "%10s %10s %12s %12s %12s\n" "phi" "K ref" "n=7" "n=9" "n=11"
    println("-" * "="^79)

    for (phi_ref, k_ref) in zick_ref
        @printf "%10.4f %10.3f" phi_ref k_ref
        for n in mesh_sizes
            try
                result = solve_sphere_case(n, phi_ref, 1.0)
                @printf " %12.3f" result.K
            catch e
                @printf " %12s" "ERROR"
            end
        end
        println()
    end

    println("-" * "="^79)
    println()

    # Detailed study for one volume fraction
    phi_detail = 0.125
    idx_ref = findfirst(p -> abs(p[1] - phi_detail) < 1e-8, zick_ref)
    k_ref_detail = zick_ref[idx_ref][2]

    println("Detailed convergence study for phi = 0.125:")
    println("-" * "="^79)
    println(@sprintf "%10s %10s %12s %12s %12s %12s" "n" "h" "K" "Ref" "RelErr" "Residual")
    println("-" * "="^79)

    for n in (7, 9, 11, 13)
        try
            result = solve_sphere_case(n, phi_detail, 1.0)
            h = 1.0 / (n - 1)
            error = abs(result.K - k_ref_detail) / k_ref_detail
            @printf "%10d %10.4f %12.3f %12.3f %12.2e %12.2e\n" n h result.K k_ref_detail error result.residual
        catch e
            println("n=$n: ERROR - $(string(e))")
        end
    end

    println("-" * "="^79)
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_study()
end
