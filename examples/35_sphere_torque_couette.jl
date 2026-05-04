using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Torque on a sphere held fixed in a Couette shear flow.
#
# The background flow is u = (γ̇·y, 0, 0) with γ̇ = 1.
# The sphere is stationary (no-slip), so the fluid exerts a torque about the
# z-axis.  The Stokes analytical formula (unbounded domain) gives:
#
#   T_z = 8π μ R³ (γ̇/2)
#
# Reference: G.G. Stokes, 1851.
# Inspired by the Basilisk test: sandbox/ghigo/src/test-stokes/torque.c

const GDOT = 1.0   # shear rate
const MU   = 1.0   # dynamic viscosity
const RHO  = 1.0   # density (Stokes — inertia-free)
const R    = 0.5   # sphere radius (d = 1)

stokes_torque_z() = 8π * MU * R^3 * (GDOT / 2)

# Level-set: positive inside sphere, negative in fluid.
sphere(x, y, z) = R - sqrt(x^2 + y^2 + z^2)

# Couette boundary conditions on a cube [-L/2, L/2]³.
# x and z are periodic; y walls prescribe the shear: u_x = ±U_wall = ±γ̇·L/2.
function couette_bcs(L::Float64)
    U_wall = GDOT * L / 2

    bc_x = BorderConditions(;
        left=Periodic(), right=Periodic(),
        bottom=Dirichlet(-U_wall), top=Dirichlet(U_wall),
        backward=Periodic(), forward=Periodic(),
    )
    bc_y = BorderConditions(;
        left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Periodic(), forward=Periodic(),
    )
    bc_z = BorderConditions(;
        left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Periodic(), forward=Periodic(),
    )
    return (bc_x, bc_y, bc_z)
end

function run_torque_case(; n::Int, L::Float64=8.0)
    grid = CartesianGrid((-L/2, -L/2, -L/2), (L/2, L/2, L/2), (n, n, n))

    model = StokesModelMono(
        grid,
        sphere,
        MU,
        RHO;
        bc_u=couette_bcs(L),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0, 0.0),
        gauge=MeanPressureGauge(),
    )

    # Warm-start: initialise ux = γ̇·y using pressure-grid cell centres as proxy.
    sys = solve_steady!(model)

    q = integrated_embedded_force(
        model, sys;
        pressure_reconstruction=:linear,
        x0=(0.0, 0.0, 0.0),
    )

    T_ref = stokes_torque_z()
    T_z   = q.torque[3]
    rel   = abs(abs(T_z) - T_ref) / T_ref
    res   = norm(sys.A * sys.x - sys.b)

    return (T_z=T_z, T_ref=T_ref, rel_err=rel, residual=res)
end

function main()
    println("Torque on a sphere in Couette shear flow (3D steady Stokes)")
    println("γ̇=$(GDOT), μ=$(MU), R=$(R), d=$(2R)")
    println("Stokes reference: T_z = 8π μ R³ (γ̇/2) = $(round(stokes_torque_z(); sigdigits=6))")
    println()

    L = 8.0  # domain side; L/d = 8 (moderate confinement)
    pts_per_d = n -> round(Int, 2R / (L / (n - 1)))

    println("  n    pts/d   T_z_num        T_ref          rel_err      order    ||Ax-b||")
    println(repeat("-", 80))

    prev_rel = NaN
    prev_h   = NaN
    for n in (17, 25, 33)
        h = L / (n - 1)
        r = run_torque_case(; n=n, L=L)
        order = isnan(prev_rel) ? NaN : log(prev_rel / r.rel_err) / log(prev_h / h)
        pd = pts_per_d(n)
        println(
            "  $(lpad(n,3))    $(lpad(pd,3))    $(lpad(round(r.T_z; sigdigits=5), 13))  ",
            "$(lpad(round(r.T_ref; sigdigits=5), 13))  $(lpad(round(r.rel_err; sigdigits=3), 10))  ",
            isnan(order) ? "   N/A" : "$(lpad(round(order; digits=2), 6))",
            "   $(round(r.residual; sigdigits=3))",
        )
        prev_rel = r.rel_err
        prev_h   = h
    end

    println()
    println("Note: confinement (L/d=$(round(Int, L/(2R)))) causes O(R/L) correction vs unbounded Stokes.")
    println("Faxén correction for sphere in linear shear: T_exact = T_Stokes (no correction to torque).")
end

main()
