using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Sphere moving infinitely slowly towards a plane wall in Stokes flow.
#
# A sphere of radius r = 0.5 (diameter d = 1) sits at a fixed gap δ above a
# no-slip plane wall.  The sphere surface moves at velocity -uref·ex (toward
# the wall).  Because Re ≪ 1 and Re ≪ δ/r, the flow is quasi-steady Stokes.
# We solve the steady problem at fixed gap and compare the x-force on the
# sphere to the Brenner (1961) and Cooley & O'Neill (1969) analytical results.
#
# Geometry: the wall is the LEFT face of the box domain (x = -L/2).
# The sphere is the only embedded boundary; the wall is a Dirichlet box face.
# All six box faces are no-slip (fluid at rest far from the sphere).
#
# References:
#   Brenner (1961), Chem. Eng. Sci. 16, 242–251.
#   Cooley & O'Neill (1969), Mathematika 16, 37–49.
# Inspired by: sandbox/ghigo/src/test-stokes/sphere-towards-wall.c

# ── Analytical corrections ───────────────────────────────────────────────────

# Tabulated from Brenner (1961), 400-term series.
const BRENNER_TABLE = [
    (0.4,   3.73562),
    (0.3,   4.61072),
    (0.2,   6.34089),
    (0.1,   11.4592),
    (0.05,  21.5858),
    (0.025, 41.7176),
    (0.01,  101.896),
]

function kappa_brenner(delta_over_r::Float64)
    for (g, k) in BRENNER_TABLE
        g ≈ delta_over_r && return k
    end
    α = acosh(1.0 + delta_over_r)
    s = 0.0
    sinhα  = sinh(α)
    sinh2α = sinh(2α)
    for n in 1:400
        num = 2sinh((2n + 1)*α) + (2n + 1)*sinh2α
        den = 4sinh((n + 0.5)*α)^2 - (2n + 1)^2 * sinhα^2
        s += n*(n + 1) / ((2n - 1)*(2n + 3)) * (num/den - 1)
    end
    return (4/3) * sinhα * s
end

# Cooley & O'Neill (1969) approximation, valid for δ ≪ r.
kappa_cooley(delta_over_r::Float64) = 1.0/delta_over_r - (1/5)*log(delta_over_r) + 0.97128

# ── Solver ───────────────────────────────────────────────────────────────────

function run_case(; delta_over_r::Float64, n::Int, L::Float64=8.0)
    r    = 0.5   # sphere radius (d = 1)
    uref = 1.0
    mu   = 1.0
    rho  = 1.0

    # Wall at left face x = -L/2; sphere centre at distance r + δ from the wall.
    wall_x    = -L/2
    sphere_cx = wall_x + r + delta_over_r * r

    # Level-set: positive inside sphere (solid), negative in fluid.
    sphere_ls(x, y, z) = r - sqrt((x - sphere_cx)^2 + y^2 + z^2)

    # All box faces no-slip: wall (left) + far-field (others) both u=0.
    bc_zero = BorderConditions(;
        left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )

    grid = CartesianGrid((-L/2, -L/2, -L/2), (L/2, L/2, L/2), (n, n, n))

    model = StokesModelMono(
        grid,
        sphere_ls,
        mu,
        rho;
        bc_u=(bc_zero, bc_zero, bc_zero),
        bc_cut=(Dirichlet(-uref), Dirichlet(0.0), Dirichlet(0.0)),
        force=(0.0, 0.0, 0.0),
        gauge=PinPressureGauge(),
    )

    sys = solve_steady!(model)

    q = integrated_embedded_force(
        model, sys;
        x0=(sphere_cx, 0.0, 0.0),
    )

    F_stokes  = 6π * mu * uref * r
    kappa_num = abs(q.force[1]) / F_stokes
    res       = norm(sys.A * sys.x - sys.b)

    h = L / (n - 1)
    return (
        kappa_num = kappa_num,
        kappa_B   = kappa_brenner(delta_over_r),
        kappa_C   = kappa_cooley(delta_over_r),
        gap_h     = (delta_over_r * r) / h,
        residual  = res,
    )
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    println("Sphere approaching a plane wall — steady 3D Stokes")
    println("F_x = κ · F_Stokes = κ · 6π μ uref r,   uref=1, μ=1, r=0.5")
    println("Wall = left box face (x = -L/2), sphere BC = Dirichlet(-uref, 0, 0)")
    println()

    # ── Part 1: mesh convergence at δ/r = 0.4 ───────────────────────────────
    println("Part 1 — mesh convergence for δ/r = 0.4, L = 8 (L/d = 8)")
    println("  n    h       gap/h   κ_num    κ_Brenner   rel_err   order    ||Ax-b||")
    println(repeat("-", 76))

    g = 0.4
    prev_rel = NaN
    prev_h   = NaN
    for n in (17, 25, 33, 49)
        L = 8.0
        h = L / (n - 1)
        r = run_case(; delta_over_r=g, n=n, L=L)
        rel = abs(r.kappa_num - r.kappa_B) / r.kappa_B
        order = isnan(prev_rel) ? NaN : log(prev_rel / rel) / log(prev_h / h)
        println(
            "  $(lpad(n, 2))   $(round(h; sigdigits=3))   $(round(r.gap_h; digits=2))   ",
            "$(round(r.kappa_num; sigdigits=4))   $(round(r.kappa_B; sigdigits=6))   ",
            "$(round(rel; sigdigits=3))   ",
            isnan(order) ? "  N/A" : "$(round(order; digits=2))",
            "   $(round(r.residual; sigdigits=3))",
        )
        prev_rel = rel
        prev_h   = h
    end

    # ── Part 2: gap-size sweep at fixed resolution ───────────────────────────
    println()
    println("Part 2 — gap-size sweep, n=33, L=8")
    println("  δ/r    gap/h   κ_num       κ_Brenner   κ_Cooley    rel_err_B   ||Ax-b||")
    println(repeat("-", 80))

    n = 33
    L = 8.0
    h = L / (n - 1)
    for delta_over_r in (0.4, 0.3, 0.2, 0.1)
        r = run_case(; delta_over_r=delta_over_r, n=n, L=L)
        rel_B = abs(r.kappa_num - r.kappa_B) / r.kappa_B
        println(
            "  $(delta_over_r)   $(round(r.gap_h; digits=2))   ",
            "$(lpad(round(r.kappa_num; sigdigits=4), 9))   ",
            "$(lpad(round(r.kappa_B; sigdigits=6), 9))   ",
            "$(lpad(round(r.kappa_C; sigdigits=6), 9))   ",
            "$(lpad(round(rel_B; sigdigits=3), 9))   $(round(r.residual; sigdigits=3))",
        )
    end

    println()
    println("Note: errors are dominated by gap resolution (gap/h < 1 for small δ/r at n=33).")
    println("Basilisk uses adaptive refinement (lmax=13, ~32 pts/d) and achieves <5% error.")
end

main()
