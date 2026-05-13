using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid, meshsize
using PenguinBCs
using PenguinStokes

# Two spheres in a 3D Couette shear flow — multi-body FSI.
#
# Two identical spheres are placed at (x0, y0 ± Δy/2, L/2) in a Couette flow
# u_x = γ̇·(y - L/2), u_y = u_z = 0.  Both spheres are free to translate and
# rotate.  In Stokes flow the hydrodynamic interaction (lubrication + far-field)
# pushes the spheres apart in y when they are close.
#
# The Couette BC is: top wall (y=L) at u_x = +Uc, bottom wall (y=0) at u_x = -Uc
# giving γ̇ = 2Uc/L.  All other faces are no-slip with u_x = γ̇·(y-L/2).
#
# This test exercises the MultiBodyFSIProblem infrastructure: shared fluid solve
# with per-body force extraction via interface-centroid proximity.

# ── Parameters ───────────────────────────────────────────────────────────────
const MU      = 1.0    # dynamic viscosity (Stokes limit)
const RHO_F   = 1.0    # fluid density
const RHO_S   = 1.5    # sphere density
const R       = 0.06   # sphere radius (d = 0.12)
const UC      = 1.0    # wall velocity; γ̇ = 2Uc/L
# DY = half the centre-to-centre distance; must satisfy DY > R (non-overlapping).
# Initial gap δ = DY - R.  We choose DY = 1.5*R → δ/R = 0.5.
const DY      = 1.5 * R

# ── BCs: Couette shear on all box faces ──────────────────────────────────────
function couette_bcs_3d(L::Float64)
    gdot = 2UC / L
    ux(x, y, z) = gdot * (y - L/2)

    bc_x = BorderConditions(;
        left=Dirichlet(ux), right=Dirichlet(ux),
        bottom=Dirichlet(-UC), top=Dirichlet(UC),
        backward=Dirichlet(ux), forward=Dirichlet(ux),
    )
    bc_y = BorderConditions(;
        left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
    bc_z = BorderConditions(;
        left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
    return (bc_x, bc_y, bc_z)
end

# ── Calibrate drag / force_sign ──────────────────────────────────────────────
function calibrate_single_drag(grid, bc, shape, center; dt)
    Uprobe = 0.1
    probe  = RigidBodyState3D(center, SVector(0.0, 0.0, Uprobe))
    sf(_t) = probe

    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, sf),
        MU, RHO_F;
        bc_u=(bc, bc, bc),
        force=(0.0, 0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple(sf, Val(3)),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys   = solve_unsteady_moving!(model, xprev; t=0.0, dt=dt, scheme=:CN)
    sm    = endtime_static_model(model)
    q     = integrated_embedded_force(sm, sys; x0=Tuple(center))

    force_sign = q.force[3] <= 0 ? 1.0 : -1.0
    kappa      = -(force_sign * q.force[3]) / Uprobe
    return (kappa=kappa, force_sign=force_sign)
end

# ── Main ─────────────────────────────────────────────────────────────────────
function main()
    L  = 1.0
    # Resolution: to resolve the gap at closest approach (δ/R ~ 0.1), you need
    # h < δ = 0.1*R = 0.006, i.e. n > L/h + 1 ~ 167.  Run at n=25 for a quick
    # demo; increase n (e.g. 65 or 129) to see proper hydrodynamic repulsion.
    n  = 25          # grid points per side; ≈ 3 pts per sphere diameter (d=0.12)
    dt = 0.01
    nsteps = 40

    grid = CartesianGrid((0.0, 0.0, 0.0), (L, L, L), (n, n, n))
    h    = minimum(meshsize(grid))
    bc   = couette_bcs_3d(L)

    # Sphere centres: same x, symmetric y-offset, centred in z.
    X1 = SVector(L/2, L/2 + DY, L/2)
    X2 = SVector(L/2, L/2 - DY, L/2)
    shape = Sphere(R)

    m      = RHO_S * (4π/3) * R^3
    I_body = (2.0/5.0) * m * R^2
    gdot   = 2UC / L

    println("Two spheres in 3D Couette shear — multi-body FSI (Stokes limit)")
    println("R=$R, ρ_s=$RHO_S, μ=$MU, γ̇=$gdot, Δy=$(2DY)")
    println("n=$n, h=$(round(h; sigdigits=3)), dt=$dt, nsteps=$nsteps")
    println("Sphere 1: X=$(Tuple(X1))  Sphere 2: X=$(Tuple(X2))")
    println()

    # Calibrate with zero background flow to get force_sign and kappa.
    bc_zero = BorderConditions(;
        left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
    cal = calibrate_single_drag(grid, bc_zero, shape, X1; dt=dt)
    println("Single-sphere drag calibration: κ=$(round(cal.kappa; sigdigits=4)), force_sign=$(cal.force_sign)")
    println()

    # Build multi-body FSI problem.
    states = [
        RigidBodyState3D(X1, SVector(0.0, 0.0, 0.0)),
        RigidBodyState3D(X2, SVector(0.0, 0.0, 0.0)),
    ]
    params_each = RigidBodyParams3D(
        m, I_body, RHO_S, shape,
        SVector(0.0, 0.0, 0.0);   # no gravity (pure hydrodynamic interaction)
        rho_fluid=RHO_F,
        buoyancy=false,
    )
    params_vec = [params_each, params_each]

    # Build the shared model with shear-flow BCs.
    # Initial statefuns needed to construct the model.
    sf1(_t) = states[1]
    sf2(_t) = states[2]
    shapes_vec = [shape, shape]
    statefuns0 = [sf1, sf2]

    model = MovingStokesModelMono(
        grid,
        multi_body_levelset(shapes_vec, statefuns0),
        MU, RHO_F;
        bc_u=bc,
        force=(0.0, 0.0, 0.0),
        bc_cut_u=multi_body_cut_bc_tuple(shapes_vec, statefuns0, Val(3)),
    )

    fsi = MultiBodyFSIProblem(
        model, states, params_vec, shapes_vec;
        force_signs=fill(cal.force_sign, 2),
        torque_signs=fill(1.0, 2),
    )

    println("step   t      X1_y       X2_y       gap/R    Vy1        Vy2        ||Ax-b||")
    println(repeat("-", 82))

    t = 0.0
    stop_margin = R + 2h

    for step in 1:nsteps
        out = step_multi_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t = out.t

        X1n = fsi.states[1].X
        X2n = fsi.states[2].X
        gap = (X1n[2] - X2n[2] - 2R) / R   # normalised gap δ/R
        Vy1 = fsi.states[1].V[2]
        Vy2 = fsi.states[2].V[2]
        res = norm(out.sys.A * out.sys.x - out.sys.b)

        println(
            "$(lpad(step,4))  $(round(t; digits=3))  ",
            "$(round(X1n[2]; sigdigits=5))  $(round(X2n[2]; sigdigits=5))  ",
            "$(round(gap; sigdigits=3))  ",
            "$(round(Vy1; sigdigits=4))  $(round(Vy2; sigdigits=4))  ",
            "$(round(res; sigdigits=3))",
        )

        # Stop if either sphere gets too close to a wall.
        for (k, s) in enumerate(fsi.states)
            X = s.X
            dwall = minimum([
                X[1] - grid.lc[1], grid.hc[1] - X[1],
                X[2] - grid.lc[2], grid.hc[2] - X[2],
                X[3] - grid.lc[3], grid.hc[3] - X[3],
            ])
            if dwall <= stop_margin
                println("Stopping: sphere $k within $(round(dwall; sigdigits=3)) of a wall")
                @goto done
            end
        end

        # Stop if spheres overlap.
        dist = norm(fsi.states[1].X - fsi.states[2].X)
        if dist < 2R - h
            println("Stopping: spheres too close (dist=$(round(dist; sigdigits=3)), 2R=$(2R))")
            @goto done
        end
    end

    @label done
    println()
    X1f = fsi.states[1].X
    X2f = fsi.states[2].X
    gap_f = (X1f[2] - X2f[2] - 2R) / R
    println("Final positions:")
    println("  Sphere 1: $(round.(Tuple(X1f); sigdigits=4))")
    println("  Sphere 2: $(round.(Tuple(X2f); sigdigits=4))")
    println("  Normalised gap δ/R = $(round(gap_f; sigdigits=4))")
    println("  ΔVy = $(round(fsi.states[1].V[2] - fsi.states[2].V[2]; sigdigits=4))")
end

main()
