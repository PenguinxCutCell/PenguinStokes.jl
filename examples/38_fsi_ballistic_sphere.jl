using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid, meshsize
using PenguinBCs
using PenguinStokes

# Ballistic sphere: 3D FSI — sphere with initial transverse velocity under gravity.
#
# A sphere denser than the fluid is launched horizontally with velocity U0·ex
# while gravity acts in the -y direction.  In the Stokes limit (Re ≪ 1) the
# equations of motion decouple:
#
#   m·dVx/dt = -κ·Vx                →  Vx(t) = U0 · exp(-t/τ)
#   m·dVy/dt = Fg - κ·Vy            →  Vy(t) = V∞·(1 - exp(-t/τ))
#
# where:
#   κ   = 6π μ r   (Stokes drag coefficient, calibrated numerically)
#   τ   = m/κ      (relaxation time)
#   Fg  = (ρ_s - ρ_f)·V_sphere·g   (buoyancy-corrected gravity)
#   V∞  = Fg/κ     (terminal vertical velocity, negative = downward)
#
# We choose parameters so that Re_d = ρ_f·|V∞|·d/μ ≈ 0.1 (Stokes limit).
#
# Inspired by sandbox/cselcuk/balistic-sphere.c

# ── Parameters ───────────────────────────────────────────────────────────────
const RHO_F   = 1.0      # fluid density
const RHO_S   = 1.5      # solid density (50% denser than fluid)
const R       = 0.1      # sphere radius
const MU      = 1.0      # dynamic viscosity  (high → Stokes limit)
const G       = 1.0      # gravitational acceleration (in -y direction)
const U0      = 0.2      # initial transverse velocity (x-direction)
# Re_d ≈ ρ_f * V∞ * 2R / μ; V∞ ≈ 2(ρ_s-ρ_f)gR²/(9μ) = 2*0.5*1*0.01/(9) ≈ 1.1e-3  → Re ≪ 1

# ── Analytical solution ───────────────────────────────────────────────────────
# Stokes drag coefficient (numerical κ replaces this in the FSI loop).
kappa_stokes() = 6π * MU * R

function analytic_velocity(t, kappa, m, Fg, U0y=0.0)
    τ    = m / kappa
    V_inf = Fg / kappa   # terminal y-velocity (negative = downward)
    Vx   = U0 * exp(-t / τ)
    Vy   = V_inf + (U0y - V_inf) * exp(-t / τ)
    return (Vx=Vx, Vy=Vy, tau=τ, V_inf=V_inf)
end

# ── FSI setup ─────────────────────────────────────────────────────────────────
function box_noslip_3d()
    return BorderConditions(;
        left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
end

function calibrate_drag(grid, bc, shape, center; dt)
    Uprobe = 0.1
    state_probe = RigidBodyState3D(center, SVector(0.0, 0.0, Uprobe))
    statefun(_t) = state_probe

    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        MU,
        RHO_F;
        bc_u=(bc, bc, bc),
        force=(0.0, 0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple(statefun, Val(3)),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=0.0, dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(center))

    force_sign = q.force[3] <= 0 ? 1.0 : -1.0
    kappa = -(force_sign * q.force[3]) / Uprobe
    return (kappa=kappa, force_sign=force_sign)
end

# ── Main ──────────────────────────────────────────────────────────────────────
function main()
    # Domain: [0,L]³, sphere starts near (L/4, 3L/4, L/2).
    L  = 1.0
    n  = 17       # coarse but 3D runs fast; ≈ 3.4 pts/d
    dt = 0.02
    nsteps = 30

    center0 = SVector(L/4, 3L/4, L/2)
    shape   = Sphere(R)

    m    = RHO_S * (4π/3) * R^3
    Fg_y = -(RHO_S - RHO_F) * (4π/3) * R^3 * G   # buoyancy-corrected, negative = down

    # Moment of inertia (isotropic sphere): I = 2/5 m r²
    I_body = (2.0/5.0) * m * R^2

    println("Ballistic sphere in 3D Stokes flow")
    println("R=$R, ρ_s=$RHO_S, ρ_f=$RHO_F, μ=$MU, g=$G")
    println("m=$(round(m; sigdigits=4)), Fg_y=$(round(Fg_y; sigdigits=4))")
    println("Stokes κ_ref = $(round(kappa_stokes(); sigdigits=4))")
    println("n=$n, dt=$dt, nsteps=$nsteps, L=$L")
    println()

    grid = CartesianGrid((0.0, 0.0, 0.0), (L, L, L), (n, n, n))
    bc   = box_noslip_3d()

    # Calibrate drag coefficient numerically.
    cal = calibrate_drag(grid, bc, shape, center0; dt=dt)
    τ_num   = m / cal.kappa
    V_inf_y = Fg_y / cal.kappa
    Re_d    = RHO_F * abs(V_inf_y) * 2R / MU
    println("Calibrated κ = $(round(cal.kappa; sigdigits=4))  (Stokes ref = $(round(kappa_stokes(); sigdigits=4)))")
    println("τ = $(round(τ_num; sigdigits=4)),  V∞_y = $(round(V_inf_y; sigdigits=4)),  Re_d ≈ $(round(Re_d; sigdigits=3))")
    println()

    # Build FSI problem: gravity in -y, sphere launches with Vx=U0.
    params = RigidBodyParams3D(
        m,
        I_body,
        RHO_S,
        shape,
        SVector(0.0, Fg_y / m, 0.0);   # specific force = g_eff (already buoyancy-corrected)
        rho_fluid=RHO_F,
        buoyancy=false,                  # we supply Fg directly as external force
    )

    state0 = RigidBodyState3D(center0, SVector(U0, 0.0, 0.0))

    statefun(_t) = state0
    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        MU,
        RHO_F;
        bc_u=(bc, bc, bc),
        force=(0.0, 0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple(statefun, Val(3)),
    )

    fsi = StokesFSIProblem(
        model,
        state0,
        params;
        pressure_reconstruction=:linear,
        force_sign=cal.force_sign,
        torque_sign=1.0,
    )

    println("step   t        Vx_num    Vx_ex    Vy_num    Vy_ex    |err_Vx|  |err_Vy|   ||Ax-b||")
    println(repeat("-", 90))

    t = 0.0
    hmin = minimum(meshsize(grid))
    stop_margin = R + 2hmin

    for step in 1:nsteps
        out = step_fsi!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t += dt

        ex = analytic_velocity(t, cal.kappa, m, Fg_y)
        eVx = abs(out.V[1] - ex.Vx)
        eVy = abs(out.V[2] - ex.Vy)
        res = norm(out.sys.A * out.sys.x - out.sys.b)

        println(
            "$(lpad(step,4))  $(round(t; digits=3))  ",
            "$(round(out.V[1]; sigdigits=4))  $(round(ex.Vx; sigdigits=4))  ",
            "$(round(out.V[2]; sigdigits=4))  $(round(ex.Vy; sigdigits=4))  ",
            "$(round(eVx; sigdigits=3))  $(round(eVy; sigdigits=3))  ",
            "$(round(res; sigdigits=3))",
        )

        # Stop if sphere approaches a wall.
        X = out.X
        dmin = minimum([
            X[1] - grid.lc[1], grid.hc[1] - X[1],
            X[2] - grid.lc[2], grid.hc[2] - X[2],
            X[3] - grid.lc[3], grid.hc[3] - X[3],
        ])
        if dmin <= stop_margin
            println("Stopping: sphere within $(round(dmin; sigdigits=3)) of a wall (threshold $stop_margin)")
            break
        end
    end

    println()
    Vx_f = fsi.state.V[1]
    Vy_f = fsi.state.V[2]
    ex   = analytic_velocity(t, cal.kappa, m, Fg_y)
    println("Final Vx: num=$(round(Vx_f; sigdigits=4))  exact=$(round(ex.Vx; sigdigits=4))  err=$(round(abs(Vx_f-ex.Vx)/abs(ex.Vx)*100; sigdigits=3))%")
    println("Final Vy: num=$(round(Vy_f; sigdigits=4))  exact=$(round(ex.Vy; sigdigits=4))  err=$(round(abs(Vy_f-ex.Vy)/abs(ex.Vy)*100; sigdigits=3))%")
    println("Trajectory: X=$(round.(Tuple(fsi.state.X); sigdigits=4))")
end

main()
