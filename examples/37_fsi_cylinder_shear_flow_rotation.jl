using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

# Cylinder freely rotating in a 2D shear flow (FSI).
#
# A cylinder of diameter d = 0.4 and density rho_s = 2*rho_f is placed at the
# centre of a unit square domain.  The background flow is a linear Couette
# shear profile:
#
#   u_x = -Uc + 2*y*Uc/L,   u_y = 0
#
# with shear rate γ̇ = 2*Uc/L.  The cylinder is free to rotate under the
# hydrodynamic torque but does not translate (TRANSLATION = 0 in Basilisk).
#
# At low Re the steady-state angular velocity of a free cylinder in a shear
# flow approaches ω_∞ = -γ̇/2 (Jeffery, 1922 for spheroids; for a circle this
# is exact for Stokes flow).  We track the transient spin-up and compare the
# final ω to this reference.
#
# Inspired by sandbox/cselcuk/shear2D_rot.c

# ── Physical parameters ───────────────────────────────────────────────────────
const UC      = 1.0    # characteristic velocity
const L       = 1.0    # domain side
const RHO_F   = 1.0    # fluid density
const RHO_S   = 2.0    # solid density (twice the fluid)
const DIAM    = 0.4    # cylinder diameter
# Stokes limit: Re → 0.  We use a large viscosity so that inertia is negligible.
const MU      = 1.0    # dynamic viscosity (high → low Re, Stokes regime)

# Shear rate and Jeffery/Stokes reference angular velocity.
const GDOT    = 2UC / L
const OMEGA_REF = -GDOT / 2   # ω_∞ for a circular cylinder in pure shear (Stokes)

# ── Shear-flow boundary conditions ───────────────────────────────────────────
# u_x = -Uc + 2y*Uc/L on left, right, top, bottom walls.
# u_y = 0 everywhere.
function shear_bcs()
    ux_wall(x, y) = -UC + 2y * UC / L

    bc_x = BorderConditions(;
        left=Dirichlet(ux_wall), right=Dirichlet(ux_wall),
        bottom=Dirichlet(ux_wall), top=Dirichlet(ux_wall),
    )
    bc_y = BorderConditions(;
        left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    return (bc_x, bc_y)
end

# ── Calibrate rotational drag coefficient ─────────────────────────────────────
# Spin the cylinder at ω = 1 in still fluid, measure the torque → kappa.
function calibrate_kappa(shape::Circle{Float64}, X0::SVector{2,Float64}; n::Int, dt::Float64)
    grid = CartesianGrid((0.0, 0.0), (L, L), (n, n))
    bc_zero = BorderConditions(;
        left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    probe = RigidBodyState2D(X0, SVector(0.0, 0.0); theta=0.0, omega=1.0)
    statefun(_t) = probe

    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        MU,
        RHO_F;
        bc_u=(bc_zero, bc_zero),
        force=(0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple_2d(statefun),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=0.0, dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; x0=Tuple(X0))

    torque_sign = q.torque <= 0 ? 1.0 : -1.0
    kappa = -(torque_sign * q.torque)   # kappa > 0: drag coefficient
    return (kappa=kappa, torque_sign=torque_sign)
end

# ── Main simulation ───────────────────────────────────────────────────────────
function main()
    n     = 65    # ~16 pts/d for d=0.4, L=1
    dt    = DIAM / UC / 200   # Tc/200 as in Basilisk
    tmax  = 2.5
    nsteps = ceil(Int, tmax / dt)
    dt    = tmax / nsteps

    X0    = SVector(L/2, L/2)
    shape = Circle(DIAM / 2)

    Re_D = RHO_F * UC * DIAM / MU
    println("Cylinder freely rotating in 2D shear flow (FSI, Stokes limit)")
    println("d=$DIAM, Re_d=$(round(Re_D; sigdigits=3)), ρ_s=$RHO_S, μ=$MU, γ̇=$GDOT")
    println("Reference ω_∞ = -γ̇/2 = $OMEGA_REF  (exact for circular cylinder in Stokes flow)")
    println("n=$n, dt=$(round(dt; sigdigits=3)), nsteps=$nsteps")
    println()

    # Calibrate rotational drag in zero-background-flow environment.
    cal = calibrate_kappa(shape, X0; n=n, dt=dt)
    println("Rotational drag calibration: κ=$(round(cal.kappa; sigdigits=5)), torque_sign=$(cal.torque_sign)")
    println()

    # Build the FSI problem with the shear-flow BCs.
    grid = CartesianGrid((0.0, 0.0), (L, L), (n, n))
    bc = shear_bcs()

    state0 = RigidBodyState2D(X0, SVector(0.0, 0.0); theta=0.0, omega=0.0)
    statefun(_t) = state0

    model = MovingStokesModelMono(
        grid,
        rigid_body_levelset(shape, statefun),
        MU,
        RHO_F;
        bc_u=bc,
        force=(0.0, 0.0),
        bc_cut_u=rigid_cut_bc_tuple_2d(statefun),
    )

    # Moment of inertia of a solid disk: I = rho_s * pi * r^4 / 2
    r_cyl = DIAM / 2
    I_body = RHO_S * π * r_cyl^4 / 2
    m_body = RHO_S * π * r_cyl^2
    params = RigidBodyParams2D(
        m_body,
        I_body,
        RHO_S,
        shape,
        SVector(0.0, 0.0);
        rho_fluid=RHO_F,
        buoyancy=false,
    )

    fsi = StokesFSIProblem2D(
        model,
        state0,
        params;
        force_sign=1.0,
        torque_sign=cal.torque_sign,
    )

    println("step      t        ω_num      ω_ref     torque     ||Ax-b||")
    println(repeat("-", 68))

    t = 0.0
    report_every = max(1, nsteps ÷ 20)
    for step in 1:nsteps
        out = step_fsi_rotation!(fsi; t=t, dt=dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)
        t = out.t
        res = norm(out.sys.A * out.sys.x - out.sys.b)

        if step % report_every == 0 || step == nsteps
            sm = endtime_static_model(fsi.model)
            q  = integrated_embedded_force(sm, out.sys; x0=Tuple(X0))
            println(
                "$(lpad(step, 5))  $(round(t; digits=3))  ",
                "$(round(out.omega; sigdigits=5))  ",
                "$(round(OMEGA_REF; sigdigits=5))  ",
                "$(round(q.torque; sigdigits=4))  ",
                "$(round(res; sigdigits=3))",
            )
        end
    end

    println()
    omega_final = fsi.state.omega
    rel_err = abs(omega_final - OMEGA_REF) / abs(OMEGA_REF)
    println("Final ω = $(round(omega_final; sigdigits=5))  (reference = $OMEGA_REF)")
    println("Relative error vs Stokes reference: $(round(rel_err * 100; sigdigits=3))%")
    println()
    println("Note: Re_d=$(RE_D) → mild inertia effects; Stokes formula is approximate.")
end

main()
