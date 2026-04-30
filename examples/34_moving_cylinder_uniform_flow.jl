using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Moving cylinder in a uniform Stokes flow.
#
# Both the fluid and the cylinder translate at constant speed U.
# The exact solution is u = (U, 0), p = 0 everywhere — the moving embedded
# boundary should introduce zero disturbance, so the velocity error must stay
# at machine-precision level throughout the simulation.
#
# Inspired by the Basilisk test: sandbox/ghigo/src/test-stokes/cylinder-steady.c
#
# Setup: fully-periodic square domain driven by a body force f = ρ*dU/dt = 0
# (U is constant).  The cylinder level-set moves at speed U in x; the embedded
# BC is Dirichlet (U, 0).  Because U is constant and the domain is periodic,
# the manufactured solution u = (U, 0), p = 0 satisfies the PDE exactly with
# no body force and no boundary layer — any residual error is purely numerical.

const UREF   = 0.912          # reference velocity
const RADIUS = 0.753 / 2      # cylinder radius  (d = 0.753)
const MU     = 0.684          # dynamic viscosity
const RHO    = 1.0            # density

periodic_bc() = BorderConditions(;
    left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

# Cylinder translates at constant speed UREF in x (periodic wrap handled by
# the level-set value being far from the interface away from the cylinder).
cylinder(x, y, t) = RADIUS - sqrt((x - mod(UREF * t + 0.5, 1.0))^2 + (y - 0.5)^2)

function max_velocity_error(model::MovingStokesModelMono{2,T}, x::AbstractVector{T}) where {T}
    cap = something(model.cap_u_end)[1]
    ux = x[model.layout.uomega[1]]
    uy = x[model.layout.uomega[2]]
    emax = zero(T)
    @inbounds for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if isfinite(V) && V > zero(T)
            emax = max(emax, abs(ux[i] - UREF), abs(uy[i]))
        end
    end
    return emax
end

function main()
    # Unit square, periodic in both directions.
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (65, 65))
    bc = periodic_bc()

    model = MovingStokesModelMono(
        grid,
        cylinder,
        MU,
        RHO;
        bc_u=(bc, bc),
        force=(0.0, 0.0),   # dU/dt = 0, no body force needed
        bc_cut_u=(Dirichlet(UREF), Dirichlet(0.0)),
    )

    nsys = last(model.layout.pomega)
    xprev = zeros(Float64, nsys)
    # Initialise to the exact solution.
    xprev[model.layout.uomega[1]] .= UREF

    tref  = (2 * RADIUS) / UREF   # d / U
    dt    = 0.01 * tref
    t_end = 2.0 * tref
    nsteps = ceil(Int, t_end / dt)
    dt = t_end / nsteps

    println("Moving cylinder in uniform Stokes flow (periodic domain)")
    println("Exact solution: u=($(UREF), 0), p=0 everywhere")
    println("grid=$(grid.n), R=$(RADIUS), μ=$(MU), U=$(UREF)")
    println("tref=$(round(tref; sigdigits=4)), dt=$(round(dt; sigdigits=4)), nsteps=$nsteps")
    println()

    t = 0.0
    emax_all = 0.0
    for step in 1:nsteps
        sys = solve_unsteady_moving!(model, xprev; t=t, dt=dt, scheme=:CN)
        xprev .= sys.x
        t += dt

        emax = max_velocity_error(model, sys.x)
        emax_all = max(emax_all, emax)
        res = norm(sys.A * sys.x - sys.b)

        if step % max(1, nsteps ÷ 10) == 0 || step == nsteps
            println("step=$(lpad(step, 4))  t/tref=$(round(t/tref; digits=3))  emax=$emax  ||Ax-b||=$res")
        end
    end

    println()
    println("Max velocity error over all steps: $emax_all")
    if emax_all < 1e-9
        println("PASS — error at machine-precision level")
    else
        println("WARN — error exceeds 1e-9, moving boundary is perturbing the flow")
    end
end

main()
