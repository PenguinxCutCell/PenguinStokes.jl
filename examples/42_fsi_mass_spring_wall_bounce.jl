# Example 40: Single circle bouncing inside a closed box with contact forces.
#
# A circle with initial velocity toward the bottom wall is allowed to bounce
# using the NormalSpringDashpotContact model.  The Stokes fluid provides
# hydrodynamic forces; contact provides wall repulsion.
#
# Timestep restriction for explicit contact:
#   dt <= 0.1 * sqrt(m_eff / k_n)
# Here m=1, k_n=1e4 → dt_contact = 0.1*sqrt(1/1e4) ≈ 1e-3.

using PenguinStokes
using PenguinBCs
using CartesianGrids: CartesianGrid
using StaticArrays: SVector
using LinearAlgebra: norm

T = Float64
n = 25
R = T(0.08)
grid = CartesianGrid((zero(T), zero(T)), (one(T), one(T)), (n, n))

# Periodic fluid BCs (box walls handled by contact, not fluid BC).
bc = BorderConditions(; left=Periodic(), right=Periodic(),
                        bottom=Periodic(), top=Periodic())

shape = Circle(R)
center0 = SVector{2,T}(0.5, 0.5)
V0 = SVector{2,T}(0.0, -0.3)  # moving toward bottom wall

body_func(x, y, t) = shape.R - hypot(x - 0.5, y - 0.5)

model = MovingStokesModelMono(
    grid, body_func, T(0.01), T(1.0);
    bc_u=(bc, bc),
    force=(zero(T), zero(T)),
    bc_cut_u=(Dirichlet(V0[1]), Dirichlet(V0[2])),
)

state = RigidBodyState(Tuple(center0), Tuple(V0))
params = RigidBodyParams(T(1.0), T(1.0), shape, SVector(0.0, 0.0); buoyancy=false)
fsi = StokesFSIProblem(model, state, params)

contact_model = NormalSpringDashpotContact(
    stiffness       = T(1e4),
    damping         = T(20.0),
    gap_tol         = zero(T),
    projection_tol  = T(1e-6),
    enable_projection = true,
)

contact_constraints = (box_contacts_from_grid(grid),)

dt = T(5e-4)
nsteps = 400

hist = simulate_fsi!(fsi;
    t0 = zero(T),
    dt = dt,
    nsteps = nsteps,
    contact_model = contact_model,
    contact_constraints = contact_constraints,
)

# Print a compact summary every 50 steps.
println("step  t       X[2]    V[2]    min_gap   Fc[2]")
for rec in hist[1:50:end]
    @printf("%4d  %.4f  %.4f  %+.4f  %+.2e  %+.2e\n",
        rec.step, rec.t, rec.X[2], rec.V[2],
        rec.contact.min_gap, rec.contact.force[2])
end
