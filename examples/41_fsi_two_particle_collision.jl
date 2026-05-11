# Example 41: Two equal circles colliding head-on in a closed box.
#
# Two circles start on opposite sides with velocities pointing toward each other.
# Both wall and pairwise particle contact are active.
#
# Timestep restriction:
#   m_eff = m1*m2/(m1+m2) = 0.5  (for equal masses m=1)
#   dt <= 0.1 * sqrt(m_eff / k_n) = 0.1*sqrt(0.5/1e4) ≈ 7e-4

using PenguinStokes
using PenguinBCs
using CartesianGrids: CartesianGrid
using StaticArrays: SVector
using LinearAlgebra: norm
using Printf

T = Float64
n = 25
R = T(0.07)
grid = CartesianGrid((zero(T), zero(T)), (one(T), one(T)), (n, n))

bc = BorderConditions(; left=Periodic(), right=Periodic(),
                        bottom=Periodic(), top=Periodic())

shape = Circle(R)
c1 = SVector{2,T}(0.3, 0.5)
c2 = SVector{2,T}(0.7, 0.5)
V1 = SVector{2,T}(0.2, 0.0)
V2 = SVector{2,T}(-0.2, 0.0)

body_func(x, y, t) = max(
    shape.R - hypot(x - c1[1], y - c1[2]),
    shape.R - hypot(x - c2[1], y - c2[2]),
)

model = MovingStokesModelMono(
    grid, body_func, T(0.01), T(1.0);
    bc_u=(bc, bc),
    force=(zero(T), zero(T)),
    bc_cut_u=(Dirichlet(zero(T)), Dirichlet(zero(T))),
)

state1 = RigidBodyState(Tuple(c1), Tuple(V1))
state2 = RigidBodyState(Tuple(c2), Tuple(V2))
params1 = RigidBodyParams(T(1.0), T(1.0), shape, SVector(0.0, 0.0); buoyancy=false)
params2 = RigidBodyParams(T(1.0), T(1.0), shape, SVector(0.0, 0.0); buoyancy=false)

fsi = MultiBodyFSIProblem(model,
    [state1, state2],
    [params1, params2],
    [shape, shape],
)

contact_model = NormalSpringDashpotContact(
    stiffness       = T(1e4),
    damping         = T(20.0),
    gap_tol         = zero(T),
    projection_tol  = T(1e-6),
    enable_projection = true,
)

contact_constraints = (
    box_contacts_from_grid(grid),
    PairwiseParticleContact(nothing),
)

dt = T(5e-4)
nsteps = 400

println("step  t       X1[1]   X2[1]   V1[1]   V2[1]  |Fc_pair|")
global t = zero(T)
for step in 1:nsteps
    out = step_multi_fsi!(fsi; t=t, dt=dt,
                          contact_model=contact_model,
                          contact_constraints=contact_constraints)
    global t = out.t
    if mod(step, 50) == 0
        X1 = fsi.states[1].X
        X2 = fsi.states[2].X
        V1x = fsi.states[1].V[1]
        V2x = fsi.states[2].V[1]
        Fc_pair = out.contact.forces[1]
        @printf("%4d  %.4f  %.4f  %.4f  %+.4f  %+.4f  %.2e\n",
            step, t, X1[1], X2[1], V1x, V2x, norm(Fc_pair))
    end
end
