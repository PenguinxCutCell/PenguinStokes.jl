module PenguinStokes

using LinearAlgebra
using SparseArrays
using StaticArrays

using CartesianGeometry: GeometricMoments, geometric_moments, nan
using CartesianGrids: CartesianGrid, SpaceTimeCartesianGrid, grid1d, meshsize
using CartesianOperators: AssembledCapacity, DiffusionOps, assembled_capacity, each_boundary_cell, periodic_flags, side_info
using PenguinBCs: AbstractBoundary, BorderConditions, Dirichlet, Neumann, Periodic, Traction, PressureOutlet, DoNothing, Symmetry, InterfaceConditions, ScalarJump, FluxJump, eval_bc, validate_borderconditions!
using PenguinSolverCore: LinearSystem, solve!

export AbstractPressureGauge, PinPressureGauge, MeanPressureGauge
export StokesLayout, StokesModelMono, staggered_velocity_grids
export StokesLayoutTwoPhase, StokesModelTwoPhase
export MovingStokesModelMono, MovingStokesModelTwoPhase
export assemble_steady!, assemble_unsteady!, solve_steady!, solve_unsteady!
export assemble_unsteady_moving!, solve_unsteady_moving!
export embedded_boundary_quantities, embedded_boundary_pressure, embedded_boundary_traction, embedded_boundary_stress, integrated_embedded_force
export RigidBodyState, RigidBodyParams, RigidBodyState2D, RigidBodyState3D
export RigidBodyParams2D, RigidBodyParams3D
export Circle, Sphere, Ellipse, Ellipsoid
export rigid_boundary_velocity, rigid_velocity_2d, rigid_velocity
export rigid_cut_bc_tuple, rigid_cut_bc_tuple_2d, rigid_body_levelset
export StokesFSIProblem, StokesFSIProblem2D
export endtime_static_model, step_fsi!, simulate_fsi!, step_fsi_rotation!, simulate_fsi_rotation!, step_fsi_strong!
export MultiBodyFSIProblem, step_multi_fsi!, multi_body_levelset, multi_body_cut_bc_tuple
export AbstractContactModel, AbstractContactConstraint
export NormalSpringDashpotContact, contact_from_restitution
export PlanarWallContact, BoxContact, box_contacts_from_grid
export PairwiseParticleContact
export contact_force, wall_contact_force, pairwise_contact_forces
export apply_contact_projection!

include("types.jl")
include("validation.jl")
include("force.jl")
include("activity.jl")
include("bc_velocity.jl")
include("bc_pressure.jl")
include("moving_geometry.jl")
include("assembly.jl")
include("analysis.jl")
include("constructors.jl")

include("orientation.jl")
include("rigidbody.jl")
include("contact.jl")
include("fsi.jl")
include("fsi_strong_coupling.jl")
include("fsi_multibody.jl")

end
