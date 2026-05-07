# Soft normal spring-dashpot contact model for rigid FSI particles.
#
# This implements an explicit contact force (Kelvin-Voigt normal model) that is
# added to the rigid-body ODE right-hand side.  It does NOT modify the Stokes
# fluid assembly.
#
# Timestep restriction: for explicit integration the contact oscillator must be
# resolved.  Practical guideline:
#
#   dt <= 0.1 * sqrt(m_eff / k_n)
#
# where:
#   - wall contact: m_eff = m (body mass)
#   - pair contact: m_eff = mi*mj / (mi + mj) (reduced mass)
#
# If dt is too large, reduce k_n, increase damping, or decrease dt.

# ── Abstract types ───────────────────────────────────────────────────────────

abstract type AbstractContactModel end
abstract type AbstractContactConstraint end

# ── Contact model ────────────────────────────────────────────────────────────

"""
    NormalSpringDashpotContact{T}

Soft normal spring-dashpot contact model (Kelvin-Voigt).

Fields:
- `stiffness`         — normal spring constant k_n  (>0)
- `damping`           — normal damping coefficient c_n (>=0)
- `gap_tol`           — contact threshold: contact is active when g < gap_tol (>=0)
- `projection_tol`    — minimum gap enforced by positional projection (>=0)
- `enable_projection` — whether to apply positional projection after ODE advance
"""
struct NormalSpringDashpotContact{T} <: AbstractContactModel
    stiffness::T
    damping::T
    gap_tol::T
    projection_tol::T
    enable_projection::Bool
end

function NormalSpringDashpotContact(;
    stiffness,
    damping,
    gap_tol=0.0,
    projection_tol=gap_tol,
    enable_projection=true,
)
    T = promote_type(typeof(stiffness), typeof(damping), typeof(gap_tol), typeof(projection_tol))
    kn = convert(T, stiffness)
    cn = convert(T, damping)
    gt = convert(T, gap_tol)
    pt = convert(T, projection_tol)
    kn > zero(T) || throw(ArgumentError("stiffness must be positive"))
    cn >= zero(T) || throw(ArgumentError("damping must be nonnegative"))
    gt >= zero(T) || throw(ArgumentError("gap_tol must be nonnegative"))
    pt >= zero(T) || throw(ArgumentError("projection_tol must be nonnegative"))
    return NormalSpringDashpotContact{T}(kn, cn, gt, pt, enable_projection)
end

"""
    contact_from_restitution(; stiffness, mass_eff, restitution, gap_tol=0.0)

Compute a `NormalSpringDashpotContact` whose damping matches a given coefficient
of restitution `e` using the Kelvin-Voigt estimate:

    c = -2 * sqrt(k * m_eff) * log(e) / sqrt(pi^2 + log(e)^2)

For e = 1 (elastic), c = 0.
"""
function contact_from_restitution(; stiffness, mass_eff, restitution, gap_tol=0.0)
    T = promote_type(typeof(stiffness), typeof(mass_eff), typeof(restitution), typeof(gap_tol))
    kn = convert(T, stiffness)
    m  = convert(T, mass_eff)
    e  = clamp(convert(T, restitution), convert(T, 1e-6), one(T))
    le = log(e)
    cn = -2 * sqrt(kn * m) * le / sqrt(convert(T, pi)^2 + le^2)
    return NormalSpringDashpotContact(; stiffness=kn, damping=cn, gap_tol=gap_tol)
end

# ── Support radius ────────────────────────────────────────────────────────────

_support_radius(shape::Circle, orientation, n) = shape.R
_support_radius(shape::Sphere, orientation, n) = shape.R
function _support_radius(shape, orientation, n)
    throw(ArgumentError("Contact support radius currently implemented only for Circle/Sphere."))
end

# ── Wall contact constraints ─────────────────────────────────────────────────

"""
    PlanarWallContact{N,T}

A planar wall defined by an inward unit normal and a point on the wall.
"""
struct PlanarWallContact{N,T} <: AbstractContactConstraint
    normal::SVector{N,T}
    point::SVector{N,T}
    name::Symbol
end

function PlanarWallContact(
    normal::NTuple{N,<:Real},
    point::NTuple{N,<:Real};
    name::Symbol=:unnamed,
) where {N}
    T = promote_type(eltype(normal), eltype(point))
    n = SVector{N,T}(normal)
    nn = norm(n)
    nn > zero(T) || throw(ArgumentError("wall normal must be nonzero"))
    return PlanarWallContact{N,T}(n / nn, SVector{N,T}(point), name)
end

"""
    BoxContact{N,T}

Collection of planar wall contacts representing a closed box.
"""
struct BoxContact{N,T} <: AbstractContactConstraint
    walls::Vector{PlanarWallContact{N,T}}
end

"""
    box_contacts_from_grid(grid; sides=:all)

Create a `BoxContact` from a `CartesianGrid`, using inward-pointing wall normals.

2D sides: :left (+x), :right (-x), :bottom (+y), :top (-y)
3D sides: additionally :back (+z), :front (-z)
"""
function box_contacts_from_grid(grid; sides=:all)
    N = length(grid.lc)
    T = eltype(grid.lc)

    all_sides_2d = [:left, :right, :bottom, :top]
    all_sides_3d = [:left, :right, :bottom, :top, :back, :front]
    all_sides = N == 2 ? all_sides_2d : all_sides_3d

    side_list = sides == :all ? all_sides : collect(sides)

    if N == 2
        walls_dict = Dict{Symbol,PlanarWallContact{2,T}}(
            :left   => PlanarWallContact((one(T), zero(T)),  (grid.lc[1], grid.lc[2]); name=:left),
            :right  => PlanarWallContact((-one(T), zero(T)), (grid.hc[1], grid.lc[2]); name=:right),
            :bottom => PlanarWallContact((zero(T), one(T)),  (grid.lc[1], grid.lc[2]); name=:bottom),
            :top    => PlanarWallContact((zero(T), -one(T)), (grid.lc[1], grid.hc[2]); name=:top),
        )
    else
        walls_dict = Dict{Symbol,PlanarWallContact{3,T}}(
            :left   => PlanarWallContact((one(T), zero(T), zero(T)),  (grid.lc[1], grid.lc[2], grid.lc[3]); name=:left),
            :right  => PlanarWallContact((-one(T), zero(T), zero(T)), (grid.hc[1], grid.lc[2], grid.lc[3]); name=:right),
            :bottom => PlanarWallContact((zero(T), one(T), zero(T)),  (grid.lc[1], grid.lc[2], grid.lc[3]); name=:bottom),
            :top    => PlanarWallContact((zero(T), -one(T), zero(T)), (grid.lc[1], grid.hc[2], grid.lc[3]); name=:top),
            :back   => PlanarWallContact((zero(T), zero(T), one(T)),  (grid.lc[1], grid.lc[2], grid.lc[3]); name=:back),
            :front  => PlanarWallContact((zero(T), zero(T), -one(T)), (grid.lc[1], grid.lc[2], grid.hc[3]); name=:front),
        )
    end

    walls = [walls_dict[s] for s in side_list]
    return BoxContact{N,T}(walls)
end

# ── Particle-particle contact ────────────────────────────────────────────────

"""
    PairwiseParticleContact

Constraint for particle-particle contact.

If `pairs === nothing`, all unique pairs (i,j) with i < j are checked (O(N^2)).
"""
struct PairwiseParticleContact <: AbstractContactConstraint
    pairs::Union{Nothing,Vector{Tuple{Int,Int}}}
end

PairwiseParticleContact() = PairwiseParticleContact(nothing)

# ── Internal setters ─────────────────────────────────────────────────────────

function _set_state_position!(state::RigidBodyState{N,T}, X::SVector{N,T}) where {N,T}
    state.X = X
    return state
end

function _set_state_position!(state::RigidBodyState2D{T}, X::SVector{2,T}) where {T}
    state.X = X
    return state
end

function _set_state_position!(state::RigidBodyState3D{T}, X::SVector{3,T}) where {T}
    state.X = X
    return state
end

function _set_state_velocity!(state::RigidBodyState{N,T}, V::SVector{N,T}) where {N,T}
    state.V = V
    return state
end

function _set_state_velocity!(state::RigidBodyState2D{T}, V::SVector{2,T}) where {T}
    state.V = V
    return state
end

function _set_state_velocity!(state::RigidBodyState3D{T}, V::SVector{3,T}) where {T}
    state.V = V
    return state
end

# ── Gap and force computations ───────────────────────────────────────────────

function _wall_gap(state, params, wall::PlanarWallContact{N,T}) where {N,T}
    X = SVector{N,T}(state_position(state))
    n = wall.normal
    xw = wall.point
    ori = state_orientation(state)
    r = _support_radius(params.shape, ori, n)
    g = dot(X - xw, n) - r
    return g
end

function _wall_normal_velocity(state, wall::PlanarWallContact{N,T}) where {N,T}
    V = SVector{N,T}(state_velocity(state))
    return dot(V, wall.normal)
end

"""
    wall_contact_force(state, params, wall::PlanarWallContact, model)

Compute the normal contact force and gap for a single planar wall.
Returns `(force, gap)`.
"""
function wall_contact_force(state, params, wall::PlanarWallContact{N,T}, model::NormalSpringDashpotContact) where {N,T}
    g = _wall_gap(state, params, wall)
    vn = _wall_normal_velocity(state, wall)
    delta = max(model.gap_tol - g, zero(T))
    Fn = max(zero(T), model.stiffness * delta - model.damping * vn)
    return Fn * wall.normal, g
end

"""
    wall_contact_force(state, params, box::BoxContact, model)

Compute the total wall contact force from all walls in a box.
Returns `(force, min_gap, ncontacts)`.
"""
function wall_contact_force(state, params, box::BoxContact{N,T}, model::NormalSpringDashpotContact) where {N,T}
    Ftotal = zero(SVector{N,T})
    min_gap = typemax(T)
    ncontacts = 0
    for wall in box.walls
        Fw, g = wall_contact_force(state, params, wall, model)
        Ftotal += Fw
        min_gap = min(min_gap, g)
        if g < model.gap_tol
            ncontacts += 1
        end
    end
    return Ftotal, min_gap, ncontacts
end

# ── Single-body contact force ────────────────────────────────────────────────

"""
    contact_force(state, params, constraints, model::NormalSpringDashpotContact)

Compute total contact force and torque on one body from all constraints.

Returns a NamedTuple:
    (force, torque, active, min_gap, ncontacts)
"""
function contact_force(
    state,
    params,
    constraints,
    model::NormalSpringDashpotContact,
)
    T = eltype(state_position(state))
    N = length(state_position(state))
    Ftotal = zero(SVector{N,T})
    min_gap = typemax(T)
    ncontacts = 0

    for c in constraints
        if c isa PlanarWallContact
            Fw, g = wall_contact_force(state, params, c, model)
            Ftotal += Fw
            min_gap = min(min_gap, g)
            if g < model.gap_tol
                ncontacts += 1
            end
        elseif c isa BoxContact
            Fw, mg, nc = wall_contact_force(state, params, c, model)
            Ftotal += Fw
            min_gap = min(min_gap, mg)
            ncontacts += nc
        end
        # PairwiseParticleContact is handled at multi-body level
    end

    zero_torque = N == 2 ? zero(T) : zero(SVector{3,T})
    return (
        force = Ftotal,
        torque = zero_torque,
        active = ncontacts > 0,
        min_gap = min_gap,
        ncontacts = ncontacts,
    )
end

# Dispatch when contact_model is nothing — return zero contribution.
function contact_force(state, params, constraints, ::Nothing)
    T = eltype(state_position(state))
    N = length(state_position(state))
    zero_torque = N == 2 ? zero(T) : zero(SVector{3,T})
    return (
        force = zero(SVector{N,T}),
        torque = zero_torque,
        active = false,
        min_gap = typemax(T),
        ncontacts = 0,
    )
end

# ── Pairwise contact forces ───────────────────────────────────────────────────

"""
    pairwise_contact_forces(states, params, pair_constraint, model)

Compute equal-and-opposite contact forces between all particle pairs.
Returns a `Vector` of `(force, torque)` NamedTuples, one per body.

Currently implemented only for Circle/Sphere shapes.
"""
function pairwise_contact_forces(
    states,
    params,
    pair_constraint::PairwiseParticleContact,
    model::NormalSpringDashpotContact,
)
    Nbodies = length(states)
    T = eltype(state_position(states[1]))
    N = length(state_position(states[1]))

    forces = [zero(SVector{N,T}) for _ in 1:Nbodies]
    zero_torque = N == 2 ? zero(T) : zero(SVector{3,T})

    pairs = if pair_constraint.pairs === nothing
        [(i, j) for i in 1:Nbodies for j in (i+1):Nbodies]
    else
        pair_constraint.pairs
    end

    for (i, j) in pairs
        Xi = SVector{N,T}(state_position(states[i]))
        Xj = SVector{N,T}(state_position(states[j]))
        Vi = SVector{N,T}(state_velocity(states[i]))
        Vj = SVector{N,T}(state_velocity(states[j]))

        d = Xj - Xi
        dist = norm(d)
        dist > zero(T) || continue  # degenerate: skip

        nij = d / dist

        ori_i = state_orientation(states[i])
        ori_j = state_orientation(states[j])
        Ri = _support_radius(params[i].shape, ori_i, -nij)
        Rj = _support_radius(params[j].shape, ori_j,  nij)

        g = dist - Ri - Rj
        vn = dot(Vj - Vi, nij)  # positive = separating

        delta = max(model.gap_tol - g, zero(T))
        Fn = max(zero(T), model.stiffness * delta - model.damping * vn)

        forces[i] += -Fn * nij
        forces[j] +=  Fn * nij
    end

    return [(force=forces[k], torque=zero_torque) for k in 1:Nbodies]
end

# ── Positional projection ─────────────────────────────────────────────────────

"""
    apply_contact_projection!(state, params, constraints, model)

After the ODE advance, project positions to remove residual penetrations.
Only positions are modified; velocities are unchanged.
"""
function apply_contact_projection!(state, params, constraints, model::NormalSpringDashpotContact)
    model.enable_projection || return nothing
    T = eltype(state_position(state))
    N = length(state_position(state))
    X = SVector{N,T}(state_position(state))

    for c in constraints
        if c isa PlanarWallContact
            X = _project_wall(X, params, c, model)
        elseif c isa BoxContact
            for wall in c.walls
                X = _project_wall(X, params, wall, model)
            end
        end
    end

    _set_state_position!(state, X)
    return nothing
end

apply_contact_projection!(state, params, constraints, ::Nothing) = nothing

function _project_wall(X::SVector{N,T}, params, wall::PlanarWallContact{N,T}, model) where {N,T}
    n = wall.normal
    xw = wall.point
    r = _support_radius(params.shape, nothing, n)
    g = dot(X - xw, n) - r
    if g < model.projection_tol
        X = X + (model.projection_tol - g) * n
    end
    return X
end

"""
    apply_contact_projection!(states, params, constraints, model)

Multi-body version: applies wall projection to each body, then pairwise
projection (repeated a few passes to resolve constraint interactions).
"""
function apply_contact_projection!(
    states::AbstractVector,
    params::AbstractVector,
    constraints,
    model::NormalSpringDashpotContact,
)
    model.enable_projection || return nothing
    Nbodies = length(states)

    # Wall projections for each body.
    for k in 1:Nbodies
        apply_contact_projection!(states[k], params[k], constraints, model)
    end

    # Pairwise projections (a few passes to handle interacting constraints).
    for c in constraints
        c isa PairwiseParticleContact || continue
        for _ in 1:3
            _project_pairs!(states, params, c, model)
        end
    end
    return nothing
end

apply_contact_projection!(states::AbstractVector, params::AbstractVector, constraints, ::Nothing) = nothing

function _project_pairs!(states, params, pair_constraint::PairwiseParticleContact, model)
    Nbodies = length(states)
    T = eltype(state_position(states[1]))
    N = length(state_position(states[1]))

    pairs = if pair_constraint.pairs === nothing
        [(i, j) for i in 1:Nbodies for j in (i+1):Nbodies]
    else
        pair_constraint.pairs
    end

    for (i, j) in pairs
        Xi = SVector{N,T}(state_position(states[i]))
        Xj = SVector{N,T}(state_position(states[j]))

        d = Xj - Xi
        dist = norm(d)
        dist > zero(T) || continue

        nij = d / dist
        Ri = _support_radius(params[i].shape, nothing, -nij)
        Rj = _support_radius(params[j].shape, nothing,  nij)

        g = dist - Ri - Rj
        if g < model.projection_tol
            # TODO: use mass-weighted split when masses are easily accessible.
            correction = convert(T, 0.5) * (model.projection_tol - g) * nij
            _set_state_position!(states[i], Xi - correction)
            _set_state_position!(states[j], Xj + correction)
        end
    end
end
