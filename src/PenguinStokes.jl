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
export MovingStokesModelMono
export assemble_steady!, assemble_unsteady!, solve_steady!, solve_unsteady!
export assemble_unsteady_moving!, solve_unsteady_moving!
export embedded_boundary_quantities, embedded_boundary_traction, embedded_boundary_stress, integrated_embedded_force
export RigidBodyState, RigidBodyParams, RigidBodyState2D, RigidBodyState3D
export RigidBodyParams2D, RigidBodyParams3D
export Circle, Sphere, Ellipse, Ellipsoid
export rigid_boundary_velocity, rigid_velocity_2d, rigid_velocity
export rigid_cut_bc_tuple, rigid_cut_bc_tuple_2d, rigid_body_levelset
export StokesFSIProblem, StokesFSIProblem2D
export endtime_static_model, step_fsi!, simulate_fsi!, step_fsi_rotation!, simulate_fsi_rotation!, step_fsi_strong!

"""
    AbstractPressureGauge

Abstract supertype for pressure nullspace constraints in Stokes systems.
Concrete options are `PinPressureGauge` and `MeanPressureGauge`.
"""
abstract type AbstractPressureGauge end

"""
    PinPressureGauge(; index=nothing)

Pressure gauge that replaces one pressure equation by a pointwise pinning
constraint `p[index] = 0` (or the first active pressure DOF when `index=nothing`).
"""
struct PinPressureGauge <: AbstractPressureGauge
    index::Union{Nothing,Int}
end
PinPressureGauge(; index::Union{Nothing,Int}=nothing) = PinPressureGauge(index)

"""
    MeanPressureGauge()

Pressure gauge that replaces one pressure equation by a zero-mean pressure
constraint over active pressure cells.
"""
struct MeanPressureGauge <: AbstractPressureGauge end

"""
    StokesLayout{N}

Unknown ordering for monophasic Stokes:
`[uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]`.
"""
struct StokesLayout{N}
    nt::Int
    uomega::NTuple{N,UnitRange{Int}}
    ugamma::NTuple{N,UnitRange{Int}}
    pomega::UnitRange{Int}
end

"""
    StokesLayoutTwoPhase{N}

Unknown ordering for fixed-interface two-phase Stokes:
`[uomega1_1; ...; uomega1_N; uomega2_1; ...; uomega2_N; ugamma1_1; ...; ugamma1_N; ugamma2_1; ...; ugamma2_N; pomega1; pomega2]`.
"""
struct StokesLayoutTwoPhase{N}
    nt::Int
    uomega1::NTuple{N,UnitRange{Int}}
    uomega2::NTuple{N,UnitRange{Int}}
    ugamma1::NTuple{N,UnitRange{Int}}
    ugamma2::NTuple{N,UnitRange{Int}}
    pomega1::UnitRange{Int}
    pomega2::UnitRange{Int}
end

@inline function Base.getproperty(layout::StokesLayoutTwoPhase, s::Symbol)
    if s === :ugamma
        return getfield(layout, :ugamma1)
    end
    return getfield(layout, s)
end

function StokesLayout(nt::Int, ::Val{N}) where {N}
    nt > 0 || throw(ArgumentError("nt must be positive"))
    start = 1
    uomega = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    ugamma = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    pomega = start:(start + nt - 1)
    return StokesLayout{N}(nt, uomega, ugamma, pomega)
end

nunknowns(layout::StokesLayout) = last(layout.pomega)
nunknowns(layout::StokesLayoutTwoPhase) = last(layout.pomega2)

function StokesLayoutTwoPhase(nt::Int, ::Val{N}) where {N}
    nt > 0 || throw(ArgumentError("nt must be positive"))
    start = 1
    uomega1 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    uomega2 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    ugamma1 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    ugamma2 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    pomega1 = start:(start + nt - 1)
    start += nt
    pomega2 = start:(start + nt - 1)
    return StokesLayoutTwoPhase{N}(nt, uomega1, uomega2, ugamma1, ugamma2, pomega1, pomega2)
end

"""
    StokesModelMono{N,T}

Monophasic cut-cell Stokes model on a staggered MAC grid.

Use constructors
`StokesModelMono(gridp, body, mu, rho; ...)` or
`StokesModelMono(cap_p, op_p, cap_u, op_u, mu, rho; ...)`.
"""
mutable struct StokesModelMono{N,T,FT,BT}
    gridp::CartesianGrid{N,T}
    gridu::NTuple{N,CartesianGrid{N,T}}
    cap_p::AssembledCapacity{N,T}
    cap_u::NTuple{N,AssembledCapacity{N,T}}
    op_p::DiffusionOps{N,T}
    op_u::NTuple{N,DiffusionOps{N,T}}
    mu::T
    rho::T
    force::FT
    bc_u::NTuple{N,BorderConditions}
    bc_p::Union{Nothing,BorderConditions}
    bc_cut::NTuple{N,AbstractBoundary}
    gauge::AbstractPressureGauge
    strong_wall_bc::Bool
    layout::StokesLayout{N}
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    body::BT
end

"""
    StokesModelTwoPhase{N,T}

Fixed-interface two-phase Stokes model with phase-wise interface velocity traces
`ugamma1`, `ugamma2` and one pressure block per phase.

Use constructors
`StokesModelTwoPhase(gridp, body, mu1, mu2; ...)` or
`StokesModelTwoPhase(cap_p1, op_p1, ..., cap_p2, op_p2, ...; ...)`.
"""
mutable struct StokesModelTwoPhase{N,T,FT1,FT2,IFT,IJT,BT}
    gridp::CartesianGrid{N,T}
    gridu::NTuple{N,CartesianGrid{N,T}}

    cap_p1::AssembledCapacity{N,T}
    cap_u1::NTuple{N,AssembledCapacity{N,T}}
    op_p1::DiffusionOps{N,T}
    op_u1::NTuple{N,DiffusionOps{N,T}}

    cap_p2::AssembledCapacity{N,T}
    cap_u2::NTuple{N,AssembledCapacity{N,T}}
    op_p2::DiffusionOps{N,T}
    op_u2::NTuple{N,DiffusionOps{N,T}}

    mu1::T
    mu2::T
    rho1::T
    rho2::T
    force1::FT1
    force2::FT2
    interface_force::IFT
    interface_jump::IJT
    bc_interface::Union{Nothing,NTuple{N,InterfaceConditions}}
    bc_u::NTuple{N,BorderConditions}
    bc_p::Union{Nothing,BorderConditions}
    gauge::AbstractPressureGauge
    strong_wall_bc::Bool
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    body::BT
    layout::StokesLayoutTwoPhase{N}
end

"""
    MovingStokesModelMono{N,T}

Unsteady monophasic moving-boundary Stokes model with prescribed embedded
boundary velocity through `bc_cut_u`.
"""
mutable struct MovingStokesModelMono{N,T,FT,BT}
    gridp::CartesianGrid{N,T}
    gridu::NTuple{N,CartesianGrid{N,T}}
    body::BT
    mu::T
    rho::T
    force::FT
    bc_u::NTuple{N,BorderConditions}
    bc_p::Union{Nothing,BorderConditions}
    bc_cut_u::NTuple{N,AbstractBoundary}
    gauge::AbstractPressureGauge
    strong_wall_bc::Bool
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    layout::StokesLayout{N}
    cap_p_slab::Union{Nothing,AssembledCapacity{N,T}}
    op_p_slab::Union{Nothing,DiffusionOps{N,T}}
    cap_p_end::Union{Nothing,AssembledCapacity{N,T}}
    op_p_end::Union{Nothing,DiffusionOps{N,T}}
    cap_u_slab::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    op_u_slab::Union{Nothing,NTuple{N,DiffusionOps{N,T}}}
    cap_u_end::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    Vun::Union{Nothing,NTuple{N,Vector{T}}}
    Vun1::Union{Nothing,NTuple{N,Vector{T}}}
end

function _default_force(::Type{T}, ::Val{N}) where {T,N}
    return ntuple(_ -> zero(T), N)
end

function _default_interface_force(::Type{T}, ::Val{N}) where {T,N}
    return ntuple(_ -> zero(T), N)
end

function _default_interface_jump(::Type{T}, ::Val{N}) where {T,N}
    return ntuple(_ -> zero(T), N)
end

function _normalize_bc_tuple(bc_u::NTuple{N,BorderConditions}) where {N}
    return bc_u
end

function _normalize_bc_tuple(bc_u::BorderConditions, ::Val{1})
    return (bc_u,)
end

function _normalize_cut_bc_tuple(bc_cut_u::NTuple{N,AbstractBoundary}) where {N}
    return bc_cut_u
end

function _normalize_cut_bc_tuple(bc_cut_u::AbstractBoundary, ::Val{1})
    return (bc_cut_u,)
end

function _normalize_cut_bc_tuple(bc_cut_u::AbstractBoundary, ::Val{N}) where {N}
    return ntuple(_ -> bc_cut_u, N)
end

function _normalize_interface_bc(::Nothing, ::Val{N}) where {N}
    return nothing
end

function _normalize_interface_bc(ic::InterfaceConditions, ::Val{N}) where {N}
    return ntuple(_ -> ic, N)
end

function _normalize_interface_bc(ic::NTuple{N,InterfaceConditions}, ::Val{N}) where {N}
    return ic
end

function _validate_stokes_interface_bcs!(
    bc_interface::Union{Nothing,NTuple{N,InterfaceConditions}},
) where {N}
    isnothing(bc_interface) && return nothing
    @inbounds for d in 1:N
        ic = bc_interface[d]
        if !(ic.scalar === nothing) && !(ic.scalar isa ScalarJump)
            throw(ArgumentError("two-phase Stokes interface scalar condition for component $d must be ScalarJump or nothing"))
        end
        if !(ic.flux === nothing) && !(ic.flux isa FluxJump)
            throw(ArgumentError("two-phase Stokes interface flux condition for component $d must be FluxJump or nothing"))
        end
    end
    return nothing
end

function _validate_pressure_bc(
    bc_p::Union{Nothing,BorderConditions},
    periodic::NTuple{N,Bool},
) where {N}
    isnothing(bc_p) && return nothing
    validate_borderconditions!(bc_p, N)
    periodic_flags(bc_p, N) == periodic ||
        throw(ArgumentError("pressure border condition periodic flags must match velocity periodic flags"))
    return bc_p
end

@inline _is_stokes_traction_bc(bc::AbstractBoundary) =
    bc isa Traction || bc isa PressureOutlet || bc isa DoNothing

@inline _is_stokes_symmetry_bc(bc::AbstractBoundary) =
    bc isa Symmetry

@inline _is_stokes_side_vector_bc(bc::AbstractBoundary) =
    _is_stokes_traction_bc(bc) || _is_stokes_symmetry_bc(bc)

@inline _is_scalar_velocity_bc(bc::AbstractBoundary) =
    bc isa Dirichlet || bc isa Neumann || bc isa Periodic

function _side_velocity_bcs(
    bc_u::NTuple{N,BorderConditions},
    side::Symbol,
    ::Type{T},
) where {N,T}
    return ntuple(comp -> get(bc_u[comp].borders, side, Neumann(zero(T))), N)
end

function _side_uses_traction_bc(
    bc_u::NTuple{N,BorderConditions},
    side::Symbol,
    ::Type{T},
) where {N,T}
    side_bcs = _side_velocity_bcs(bc_u, side, T)
    return any(_is_stokes_traction_bc, side_bcs)
end

function _side_uses_symmetry_bc(
    bc_u::NTuple{N,BorderConditions},
    side::Symbol,
    ::Type{T},
) where {N,T}
    side_bcs = _side_velocity_bcs(bc_u, side, T)
    return any(_is_stokes_symmetry_bc, side_bcs)
end

function _side_uses_vector_bc(
    bc_u::NTuple{N,BorderConditions},
    side::Symbol,
    ::Type{T},
) where {N,T}
    side_bcs = _side_velocity_bcs(bc_u, side, T)
    return any(_is_stokes_side_vector_bc, side_bcs)
end

function _pressure_side_conflicts_with_vector_bc(
    bc_p::Union{Nothing,BorderConditions},
    side::Symbol,
)
    isnothing(bc_p) && return false
    return haskey(bc_p.borders, side)
end

function _validate_stokes_box_bcs!(
    bc_u::NTuple{N,BorderConditions},
    bc_p::Union{Nothing,BorderConditions},
    periodic::NTuple{N,Bool},
    ::Type{T},
) where {N,T}
    pairs = _side_pairs(N)
    for d in 1:N
        side_lo, side_hi = pairs[d]
        for side in (side_lo, side_hi)
            side_bcs = _side_velocity_bcs(bc_u, side, T)
            traction_mask = ntuple(comp -> _is_stokes_traction_bc(side_bcs[comp]), N)
            symmetry_mask = ntuple(comp -> _is_stokes_symmetry_bc(side_bcs[comp]), N)
            vector_mask = ntuple(comp -> _is_stokes_side_vector_bc(side_bcs[comp]), N)
            any_traction = any(traction_mask)
            any_symmetry = any(symmetry_mask)
            any_vector = any(vector_mask)

            for comp in 1:N
                bc = side_bcs[comp]
                (_is_scalar_velocity_bc(bc) || _is_stokes_side_vector_bc(bc)) ||
                    throw(ArgumentError("unsupported velocity boundary type $(typeof(bc)) on side `$side`"))
            end

            if any_symmetry
                all(symmetry_mask) ||
                    throw(ArgumentError("Symmetry is a side-level Stokes BC and must be set on all velocity components of side `$side`"))
                any_traction &&
                    throw(ArgumentError("side `$side` cannot mix Symmetry with traction-type BCs"))
            end

            if any_traction
                all(traction_mask) ||
                    throw(ArgumentError("traction-type Stokes BC on side `$side` must be set on all velocity components"))
                any_symmetry &&
                    throw(ArgumentError("side `$side` cannot mix traction-type BCs with Symmetry"))
            end

            any_vector || continue
            periodic[d] &&
                throw(ArgumentError("side-level Stokes BC on side `$side` is incompatible with periodic boundaries"))
            _pressure_side_conflicts_with_vector_bc(bc_p, side) &&
                throw(ArgumentError("pressure BC on side `$side` conflicts with side-level Stokes velocity BC"))
        end
    end
    return nothing
end

function _side_pairs(N::Int)
    if N == 1
        return ((:left, :right),)
    elseif N == 2
        return ((:left, :right), (:bottom, :top))
    elseif N == 3
        return ((:left, :right), (:bottom, :top), (:backward, :forward))
    end
    throw(ArgumentError("unsupported dimension N=$N; expected 1, 2, or 3"))
end

function _periodic_velocity_flags(bc_u::NTuple{N,BorderConditions}) where {N}
    flags = periodic_flags(bc_u[1], N)
    @inbounds for d in 2:N
        periodic_flags(bc_u[d], N) == flags ||
            throw(ArgumentError("all velocity component border conditions must share identical periodic flags"))
    end
    return flags
end

"""
    staggered_velocity_grids(gridp)

Build component-wise staggered velocity grids from a pressure grid `gridp`
by half-cell shifting each component grid along its own axis.
"""
function staggered_velocity_grids(gridp::CartesianGrid{N,T}) where {N,T}
    h = meshsize(gridp)
    return ntuple(d -> begin
        lc = ntuple(k -> k == d ? gridp.lc[k] - h[k] / T(2) : gridp.lc[k], N)
        hc = ntuple(k -> k == d ? gridp.hc[k] - h[k] / T(2) : gridp.hc[k], N)
        CartesianGrid(lc, hc, gridp.n)
    end, N)
end

function _force_component(force, d::Int, x::SVector{N,T}, t::T) where {N,T}
    if force isa Number
        return convert(T, force)
    elseif force isa NTuple{N,Any}
        fd = force[d]
        if fd isa Number
            return convert(T, fd)
        elseif fd isa Function
            if applicable(fd, x..., t)
                return convert(T, fd(x..., t))
            elseif applicable(fd, x...)
                return convert(T, fd(x...))
            end
            throw(ArgumentError("velocity forcing callback for component $d must accept (x...) or (x..., t)"))
        end
        throw(ArgumentError("unsupported forcing entry type $(typeof(fd)) for component $d"))
    elseif force isa Function
        if applicable(force, x..., t)
            y = force(x..., t)
            if y isa Number
                return convert(T, y)
            end
            return convert(T, y[d])
        elseif applicable(force, x...)
            y = force(x...)
            if y isa Number
                return convert(T, y)
            end
            return convert(T, y[d])
        end
        throw(ArgumentError("velocity forcing callback must accept (x...) or (x..., t)"))
    end
    throw(ArgumentError("unsupported forcing type $(typeof(force))"))
end

function _force_values(model::StokesModelMono{N,T}, d::Int, t::T) where {N,T}
    nt = model.cap_u[d].ntotal
    out = Vector{T}(undef, nt)
    cap = model.cap_u[d]
    @inbounds for i in 1:nt
        out[i] = _force_component(model.force, d, cap.C_ω[i], t)
    end
    return out
end

function _force_values(model::StokesModelTwoPhase{N,T}, phase::Int, d::Int, t::T) where {N,T}
    if phase == 1
        cap = model.cap_u1[d]
        force = model.force1
    elseif phase == 2
        cap = model.cap_u2[d]
        force = model.force2
    else
        throw(ArgumentError("phase must be 1 or 2"))
    end
    nt = cap.ntotal
    out = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        out[i] = _force_component(force, d, cap.C_ω[i], t)
    end
    return out
end

function _force_values(
    model::MovingStokesModelMono{N,T},
    cap::AssembledCapacity{N,T},
    d::Int,
    t::T,
) where {N,T}
    nt = cap.ntotal
    out = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        out[i] = _force_component(model.force, d, cap.C_ω[i], t)
    end
    return out
end

function _interface_force_component(interface_force, d::Int, x::SVector{N,T}, t::T) where {N,T}
    if interface_force isa Number
        return convert(T, interface_force)
    elseif interface_force isa NTuple{N,Any}
        fd = interface_force[d]
        if fd isa Number
            return convert(T, fd)
        elseif fd isa Function
            if applicable(fd, x..., t)
                return convert(T, fd(x..., t))
            elseif applicable(fd, x...)
                return convert(T, fd(x...))
            elseif applicable(fd, x, t)
                return convert(T, fd(x, t))
            elseif applicable(fd, x)
                return convert(T, fd(x))
            end
            throw(ArgumentError("interface forcing callback for component $d must accept (x...), (x..., t), (x), or (x, t)"))
        end
        throw(ArgumentError("unsupported interface forcing entry type $(typeof(fd)) for component $d"))
    elseif interface_force isa Function
        if applicable(interface_force, x..., t)
            y = interface_force(x..., t)
            return y isa Number ? convert(T, y) : convert(T, y[d])
        elseif applicable(interface_force, x...)
            y = interface_force(x...)
            return y isa Number ? convert(T, y) : convert(T, y[d])
        elseif applicable(interface_force, x, t)
            y = interface_force(x, t)
            return y isa Number ? convert(T, y) : convert(T, y[d])
        elseif applicable(interface_force, x)
            y = interface_force(x)
            return y isa Number ? convert(T, y) : convert(T, y[d])
        end
        throw(ArgumentError("interface forcing callback must accept (x...), (x..., t), (x), or (x, t)"))
    end
    throw(ArgumentError("unsupported interface forcing type $(typeof(interface_force))"))
end

function _cut_values(cap::AssembledCapacity{N,T}, bc_cut::AbstractBoundary, t::T) where {N,T}
    out = zeros(T, cap.ntotal)
    if bc_cut isa Dirichlet
        @inbounds for i in eachindex(out)
            out[i] = convert(T, eval_bc(bc_cut.value, cap.C_γ[i], t))
        end
        return out
    end
    if bc_cut isa Neumann || bc_cut isa Periodic
        throw(ArgumentError("cut-cell velocity condition currently supports Dirichlet only"))
    end
    throw(ArgumentError("unsupported cut-cell velocity condition type $(typeof(bc_cut))"))
end

function _insert_block!(A::SparseMatrixCSC{T,Int}, rows::UnitRange{Int}, cols::UnitRange{Int}, B::SparseMatrixCSC{T,Int}) where {T}
    size(B, 1) == length(rows) || throw(DimensionMismatch("block rows do not match target range"))
    size(B, 2) == length(cols) || throw(DimensionMismatch("block cols do not match target range"))
    @inbounds for j in 1:size(B, 2)
        for p in nzrange(B, j)
            i = B.rowval[p]
            A[rows[i], cols[j]] = A[rows[i], cols[j]] + B.nzval[p]
        end
    end
    return A
end

function _insert_vec!(b::Vector{T}, rows::UnitRange{Int}, v::AbstractVector{T}) where {T}
    length(v) == length(rows) || throw(DimensionMismatch("vector block length mismatch"))
    @inbounds for i in eachindex(v)
        b[rows[i]] += v[i]
    end
    return b
end

function _cell_activity_masks(cap::AssembledCapacity{N,T}) where {N,T}
    nt = cap.ntotal
    active_omega = BitVector(undef, nt)
    active_gamma = BitVector(undef, nt)
    li = LinearIndices(cap.nnodes)
    @inbounds for I in CartesianIndices(cap.nnodes)
        i = li[I]
        physical = true
        for d in 1:N
            if I[d] == cap.nnodes[d]
                physical = false
                break
            end
        end
        if !physical
            active_omega[i] = false
            active_gamma[i] = false
            continue
        end
        v = cap.buf.V[i]
        g = cap.buf.Γ[i]
        active_omega[i] = isfinite(v) && v > zero(T)
        active_gamma[i] = isfinite(g) && g > zero(T)
    end
    return active_omega, active_gamma
end

function _pressure_activity(cap::AssembledCapacity{N,T}) where {N,T}
    active = BitVector(undef, cap.ntotal)
    li = LinearIndices(cap.nnodes)
    @inbounds for I in CartesianIndices(cap.nnodes)
        i = li[I]
        physical = true
        for d in 1:N
            # Node-padded layout: only the last layer is a halo.
            if I[d] == cap.nnodes[d]
                physical = false
                break
            end
        end
        if !physical
            active[i] = false
            continue
        end
        v = cap.buf.V[i]
        active[i] = isfinite(v) && v > zero(T)
    end
    return active
end

function _pressure_interior_activity(cap::AssembledCapacity{N,T}) where {N,T}
    active = BitVector(undef, cap.ntotal)
    li = LinearIndices(cap.nnodes)
    @inbounds for I in CartesianIndices(cap.nnodes)
        i = li[I]
        physical = true
        interior = true
        for d in 1:N
            # Node-padded layout: only the last layer is a halo.
            if I[d] == cap.nnodes[d]
                physical = false
                break
            end
            # Prefer not to pin/replace pressure equations on the outermost
            # physical ring when an interior coupled row exists.
            if !(1 < I[d] < (cap.nnodes[d] - 1))
                interior = false
            end
        end
        if !physical
            active[i] = false
            continue
        end
        v = cap.buf.V[i]
        active[i] = interior && isfinite(v) && v > zero(T)
    end
    return active
end

function _prune_uncoupled_active!(active::BitVector, A::SparseMatrixCSC{T,Int}) where {T}
    n = size(A, 2)
    length(active) == n || throw(DimensionMismatch("active-mask length must match matrix size"))
    changed = true
    while changed
        changed = false
        @inbounds for j in 1:n
            active[j] || continue
            coupled = false
            for ptr in nzrange(A, j)
                row = A.rowval[ptr]
                if active[row] && A.nzval[ptr] != zero(T)
                    coupled = true
                    break
                end
            end
            if !coupled
                active[j] = false
                changed = true
            end
        end
    end
    return active
end

function _stokes_row_activity(model::StokesModelMono{N,T}, A::SparseMatrixCSC{T,Int}) where {N,T}
    layout = model.layout
    active = falses(nunknowns(layout))
    @inbounds for d in 1:N
        aomega, agamma = _cell_activity_masks(model.cap_u[d])
        for i in 1:model.cap_u[d].ntotal
            active[layout.uomega[d][i]] = aomega[i]
            active[layout.ugamma[d][i]] = agamma[i]
        end
    end
    pactive = _pressure_activity(model.cap_p)
    pfirst = first(layout.pomega)
    plast = last(layout.pomega)
    @inbounds for i in 1:model.cap_p.ntotal
        pactive[i] || continue
        col = layout.pomega[i]
        coupled = false
        for ptr in nzrange(A, col)
            row = A.rowval[ptr]
            if (row < pfirst || row > plast) && active[row] && A.nzval[ptr] != zero(T)
                coupled = true
                break
            end
        end
        pactive[i] = coupled
    end
    @inbounds for i in 1:model.cap_p.ntotal
        active[layout.pomega[i]] = pactive[i]
    end
    return _prune_uncoupled_active!(active, A)
end

function _stokes_row_activity(model::StokesModelTwoPhase{N,T}, A::SparseMatrixCSC{T,Int}) where {N,T}
    layout = model.layout
    active = falses(nunknowns(layout))
    _, agamma_p = _cell_activity_masks(model.cap_p1)

    @inbounds for d in 1:N
        aomega1, agamma1 = _cell_activity_masks(model.cap_u1[d])
        aomega2, agamma2 = _cell_activity_masks(model.cap_u2[d])
        for i in 1:model.cap_u1[d].ntotal
            active[layout.uomega1[d][i]] = aomega1[i]
            active[layout.uomega2[d][i]] = aomega2[i]
            has_gamma = (agamma1[i] || agamma2[i]) && agamma_p[i]
            active[layout.ugamma1[d][i]] = has_gamma
            active[layout.ugamma2[d][i]] = has_gamma
        end
    end

    p1active = _pressure_activity(model.cap_p1)
    p2active = _pressure_activity(model.cap_p2)
    u1support = ntuple(d -> _cell_activity_masks(model.cap_u1[d])[1], N)
    u2support = ntuple(d -> _cell_activity_masks(model.cap_u2[d])[1], N)

    @inbounds for i in 1:model.cap_p1.ntotal
        p1active[i] || continue
        supported = false
        for d in 1:N
            if u1support[d][i]
                supported = true
                break
            end
        end
        p1active[i] = supported
    end

    @inbounds for i in 1:model.cap_p2.ntotal
        p2active[i] || continue
        supported = false
        for d in 1:N
            if u2support[d][i]
                supported = true
                break
            end
        end
        p2active[i] = supported
    end

    p1first = first(layout.pomega1)
    p1last = last(layout.pomega1)
    p2first = first(layout.pomega2)
    p2last = last(layout.pomega2)

    @inbounds for i in 1:model.cap_p1.ntotal
        p1active[i] || continue
        col = layout.pomega1[i]
        coupled = false
        for ptr in nzrange(A, col)
            row = A.rowval[ptr]
            if (row < p1first || row > p1last) && active[row] && A.nzval[ptr] != zero(T)
                coupled = true
                break
            end
        end
        p1active[i] = coupled
    end

    @inbounds for i in 1:model.cap_p2.ntotal
        p2active[i] || continue
        col = layout.pomega2[i]
        coupled = false
        for ptr in nzrange(A, col)
            row = A.rowval[ptr]
            if (row < p2first || row > p2last) && active[row] && A.nzval[ptr] != zero(T)
                coupled = true
                break
            end
        end
        p2active[i] = coupled
    end

    @inbounds for i in 1:model.cap_p1.ntotal
        active[layout.pomega1[i]] = p1active[i]
        active[layout.pomega2[i]] = p2active[i]
    end
    return _prune_uncoupled_active!(active, A)
end

function _stokes_row_activity(model::MovingStokesModelMono{N,T}, A::SparseMatrixCSC{T,Int}) where {N,T}
    isnothing(model.cap_u_end) && throw(ArgumentError("moving model velocity end-capacity cache is not built"))
    isnothing(model.cap_p_end) && throw(ArgumentError("moving model pressure end-capacity cache is not built"))
    cap_u_end = something(model.cap_u_end)
    cap_p_end = something(model.cap_p_end)

    layout = model.layout
    active = falses(nunknowns(layout))
    @inbounds for d in 1:N
        aomega, agamma = _cell_activity_masks(cap_u_end[d])
        for i in 1:cap_u_end[d].ntotal
            active[layout.uomega[d][i]] = aomega[i]
            active[layout.ugamma[d][i]] = agamma[i]
        end
    end

    pactive = _pressure_activity(cap_p_end)
    pfirst = first(layout.pomega)
    plast = last(layout.pomega)
    @inbounds for i in 1:cap_p_end.ntotal
        pactive[i] || continue
        col = layout.pomega[i]
        coupled = false
        for ptr in nzrange(A, col)
            row = A.rowval[ptr]
            if (row < pfirst || row > plast) && active[row] && A.nzval[ptr] != zero(T)
                coupled = true
                break
            end
        end
        pactive[i] = coupled
    end
    @inbounds for i in 1:cap_p_end.ntotal
        active[layout.pomega[i]] = pactive[i]
    end
    return _prune_uncoupled_active!(active, A)
end

function _apply_row_identity_constraints!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    active_rows::BitVector,
) where {T}
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("row-identity constraints require square matrix"))
    length(b) == n || throw(ArgumentError("rhs length mismatch"))
    length(active_rows) == n || throw(ArgumentError("active row mask length mismatch"))

    p = Vector{T}(undef, n)
    @inbounds for i in 1:n
        ai = active_rows[i]
        p[i] = ai ? zero(T) : one(T)
        ai || (b[i] = zero(T))
    end

    @inbounds for j in 1:n
        aj = active_rows[j]
        for k in nzrange(A, j)
            if !(aj && active_rows[A.rowval[k]])
                A.nzval[k] = zero(T)
            end
        end
    end
    dropzeros!(A)
    return A + spdiagm(0 => p), b
end

function _enforce_dirichlet!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, row::Int, col::Int, value::T) where {T}
    @inbounds for ptr in nzrange(A, col)
        r = A.rowval[ptr]
        r == row && continue
        coeff = A.nzval[ptr]
        if coeff != zero(T)
            b[r] -= coeff * value
            A.nzval[ptr] = zero(T)
        end
    end
    @inbounds for j in 1:size(A, 2)
        A[row, j] = zero(T)
    end
    A[row, col] = one(T)
    b[row] = value
    return A, b
end

function _set_sparse_row!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    row::Int,
    cols::AbstractVector{Int},
    vals::AbstractVector{T},
    rhs::T,
) where {T}
    length(cols) == length(vals) || throw(DimensionMismatch("row values length mismatch"))
    @inbounds for j in 1:size(A, 2)
        A[row, j] = zero(T)
    end
    @inbounds for k in eachindex(cols)
        A[row, cols[k]] = vals[k]
    end
    b[row] = rhs
    return A, b
end

@inline function _iadd_term!(cols::Vector{Int}, vals::Vector{T}, col::Int, val::T) where {T}
    iszero(val) && return nothing
    @inbounds for k in eachindex(cols)
        if cols[k] == col
            vals[k] += val
            return nothing
        end
    end
    push!(cols, col)
    push!(vals, val)
    return nothing
end

function _append_scaled_terms!(
    cols::Vector{Int},
    vals::Vector{T},
    tcols::Vector{Int},
    tvals::Vector{T},
    scale::T,
) where {T}
    @inbounds for k in eachindex(tcols)
        _iadd_term!(cols, vals, tcols[k], scale * tvals[k])
    end
    return nothing
end

@inline function _is_physical_cart_index(I::CartesianIndex{N}, nnodes::NTuple{N,Int}) where {N}
    @inbounds for d in 1:N
        if I[d] < 1 || I[d] >= nnodes[d]
            return false
        end
    end
    return true
end

@inline function _shift_index(I::CartesianIndex{N}, d::Int, δ::Int) where {N}
    return CartesianIndex(ntuple(k -> (k == d ? I[k] + δ : I[k]), N))
end

@inline function _is_active_physical(
    cap::AssembledCapacity{N,T},
    I::CartesianIndex{N},
) where {N,T}
    _is_physical_cart_index(I, cap.nnodes) || return false
    li = LinearIndices(cap.nnodes)
    i = li[I]
    v = cap.buf.V[i]
    return isfinite(v) && v > zero(T)
end

function _pressure_boundary_value_terms(
    cap_p::AssembledCapacity{N,T},
    pomega::UnitRange{Int},
    I::CartesianIndex{N},
    d::Int,
    is_high::Bool,
    sign_n::Real,
) where {N,T}
    cols = Int[]
    vals = T[]
    inward = is_high ? -1 : 1
    li = LinearIndices(cap_p.nnodes)

    _is_active_physical(cap_p, I) || return cols, vals
    i1 = li[I]
    I2 = _shift_index(I, d, inward)
    if _is_active_physical(cap_p, I2)
        i2 = li[I2]
        _iadd_term!(cols, vals, pomega[i1], convert(T, -sign_n) * convert(T, 1.5))
        _iadd_term!(cols, vals, pomega[i2], convert(T, -sign_n) * convert(T, -0.5))
    else
        _iadd_term!(cols, vals, pomega[i1], convert(T, -sign_n))
    end
    return cols, vals
end

function _normal_derivative_terms_collocated(
    cap_u::AssembledCapacity{N,T},
    row_uomega::UnitRange{Int},
    I::CartesianIndex{N},
    d::Int,
    is_high::Bool,
) where {N,T}
    cols = Int[]
    vals = T[]
    li = LinearIndices(cap_u.nnodes)
    Δd = abs(cap_u.xyz[d][2] - cap_u.xyz[d][1])
    inward = is_high ? -1 : 1

    I1 = I
    I2 = _shift_index(I1, d, inward)
    I3 = _shift_index(I2, d, inward)
    if _is_active_physical(cap_u, I1) && _is_active_physical(cap_u, I2) && _is_active_physical(cap_u, I3)
        i1 = li[I1]
        i2 = li[I2]
        i3 = li[I3]
        s = inv(convert(T, 2) * Δd)
        _iadd_term!(cols, vals, row_uomega[i1], convert(T, 3) * s)
        _iadd_term!(cols, vals, row_uomega[i2], convert(T, -4) * s)
        _iadd_term!(cols, vals, row_uomega[i3], s)
        return cols, vals
    end
    if _is_active_physical(cap_u, I1) && _is_active_physical(cap_u, I2)
        i1 = li[I1]
        i2 = li[I2]
        s = inv(Δd)
        _iadd_term!(cols, vals, row_uomega[i1], s)
        _iadd_term!(cols, vals, row_uomega[i2], -s)
    end
    return cols, vals
end

function _normal_derivative_terms_halfshifted(
    cap_u::AssembledCapacity{N,T},
    row_uomega::UnitRange{Int},
    I::CartesianIndex{N},
    d::Int,
    is_high::Bool,
) where {N,T}
    cols = Int[]
    vals = T[]
    li = LinearIndices(cap_u.nnodes)
    Δd = abs(cap_u.xyz[d][2] - cap_u.xyz[d][1])
    inward = is_high ? -1 : 1

    I1 = I
    I2 = _shift_index(I1, d, inward)
    I3 = _shift_index(I2, d, inward)
    if _is_active_physical(cap_u, I1) && _is_active_physical(cap_u, I2) && _is_active_physical(cap_u, I3)
        i1 = li[I1]
        i2 = li[I2]
        i3 = li[I3]
        s = inv(Δd)
        _iadd_term!(cols, vals, row_uomega[i1], convert(T, 2) * s)
        _iadd_term!(cols, vals, row_uomega[i2], convert(T, -3) * s)
        _iadd_term!(cols, vals, row_uomega[i3], s)
        return cols, vals
    end
    if _is_active_physical(cap_u, I1) && _is_active_physical(cap_u, I2)
        i1 = li[I1]
        i2 = li[I2]
        s = inv(Δd)
        _iadd_term!(cols, vals, row_uomega[i1], s)
        _iadd_term!(cols, vals, row_uomega[i2], -s)
    end
    return cols, vals
end

function _tangential_derivative_terms_on_boundary_layer(
    cap_ud::AssembledCapacity{N,T},
    row_udomega::UnitRange{Int},
    cap_ref::AssembledCapacity{N,T},
    I_ref::CartesianIndex{N},
    d::Int,
    is_high::Bool,
    dir::Int,
    x_d::T,
) where {N,T}
    cols = Int[]
    vals = T[]
    li_ud = LinearIndices(cap_ud.nnodes)
    li_ref = LinearIndices(cap_ref.nnodes)

    iref = li_ref[I_ref]
    x0 = cap_ref.C_ω[iref][dir]

    I_base = CartesianIndex(ntuple(k -> begin
        if k == d
            is_high ? (cap_ud.nnodes[k] - 1) : 1
        else
            I_ref[k]
        end
    end, N))
    _is_physical_cart_index(I_base, cap_ud.nnodes) || return cols, vals

    idx = Int[]
    xcoord = T[]
    nphys = cap_ud.nnodes[dir] - 1
    for j in 1:nphys
        Ij = CartesianIndex(ntuple(k -> (k == dir ? j : I_base[k]), N))
        _is_active_physical(cap_ud, Ij) || continue
        ij = li_ud[Ij]
        push!(idx, j)
        push!(xcoord, cap_ud.C_ω[ij][dir])
    end

    length(idx) >= 2 || return cols, vals

    dist = abs.(xcoord .- x0)
    perm = sortperm(dist)
    nst = min(3, length(perm))
    keep = perm[1:nst]
    js = idx[keep]
    xs = xcoord[keep]

    # Keep monotone x-order for a stable tiny Vandermonde.
    p = sortperm(xs)
    js = js[p]
    xs = xs[p]

    nst_used = length(xs)
    M = Matrix{T}(undef, nst_used, nst_used)
    rhs = zeros(T, nst_used)
    rhs[min(2, nst_used)] = one(T)
    @inbounds for r in 1:nst_used
        for c in 1:nst_used
            M[r, c] = (xs[c] - x0)^(r - 1)
        end
    end
    coeff_t = M \ rhs

    # Differentiate the boundary trace u_n|_{x=x_d} tangentially.
    inward = is_high ? -1 : 1
    tol = sqrt(eps(T)) * max(one(T), abs(x_d))
    @inbounds for (j, c_t) in zip(js, coeff_t)
        I1 = CartesianIndex(ntuple(k -> (k == dir ? j : I_base[k]), N))
        _is_active_physical(cap_ud, I1) || continue
        i1 = li_ud[I1]
        x1 = cap_ud.C_ω[i1][d]
        dist = abs(x_d - x1)

        if dist <= tol
            _iadd_term!(cols, vals, row_udomega[i1], c_t)
            continue
        end

        I2 = _shift_index(I1, d, inward)
        if !_is_active_physical(cap_ud, I2)
            _iadd_term!(cols, vals, row_udomega[i1], c_t)
            continue
        end
        i2 = li_ud[I2]
        x2 = cap_ud.C_ω[i2][d]
        denom = x1 - x2
        if abs(denom) <= eps(T)
            _iadd_term!(cols, vals, row_udomega[i1], c_t)
            continue
        end
        c1 = (x_d - x2) / denom
        c2 = (x_d - x1) / (x2 - x1)
        _iadd_term!(cols, vals, row_udomega[i1], c_t * c1)
        _iadd_term!(cols, vals, row_udomega[i2], c_t * c2)
    end
    return cols, vals
end

function _traction_component(
    side_bc::AbstractBoundary,
    comp::Int,
    side::Symbol,
    xb::SVector{N,T},
    t::T,
    ::Val{N},
    ::Type{T},
) where {N,T}
    d, _, sign_n = side_info(side, N)
    if side_bc isa Traction
        τ = eval_bc(side_bc.value, xb, t)
        if τ isa Number
            return convert(T, τ)
        end
        return convert(T, τ[comp])
    elseif side_bc isa PressureOutlet
        pout = convert(T, eval_bc(side_bc.value, xb, t))
        return comp == d ? (-convert(T, sign_n) * pout) : zero(T)
    elseif side_bc isa DoNothing
        return zero(T)
    end
    throw(ArgumentError("unsupported traction boundary type $(typeof(side_bc)) on side `$side`"))
end

function _resolve_side_traction(
    side_bcs::NTuple{N,AbstractBoundary},
    side::Symbol,
    xb::SVector{N,T},
    t::T,
    ::Val{N},
    ::Type{T},
) where {N,T}
    return SVector{N,T}(ntuple(comp -> _traction_component(side_bcs[comp], comp, side, xb, t, Val(N), T), N))
end

function _apply_traction_box_bc_block!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    gridp::CartesianGrid{N,T},
    cap_p::AssembledCapacity{N,T},
    cap_u::NTuple{N,AssembledCapacity{N,T}},
    μ::T,
    bc_u::NTuple{N,BorderConditions},
    row_uomega::NTuple{N,UnitRange{Int}},
    pomega::UnitRange{Int},
    periodic::NTuple{N,Bool},
    t::T,
    locked_rows::BitVector,
) where {N,T}
    pairs = _side_pairs(N)
    for d in 1:N
        periodic[d] && continue
        side_lo, side_hi = pairs[d]
        for side in (side_lo, side_hi)
            _side_uses_traction_bc(bc_u, side, T) || continue
            side_bcs = _side_velocity_bcs(bc_u, side, T)
            _, is_high, sign_n = side_info(side, N)
            x_d = is_high ? gridp.hc[d] : gridp.lc[d]

            @inbounds for comp in 1:N
                _is_stokes_traction_bc(side_bcs[comp]) || continue

                cap_comp = cap_u[comp]
                cap_d = cap_u[d]
                cap_comp.nnodes == cap_p.nnodes ||
                    throw(ArgumentError("velocity/pressure capacities must share nnodes on traction side `$side`"))
                cap_d.nnodes == cap_comp.nnodes ||
                    throw(ArgumentError("velocity capacities must share nnodes for traction coupling"))

                li = LinearIndices(cap_comp.nnodes)
                for I in each_boundary_cell(cap_comp.nnodes, side)
                    i = li[I]
                    Vi = cap_comp.buf.V[i]
                    Aface = cap_comp.buf.A[d][i]
                    if !(isfinite(Vi) && Vi > zero(T) && isfinite(Aface) && Aface > zero(T))
                        continue
                    end

                    x = cap_comp.C_ω[i]
                    xb = SVector{N,T}(ntuple(k -> (k == d ? x_d : x[k]), N))
                    traction_vec = _resolve_side_traction(side_bcs, side, xb, t, Val(N), T)
                    rhs = traction_vec[comp]

                    cols = Int[]
                    vals = T[]
                    if comp == d
                        pcols, pvals = _pressure_boundary_value_terms(cap_p, pomega, I, d, is_high, sign_n)
                        _append_scaled_terms!(cols, vals, pcols, pvals, one(T))
                        dist = abs(x[d] - x_d)
                        tol = sqrt(eps(T)) * max(one(T), abs(x_d))
                        if dist <= tol
                            ncols, nvals = _normal_derivative_terms_collocated(cap_comp, row_uomega[comp], I, d, is_high)
                        else
                            ncols, nvals = _normal_derivative_terms_halfshifted(cap_comp, row_uomega[comp], I, d, is_high)
                        end
                        _append_scaled_terms!(cols, vals, ncols, nvals, convert(T, 2) * μ)
                    else
                        ncols, nvals = _normal_derivative_terms_halfshifted(cap_comp, row_uomega[comp], I, d, is_high)
                        _append_scaled_terms!(cols, vals, ncols, nvals, μ)
                        tcols, tvals = _tangential_derivative_terms_on_boundary_layer(
                            cap_d,
                            row_uomega[d],
                            cap_comp,
                            I,
                            d,
                            is_high,
                            comp,
                            x_d,
                        )
                        _append_scaled_terms!(cols, vals, tcols, tvals, μ)
                    end

                    isempty(cols) && continue
                    row = row_uomega[comp][i]
                    _set_sparse_row!(A, b, row, cols, vals, rhs)
                    locked_rows[row] = true
                end
            end
        end
    end

    return A, b
end

function _apply_traction_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelMono{N,T},
    t::T,
) where {N,T}
    locked_rows = falses(size(A, 1))
    _apply_traction_box_bc_block!(
        A,
        b,
        model.gridp,
        model.cap_p,
        model.cap_u,
        model.mu,
        model.bc_u,
        model.layout.uomega,
        model.layout.pomega,
        model.periodic,
        t,
        locked_rows,
    )
    return locked_rows
end

function _apply_traction_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelTwoPhase{N,T},
    t::T,
) where {N,T}
    locked_rows = falses(size(A, 1))
    _apply_traction_box_bc_block!(
        A,
        b,
        model.gridp,
        model.cap_p1,
        model.cap_u1,
        model.mu1,
        model.bc_u,
        model.layout.uomega1,
        model.layout.pomega1,
        model.periodic,
        t,
        locked_rows,
    )
    _apply_traction_box_bc_block!(
        A,
        b,
        model.gridp,
        model.cap_p2,
        model.cap_u2,
        model.mu2,
        model.bc_u,
        model.layout.uomega2,
        model.layout.pomega2,
        model.periodic,
        t,
        locked_rows,
    )
    return locked_rows
end

function _apply_traction_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelMono{N,T},
    t::T,
) where {N,T}
    isnothing(model.cap_p_end) && throw(ArgumentError("moving model pressure end-capacity cache is not built"))
    isnothing(model.cap_u_end) && throw(ArgumentError("moving model velocity end-capacity cache is not built"))
    cap_p_end = something(model.cap_p_end)
    cap_u_end = something(model.cap_u_end)
    locked_rows = falses(size(A, 1))
    _apply_traction_box_bc_block!(
        A,
        b,
        model.gridp,
        cap_p_end,
        cap_u_end,
        model.mu,
        model.bc_u,
        model.layout.uomega,
        model.layout.pomega,
        model.periodic,
        t,
        locked_rows,
    )
    return locked_rows
end

function _apply_symmetry_box_bc_block!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    gridp::CartesianGrid{N,T},
    cap_u::NTuple{N,AssembledCapacity{N,T}},
    μ::T,
    bc_u::NTuple{N,BorderConditions},
    row_uomega::NTuple{N,UnitRange{Int}},
    periodic::NTuple{N,Bool},
    t::T,
    locked_rows::BitVector,
) where {N,T}
    pairs = _side_pairs(N)
    for d in 1:N
        periodic[d] && continue
        side_lo, side_hi = pairs[d]
        for side in (side_lo, side_hi)
            _side_uses_symmetry_bc(bc_u, side, T) || continue
            side_bcs = _side_velocity_bcs(bc_u, side, T)
            all(bc -> bc isa Symmetry, side_bcs) || continue

            _, is_high, _ = side_info(side, N)
            x_d = is_high ? gridp.hc[d] : gridp.lc[d]

            @inbounds for comp in 1:N
                cap_comp = cap_u[comp]
                cap_d = cap_u[d]
                cap_d.nnodes == cap_comp.nnodes ||
                    throw(ArgumentError("velocity capacities must share nnodes for symmetry coupling"))
                li = LinearIndices(cap_comp.nnodes)
                for I in each_boundary_cell(cap_comp.nnodes, side)
                    i = li[I]
                    Vi = cap_comp.buf.V[i]
                    Aface = cap_comp.buf.A[d][i]
                    if !(isfinite(Vi) && Vi > zero(T) && isfinite(Aface) && Aface > zero(T))
                        continue
                    end

                    row = row_uomega[comp][i]
                    locked_rows[row] && continue

                    if comp == d
                        _enforce_dirichlet!(A, b, row, row_uomega[comp][i], zero(T))
                        locked_rows[row] = true
                        continue
                    end

                    cols = Int[]
                    vals = T[]
                    ncols, nvals = _normal_derivative_terms_halfshifted(cap_comp, row_uomega[comp], I, d, is_high)
                    _append_scaled_terms!(cols, vals, ncols, nvals, μ)
                    tcols, tvals = _tangential_derivative_terms_on_boundary_layer(
                        cap_d,
                        row_uomega[d],
                        cap_comp,
                        I,
                        d,
                        is_high,
                        comp,
                        x_d,
                    )
                    _append_scaled_terms!(cols, vals, tcols, tvals, μ)
                    isempty(cols) && continue

                    _set_sparse_row!(A, b, row, cols, vals, zero(T))
                    locked_rows[row] = true
                end
            end
        end
    end
    return A, b
end

function _apply_symmetry_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelMono{N,T},
    t::T,
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    locks = isnothing(locked_rows) ? falses(size(A, 1)) : locked_rows
    _apply_symmetry_box_bc_block!(
        A,
        b,
        model.gridp,
        model.cap_u,
        model.mu,
        model.bc_u,
        model.layout.uomega,
        model.periodic,
        t,
        locks,
    )
    return locks
end

function _apply_symmetry_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelTwoPhase{N,T},
    t::T,
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    locks = isnothing(locked_rows) ? falses(size(A, 1)) : locked_rows
    _apply_symmetry_box_bc_block!(
        A,
        b,
        model.gridp,
        model.cap_u1,
        model.mu1,
        model.bc_u,
        model.layout.uomega1,
        model.periodic,
        t,
        locks,
    )
    _apply_symmetry_box_bc_block!(
        A,
        b,
        model.gridp,
        model.cap_u2,
        model.mu2,
        model.bc_u,
        model.layout.uomega2,
        model.periodic,
        t,
        locks,
    )
    return locks
end

function _apply_symmetry_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelMono{N,T},
    t::T,
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    isnothing(model.cap_u_end) && throw(ArgumentError("moving model velocity end-capacity cache is not built"))
    cap_u_end = something(model.cap_u_end)
    locks = isnothing(locked_rows) ? falses(size(A, 1)) : locked_rows
    _apply_symmetry_box_bc_block!(
        A,
        b,
        model.gridp,
        cap_u_end,
        model.mu,
        model.bc_u,
        model.layout.uomega,
        model.periodic,
        t,
        locks,
    )
    return locks
end

function _apply_component_velocity_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    comp::Int,
    strong_wall_bc::Bool,
    gridp::CartesianGrid{N,T},
    cap::AssembledCapacity{N,T},
    mu::T,
    bc::BorderConditions,
    var_uomega::UnitRange{Int},
    row_uomega::UnitRange{Int},
    t::T,
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    validate_borderconditions!(bc, N)
    li = LinearIndices(cap.nnodes)
    constrained = falses(cap.ntotal)

    for (side, side_bc) in bc.borders
        side_bc isa AbstractBoundary ||
            throw(ArgumentError("velocity border condition `$side` only supports valid boundary types"))
        _is_stokes_side_vector_bc(side_bc) && continue
        d, is_high, _ = side_info(side, N)
        side_bc isa Periodic && continue

        xyz_d = cap.xyz[d]
        length(xyz_d) >= 2 || throw(ArgumentError("velocity grid must have at least 2 nodes in each direction"))
        Δd = abs(xyz_d[2] - xyz_d[1])
        δ = Δd / T(2)
        x_d = is_high ? gridp.hc[d] : gridp.lc[d]

        for I in each_boundary_cell(cap.nnodes, side)
            i = li[I]
            constrained[i] && continue
            Aface = cap.buf.A[d][i]
            if !isfinite(Aface) || iszero(Aface)
                continue
            end

            x = cap.C_ω[i]
            xb = SVector{N,T}(ntuple(k -> (k == d ? x_d : x[k]), N))
            row = row_uomega[i]
            if !isnothing(locked_rows) && locked_rows[row]
                continue
            end

            if side_bc isa Dirichlet
                val = convert(T, eval_bc(side_bc.value, xb, t))
                dist = abs(x[d] - x_d)
                tol = sqrt(eps(T)) * max(one(T), Δd)
                collocated_wall = dist <= tol
                if collocated_wall && (d == comp) && strong_wall_bc
                    _enforce_dirichlet!(A, b, row, var_uomega[i], val)
                    constrained[i] = true
                else
                    δeff = collocated_wall ? δ : dist
                    a = mu * Aface / δeff
                    A[row, var_uomega[i]] = A[row, var_uomega[i]] + a
                    b[row] += a * val
                end
            elseif side_bc isa Neumann
                g = convert(T, eval_bc(side_bc.value, xb, t))
                iszero(g) && continue
                b[row] += mu * g * Aface
            else
                throw(ArgumentError("velocity border condition `$side` only supports Dirichlet/Neumann/Periodic"))
            end
        end
    end

    return A, b
end

function _apply_velocity_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelMono{N,T},
    t::T;
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    layout = model.layout
    for d in 1:N
        _apply_component_velocity_box_bc!(
            A,
            b,
            d,
            model.strong_wall_bc,
            model.gridp,
            model.cap_u[d],
            model.mu,
            model.bc_u[d],
            layout.uomega[d],
            layout.uomega[d],
            t,
            locked_rows,
        )
    end
    return A, b
end

function _apply_velocity_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelTwoPhase{N,T},
    t::T;
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    layout = model.layout
    for d in 1:N
        _apply_component_velocity_box_bc!(
            A,
            b,
            d,
            model.strong_wall_bc,
            model.gridp,
            model.cap_u1[d],
            model.mu1,
            model.bc_u[d],
            layout.uomega1[d],
            layout.uomega1[d],
            t,
            locked_rows,
        )
        _apply_component_velocity_box_bc!(
            A,
            b,
            d,
            model.strong_wall_bc,
            model.gridp,
            model.cap_u2[d],
            model.mu2,
            model.bc_u[d],
            layout.uomega2[d],
            layout.uomega2[d],
            t,
            locked_rows,
        )
    end
    return A, b
end

function _apply_velocity_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelMono{N,T},
    t::T;
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    isnothing(model.cap_u_end) && throw(ArgumentError("moving model velocity end-capacity cache is not built"))
    cap_u_end = something(model.cap_u_end)
    layout = model.layout
    for d in 1:N
        _apply_component_velocity_box_bc!(
            A,
            b,
            d,
            model.strong_wall_bc,
            model.gridp,
            cap_u_end[d],
            model.mu,
            model.bc_u[d],
            layout.uomega[d],
            layout.uomega[d],
            t,
            locked_rows,
        )
    end
    return A, b
end

function _pressure_side_bc(bc_p::Union{Nothing,BorderConditions}, side::Symbol, ::Type{T}) where {T}
    isnothing(bc_p) && return Neumann(zero(T))
    return get(bc_p.borders, side, Neumann(zero(T)))
end

function _pressure_neumann_rhs(
    gridp::CartesianGrid{N,T},
    side::Symbol,
    side_bc::Neumann,
    x::SVector{N,T},
    t::T,
) where {N,T}
    d, _, sign_n = side_info(side, N)
    x_d = sign_n > 0 ? gridp.hc[d] : gridp.lc[d]
    xb = SVector{N,T}(ntuple(k -> (k == d ? x_d : x[k]), N))
    g_n = convert(T, eval_bc(side_bc.value, xb, t))
    return convert(T, sign_n) * g_n
end

function _apply_pressure_box_bc_block!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    gridp::CartesianGrid{N,T},
    cap::AssembledCapacity{N,T},
    pomega::UnitRange{Int},
    periodic::NTuple{N,Bool},
    bc_u::NTuple{N,BorderConditions},
    bc_p::BorderConditions,
    t::T,
) where {N,T}
    pairs = _side_pairs(N)
    li = LinearIndices(cap.nnodes)
    constrained = falses(cap.ntotal)

    @inbounds for d in 1:N
        periodic[d] && continue
        side_lo, side_hi = pairs[d]
        for side in (side_lo, side_hi)
            _side_uses_vector_bc(bc_u, side, T) && continue
            side_bc = _pressure_side_bc(bc_p, side, T)
            side_bc isa Periodic && continue
            Δd = abs(cap.xyz[d][2] - cap.xyz[d][1])
            for I in each_boundary_cell(cap.nnodes, side)
                i = li[I]
                constrained[i] && continue
                V = cap.buf.V[i]
                if !(isfinite(V) && V > zero(T))
                    continue
                end
                row = pomega[i]
                if side_bc isa Dirichlet
                    x = cap.C_ω[i]
                    val = convert(T, eval_bc(side_bc.value, x, t))
                    _set_sparse_row!(
                        A,
                        b,
                        row,
                        Int[pomega[i]],
                        T[one(T)],
                        val,
                    )
                    constrained[i] = true
                elseif side_bc isa Neumann
                    x = cap.C_ω[i]
                    g = _pressure_neumann_rhs(gridp, side, side_bc, x, t)
                    is_high = side in (:right, :top, :forward)
                    I2 = CartesianIndex(ntuple(k -> k == d ? (is_high ? I[k] - 1 : I[k] + 1) : I[k], N))
                    I3 = CartesianIndex(ntuple(k -> k == d ? (is_high ? I[k] - 2 : I[k] + 2) : I[k], N))
                    use_second = true
                    @inbounds for k in 1:N
                        if I2[k] < 1 || I2[k] > cap.nnodes[k] || I3[k] < 1 || I3[k] > cap.nnodes[k]
                            use_second = false
                            break
                        end
                    end
                    if use_second
                        i2 = li[I2]
                        i3 = li[I3]
                        if is_high
                            coeff = T[2, -3, 1]
                        else
                            coeff = T[-2, 3, -1]
                        end
                        cols = Int[pomega[i], pomega[i2], pomega[i3]]
                        _set_sparse_row!(A, b, row, cols, coeff, Δd * g)
                    else
                        i2 = li[I2]
                        if is_high
                            coeff = T[1, -1]
                        else
                            coeff = T[-1, 1]
                        end
                        cols = Int[pomega[i], pomega[i2]]
                        _set_sparse_row!(A, b, row, cols, coeff, Δd * g)
                    end
                    constrained[i] = true
                else
                    throw(ArgumentError("pressure border condition `$side` only supports Dirichlet/Neumann/Periodic"))
                end
            end
        end
    end
    return A, b
end

function _apply_pressure_box_bc!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelMono{N,T}, t::T) where {N,T}
    model.strong_wall_bc || return A, b
    isnothing(model.bc_p) && return A, b
    return _apply_pressure_box_bc_block!(
        A, b, model.gridp, model.cap_p, model.layout.pomega, model.periodic, model.bc_u, model.bc_p, t
    )
end

function _apply_pressure_box_bc!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelTwoPhase{N,T}, t::T) where {N,T}
    model.strong_wall_bc || return A, b
    isnothing(model.bc_p) && return A, b
    _apply_pressure_box_bc_block!(
        A, b, model.gridp, model.cap_p1, model.layout.pomega1, model.periodic, model.bc_u, model.bc_p, t
    )
    _apply_pressure_box_bc_block!(
        A, b, model.gridp, model.cap_p2, model.layout.pomega2, model.periodic, model.bc_u, model.bc_p, t
    )
    return A, b
end

function _apply_pressure_box_bc!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::MovingStokesModelMono{N,T}, t::T) where {N,T}
    model.strong_wall_bc || return A, b
    isnothing(model.bc_p) && return A, b
    isnothing(model.cap_p_end) && throw(ArgumentError("moving model pressure end-capacity cache is not built"))
    cap_p_end = something(model.cap_p_end)
    return _apply_pressure_box_bc_block!(
        A, b, model.gridp, cap_p_end, model.layout.pomega, model.periodic, model.bc_u, model.bc_p, t
    )
end

function _first_active_pressure_index(model::StokesModelMono{N,T}) where {N,T}
    pactive = _pressure_activity(model.cap_p)
    idx = findfirst(pactive)
    return isnothing(idx) ? 1 : idx
end

function _first_active_pressure_index(cap::AssembledCapacity{N,T}) where {N,T}
    pactive = _pressure_activity(cap)
    idx = findfirst(pactive)
    return isnothing(idx) ? 1 : idx
end

function _first_coupled_pressure_index(
    A::SparseMatrixCSC{T,Int},
    pomega::UnitRange{Int},
    pactive::BitVector,
) where {T}
    pset = pomega
    @inbounds for local_i in 1:length(pomega)
        pactive[local_i] || continue
        col = pomega[local_i]
        coupled = false
        for ptr in nzrange(A, col)
            row = A.rowval[ptr]
            if row < first(pset) || row > last(pset)
                if A.nzval[ptr] != zero(T)
                    coupled = true
                    break
                end
            end
        end
        coupled && return local_i
    end
    return nothing
end

function _default_pressure_gauge_index(
    cap::AssembledCapacity{N,T},
    A::SparseMatrixCSC{T,Int},
    pomega::UnitRange{Int},
) where {N,T}
    pactive = _pressure_activity(cap)
    pinterior = _pressure_interior_activity(cap)
    pcand = pactive .& pinterior

    coupled = _first_coupled_pressure_index(A, pomega, pcand)
    !isnothing(coupled) && return coupled

    coupled = _first_coupled_pressure_index(A, pomega, pactive)
    !isnothing(coupled) && return coupled

    idx = findfirst(pcand)
    !isnothing(idx) && return idx
    return _first_active_pressure_index(cap)
end

function _apply_pin_pressure_gauge!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    pomega::UnitRange{Int},
    cap::AssembledCapacity{N,T},
    gauge::PinPressureGauge,
) where {N,T}
    nt = cap.ntotal
    idx = isnothing(gauge.index) ? _default_pressure_gauge_index(cap, A, pomega) : gauge.index
    1 <= idx <= nt || throw(ArgumentError("pressure pin index must be in 1:$nt"))
    row = pomega[idx]
    col = pomega[idx]
    _enforce_dirichlet!(A, b, row, col, zero(T))
    return A, b
end

function _apply_mean_pressure_gauge!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    pomega::UnitRange{Int},
    cap::AssembledCapacity{N,T},
) where {N,T}
    idx0 = _default_pressure_gauge_index(cap, A, pomega)
    row = pomega[idx0]

    @inbounds for j in 1:size(A, 2)
        A[row, j] = zero(T)
    end

    pactive = _pressure_activity(cap)
    active_idx = findall(pactive)
    if isempty(active_idx)
        A[row, pomega[idx0]] = one(T)
        b[row] = zero(T)
        return A, b
    end

    vols = Vector{T}(undef, length(active_idx))
    s = zero(T)
    @inbounds for k in eachindex(active_idx)
        v = cap.buf.V[active_idx[k]]
        vk = (isfinite(v) && v > zero(T)) ? v : zero(T)
        vols[k] = vk
        s += vk
    end

    if !(isfinite(s) && s > zero(T))
        w = inv(convert(T, length(active_idx)))
        @inbounds for i in active_idx
            A[row, pomega[i]] = w
        end
    else
        @inbounds for (k, i) in enumerate(active_idx)
            A[row, pomega[i]] = vols[k] / s
        end
    end

    b[row] = zero(T)
    return A, b
end

function _apply_pressure_gauge!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelMono{N,T}) where {N,T}
    layout = model.layout

    if model.gauge isa PinPressureGauge
        return _apply_pin_pressure_gauge!(A, b, layout.pomega, model.cap_p, model.gauge)
    elseif model.gauge isa MeanPressureGauge
        return _apply_mean_pressure_gauge!(A, b, layout.pomega, model.cap_p)
    end

    throw(ArgumentError("unsupported pressure gauge type $(typeof(model.gauge))"))
end

function _apply_pressure_gauge!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelTwoPhase{N,T}) where {N,T}
    layout = model.layout

    if model.gauge isa PinPressureGauge
        return _apply_pin_pressure_gauge!(A, b, layout.pomega1, model.cap_p1, model.gauge)
    elseif model.gauge isa MeanPressureGauge
        return _apply_mean_pressure_gauge!(A, b, layout.pomega1, model.cap_p1)
    end

    throw(ArgumentError("unsupported pressure gauge type $(typeof(model.gauge))"))
end

function _apply_pressure_gauge!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::MovingStokesModelMono{N,T}) where {N,T}
    isnothing(model.cap_p_end) && throw(ArgumentError("moving model pressure end-capacity cache is not built"))
    cap_p_end = something(model.cap_p_end)

    layout = model.layout

    if model.gauge isa PinPressureGauge
        return _apply_pin_pressure_gauge!(A, b, layout.pomega, cap_p_end, model.gauge)
    elseif model.gauge isa MeanPressureGauge
        return _apply_mean_pressure_gauge!(A, b, layout.pomega, cap_p_end)
    end

    throw(ArgumentError("unsupported pressure gauge type $(typeof(model.gauge))"))
end

function _theta_from_scheme(::Type{T}, scheme) where {T}
    if scheme isa Symbol
        if scheme === :BE
            return one(T)
        elseif scheme === :CN
            return convert(T, 0.5)
        end
        throw(ArgumentError("unknown scheme `$scheme`; expected :BE or :CN"))
    elseif scheme isa Real
        theta = convert(T, scheme)
        zero(T) <= theta <= one(T) || throw(ArgumentError("theta must be in [0,1]"))
        return theta
    end
    throw(ArgumentError("scheme must be Symbol (:BE/:CN) or numeric theta"))
end

psip_cn(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 0.5
psim_cn(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 0.5
psip_be(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 1.0
psim_be(Vn, Vn1) = 0.0

function _psi_functions(scheme)
    if scheme isa Symbol
        if scheme === :CN
            return psip_cn, psim_cn
        elseif scheme === :BE
            return psip_be, psim_be
        end
    elseif scheme isa Real
        θ = Float64(scheme)
        psip = (Vn, Vn1) -> (iszero(Vn) && iszero(Vn1) ? 0.0 : θ)
        psim = (Vn, Vn1) -> (iszero(Vn) && iszero(Vn1) ? 0.0 : (1.0 - θ))
        return psip, psim
    end
    throw(ArgumentError("moving scheme must be :BE, :CN, or numeric θ"))
end

function _eval_levelset_time(body, x::SVector{N,T}, t::T) where {N,T}
    if applicable(body, x..., t)
        return convert(T, body(x..., t))
    elseif applicable(body, x...)
        return convert(T, body(x...))
    end
    throw(ArgumentError("level-set callback must accept (x...) or (x..., t)"))
end

function _space_moments_at_time(
    model::MovingStokesModelMono{N,T},
    xyz_space::NTuple{N,AbstractVector{T}},
    t::T,
) where {N,T}
    body_t = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    return geometric_moments(body_t, xyz_space, T, nan; method=model.geom_method)
end

function _slice_spacetime_to_space(
    vec_st::AbstractVector,
    nn_space::NTuple{N,Int},
    nt::Int,
    it::Int,
) where {N}
    dims_st = (nn_space..., nt)
    li_st = LinearIndices(dims_st)
    li_sp = LinearIndices(nn_space)
    out = similar(vec_st, prod(nn_space))
    @inbounds for I in CartesianIndices(nn_space)
        out[li_sp[I]] = vec_st[li_st[Tuple(I)..., it]]
    end
    return out
end

function reduce_slab_to_space(
    m_st::GeometricMoments{N1,T},
    nn_space::NTuple{N,Int},
) where {N1,N,T}
    N1 == N + 1 || throw(ArgumentError("expected slab moments dimension $(N + 1), got $N1"))
    nt = length(m_st.xyz[N1])
    nt == 2 || throw(ArgumentError("space-time reduction expects 2 time nodes, got $nt"))

    V = _slice_spacetime_to_space(m_st.V, nn_space, nt, 1)
    Γ = _slice_spacetime_to_space(m_st.interface_measure, nn_space, nt, 1)
    ctype = _slice_spacetime_to_space(m_st.cell_type, nn_space, nt, 1)
    A = ntuple(d -> _slice_spacetime_to_space(m_st.A[d], nn_space, nt, 1), N)
    B = ntuple(d -> _slice_spacetime_to_space(m_st.B[d], nn_space, nt, 1), N)
    W = ntuple(d -> _slice_spacetime_to_space(m_st.W[d], nn_space, nt, 1), N)

    bary_st = _slice_spacetime_to_space(m_st.barycenter, nn_space, nt, 1)
    baryγ_st = _slice_spacetime_to_space(m_st.barycenter_interface, nn_space, nt, 1)
    nγ_st = _slice_spacetime_to_space(m_st.interface_normal, nn_space, nt, 1)

    bary = Vector{SVector{N,T}}(undef, length(V))
    baryγ = Vector{SVector{N,T}}(undef, length(V))
    nγ = Vector{SVector{N,T}}(undef, length(V))
    @inbounds for i in eachindex(V)
        bi = bary_st[i]
        bγi = baryγ_st[i]
        ni = nγ_st[i]
        bary[i] = SVector{N,T}(ntuple(d -> bi[d], N))
        baryγ[i] = SVector{N,T}(ntuple(d -> bγi[d], N))
        nγ[i] = SVector{N,T}(ntuple(d -> ni[d], N))
    end

    xyz = ntuple(d -> collect(T, m_st.xyz[d]), N)
    return GeometricMoments(V, bary, Γ, ctype, baryγ, nγ, A, B, W, xyz)
end

function _build_moving_slab!(
    model::MovingStokesModelMono{N,T},
    t::T,
    dt::T,
) where {N,T}
    xyz_p = grid1d(model.gridp)
    body_n = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    body_n1 = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t + dt)
    moms_p_n1 = _space_moments_at_time(model, xyz_p, t + dt)

    stgrid_p = SpaceTimeCartesianGrid(model.gridp, T[t, t + dt])
    xyz_st_p = grid1d(stgrid_p)
    body_st = (x...) -> begin
        xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
        _eval_levelset_time(model.body, xs, convert(T, x[N + 1]))
    end
    moms_p_st = geometric_moments(body_st, xyz_st_p, T, nan; method=model.geom_method)
    moms_p_slab = reduce_slab_to_space(moms_p_st, model.gridp.n)
    cap_p_slab = assembled_capacity(moms_p_slab; bc=zero(T))
    op_p_slab = DiffusionOps(cap_p_slab; periodic=model.periodic)
    cap_p_end = assembled_capacity(moms_p_n1; bc=zero(T))
    op_p_end = DiffusionOps(cap_p_end; periodic=model.periodic)

    ugeom = ntuple(d -> begin
        xyz_u = grid1d(model.gridu[d])
        moms_u_n = geometric_moments(body_n, xyz_u, T, nan; method=model.geom_method)
        moms_u_n1 = geometric_moments(body_n1, xyz_u, T, nan; method=model.geom_method)
        stgrid_u = SpaceTimeCartesianGrid(model.gridu[d], T[t, t + dt])
        xyz_st_u = grid1d(stgrid_u)
        moms_u_st = geometric_moments(body_st, xyz_st_u, T, nan; method=model.geom_method)
        moms_u_slab = reduce_slab_to_space(moms_u_st, model.gridu[d].n)
        (
            cap_slab=assembled_capacity(moms_u_slab; bc=zero(T)),
            cap_end=assembled_capacity(moms_u_n1; bc=zero(T)),
            Vn=Vector{T}(moms_u_n.V),
            Vn1=Vector{T}(moms_u_n1.V),
        )
    end, N)
    cap_u_slab = ntuple(d -> ugeom[d].cap_slab, N)
    op_u_slab = ntuple(d -> DiffusionOps(cap_u_slab[d]; periodic=model.periodic), N)
    cap_u_end = ntuple(d -> ugeom[d].cap_end, N)
    Vun = ntuple(d -> ugeom[d].Vn, N)
    Vun1 = ntuple(d -> ugeom[d].Vn1, N)

    model.cap_p_slab = cap_p_slab
    model.op_p_slab = op_p_slab
    model.cap_p_end = cap_p_end
    model.op_p_end = op_p_end
    model.cap_u_slab = cap_u_slab
    model.op_u_slab = op_u_slab
    model.cap_u_end = cap_u_end
    model.Vun = Vun
    model.Vun1 = Vun1
    return model
end

function _expand_prev_state(model::StokesModelMono{N,T}, x_prev::AbstractVector) where {N,T}
    nsys = nunknowns(model.layout)
    nt = model.layout.nt
    if length(x_prev) == nsys
        return Vector{T}(x_prev)
    elseif length(x_prev) == N * nt
        x = zeros(T, nsys)
        @inbounds for d in 1:N
            src = ((d - 1) * nt + 1):(d * nt)
            x[model.layout.uomega[d]] .= x_prev[src]
        end
        return x
    end
    throw(DimensionMismatch("x_prev length must be $nsys (full state) or $(N * nt) (uomega blocks)"))
end

function _expand_prev_state(model::StokesModelTwoPhase{N,T}, x_prev::AbstractVector) where {N,T}
    nsys = nunknowns(model.layout)
    length(x_prev) == nsys ||
        throw(DimensionMismatch("x_prev length must be $nsys for two-phase full state"))
    return Vector{T}(x_prev)
end

function _expand_prev_state(model::MovingStokesModelMono{N,T}, x_prev::AbstractVector) where {N,T}
    nsys = nunknowns(model.layout)
    nt = model.layout.nt
    if length(x_prev) == nsys
        return Vector{T}(x_prev)
    elseif length(x_prev) == N * nt
        x = zeros(T, nsys)
        @inbounds for d in 1:N
            src = ((d - 1) * nt + 1):(d * nt)
            x[model.layout.uomega[d]] .= x_prev[src]
        end
        return x
    end
    throw(DimensionMismatch("x_prev length must be $nsys (full state) or $(N * nt) (uomega blocks)"))
end

function _stokes_blocks(model::StokesModelMono{N,T}) where {N,T}
    nt = model.cap_p.ntotal

    grad_full = model.op_p.G + model.op_p.H
    size(grad_full, 1) == N * nt ||
        throw(ArgumentError("pressure gradient rows ($(size(grad_full, 1))) must equal N*nt ($(N * nt))"))

    visc_omega = ntuple(d -> begin
        opu = model.op_u[d]
        model.mu * (opu.G' * (opu.Winv * opu.G))
    end, N)

    visc_gamma = ntuple(d -> begin
        opu = model.op_u[d]
        model.mu * (opu.G' * (opu.Winv * opu.H))
    end, N)

    grad = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full[rows, :])
        if !model.periodic[d]
            capd = model.cap_u[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    div_omega = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = model.op_p.G[rows, :]
        hp = model.op_p.H[rows, :]
        -(gp' + hp')
    end, N)

    div_gamma = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = model.op_p.H[rows, :]
        sparse(hp')
    end, N)

    mass = ntuple(d -> model.rho .* model.cap_u[d].buf.V, N)

    return (; nt, visc_omega, visc_gamma, grad, div_omega, div_gamma, mass)
end

function _assemble_core!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelMono{N,T}, blocks, t::T) where {N,T}
    nt = blocks.nt
    layout = model.layout

    @inbounds for d in 1:N
        _insert_block!(A, layout.uomega[d], layout.uomega[d], blocks.visc_omega[d])
        _insert_block!(A, layout.uomega[d], layout.ugamma[d], blocks.visc_gamma[d])
        _insert_block!(A, layout.uomega[d], layout.pomega, blocks.grad[d])

        tie = spdiagm(0 => ones(T, nt))
        _insert_block!(A, layout.ugamma[d], layout.ugamma[d], tie)

        force_vec = _force_values(model, d, t)
        _insert_vec!(b, layout.uomega[d], model.cap_u[d].V * force_vec)

        cut_vec = _cut_values(model.cap_u[d], model.bc_cut[d], t)
        _insert_vec!(b, layout.ugamma[d], cut_vec)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], blocks.div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], blocks.div_gamma[d])
    end

    return A, b
end

function _stokes_phase_blocks(
    cap_p::AssembledCapacity{N,T},
    op_p::DiffusionOps{N,T},
    cap_u::NTuple{N,AssembledCapacity{N,T}},
    op_u::NTuple{N,DiffusionOps{N,T}},
    mu::T,
    rho::T,
    periodic::NTuple{N,Bool},
) where {N,T}
    nt = cap_p.ntotal

    grad_full = op_p.G + op_p.H
    size(grad_full, 1) == N * nt ||
        throw(ArgumentError("pressure gradient rows ($(size(grad_full, 1))) must equal N*nt ($(N * nt))"))

    visc_omega = ntuple(d -> begin
        opud = op_u[d]
        mu * (opud.G' * (opud.Winv * opud.G))
    end, N)

    visc_gamma = ntuple(d -> begin
        opud = op_u[d]
        mu * (opud.G' * (opud.Winv * opud.H))
    end, N)

    grad = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full[rows, :])
        if !periodic[d]
            capd = cap_u[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    div_omega = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = op_p.G[rows, :]
        hp = op_p.H[rows, :]
        -(gp' + hp')
    end, N)

    div_gamma = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = op_p.H[rows, :]
        sparse(hp')
    end, N)

    deriv_omega = ntuple(β -> begin
        gω = op_u[β].Winv * op_u[β].G
        ntuple(b -> begin
            rows = ((b - 1) * nt + 1):(b * nt)
            sparse(gω[rows, :])
        end, N)
    end, N)

    deriv_gamma = ntuple(β -> begin
        gγ = op_u[β].Winv * op_u[β].H
        ntuple(b -> begin
            rows = ((b - 1) * nt + 1):(b * nt)
            sparse(gγ[rows, :])
        end, N)
    end, N)

    gamma = zeros(T, nt)
    normals = ntuple(_ -> zeros(T, nt), N)
    @inbounds for i in 1:nt
        Γi = cap_p.buf.Γ[i]
        if !(isfinite(Γi) && Γi > zero(T))
            continue
        end
        gamma[i] = Γi
        nγ = cap_p.n_γ[i]
        for d in 1:N
            nd = nγ[d]
            normals[d][i] = isfinite(nd) ? nd : zero(T)
        end
    end

    mass = ntuple(d -> rho .* cap_u[d].buf.V, N)

    return (
        nt=nt,
        visc_omega=visc_omega,
        visc_gamma=visc_gamma,
        grad=grad,
        div_omega=div_omega,
        div_gamma=div_gamma,
        deriv_omega=deriv_omega,
        deriv_gamma=deriv_gamma,
        gamma=gamma,
        normals=normals,
        mass=mass,
    )
end

function _stokes_blocks(model::StokesModelTwoPhase{N,T}) where {N,T}
    phase1 = _stokes_phase_blocks(
        model.cap_p1,
        model.op_p1,
        model.cap_u1,
        model.op_u1,
        model.mu1,
        model.rho1,
        model.periodic,
    )
    phase2 = _stokes_phase_blocks(
        model.cap_p2,
        model.op_p2,
        model.cap_u2,
        model.op_u2,
        model.mu2,
        model.rho2,
        model.periodic,
    )
    return (; nt=model.cap_p1.ntotal, phase1, phase2)
end

function _interface_condition_values(
    model::StokesModelTwoPhase{N,T},
    d::Int,
    gamma::AbstractVector{T},
    t::T,
) where {N,T}
    nt = length(gamma)
    α1 = ones(T, nt)
    α2 = ones(T, nt)
    β1 = ones(T, nt)
    β2 = ones(T, nt)
    rhs_jump = zeros(T, nt)
    rhs_trac = zeros(T, nt)

    ic = isnothing(model.bc_interface) ? nothing : model.bc_interface[d]
    sbc = isnothing(ic) ? nothing : ic.scalar
    fbc = isnothing(ic) ? nothing : ic.flux
    if !(sbc === nothing) && !(sbc isa ScalarJump)
        throw(ArgumentError("two-phase Stokes interface scalar condition for component $d must be ScalarJump or nothing"))
    end
    if !(fbc === nothing) && !(fbc isa FluxJump)
        throw(ArgumentError("two-phase Stokes interface flux condition for component $d must be FluxJump or nothing"))
    end

    @inbounds for i in 1:nt
        Γi = gamma[i]
        if !(isfinite(Γi) && Γi > zero(T))
            continue
        end
        xi = model.cap_p1.C_γ[i]
        if sbc === nothing
            rhs_jump[i] = _interface_force_component(model.interface_jump, d, xi, t)
        else
            sb = sbc::ScalarJump
            α1[i] = convert(T, eval_bc(sb.α₁, xi, t))
            α2[i] = convert(T, eval_bc(sb.α₂, xi, t))
            rhs_jump[i] = convert(T, eval_bc(sb.value, xi, t))
        end
        if fbc === nothing
            rhs_trac[i] = Γi * _interface_force_component(model.interface_force, d, xi, t)
        else
            fb = fbc::FluxJump
            β1[i] = convert(T, eval_bc(fb.β₁, xi, t))
            β2[i] = convert(T, eval_bc(fb.β₂, xi, t))
            rhs_trac[i] = Γi * convert(T, eval_bc(fb.value, xi, t))
        end
    end
    return α1, α2, β1, β2, rhs_jump, rhs_trac
end

function _assemble_interface_traction_rows!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::StokesModelTwoPhase{N,T},
    blocks,
    t::T,
) where {N,T}
    layout = model.layout
    phase1 = blocks.phase1
    phase2 = blocks.phase2

    @inbounds for α in 1:N
        rows_trac = layout.ugamma1[α]
        rows_jump = layout.ugamma2[α]
        α1, α2, β1, β2, rhs_jump, rhs_trac = _interface_condition_values(model, α, phase1.gamma, t)

        Tp1 = spdiagm(0 => -(β1 .* phase1.gamma .* phase1.normals[α]))
        Tp2 = spdiagm(0 => -(β2 .* phase2.gamma .* phase2.normals[α]))
        _insert_block!(A, rows_trac, layout.pomega1, Tp1)
        _insert_block!(A, rows_trac, layout.pomega2, Tp2)

        for β in 1:N
            w1 = model.mu1 .* (β1 .* phase1.gamma .* phase1.normals[β])
            D1 = spdiagm(0 => w1)
            _insert_block!(A, rows_trac, layout.uomega1[α], sparse(D1 * phase1.deriv_omega[α][β]))
            _insert_block!(A, rows_trac, layout.uomega1[β], sparse(D1 * phase1.deriv_omega[β][α]))
            _insert_block!(A, rows_trac, layout.ugamma1[α], sparse(D1 * phase1.deriv_gamma[α][β]))
            _insert_block!(A, rows_trac, layout.ugamma1[β], sparse(D1 * phase1.deriv_gamma[β][α]))

            w2 = model.mu2 .* (β2 .* phase2.gamma .* phase2.normals[β])
            D2 = spdiagm(0 => w2)
            _insert_block!(A, rows_trac, layout.uomega2[α], sparse(D2 * phase2.deriv_omega[α][β]))
            _insert_block!(A, rows_trac, layout.uomega2[β], sparse(D2 * phase2.deriv_omega[β][α]))
            _insert_block!(A, rows_trac, layout.ugamma2[α], sparse(D2 * phase2.deriv_gamma[α][β]))
            _insert_block!(A, rows_trac, layout.ugamma2[β], sparse(D2 * phase2.deriv_gamma[β][α]))
        end

        _insert_vec!(b, rows_trac, rhs_trac)

        _insert_block!(A, rows_jump, layout.ugamma1[α], spdiagm(0 => α1))
        _insert_block!(A, rows_jump, layout.ugamma2[α], spdiagm(0 => -α2))
        _insert_vec!(b, rows_jump, rhs_jump)
    end

    return A, b
end

function _assemble_core!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelTwoPhase{N,T}, blocks, t::T) where {N,T}
    layout = model.layout
    phase1 = blocks.phase1
    phase2 = blocks.phase2

    @inbounds for d in 1:N
        _insert_block!(A, layout.uomega1[d], layout.uomega1[d], phase1.visc_omega[d])
        _insert_block!(A, layout.uomega1[d], layout.ugamma1[d], phase1.visc_gamma[d])
        _insert_block!(A, layout.uomega1[d], layout.pomega1, phase1.grad[d])
        f1 = _force_values(model, 1, d, t)
        _insert_vec!(b, layout.uomega1[d], model.cap_u1[d].V * f1)

        _insert_block!(A, layout.uomega2[d], layout.uomega2[d], phase2.visc_omega[d])
        _insert_block!(A, layout.uomega2[d], layout.ugamma2[d], phase2.visc_gamma[d])
        _insert_block!(A, layout.uomega2[d], layout.pomega2, phase2.grad[d])
        f2 = _force_values(model, 2, d, t)
        _insert_vec!(b, layout.uomega2[d], model.cap_u2[d].V * f2)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega1, layout.uomega1[d], phase1.div_omega[d])
        _insert_block!(A, layout.pomega1, layout.ugamma1[d], phase1.div_gamma[d])
        _insert_block!(A, layout.pomega2, layout.uomega2[d], phase2.div_omega[d])
        _insert_block!(A, layout.pomega2, layout.ugamma2[d], phase2.div_gamma[d])
    end

    _assemble_interface_traction_rows!(A, b, model, blocks, t)
    return A, b
end

"""
    assemble_steady!(sys, model, t=0)

Assemble steady Stokes linear system into `sys` for `StokesModelMono` or
`StokesModelTwoPhase` at time `t`.

Mutates `sys.A` and `sys.b` in place.
"""
function assemble_steady!(sys::LinearSystem{T}, model::StokesModelTwoPhase{N,T}, t::T=zero(T)) where {N,T}
    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    _assemble_core!(A, b, model, blocks, t)
    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t)
    _apply_symmetry_box_bc!(A, b, model, t, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

"""
    assemble_unsteady!(sys, model, x_prev, t, dt; scheme=:BE)

Assemble one unsteady theta-step system for `StokesModelMono` or
`StokesModelTwoPhase`.

`scheme` supports `:BE`, `:CN`, or numeric `theta ∈ [0,1]`.
"""
function assemble_unsteady!(
    sys::LinearSystem{T},
    model::StokesModelTwoPhase{N,T},
    x_prev::AbstractVector,
    t::T,
    dt::T;
    scheme=:BE,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    theta = _theta_from_scheme(T, scheme)
    t_next = t + dt

    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)
    layout = model.layout
    xfull_prev = _expand_prev_state(model, x_prev)

    phase1 = blocks.phase1
    phase2 = blocks.phase2

    @inbounds for d in 1:N
        mdt1 = phase1.mass[d] ./ dt
        block11 = theta .* phase1.visc_omega[d] + spdiagm(0 => mdt1)
        block1g = theta .* phase1.visc_gamma[d]
        _insert_block!(A, layout.uomega1[d], layout.uomega1[d], block11)
        _insert_block!(A, layout.uomega1[d], layout.ugamma1[d], block1g)
        _insert_block!(A, layout.uomega1[d], layout.pomega1, phase1.grad[d])

        u1_prev = Vector{T}(xfull_prev[layout.uomega1[d]])
        ug1_prev = Vector{T}(xfull_prev[layout.ugamma1[d]])
        rhs1 = mdt1 .* u1_prev
        if theta != one(T)
            rhs1 .-= (one(T) - theta) .* (phase1.visc_omega[d] * u1_prev + phase1.visc_gamma[d] * ug1_prev)
        end
        f1_prev = _force_values(model, 1, d, t)
        f1_next = _force_values(model, 1, d, t_next)
        f1_theta = theta .* f1_next .+ (one(T) - theta) .* f1_prev
        rhs1 .+= model.cap_u1[d].V * f1_theta
        _insert_vec!(b, layout.uomega1[d], rhs1)

        mdt2 = phase2.mass[d] ./ dt
        block22 = theta .* phase2.visc_omega[d] + spdiagm(0 => mdt2)
        block2g = theta .* phase2.visc_gamma[d]
        _insert_block!(A, layout.uomega2[d], layout.uomega2[d], block22)
        _insert_block!(A, layout.uomega2[d], layout.ugamma2[d], block2g)
        _insert_block!(A, layout.uomega2[d], layout.pomega2, phase2.grad[d])

        u2_prev = Vector{T}(xfull_prev[layout.uomega2[d]])
        ug2_prev = Vector{T}(xfull_prev[layout.ugamma2[d]])
        rhs2 = mdt2 .* u2_prev
        if theta != one(T)
            rhs2 .-= (one(T) - theta) .* (phase2.visc_omega[d] * u2_prev + phase2.visc_gamma[d] * ug2_prev)
        end
        f2_prev = _force_values(model, 2, d, t)
        f2_next = _force_values(model, 2, d, t_next)
        f2_theta = theta .* f2_next .+ (one(T) - theta) .* f2_prev
        rhs2 .+= model.cap_u2[d].V * f2_theta
        _insert_vec!(b, layout.uomega2[d], rhs2)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega1, layout.uomega1[d], phase1.div_omega[d])
        _insert_block!(A, layout.pomega1, layout.ugamma1[d], phase1.div_gamma[d])
        _insert_block!(A, layout.pomega2, layout.uomega2[d], phase2.div_omega[d])
        _insert_block!(A, layout.pomega2, layout.ugamma2[d], phase2.div_gamma[d])
    end

    _assemble_interface_traction_rows!(A, b, model, blocks, t_next)
    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t_next)
    _apply_symmetry_box_bc!(A, b, model, t_next, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t_next; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

function assemble_steady!(sys::LinearSystem{T}, model::StokesModelMono{N,T}, t::T=zero(T)) where {N,T}
    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    _assemble_core!(A, b, model, blocks, t)
    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t)
    _apply_symmetry_box_bc!(A, b, model, t, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

function assemble_unsteady!(
    sys::LinearSystem{T},
    model::StokesModelMono{N,T},
    x_prev::AbstractVector,
    t::T,
    dt::T;
    scheme=:BE,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    theta = _theta_from_scheme(T, scheme)
    t_next = t + dt

    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    layout = model.layout
    nt = blocks.nt
    xfull_prev = _expand_prev_state(model, x_prev)

    @inbounds for d in 1:N
        mdt = blocks.mass[d] ./ dt
        block_oo = theta * blocks.visc_omega[d] + spdiagm(0 => mdt)
        block_og = theta * blocks.visc_gamma[d]

        _insert_block!(A, layout.uomega[d], layout.uomega[d], block_oo)
        _insert_block!(A, layout.uomega[d], layout.ugamma[d], block_og)
        _insert_block!(A, layout.uomega[d], layout.pomega, blocks.grad[d])

        tie = spdiagm(0 => ones(T, nt))
        _insert_block!(A, layout.ugamma[d], layout.ugamma[d], tie)

        uo_prev = Vector{T}(xfull_prev[layout.uomega[d]])
        ug_prev = Vector{T}(xfull_prev[layout.ugamma[d]])

        rhs = mdt .* uo_prev
        if theta != one(T)
            rhs .-= (one(T) - theta) .* (blocks.visc_omega[d] * uo_prev + blocks.visc_gamma[d] * ug_prev)
        end

        f_prev = _force_values(model, d, t)
        f_next = _force_values(model, d, t_next)
        ftheta = theta .* f_next .+ (one(T) - theta) .* f_prev
        rhs .+= model.cap_u[d].V * ftheta
        _insert_vec!(b, layout.uomega[d], rhs)

        cut_next = _cut_values(model.cap_u[d], model.bc_cut[d], t_next)
        _insert_vec!(b, layout.ugamma[d], cut_next)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], blocks.div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], blocks.div_gamma[d])
    end

    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t_next)
    _apply_symmetry_box_bc!(A, b, model, t_next, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t_next; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

"""
    assemble_unsteady_moving!(sys, model, x_prev, t, dt; scheme=:CN)

Assemble one unsteady moving-boundary theta-step system for
`MovingStokesModelMono`, using slab geometry over `[t, t+dt]` and end-time box
BC/gauge application.
"""
function assemble_unsteady_moving!(
    sys::LinearSystem{T},
    model::MovingStokesModelMono{N,T},
    x_prev::AbstractVector,
    t::T,
    dt::T;
    scheme=:CN,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    theta = _theta_from_scheme(T, scheme)
    psip, psim = _psi_functions(scheme)
    t_next = t + dt

    _build_moving_slab!(model, t, dt)

    op_p_slab = something(model.op_p_slab)
    op_p_end = something(model.op_p_end)
    cap_u_slab = something(model.cap_u_slab)
    op_u_slab = something(model.op_u_slab)
    cap_u_end = something(model.cap_u_end)
    Vun = something(model.Vun)
    Vun1 = something(model.Vun1)

    nt = model.layout.nt
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)
    layout = model.layout
    xfull_prev = _expand_prev_state(model, x_prev)

    grad_full = op_p_slab.G + op_p_slab.H
    size(grad_full, 1) == N * nt ||
        throw(ArgumentError("pressure gradient rows ($(size(grad_full, 1))) must equal N*nt ($(N * nt))"))

    grad = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gd = sparse(-grad_full[rows, :])
        if !model.periodic[d]
            capd = cap_u_end[d]
            li = LinearIndices(capd.nnodes)
            @inbounds for I in CartesianIndices(capd.nnodes)
                if I[d] != 1
                    continue
                end
                physical = true
                for k in 1:N
                    if I[k] == capd.nnodes[k]
                        physical = false
                        break
                    end
                end
                physical || continue
                II = CartesianIndex(ntuple(k -> (k == d ? I[k] + 1 : I[k]), N))
                i = li[I]
                j = li[II]
                Aface = capd.buf.A[d][i]
                if !isfinite(Aface) || iszero(Aface)
                    continue
                end
                gd[i, i] = gd[i, i] + 2 * Aface
                gd[i, j] = gd[i, j] - Aface
            end
        end
        gd
    end, N)

    div_omega = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        gp = op_p_end.G[rows, :]
        hp = op_p_end.H[rows, :]
        -(gp' + hp')
    end, N)

    div_gamma = ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        hp = op_p_end.H[rows, :]
        sparse(hp')
    end, N)

    @inbounds for d in 1:N
        opud = op_u_slab[d]
        K = model.mu * (opud.G' * (opud.Winv * opud.G))
        C = model.mu * (opud.G' * (opud.Winv * opud.H))
        M1 = spdiagm(0 => model.rho .* Vun1[d])
        M0 = spdiagm(0 => model.rho .* Vun[d])
        Ψp = spdiagm(0 => T[psip(Vun[d][i], Vun1[d][i]) for i in 1:nt])
        Ψm = spdiagm(0 => T[psim(Vun[d][i], Vun1[d][i]) for i in 1:nt])

        # Ψp/Ψm already encode the selected temporal scheme for moving-slab terms.
        # Do not apply θ a second time to K/C contributions.
        A_oo = M1 + (K * Ψp)
        A_og = -(M1 - M0) + (C * Ψp)
        _insert_block!(A, layout.uomega[d], layout.uomega[d], A_oo)
        _insert_block!(A, layout.uomega[d], layout.ugamma[d], A_og)
        _insert_block!(A, layout.uomega[d], layout.pomega, grad[d])

        tie = spdiagm(0 => ones(T, nt))
        _insert_block!(A, layout.ugamma[d], layout.ugamma[d], tie)

        uω_prev = Vector{T}(xfull_prev[layout.uomega[d]])
        uγ_prev = Vector{T}(xfull_prev[layout.ugamma[d]])
        rhs = (M0 - (K * Ψm)) * uω_prev
        rhs .-= (C * Ψm) * uγ_prev

        f_prev = _force_values(model, cap_u_end[d], d, t)
        f_next = _force_values(model, cap_u_end[d], d, t_next)
        rhs .+= theta .* (cap_u_slab[d].V * f_next) .+ (one(T) - theta) .* (cap_u_slab[d].V * f_prev)
        _insert_vec!(b, layout.uomega[d], rhs)

        cut_next = _cut_values(cap_u_end[d], model.bc_cut_u[d], t_next)
        _insert_vec!(b, layout.ugamma[d], cut_next)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], div_gamma[d])
    end

    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t_next)
    _apply_symmetry_box_bc!(A, b, model, t_next, traction_locked_rows)
    _apply_velocity_box_bc!(A, b, model, t_next; locked_rows=traction_locked_rows)
    _apply_pressure_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

"""
    solve_steady!(model; t=0, method=:direct, kwargs...)

Assemble and solve a steady Stokes system for `StokesModelMono` or
`StokesModelTwoPhase`. Returns a `LinearSystem` with solution in `sys.x`.
"""
function solve_steady!(
    model::StokesModelMono{N,T};
    t::T=zero(T),
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_steady!(sys, model, t)
    solve!(sys; method=method, kwargs...)
    return sys
end

function solve_steady!(
    model::StokesModelTwoPhase{N,T};
    t::T=zero(T),
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_steady!(sys, model, t)
    solve!(sys; method=method, kwargs...)
    return sys
end

"""
    solve_unsteady!(model, x_prev; t=0, dt, scheme=:BE, method=:direct, kwargs...)

Assemble and solve one unsteady theta-step for `StokesModelMono` or
`StokesModelTwoPhase`.
"""
function solve_unsteady!(
    model::StokesModelMono{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    scheme=:BE,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_unsteady!(sys, model, x_prev, t, dt; scheme=scheme)
    solve!(sys; method=method, kwargs...)
    return sys
end

function solve_unsteady!(
    model::StokesModelTwoPhase{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    scheme=:BE,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_unsteady!(sys, model, x_prev, t, dt; scheme=scheme)
    solve!(sys; method=method, kwargs...)
    return sys
end

"""
    solve_unsteady_moving!(model, x_prev; t=0, dt, scheme=:CN, method=:direct, kwargs...)

Assemble and solve one unsteady moving-boundary theta-step for
`MovingStokesModelMono`.
"""
function solve_unsteady_moving!(
    model::MovingStokesModelMono{N,T},
    x_prev::AbstractVector;
    t::T=zero(T),
    dt::T,
    scheme=:CN,
    method::Symbol=:direct,
    kwargs...,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
    assemble_unsteady_moving!(sys, model, x_prev, t, dt; scheme=scheme)
    solve!(sys; method=method, kwargs...)
    return sys
end

@inline function _zero_svector(::Type{T}, ::Val{N}) where {T,N}
    return SVector{N,T}(ntuple(_ -> zero(T), N))
end

@inline function _is_physical_index(cap::AssembledCapacity{N,T}, i::Int) where {N,T}
    return isfinite(cap.buf.V[i]) && cap.buf.V[i] > zero(T)
end

@inline function _is_interface_index(cap::AssembledCapacity{N,T}, i::Int) where {N,T}
    _is_physical_index(cap, i) || return false
    Γi = cap.buf.Γ[i]
    if !(isfinite(Γi) && Γi > zero(T))
        return false
    end
    xγ = cap.C_γ[i]
    nγ = cap.n_γ[i]
    @inbounds for k in 1:N
        if !(isfinite(xγ[k]) && isfinite(nγ[k]))
            return false
        end
    end
    return true
end

function _scalar_gradient(
    op::DiffusionOps{N,T},
    xω::AbstractVector{T},
    xγ::AbstractVector{T},
) where {N,T}
    nt = length(xω)
    length(xγ) == nt || throw(DimensionMismatch("xγ length must match xω length"))
    g = op.Winv * (op.G * xω + op.H * xγ)
    length(g) == N * nt || throw(DimensionMismatch("gradient length mismatch"))
    return ntuple(d -> begin
        rows = ((d - 1) * nt + 1):(d * nt)
        Vector{T}(g[rows])
    end, N)
end

function _as_origin(x0, ::Type{T}, ::Val{N}) where {T,N}
    if isnothing(x0)
        return _zero_svector(T, Val(N))
    end
    return SVector{N,T}(ntuple(k -> convert(T, x0[k]), N))
end

function _pressure_trace(
    model::StokesModelMono{N,T},
    pω::AbstractVector{T},
    grad_p,
    i::Int,
    xγ::SVector{N,T},
    reconstruction::Symbol,
) where {N,T}
    pγ = pω[i]
    if reconstruction === :none
        return pγ
    elseif reconstruction === :linear
        xω = model.cap_p.C_ω[i]
        @inbounds for k in 1:N
            pγ += grad_p[k][i] * (xγ[k] - xω[k])
        end
        return pγ
    end
    throw(ArgumentError("unknown pressure reconstruction `$reconstruction`; expected :none or :linear"))
end

@inline function _stress_tensor(
    μ::T,
    G::SMatrix{N,N,T},
    p::T,
) where {N,T}
    I_N = SMatrix{N,N,T}(Matrix{T}(I, N, N))
    return μ .* (G + transpose(G)) - p .* I_N
end

@inline function _traction_from_stress(
    σ::SMatrix{N,N,T},
    n::SVector{N,T},
) where {N,T}
    return σ * n
end

"""
    embedded_boundary_quantities(model, x; mu=model.mu, pressure_reconstruction=:linear, x0=nothing)

Compute cut-boundary stress/traction diagnostics on the staggered velocity grids.

Returns a named tuple with:
- `stress`: stress tensors on pressure-grid cut cells
- `traction`: traction vectors on pressure-grid cut cells
- `force_density`: integrated force vectors (`traction * Γ`) on pressure-grid cut cells
- `interface_indices`: active cut-cell indices on the pressure grid
- `force`: total integrated force vector
- `force_pressure`, `force_viscous`: pressure/viscous splits of `force`
- `torque`: scalar in 2D or vector in 3D about `x0`
"""
function embedded_boundary_quantities(
    model::StokesModelMono{N,T},
    x::AbstractVector;
    mu::Real=model.mu,
    pressure_reconstruction::Symbol=:linear,
    x0=nothing,
) where {N,T}
    nsys = nunknowns(model.layout)
    length(x) == nsys || throw(DimensionMismatch("state length must be $nsys"))
    μ = convert(T, mu)
    origin = _as_origin(x0, T, Val(N))
    nt = model.cap_p.ntotal

    layout = model.layout
    uω = ntuple(d -> Vector{T}(x[layout.uomega[d]]), N)
    uγ = ntuple(d -> Vector{T}(x[layout.ugamma[d]]), N)
    pω = Vector{T}(x[layout.pomega])

    grad_u = ntuple(d -> _scalar_gradient(model.op_u[d], uω[d], uγ[d]), N)
    pγ_seed = zeros(T, nt)
    if pressure_reconstruction === :linear
        pγ_seed .= pω
    elseif pressure_reconstruction !== :none
        throw(ArgumentError("unknown pressure reconstruction `$pressure_reconstruction`; expected :none or :linear"))
    end
    grad_p = _scalar_gradient(model.op_p, pω, pγ_seed)

    zero_vec = _zero_svector(T, Val(N))
    zero_sig = zero(SMatrix{N,N,T,N * N})

    stress = fill(zero_sig, nt)
    traction = fill(zero_vec, nt)
    force_density = fill(zero_vec, nt)
    interface_indices = Int[]

    F = zeros(T, N)
    Fp = zeros(T, N)
    Fμ = zeros(T, N)

    τ2 = zero(T)
    τ3 = SVector{3,T}(zero(T), zero(T), zero(T))

    cap = model.cap_p
    @inbounds for i in 1:cap.ntotal
        _is_interface_index(cap, i) || continue
        push!(interface_indices, i)

        Γi = cap.buf.Γ[i]
        n = cap.n_γ[i]
        xγi = cap.C_γ[i]
        pγi = _pressure_trace(model, pω, grad_p, i, xγi, pressure_reconstruction)

        G = MMatrix{N,N,T}(undef)
        for a in 1:N, b in 1:N
            G[a, b] = grad_u[a][b][i]
        end
        σ = _stress_tensor(μ, SMatrix{N,N,T}(G), pγi)
        tvec = _traction_from_stress(σ, n)
        tp = -pγi .* n
        tv = tvec - tp
        fvec = Γi .* tvec
        fpvec = Γi .* tp
        fvvec = Γi .* tv

        stress[i] = σ
        traction[i] = tvec
        force_density[i] = fvec

        F .+= fvec
        Fp .+= fpvec
        Fμ .+= fvvec

        if N == 2
            r = xγi - origin
            τ2 += r[1] * fvec[2] - r[2] * fvec[1]
        elseif N == 3
            r = SVector{3,T}(xγi[1] - origin[1], xγi[2] - origin[2], xγi[3] - origin[3])
            τ3 += cross(r, SVector{3,T}(fvec[1], fvec[2], fvec[3]))
        end
    end

    Ftot = SVector{N,T}(Tuple(F))
    Fptot = SVector{N,T}(Tuple(Fp))
    Fμtot = SVector{N,T}(Tuple(Fμ))
    torque = N == 2 ? τ2 : (N == 3 ? τ3 : nothing)

    return (
        stress=stress,
        traction=traction,
        force_density=force_density,
        interface_indices=interface_indices,
        force=Ftot,
        force_pressure=Fptot,
        force_viscous=Fμtot,
        torque=torque,
    )
end

function embedded_boundary_quantities(
    model::StokesModelMono{N,T},
    sys::LinearSystem{T};
    kwargs...,
) where {N,T}
    return embedded_boundary_quantities(model, sys.x; kwargs...)
end

"""
    embedded_boundary_traction(model, x; kwargs...)

Return pressure-grid traction vectors on embedded boundary cells.
"""
function embedded_boundary_traction(model::StokesModelMono, x::AbstractVector; kwargs...)
    return embedded_boundary_quantities(model, x; kwargs...).traction
end

function embedded_boundary_traction(model::StokesModelMono{N,T}, sys::LinearSystem{T}; kwargs...) where {N,T}
    return embedded_boundary_traction(model, sys.x; kwargs...)
end

"""
    embedded_boundary_stress(model, x; kwargs...)

Return pressure-grid stress tensors on embedded boundary cells.
"""
function embedded_boundary_stress(model::StokesModelMono, x::AbstractVector; kwargs...)
    return embedded_boundary_quantities(model, x; kwargs...).stress
end

function embedded_boundary_stress(model::StokesModelMono{N,T}, sys::LinearSystem{T}; kwargs...) where {N,T}
    return embedded_boundary_stress(model, sys.x; kwargs...)
end

"""
    integrated_embedded_force(model, x; kwargs...)

Return integrated embedded-boundary force components and torque.
"""
function integrated_embedded_force(model::StokesModelMono, x::AbstractVector; kwargs...)
    q = embedded_boundary_quantities(model, x; kwargs...)
    return (force=q.force, force_pressure=q.force_pressure, force_viscous=q.force_viscous, torque=q.torque)
end

function integrated_embedded_force(model::StokesModelMono{N,T}, sys::LinearSystem{T}; kwargs...) where {N,T}
    return integrated_embedded_force(model, sys.x; kwargs...)
end

"""
    StokesModelMono(gridp, body, mu, rho; kwargs...)
    StokesModelMono(cap_p, op_p, cap_u, op_u, mu, rho; kwargs...)

Construct a monophasic Stokes model.

Key keywords:
- `force`: body force (constant tuple/scalar or callback)
- `bc_u`: per-component outer velocity BCs
- `bc_p`: optional pressure BCs on box walls
- `bc_cut`: cut-boundary velocity BC (single boundary or per-component tuple;
  currently `Dirichlet` values are supported for cut rows)
- `gauge`: pressure gauge (`PinPressureGauge` or `MeanPressureGauge`)
- `strong_wall_bc`: enable strong elimination for collocated Dirichlet wall rows
"""
function StokesModelMono(
    cap_p::AssembledCapacity{N,T},
    op_p::DiffusionOps{N,T},
    cap_u::NTuple{N,AssembledCapacity{N,T}},
    op_u::NTuple{N,DiffusionOps{N,T}},
    mu::Real,
    rho::Real;
    force=_default_force(T, Val(N)),
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    bc_cut::Union{AbstractBoundary,NTuple{N,AbstractBoundary}}=ntuple(_ -> Dirichlet(zero(T)), N),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:prebuilt,
    body=nothing,
) where {N,T}
    nt = cap_p.ntotal
    @inbounds for d in 1:N
        cap_u[d].ntotal == nt || throw(ArgumentError("all velocity capacities must have same ntotal as pressure"))
    end

    pflags = _periodic_velocity_flags(bc_u)
    pflags == periodic_flags(bc_u[1], N) || throw(ArgumentError("invalid periodic flags"))
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)

    gridp = CartesianGrid(
        ntuple(d -> cap_p.xyz[d][1], N),
        ntuple(d -> cap_p.xyz[d][end], N),
        cap_p.nnodes,
    )
    gridu = ntuple(d -> CartesianGrid(
        ntuple(k -> cap_u[d].xyz[k][1], N),
        ntuple(k -> cap_u[d].xyz[k][end], N),
        cap_u[d].nnodes,
    ), N)

    layout = StokesLayout(nt, Val(N))
    bc_cut_tuple = bc_cut isa NTuple ? bc_cut : _normalize_cut_bc_tuple(bc_cut, Val(N))
    return StokesModelMono{N,T,typeof(force),typeof(body)}(
        gridp,
        gridu,
        cap_p,
        cap_u,
        op_p,
        op_u,
        convert(T, mu),
        convert(T, rho),
        force,
        bc_u,
        bc_p,
        bc_cut_tuple,
        gauge,
        strong_wall_bc,
        layout,
        pflags,
        geom_method,
        body,
    )
end

function StokesModelMono(
    gridp::CartesianGrid{N,T},
    body,
    mu::Real,
    rho::Real;
    force=_default_force(T, Val(N)),
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    bc_cut::Union{AbstractBoundary,NTuple{N,AbstractBoundary}}=ntuple(_ -> Dirichlet(zero(T)), N),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:vofijul,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    gridu = staggered_velocity_grids(gridp)

    mp = geometric_moments(body, grid1d(gridp), T, nan; method=geom_method)
    mu_mom = ntuple(d -> geometric_moments(body, grid1d(gridu[d]), T, nan; method=geom_method), N)

    cap_p = assembled_capacity(mp; bc=zero(T))
    cap_u = ntuple(d -> assembled_capacity(mu_mom[d]; bc=zero(T)), N)

    op_p = DiffusionOps(cap_p; periodic=pflags)
    op_u = ntuple(d -> DiffusionOps(cap_u[d]; periodic=pflags), N)

    nt = cap_p.ntotal
    @inbounds for d in 1:N
        cap_u[d].ntotal == nt || throw(ArgumentError("all velocity capacities must have same ntotal as pressure"))
    end

    layout = StokesLayout(nt, Val(N))
    bc_cut_tuple = bc_cut isa NTuple ? bc_cut : _normalize_cut_bc_tuple(bc_cut, Val(N))

    return StokesModelMono{N,T,typeof(force),typeof(body)}(
        gridp,
        gridu,
        cap_p,
        cap_u,
        op_p,
        op_u,
        convert(T, mu),
        convert(T, rho),
        force,
        bc_u,
        bc_p,
        bc_cut_tuple,
        gauge,
        strong_wall_bc,
        layout,
        pflags,
        geom_method,
        body,
    )
end

"""
    MovingStokesModelMono(gridp, body, mu, rho; kwargs...)

Construct a moving-boundary monophasic Stokes model.

`body(x..., t)` defines moving geometry and `bc_cut_u` prescribes per-component
embedded-boundary velocity at `t_{n+1}` in `assemble_unsteady_moving!`.
"""
function MovingStokesModelMono(
    gridp::CartesianGrid{N,T},
    body,
    mu::Real,
    rho::Real;
    force=_default_force(T, Val(N)),
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    bc_cut_u::Union{AbstractBoundary,NTuple{N,AbstractBoundary}}=ntuple(_ -> Dirichlet(zero(T)), N),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:vofijul,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    gridu = staggered_velocity_grids(gridp)
    nt = prod(gridp.n)
    layout = StokesLayout(nt, Val(N))
    bc_cut_tuple = bc_cut_u isa NTuple ? bc_cut_u : _normalize_cut_bc_tuple(bc_cut_u, Val(N))

    return MovingStokesModelMono{N,T,typeof(force),typeof(body)}(
        gridp,
        gridu,
        body,
        convert(T, mu),
        convert(T, rho),
        force,
        bc_u,
        bc_p,
        bc_cut_tuple,
        gauge,
        strong_wall_bc,
        pflags,
        geom_method,
        layout,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function _check_two_phase_interface_consistency(
    cap_p1::AssembledCapacity{N,T},
    cap_p2::AssembledCapacity{N,T},
) where {N,T}
    cap_p1.ntotal == cap_p2.ntotal ||
        throw(ArgumentError("two-phase pressure capacities must have same ntotal"))
    cap_p1.nnodes == cap_p2.nnodes ||
        throw(ArgumentError("two-phase pressure capacities must share nnodes"))
    tol_Γ_abs = sqrt(eps(T))
    tol_Γ_rel = convert(T, 1e-6)
    tol_n = convert(T, 1e-6)
    @inbounds for i in 1:cap_p1.ntotal
        Γ1 = cap_p1.buf.Γ[i]
        Γ2 = cap_p2.buf.Γ[i]
        has1 = isfinite(Γ1) && Γ1 > zero(T)
        has2 = isfinite(Γ2) && Γ2 > zero(T)
        has1 == has2 || throw(ArgumentError("inconsistent interface support between phases at cell index $i"))
        has1 || continue
        isapprox(Γ1, Γ2; atol=tol_Γ_abs, rtol=tol_Γ_rel) ||
            throw(ArgumentError("inconsistent interface measure between phases at cell index $i"))
        n1 = cap_p1.n_γ[i]
        n2 = cap_p2.n_γ[i]
        norm(n1 + n2) <= tol_n ||
            throw(ArgumentError("phase-interface normals are not opposite at cell index $i"))
    end
    return nothing
end

"""
    StokesModelTwoPhase(gridp, body, mu1, mu2; kwargs...)
    StokesModelTwoPhase(cap_p1, op_p1, cap_u1, op_u1, cap_p2, op_p2, cap_u2, op_u2, mu1, mu2, rho1, rho2; kwargs...)

Construct a fixed-interface two-phase Stokes model with phase-wise interface
trace unknowns and phase-wise pressure blocks.

Key keywords:
- `rho1`, `rho2`: phase densities (for unsteady terms)
- `force1`, `force2`: phase body forces
- `interface_jump`: interface velocity jump forcing callback/value for `[u] = f`
- `interface_force`: interface traction-jump forcing callback/value for `[traction] = g`
- `bc_interface`: optional per-component `InterfaceConditions` (`ScalarJump`/`FluxJump`)
- `bc_u`, `bc_p`, `gauge`, `strong_wall_bc`: box BC and gauge controls
"""
function StokesModelTwoPhase(
    cap_p1::AssembledCapacity{N,T},
    op_p1::DiffusionOps{N,T},
    cap_u1::NTuple{N,AssembledCapacity{N,T}},
    op_u1::NTuple{N,DiffusionOps{N,T}},
    cap_p2::AssembledCapacity{N,T},
    op_p2::DiffusionOps{N,T},
    cap_u2::NTuple{N,AssembledCapacity{N,T}},
    op_u2::NTuple{N,DiffusionOps{N,T}},
    mu1::Real,
    mu2::Real,
    rho1::Real,
    rho2::Real;
    force1=_default_force(T, Val(N)),
    force2=_default_force(T, Val(N)),
    interface_jump=_default_interface_jump(T, Val(N)),
    interface_force=_default_interface_force(T, Val(N)),
    bc_interface::Union{Nothing,InterfaceConditions,NTuple{N,InterfaceConditions}}=nothing,
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:prebuilt,
    body=nothing,
    check_interface::Bool=true,
) where {N,T}
    nt = cap_p1.ntotal
    cap_p2.ntotal == nt || throw(ArgumentError("phase-2 pressure capacity must have same ntotal as phase-1"))
    @inbounds for d in 1:N
        cap_u1[d].ntotal == nt || throw(ArgumentError("all phase-1 velocity capacities must match pressure ntotal"))
        cap_u2[d].ntotal == nt || throw(ArgumentError("all phase-2 velocity capacities must match pressure ntotal"))
    end

    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    bc_interface_tuple = _normalize_interface_bc(bc_interface, Val(N))
    _validate_stokes_interface_bcs!(bc_interface_tuple)

    check_interface && _check_two_phase_interface_consistency(cap_p1, cap_p2)

    gridp = CartesianGrid(
        ntuple(d -> cap_p1.xyz[d][1], N),
        ntuple(d -> cap_p1.xyz[d][end], N),
        cap_p1.nnodes,
    )
    gridu = ntuple(d -> CartesianGrid(
        ntuple(k -> cap_u1[d].xyz[k][1], N),
        ntuple(k -> cap_u1[d].xyz[k][end], N),
        cap_u1[d].nnodes,
    ), N)

    layout = StokesLayoutTwoPhase(nt, Val(N))
    return StokesModelTwoPhase{
        N,T,typeof(force1),typeof(force2),typeof(interface_force),typeof(interface_jump),typeof(body)
    }(
        gridp,
        gridu,
        cap_p1,
        cap_u1,
        op_p1,
        op_u1,
        cap_p2,
        cap_u2,
        op_p2,
        op_u2,
        convert(T, mu1),
        convert(T, mu2),
        convert(T, rho1),
        convert(T, rho2),
        force1,
        force2,
        interface_force,
        interface_jump,
        bc_interface_tuple,
        bc_u,
        bc_p,
        gauge,
        strong_wall_bc,
        pflags,
        geom_method,
        body,
        layout,
    )
end

function StokesModelTwoPhase(
    gridp::CartesianGrid{N,T},
    body,
    mu1::Real,
    mu2::Real;
    rho1::Real=one(T),
    rho2::Real=one(T),
    force1=_default_force(T, Val(N)),
    force2=_default_force(T, Val(N)),
    interface_jump=_default_interface_jump(T, Val(N)),
    interface_force=_default_interface_force(T, Val(N)),
    bc_interface::Union{Nothing,InterfaceConditions,NTuple{N,InterfaceConditions}}=nothing,
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:vofijul,
    check_interface::Bool=true,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    gridu = staggered_velocity_grids(gridp)

    body2 = (x...) -> -body(x...)

    mp1 = geometric_moments(body, grid1d(gridp), T, nan; method=geom_method)
    mp2 = geometric_moments(body2, grid1d(gridp), T, nan; method=geom_method)
    mom_u1 = ntuple(d -> geometric_moments(body, grid1d(gridu[d]), T, nan; method=geom_method), N)
    mom_u2 = ntuple(d -> geometric_moments(body2, grid1d(gridu[d]), T, nan; method=geom_method), N)

    cap_p1 = assembled_capacity(mp1; bc=zero(T))
    cap_p2 = assembled_capacity(mp2; bc=zero(T))
    cap_u1 = ntuple(d -> assembled_capacity(mom_u1[d]; bc=zero(T)), N)
    cap_u2 = ntuple(d -> assembled_capacity(mom_u2[d]; bc=zero(T)), N)

    op_p1 = DiffusionOps(cap_p1; periodic=pflags)
    op_p2 = DiffusionOps(cap_p2; periodic=pflags)
    op_u1 = ntuple(d -> DiffusionOps(cap_u1[d]; periodic=pflags), N)
    op_u2 = ntuple(d -> DiffusionOps(cap_u2[d]; periodic=pflags), N)

    return StokesModelTwoPhase(
        cap_p1,
        op_p1,
        cap_u1,
        op_u1,
        cap_p2,
        op_p2,
        cap_u2,
        op_u2,
        mu1,
        mu2,
        rho1,
        rho2;
        force1=force1,
        force2=force2,
        interface_jump=interface_jump,
        interface_force=interface_force,
        bc_interface=bc_interface,
        bc_u=bc_u,
        bc_p=bc_p,
        gauge=gauge,
        strong_wall_bc=strong_wall_bc,
        geom_method=geom_method,
        body=body,
        check_interface=check_interface,
    )
end

include("orientation.jl")
include("rigidbody.jl")
include("fsi.jl")
include("fsi_strong_coupling.jl")

end
