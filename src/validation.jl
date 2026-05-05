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
