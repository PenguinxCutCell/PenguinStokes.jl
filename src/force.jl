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

function _force_values(
    model::MovingStokesModelTwoPhase{N,T},
    cap::AssembledCapacity{N,T},
    phase::Int,
    d::Int,
    t::T,
) where {N,T}
    force = if phase == 1
        model.force1
    elseif phase == 2
        model.force2
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

@inline function _as_interface_force_vector(y, ::Val{N}, ::Type{T}) where {N,T}
    if y isa Number
        vy = convert(T, y)
        return SVector{N,T}(ntuple(_ -> vy, N))
    elseif y isa Tuple || y isa AbstractVector
        length(y) == N ||
            throw(ArgumentError("interface forcing callback must return scalar or length-$N vector"))
        return SVector{N,T}(ntuple(d -> convert(T, y[d]), N))
    end
    throw(ArgumentError("interface forcing callback must return scalar or length-$N vector"))
end

function _interface_force_vector(interface_force, x::SVector{N,T}, t::T) where {N,T}
    if interface_force isa Number
        fv = convert(T, interface_force)
        return SVector{N,T}(ntuple(_ -> fv, N))
    elseif interface_force isa NTuple{N,Any}
        return SVector{N,T}(ntuple(d -> begin
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
        end, N))
    elseif interface_force isa Function
        if applicable(interface_force, x..., t)
            return _as_interface_force_vector(interface_force(x..., t), Val(N), T)
        elseif applicable(interface_force, x...)
            return _as_interface_force_vector(interface_force(x...), Val(N), T)
        elseif applicable(interface_force, x, t)
            return _as_interface_force_vector(interface_force(x, t), Val(N), T)
        elseif applicable(interface_force, x)
            return _as_interface_force_vector(interface_force(x), Val(N), T)
        end
        throw(ArgumentError("interface forcing callback must accept (x...), (x..., t), (x), or (x, t)"))
    end
    throw(ArgumentError("unsupported interface forcing type $(typeof(interface_force))"))
end

function _interface_force_vector_with_normal(interface_force, x::SVector{N,T}, n::SVector{N,T}, t::T) where {N,T}
    """
    Call interface_force callback with optional discrete normal vector.
    Supports signatures: (x..., nx, ny, ..., t), (x..., nx, ny, ...), and fallback to _interface_force_vector.
    """
    if interface_force isa Number
        fv = convert(T, interface_force)
        return SVector{N,T}(ntuple(_ -> fv, N))
    elseif interface_force isa NTuple{N,Any}
        # For tuple forcing, use component-wise callbacks; they don't support normals
        return _interface_force_vector(interface_force, x, t)
    elseif interface_force isa Function
        # Try signatures with discrete normal first
        if N == 2
            if applicable(interface_force, x[1], x[2], n[1], n[2], t)
                return _as_interface_force_vector(interface_force(x[1], x[2], n[1], n[2], t), Val(N), T)
            elseif applicable(interface_force, x[1], x[2], n[1], n[2])
                return _as_interface_force_vector(interface_force(x[1], x[2], n[1], n[2]), Val(N), T)
            end
        elseif N == 3
            if applicable(interface_force, x[1], x[2], x[3], n[1], n[2], n[3], t)
                return _as_interface_force_vector(interface_force(x[1], x[2], x[3], n[1], n[2], n[3], t), Val(N), T)
            elseif applicable(interface_force, x[1], x[2], x[3], n[1], n[2], n[3])
                return _as_interface_force_vector(interface_force(x[1], x[2], x[3], n[1], n[2], n[3]), Val(N), T)
            end
        end
        # Fallback to position-only signatures
        return _interface_force_vector(interface_force, x, t)
    end
    throw(ArgumentError("unsupported interface forcing type $(typeof(interface_force))"))
end

@inline function _consistent_interface_force_component(
    fγ::SVector{N,T},
    nγ::SVector{N,T},
    d::Int,
) where {N,T}
    fnorm = norm(fγ)
    nnorm = norm(nγ)
    if fnorm > zero(T) && nnorm > zero(T)
        c = dot(fγ, nγ) / (fnorm * nnorm)
        # When the prescribed traction is (nearly) normal, use the same
        # discrete normal moments as the pressure coupling to preserve exact
        # local force balance for constant jump states.
        if abs(abs(c) - one(T)) <= convert(T, 1e-6)
            s = c >= zero(T) ? one(T) : -one(T)
            return s * fnorm * nγ[d]
        end
    end
    return fγ[d]
end
