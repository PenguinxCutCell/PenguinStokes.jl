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

function _apply_pressure_box_bc!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::MovingStokesModelTwoPhase{N,T}, t::T) where {N,T}
    model.strong_wall_bc || return A, b
    isnothing(model.bc_p) && return A, b
    isnothing(model.cap_p1_end) && throw(ArgumentError("moving two-phase model phase-1 pressure end-capacity cache is not built"))
    isnothing(model.cap_p2_end) && throw(ArgumentError("moving two-phase model phase-2 pressure end-capacity cache is not built"))
    cap_p1_end = something(model.cap_p1_end)
    cap_p2_end = something(model.cap_p2_end)
    _apply_pressure_box_bc_block!(
        A, b, model.gridp, cap_p1_end, model.layout.pomega1, model.periodic, model.bc_u, model.bc_p, t
    )
    _apply_pressure_box_bc_block!(
        A, b, model.gridp, cap_p2_end, model.layout.pomega2, model.periodic, model.bc_u, model.bc_p, t
    )
    return A, b
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

function _apply_pressure_gauge!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::MovingStokesModelTwoPhase{N,T}) where {N,T}
    isnothing(model.cap_p1_end) && throw(ArgumentError("moving two-phase model phase-1 pressure end-capacity cache is not built"))
    cap_p1_end = something(model.cap_p1_end)

    layout = model.layout
    if model.gauge isa PinPressureGauge
        return _apply_pin_pressure_gauge!(A, b, layout.pomega1, cap_p1_end, model.gauge)
    elseif model.gauge isa MeanPressureGauge
        return _apply_mean_pressure_gauge!(A, b, layout.pomega1, cap_p1_end)
    end

    throw(ArgumentError("unsupported pressure gauge type $(typeof(model.gauge))"))
end
