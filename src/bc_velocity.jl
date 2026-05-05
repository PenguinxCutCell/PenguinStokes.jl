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

function _apply_traction_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelTwoPhase{N,T},
    t::T,
) where {N,T}
    isnothing(model.cap_p1_end) && throw(ArgumentError("moving two-phase model phase-1 pressure end-capacity cache is not built"))
    isnothing(model.cap_u1_end) && throw(ArgumentError("moving two-phase model phase-1 velocity end-capacity cache is not built"))
    isnothing(model.cap_p2_end) && throw(ArgumentError("moving two-phase model phase-2 pressure end-capacity cache is not built"))
    isnothing(model.cap_u2_end) && throw(ArgumentError("moving two-phase model phase-2 velocity end-capacity cache is not built"))
    cap_p1_end = something(model.cap_p1_end)
    cap_u1_end = something(model.cap_u1_end)
    cap_p2_end = something(model.cap_p2_end)
    cap_u2_end = something(model.cap_u2_end)
    locked_rows = falses(size(A, 1))
    _apply_traction_box_bc_block!(
        A,
        b,
        model.gridp,
        cap_p1_end,
        cap_u1_end,
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
        cap_p2_end,
        cap_u2_end,
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

function _apply_symmetry_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelTwoPhase{N,T},
    t::T,
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    isnothing(model.cap_u1_end) && throw(ArgumentError("moving two-phase model phase-1 velocity end-capacity cache is not built"))
    isnothing(model.cap_u2_end) && throw(ArgumentError("moving two-phase model phase-2 velocity end-capacity cache is not built"))
    cap_u1_end = something(model.cap_u1_end)
    cap_u2_end = something(model.cap_u2_end)
    locks = isnothing(locked_rows) ? falses(size(A, 1)) : locked_rows
    _apply_symmetry_box_bc_block!(
        A,
        b,
        model.gridp,
        cap_u1_end,
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
        cap_u2_end,
        model.mu2,
        model.bc_u,
        model.layout.uomega2,
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

function _apply_velocity_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    model::MovingStokesModelTwoPhase{N,T},
    t::T;
    locked_rows::Union{Nothing,BitVector}=nothing,
) where {N,T}
    isnothing(model.cap_u1_end) && throw(ArgumentError("moving two-phase model phase-1 velocity end-capacity cache is not built"))
    isnothing(model.cap_u2_end) && throw(ArgumentError("moving two-phase model phase-2 velocity end-capacity cache is not built"))
    cap_u1_end = something(model.cap_u1_end)
    cap_u2_end = something(model.cap_u2_end)
    layout = model.layout
    for d in 1:N
        _apply_component_velocity_box_bc!(
            A,
            b,
            d,
            model.strong_wall_bc,
            model.gridp,
            cap_u1_end[d],
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
            cap_u2_end[d],
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

