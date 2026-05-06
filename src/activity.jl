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
    isnothing(model.cap_u_slab) && throw(ArgumentError("moving model velocity slab-capacity cache is not built"))
    isnothing(model.Vun) && throw(ArgumentError("moving model previous velocity-volume cache is not built"))
    isnothing(model.Vun1) && throw(ArgumentError("moving model end velocity-volume cache is not built"))
    isnothing(model.cap_p_slab) && throw(ArgumentError("moving model pressure slab-capacity cache is not built"))
    cap_u_end = something(model.cap_u_end)
    cap_u_slab = something(model.cap_u_slab)
    Vun = something(model.Vun)
    Vun1 = something(model.Vun1)
    # Use the slab pressure capacity for activity: a pressure cell is active when it had
    # nonzero fluid volume during the time step [t, t+dt].  Cells that existed only at
    # the start time (V_end=0 but V_slab>0) still carry a valid incompressibility
    # constraint from the slab divergence operator; cells that are completely empty in
    # the slab (V_slab=0) are genuinely inactive.
    cap_p_slab = something(model.cap_p_slab)

    layout = model.layout
    active = falses(nunknowns(layout))
    @inbounds for d in 1:N
        aomega_end, agamma_end = _cell_activity_masks(cap_u_end[d])
        aomega_slab, agamma_slab = _cell_activity_masks(cap_u_slab[d])
        for i in 1:cap_u_end[d].ntotal
            active[layout.uomega[d][i]] = aomega_end[i] || aomega_slab[i]
            vdelta = abs(Vun1[d][i] - Vun[d][i])
            volume_changed = isfinite(vdelta) && vdelta > sqrt(eps(T)) * max(one(T), abs(Vun[d][i]), abs(Vun1[d][i]))
            active[layout.ugamma[d][i]] = agamma_end[i] || agamma_slab[i] || volume_changed
        end
    end

    # Deactivate pressure cells whose slab volume is below a small-cell threshold.
    # Cells with V_slab ≪ h^N (sliver cells at the domain boundary) have near-zero
    # gradient coupling in the momentum rows, making their pressure ill-determined.
    h = minimum(meshsize(model.gridp))
    eps_cut = convert(T, 1e-3)
    min_vol = eps_cut * h^N
    pactive = BitVector(undef, cap_p_slab.ntotal)
    @inbounds for i in 1:cap_p_slab.ntotal
        v = cap_p_slab.buf.V[i]
        pactive[i] = isfinite(v) && v >= min_vol
    end
    # Also deactivate halo nodes.
    li = LinearIndices(cap_p_slab.nnodes)
    @inbounds for I in CartesianIndices(cap_p_slab.nnodes)
        i = li[I]
        for d in 1:N
            if I[d] == cap_p_slab.nnodes[d]
                pactive[i] = false
                break
            end
        end
    end

    pfirst = first(layout.pomega)
    plast = last(layout.pomega)
    @inbounds for i in 1:cap_p_slab.ntotal
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
    @inbounds for i in 1:cap_p_slab.ntotal
        active[layout.pomega[i]] = pactive[i]
    end

    # The moving pressure-slab divergence can reference trace columns that are
    # not cut at the same velocity-grid index at t^{n+1}.  Keep every trace
    # column that is actually used by an active equation; otherwise column
    # pruning destroys the constant-velocity cancellation in the slab operator.
    @inbounds for d in 1:N
        for i in 1:cap_u_end[d].ntotal
            col = layout.ugamma[d][i]
            active[col] && continue
            coupled = false
            for ptr in nzrange(A, col)
                row = A.rowval[ptr]
                if active[row] && A.nzval[ptr] != zero(T)
                    coupled = true
                    break
                end
            end
            active[col] = coupled
        end
    end

    return _prune_uncoupled_active!(active, A)
end

function _stokes_row_activity(model::MovingStokesModelTwoPhase{N,T}, A::SparseMatrixCSC{T,Int}) where {N,T}
    isnothing(model.cap_u1_end) && throw(ArgumentError("moving two-phase model phase-1 velocity end-capacity cache is not built"))
    isnothing(model.cap_u2_end) && throw(ArgumentError("moving two-phase model phase-2 velocity end-capacity cache is not built"))
    isnothing(model.cap_p1_slab) && throw(ArgumentError("moving two-phase model phase-1 pressure slab-capacity cache is not built"))
    isnothing(model.cap_p2_slab) && throw(ArgumentError("moving two-phase model phase-2 pressure slab-capacity cache is not built"))
    cap_u1_end = something(model.cap_u1_end)
    cap_u2_end = something(model.cap_u2_end)
    # Use slab capacities for pressure activity: a cell is active when it had nonzero
    # fluid volume during [t, t+dt], which is correctly captured by the slab geometry.
    cap_p1_slab = something(model.cap_p1_slab)
    cap_p2_slab = something(model.cap_p2_slab)
    cap_p1_end = something(model.cap_p1_end)

    layout = model.layout
    active = falses(nunknowns(layout))
    _, agamma_p = _cell_activity_masks(cap_p1_end)

    @inbounds for d in 1:N
        aomega1, agamma1 = _cell_activity_masks(cap_u1_end[d])
        aomega2, agamma2 = _cell_activity_masks(cap_u2_end[d])
        for i in 1:cap_u1_end[d].ntotal
            active[layout.uomega1[d][i]] = aomega1[i]
            active[layout.uomega2[d][i]] = aomega2[i]
            has_gamma = (agamma1[i] || agamma2[i]) && (agamma_p[i] || aomega1[i] || aomega2[i])
            active[layout.ugamma1[d][i]] = has_gamma
            active[layout.ugamma2[d][i]] = has_gamma
        end
    end

    # Deactivate pressure cells below the small-cell volume threshold.  Moving
    # two-phase CN is especially sensitive to partially activated slab-pressure
    # islands: they create pressure-dominated near-null modes and amplify trace
    # truncation error.  Keep this cutoff conservative unless the pressure
    # block is stabilized/scaled consistently with the time-integrated rows.
    h = minimum(meshsize(model.gridp))
    eps_cut = convert(T, 2e-3)
    min_vol = eps_cut * h^N
    li = LinearIndices(cap_p1_slab.nnodes)

    function _slab_pactive(cap_p_slab)
        pa = BitVector(undef, cap_p_slab.ntotal)
        @inbounds for I in CartesianIndices(cap_p_slab.nnodes)
            i = li[I]
            halo = any(d -> I[d] == cap_p_slab.nnodes[d], 1:N)
            if halo
                pa[i] = false
                continue
            end
            v = cap_p_slab.buf.V[i]
            pa[i] = isfinite(v) && v >= min_vol
        end
        return pa
    end
    p1active = _slab_pactive(cap_p1_slab)
    p2active = _slab_pactive(cap_p2_slab)

    p1first = first(layout.pomega1)
    p1last = last(layout.pomega1)
    p2first = first(layout.pomega2)
    p2last = last(layout.pomega2)

    @inbounds for i in 1:cap_p1_slab.ntotal
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

    @inbounds for i in 1:cap_p2_slab.ntotal
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

    @inbounds for i in 1:cap_p1_slab.ntotal
        active[layout.pomega1[i]] = p1active[i]
        active[layout.pomega2[i]] = p2active[i]
    end
    return _prune_uncoupled_active!(active, A)
end
