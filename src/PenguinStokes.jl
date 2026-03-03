module PenguinStokes

using LinearAlgebra
using SparseArrays
using StaticArrays

using CartesianGeometry: GeometricMoments, geometric_moments, nan
using CartesianGrids: CartesianGrid, grid1d, meshsize
using CartesianOperators: AssembledCapacity, DiffusionOps, assembled_capacity, each_boundary_cell, periodic_flags, side_info
using PenguinBCs: AbstractBoundary, BorderConditions, Dirichlet, Neumann, Periodic, eval_bc, validate_borderconditions!
using PenguinSolverCore: LinearSystem, solve!

export AbstractPressureGauge, PinPressureGauge, MeanPressureGauge
export StokesLayout, StokesModelMono, staggered_velocity_grids
export assemble_steady!, assemble_unsteady!, solve_steady!, solve_unsteady!

abstract type AbstractPressureGauge end

struct PinPressureGauge <: AbstractPressureGauge
    index::Union{Nothing,Int}
end
PinPressureGauge(; index::Union{Nothing,Int}=nothing) = PinPressureGauge(index)

struct MeanPressureGauge <: AbstractPressureGauge end

struct StokesLayout{N}
    nt::Int
    uomega::NTuple{N,UnitRange{Int}}
    ugamma::NTuple{N,UnitRange{Int}}
    pomega::UnitRange{Int}
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
    bc_cut::AbstractBoundary
    gauge::AbstractPressureGauge
    layout::StokesLayout{N}
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    body::BT
end

function _default_force(::Type{T}, ::Val{N}) where {T,N}
    return ntuple(_ -> zero(T), N)
end

function _normalize_bc_tuple(bc_u::NTuple{N,BorderConditions}) where {N}
    return bc_u
end

function _normalize_bc_tuple(bc_u::BorderConditions, ::Val{1})
    return (bc_u,)
end

function _periodic_velocity_flags(bc_u::NTuple{N,BorderConditions}) where {N}
    flags = periodic_flags(bc_u[1], N)
    @inbounds for d in 2:N
        periodic_flags(bc_u[d], N) == flags ||
            throw(ArgumentError("all velocity component border conditions must share identical periodic flags"))
    end
    return flags
end

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
            if (row < pfirst || row > plast) && A.nzval[ptr] != zero(T)
                coupled = true
                break
            end
        end
        pactive[i] = coupled
    end
    @inbounds for i in 1:model.cap_p.ntotal
        active[layout.pomega[i]] = pactive[i]
    end
    return active
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

function _apply_component_velocity_box_bc!(
    A::SparseMatrixCSC{T,Int},
    b::Vector{T},
    gridp::CartesianGrid{N,T},
    cap::AssembledCapacity{N,T},
    mu::T,
    bc::BorderConditions,
    var_uomega::UnitRange{Int},
    row_uomega::UnitRange{Int},
    t::T,
) where {N,T}
    validate_borderconditions!(bc, N)
    li = LinearIndices(cap.nnodes)

    for (side, side_bc) in bc.borders
        side_bc isa AbstractBoundary ||
            throw(ArgumentError("velocity border condition `$side` only supports Dirichlet/Neumann/Periodic"))
        d, is_high, _ = side_info(side, N)
        side_bc isa Periodic && continue

        xyz_d = cap.xyz[d]
        length(xyz_d) >= 2 || throw(ArgumentError("velocity grid must have at least 2 nodes in each direction"))
        Δd = abs(xyz_d[2] - xyz_d[1])
        δ = Δd / T(2)
        x_d = is_high ? gridp.hc[d] : gridp.lc[d]

        for I in each_boundary_cell(cap.nnodes, side)
            i = li[I]
            Aface = cap.buf.A[d][i]
            if !isfinite(Aface) || iszero(Aface)
                continue
            end

            x = cap.C_ω[i]
            xb = SVector{N,T}(ntuple(k -> (k == d ? x_d : x[k]), N))
            row = row_uomega[i]

            if side_bc isa Dirichlet
                val = convert(T, eval_bc(side_bc.value, xb, t))
                a = mu * Aface / δ
                A[row, var_uomega[i]] = A[row, var_uomega[i]] + a
                b[row] += a * val
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

function _apply_velocity_box_bc!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelMono{N,T}, t::T) where {N,T}
    layout = model.layout
    for d in 1:N
        _apply_component_velocity_box_bc!(
            A,
            b,
            model.gridp,
            model.cap_u[d],
            model.mu,
            model.bc_u[d],
            layout.uomega[d],
            layout.uomega[d],
            t,
        )
    end
    return A, b
end

function _first_active_pressure_index(model::StokesModelMono{N,T}) where {N,T}
    pactive = _pressure_activity(model.cap_p)
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

function _apply_pressure_gauge!(A::SparseMatrixCSC{T,Int}, b::Vector{T}, model::StokesModelMono{N,T}) where {N,T}
    layout = model.layout
    nt = model.cap_p.ntotal
    row = first(layout.pomega)

    if model.gauge isa PinPressureGauge
        idx = if !isnothing(model.gauge.index)
            model.gauge.index
        else
            pactive = _pressure_activity(model.cap_p)
            coupled = _first_coupled_pressure_index(A, layout.pomega, pactive)
            isnothing(coupled) ? _first_active_pressure_index(model) : coupled
        end
        1 <= idx <= nt || throw(ArgumentError("pressure pin index must be in 1:$nt"))
        col = layout.pomega[idx]
        _enforce_dirichlet!(A, b, row, col, zero(T))
        return A, b
    elseif model.gauge isa MeanPressureGauge
        @inbounds for j in 1:size(A, 2)
            A[row, j] = zero(T)
        end
        pactive = _pressure_activity(model.cap_p)
        active_idx = findall(pactive)
        if isempty(active_idx)
            A[row, layout.pomega[1]] = one(T)
        else
            w = inv(convert(T, length(active_idx)))
            @inbounds for i in active_idx
                A[row, layout.pomega[i]] = w
            end
        end
        b[row] = zero(T)
        return A, b
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
        -grad_full[rows, :]
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

        cut_vec = _cut_values(model.cap_u[d], model.bc_cut, t)
        _insert_vec!(b, layout.ugamma[d], cut_vec)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], blocks.div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], blocks.div_gamma[d])
    end

    return A, b
end

function assemble_steady!(sys::LinearSystem{T}, model::StokesModelMono{N,T}, t::T=zero(T)) where {N,T}
    blocks = _stokes_blocks(model)
    nsys = nunknowns(model.layout)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)

    _assemble_core!(A, b, model, blocks, t)
    _apply_velocity_box_bc!(A, b, model, t)
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

        cut_next = _cut_values(model.cap_u[d], model.bc_cut, t_next)
        _insert_vec!(b, layout.ugamma[d], cut_next)
    end

    @inbounds for d in 1:N
        _insert_block!(A, layout.pomega, layout.uomega[d], blocks.div_omega[d])
        _insert_block!(A, layout.pomega, layout.ugamma[d], blocks.div_gamma[d])
    end

    _apply_velocity_box_bc!(A, b, model, t_next)
    _apply_pressure_gauge!(A, b, model)

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)

    sys.A = A
    sys.b = b
    length(sys.x) == nsys || (sys.x = zeros(T, nsys))
    sys.cache = nothing
    return sys
end

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

function StokesModelMono(
    cap_p::AssembledCapacity{N,T},
    op_p::DiffusionOps{N,T},
    cap_u::NTuple{N,AssembledCapacity{N,T}},
    op_u::NTuple{N,DiffusionOps{N,T}},
    mu::Real,
    rho::Real;
    force=_default_force(T, Val(N)),
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_cut::AbstractBoundary=Dirichlet(zero(T)),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    geom_method::Symbol=:prebuilt,
    body=nothing,
) where {N,T}
    nt = cap_p.ntotal
    @inbounds for d in 1:N
        cap_u[d].ntotal == nt || throw(ArgumentError("all velocity capacities must have same ntotal as pressure"))
    end

    pflags = _periodic_velocity_flags(bc_u)
    pflags == periodic_flags(bc_u[1], N) || throw(ArgumentError("invalid periodic flags"))

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
        bc_cut,
        gauge,
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
    bc_cut::AbstractBoundary=Dirichlet(zero(T)),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    geom_method::Symbol=:vofijul,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
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
        bc_cut,
        gauge,
        layout,
        pflags,
        geom_method,
        body,
    )
end

end
