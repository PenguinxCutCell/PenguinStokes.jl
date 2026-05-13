function _dynamic_row_mask(
    A::SparseMatrixCSC{T,Int},
    b::AbstractVector{T},
    rows::UnitRange{Int},
) where {T}
    mask = BitVector(undef, length(rows))
    @inbounds for (k, row) in enumerate(rows)
        mask[k] = !_is_zero_identity_row(A, b, row)
    end
    return mask
end

function _gauge_candidate_rows(
    A::SparseMatrixCSC{T,Int},
    layout::StokesLayoutTwoPhase{N},
) where {N,T}
    rows = Int[]
    pstart = first(layout.pomega1)
    @inbounds for row in layout.pomega1
        idx, _ = findnz(A[row, :])
        isempty(idx) && continue
        has_vel = false
        has_p = false
        for col in idx
            if col < pstart
                has_vel = true
            else
                has_p = true
            end
        end
        (!has_vel && has_p) && push!(rows, row)
    end
    return rows
end

"""
    build_static_circle_equilibrium_state(model; sigma, R, gauge=:mean, sys=nothing)

Build an exact-candidate equilibrium state for fixed-interface two-phase static
circle balance:
- zero bulk and interface velocities
- phase-wise constant pressures with `p_in - p_out = sigma / R`
- gauge row satisfied on the assembled steady system.
"""
function build_static_circle_equilibrium_state(
    model::StokesModelTwoPhase{N,T};
    sigma::Real,
    R::Real,
    gauge::Symbol=:mean,
    sys::Union{Nothing,LinearSystem{T}}=nothing,
) where {N,T}
    nsys = nunknowns(model.layout)
    sys_local = if isnothing(sys)
        tmp = LinearSystem(spzeros(T, nsys, nsys), zeros(T, nsys))
        assemble_steady!(tmp, model, zero(T))
        tmp
    else
        sys
    end

    A = sys_local.A
    b = sys_local.b
    layout = model.layout
    dp = convert(T, sigma / R)

    dyn_p1 = _dynamic_row_mask(A, b, layout.pomega1)
    dyn_p2 = _dynamic_row_mask(A, b, layout.pomega2)

    gauge === :mean || gauge === :pin ||
        throw(ArgumentError("gauge must be :mean or :pin"))

    # Start from a physically balanced pressure split for the two phases.
    p1c = dp
    p2c = zero(T)
    if gauge === :pin
        # For pin gauge on phase-1, an additional global shift is applied below
        # to satisfy the pinned row exactly.
        p1c = dp
        p2c = zero(T)
    end

    xeq = zeros(T, nsys)
    shift_cols = Int[]
    @inbounds for i in 1:length(layout.pomega1)
        if dyn_p1[i]
            col = layout.pomega1[i]
            xeq[col] = p1c
            push!(shift_cols, col)
        end
        if dyn_p2[i]
            col = layout.pomega2[i]
            xeq[col] = p2c
            push!(shift_cols, col)
        end
    end

    # Enforce gauge rows exactly by shifting both phase constants together.
    gauge_rows = _gauge_candidate_rows(A, layout)
    atol = sqrt(eps(T))
    @inbounds for row in gauge_rows
        isempty(shift_cols) && break
        denom = zero(T)
        for col in shift_cols
            denom += A[row, col]
        end
        abs(denom) <= atol && continue

        idx, vals = findnz(A[row, :])
        rrow = -b[row]
        for k in eachindex(idx)
            rrow += vals[k] * xeq[idx[k]]
        end
        s = rrow / denom
        for col in shift_cols
            xeq[col] -= s
        end
    end

    return xeq
end

function _assemble_two_phase_stages(
    model::StokesModelTwoPhase{N,T},
    t::T,
) where {N,T}
    nsys = nunknowns(model.layout)
    blocks = _stokes_blocks(model)
    A = spzeros(T, nsys, nsys)
    b = zeros(T, nsys)
    stages = Dict{Symbol,Tuple{SparseMatrixCSC{T,Int},Vector{T}}}()

    _assemble_core!(A, b, model, blocks, t)
    stages[:core] = (copy(A), copy(b))

    traction_locked_rows = _apply_traction_box_bc!(A, b, model, t)
    stages[:traction_bc] = (copy(A), copy(b))

    _apply_symmetry_box_bc!(A, b, model, t, traction_locked_rows)
    stages[:symmetry_bc] = (copy(A), copy(b))

    _apply_velocity_box_bc!(A, b, model, t; locked_rows=traction_locked_rows)
    stages[:velocity_bc] = (copy(A), copy(b))

    _apply_pressure_box_bc!(A, b, model, t)
    stages[:pressure_bc] = (copy(A), copy(b))

    _apply_pressure_gauge!(A, b, model)
    stages[:gauge] = (copy(A), copy(b))

    active_rows = _stokes_row_activity(model, A)
    A, b = _apply_row_identity_constraints!(A, b, active_rows)
    stages[:row_identity] = (A, b)
    return stages, active_rows
end

"""
    exact_equilibrium_residual(sys, model, xeq; t=0)

Return exact-candidate defect diagnostics `A*xeq - b` on the assembled system.
Also returns residuals for intermediate assembly stages and stage increments.
"""
function exact_equilibrium_residual(
    sys::LinearSystem{T},
    model::StokesModelTwoPhase{N,T},
    xeq::AbstractVector{T};
    t::T=zero(T),
) where {N,T}
    length(xeq) == nunknowns(model.layout) ||
        throw(DimensionMismatch("xeq length must match system unknown count"))

    r = sys.A * xeq - sys.b
    stages, active_rows = _assemble_two_phase_stages(model, t)

    stage_residuals = Dict{Symbol,Vector{T}}()
    for (name, (A, b)) in stages
        stage_residuals[name] = A * xeq - b
    end

    order = (:core, :traction_bc, :symmetry_bc, :velocity_bc, :pressure_bc, :gauge, :row_identity)
    incr = Dict{Symbol,Vector{T}}()
    prev = zeros(T, length(r))
    for name in order
        curr = stage_residuals[name]
        incr[name] = curr .- prev
        prev = curr
    end

    return (
        residual=r,
        stage_residuals=stage_residuals,
        incremental_residuals=incr,
        active_rows=active_rows,
    )
end

function _row_coords_for_block(
    model::StokesModelTwoPhase{N,T},
    block::Symbol,
    local_i::Int,
) where {N,T}
    if block === :uomega1_x
        return model.cap_u1[1].C_ω[local_i]
    elseif block === :uomega1_y
        return model.cap_u1[2].C_ω[local_i]
    elseif block === :uomega2_x
        return model.cap_u2[1].C_ω[local_i]
    elseif block === :uomega2_y
        return model.cap_u2[2].C_ω[local_i]
    elseif block === :ugamma1_x || block === :ugamma1_y || block === :ugamma2_x || block === :ugamma2_y
        return model.cap_p1.C_γ[local_i]
    elseif block === :pomega1
        return model.cap_p1.C_ω[local_i]
    elseif block === :pomega2
        return model.cap_p2.C_ω[local_i]
    end
    return nothing
end

"""
    block_residual_report(sys, model, r; topk=10, io=stdout)

Report block-wise residual norms and worst rows for a two-phase static audit.
"""
function block_residual_report(
    sys::LinearSystem{T},
    model::StokesModelTwoPhase{N,T},
    r::AbstractVector{T};
    topk::Int=10,
    io::IO=stdout,
) where {N,T}
    layout = model.layout
    blocks = (
        (:uomega1_x, layout.uomega1[1]),
        (:uomega1_y, layout.uomega1[2]),
        (:uomega2_x, layout.uomega2[1]),
        (:uomega2_y, layout.uomega2[2]),
        (:ugamma1_x, layout.ugamma1[1]),
        (:ugamma1_y, layout.ugamma1[2]),
        (:ugamma2_x, layout.ugamma2[1]),
        (:ugamma2_y, layout.ugamma2[2]),
        (:pomega1, layout.pomega1),
        (:pomega2, layout.pomega2),
    )

    summary = Dict{Symbol,NamedTuple}()
    println(io, "Block residual report:")
    for (name, rows) in blocks
        rv = @view r[rows]
        maxabs = maximum(abs, rv)
        l2 = norm(rv)
        active_count = 0
        @inbounds for row in rows
            active_count += !_is_zero_identity_row(sys.A, sys.b, row)
        end
        println(io, "  ", name, ": maxabs=", maxabs, ", l2=", l2, ", active_rows=", active_count, "/", length(rows))

        order = sortperm(abs.(rv); rev=true)
        nshow = min(topk, length(order))
        shown = 0
        @inbounds for k in 1:nshow
            local_i = order[k]
            val = rv[local_i]
            abs(val) == zero(T) && continue
            row = rows[local_i]
            x = _row_coords_for_block(model, name, local_i)
            println(io, "    row=", row, " local=", local_i, " resid=", val, " x=", x)
            shown += 1
        end
        shown == 0 && println(io, "    top rows are zero.")

        summary[name] = (maxabs=maxabs, l2=l2, active_rows=active_count, total_rows=length(rows))
    end
    return summary
end

@inline function _mean_values(vals::AbstractVector{T}, idx::Vector{Int}) where {T}
    isempty(idx) && return zero(T)
    s = zero(T)
    @inbounds for i in idx
        s += vals[i]
    end
    return s / convert(T, length(idx))
end

@inline function _std_values(vals::AbstractVector{T}, idx::Vector{Int}) where {T}
    n = length(idx)
    n <= 1 && return zero(T)
    μ = _mean_values(vals, idx)
    s2 = zero(T)
    @inbounds for i in idx
        d = vals[i] - μ
        s2 += d * d
    end
    return sqrt(s2 / convert(T, n - 1))
end

@inline function _weighted_mean(vals::AbstractVector{T}, idx::Vector{Int}, w::AbstractVector{T}) where {T}
    isempty(idx) && return zero(T)
    s = zero(T)
    ws = zero(T)
    @inbounds for i in idx
        wi = w[i]
        vi = vals[i]
        s += wi * vi
        ws += wi
    end
    return ws > zero(T) ? s / ws : _mean_values(vals, idx)
end

function _phase_pressure_stats(
    p::AbstractVector{T},
    cap::AssembledCapacity{N,T},
) where {N,T}
    idx_all = findall(_pressure_activity(cap))
    idx_cut = Int[]
    idx_full = Int[]
    @inbounds for i in idx_all
        Γi = cap.buf.Γ[i]
        if isfinite(Γi) && Γi > zero(T)
            push!(idx_cut, i)
        else
            push!(idx_full, i)
        end
    end

    V = cap.buf.V
    stats = Dict{Symbol,NamedTuple}()
    for (tag, idx) in ((:all, idx_all), (:full, idx_full), (:cut, idx_cut))
        if isempty(idx)
            stats[tag] = (count=0, mean=zero(T), mean_weighted=zero(T), std=zero(T), min=zero(T), max=zero(T))
            continue
        end
        μ = _mean_values(p, idx)
        σ = _std_values(p, idx)
        pmin = p[idx[1]]
        pmax = p[idx[1]]
        @inbounds for i in idx
            v = p[i]
            v < pmin && (pmin = v)
            v > pmax && (pmax = v)
        end
        stats[tag] = (
            count=length(idx),
            mean=μ,
            mean_weighted=_weighted_mean(p, idx, V),
            std=σ,
            min=pmin,
            max=pmax,
        )
    end
    return stats
end

"""
    pressure_flatness_report(model, sys; io=stdout, verbose=false)

Return phase-wise pressure flatness diagnostics on active pressure cells, with
all/full/cut splits and simple/volume-weighted jump estimates.
"""
function pressure_flatness_report(
    model::StokesModelTwoPhase{N,T},
    sys::LinearSystem{T};
    io::IO=stdout,
    verbose::Bool=false,
) where {N,T}
    p1 = sys.x[model.layout.pomega1]
    p2 = sys.x[model.layout.pomega2]
    s1 = _phase_pressure_stats(p1, model.cap_p1)
    s2 = _phase_pressure_stats(p2, model.cap_p2)

    jump = Dict{Symbol,NamedTuple}()
    for tag in (:all, :full, :cut)
        jump[tag] = (
            simple=s1[tag].mean - s2[tag].mean,
            weighted=s1[tag].mean_weighted - s2[tag].mean_weighted,
        )
    end

    if verbose
        println(io, "Pressure flatness report:")
        for tag in (:all, :full, :cut)
            println(io, "  ", tag, ": p1(mean,std,min,max)=(", s1[tag].mean, ", ", s1[tag].std, ", ", s1[tag].min, ", ", s1[tag].max, ")",
                ", p2(mean,std,min,max)=(", s2[tag].mean, ", ", s2[tag].std, ", ", s2[tag].min, ", ", s2[tag].max, ")",
                ", jump(simple,weighted)=(", jump[tag].simple, ", ", jump[tag].weighted, ")")
        end
    end

    return (phase1=s1, phase2=s2, jump=jump)
end

"""
    local_capillary_balance_report(model, sigma, R; center=SVector(0,0), topk=20, io=stdout)

Local interface force-balance audit on active interface-support cells.
"""
function local_capillary_balance_report(
    model::StokesModelTwoPhase{N,T},
    sigma::Real,
    R::Real;
    center::SVector{N,<:Real}=SVector{N,Float64}(ntuple(_ -> 0.0, N)),
    t::T=zero(T),
    topk::Int=20,
    io::IO=stdout,
) where {N,T}
    dp = convert(T, sigma / R)
    xc = SVector{N,T}(ntuple(d -> convert(T, center[d]), N))
    iface = findall(isfinite.(model.cap_p1.buf.Γ) .& (model.cap_p1.buf.Γ .> zero(T)))

    rows = Vector{NamedTuple}(undef, length(iface))
    @inbounds for (k, i) in enumerate(iface)
        Γi = model.cap_p1.buf.Γ[i]
        xγ = model.cap_p1.C_γ[i]
        nbar = model.cap_p1.n_γ[i]
        fraw = _interface_force_vector(model.interface_force, xγ, t)
        fdisc = SVector{N,T}(ntuple(d -> _consistent_interface_force_component(fraw, nbar, d), N))

        dx = xγ - xc
        rr = norm(dx)
        ncir = rr > zero(T) ? dx / rr : SVector{N,T}(ntuple(_ -> zero(T), N))

        Fraw = Γi .* fraw
        Fdisc = Γi .* fdisc
        Fpress = -Γi * dp .* nbar
        rows[k] = (
            i=i,
            gamma=Γi,
            xgamma=xγ,
            n_discrete=nbar,
            n_circle=ncir,
            n_mismatch=nbar - ncir,
            force_raw=Fraw,
            force_discrete=Fdisc,
            force_pressure=Fpress,
            force_mismatch_raw=Fraw - Fpress,
            force_mismatch_discrete=Fdisc - Fpress,
        )
    end

    order = sortperm(rows; by=r -> norm(r.force_mismatch_raw), rev=true)
    println(io, "Local capillary balance report:")
    println(io, "  interface cells=", length(rows), ", dp_th=", dp)
    for k in 1:min(topk, length(order))
        r = rows[order[k]]
        println(io, "  i=", r.i, " Γ=", r.gamma, " xγ=", r.xgamma,
            " nΔ=", r.n_mismatch,
            " Fraw-Fp=", r.force_mismatch_raw,
            " Fdisc-Fp=", r.force_mismatch_discrete)
    end
    return rows
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

function _pressure_gradient_reconstruction(
    model::StokesModelMono{N,T},
    pω::AbstractVector{T},
    reconstruction::Symbol,
) where {N,T}
    nt = model.cap_p.ntotal
    pγ_seed = zeros(T, nt)
    if reconstruction === :linear
        pγ_seed .= pω
    elseif reconstruction !== :none
        throw(ArgumentError("unknown pressure reconstruction `$reconstruction`; expected :none or :linear"))
    end
    return _scalar_gradient(model.op_p, pω, pγ_seed)
end

"""
    embedded_boundary_pressure(model, x; pressure_reconstruction=:linear)
    embedded_boundary_pressure(model, sys; pressure_reconstruction=:linear)

Reconstruct pressure traces on embedded-boundary pressure cells for
`StokesModelMono`.

The pressure unknown is the bulk/cell pressure `pω`, and interface pressure is
reconstructed at the embedded-interface centroid `C_γ`:
- `:none`: `pγ = pω[i]`
- `:linear`: `pγ = pω[i] + ∇p[i] ⋅ (C_γ[i] - C_ω[i])`

Returns a named tuple with:
- `pressure`: reconstructed `pγ` on pressure cells (`NaN` away from interface)
- `interface_indices`: active embedded-interface pressure-cell indices
- `centers`: interface centroids `C_γ` on active indices
- `normals`: interface normals `n_γ` on active indices
- `measure`: interface measure `Γ` on active indices
- `force_density`: pressure-force contributions `(-pγ*n_γ)*Γ` on pressure cells
- `force`: integrated pressure force
"""
function embedded_boundary_pressure(
    model::StokesModelMono{N,T},
    x::AbstractVector;
    pressure_reconstruction::Symbol=:linear,
) where {N,T}
    nsys = nunknowns(model.layout)
    length(x) == nsys || throw(DimensionMismatch("state length must be $nsys"))
    nt = model.cap_p.ntotal

    pω = Vector{T}(x[model.layout.pomega])
    grad_p = _pressure_gradient_reconstruction(model, pω, pressure_reconstruction)

    pressure = fill(T(NaN), nt)
    zero_vec = _zero_svector(T, Val(N))
    force_density = fill(zero_vec, nt)
    interface_indices = Int[]
    centers = SVector{N,T}[]
    normals = SVector{N,T}[]
    measure = T[]

    Fp = zeros(T, N)

    cap = model.cap_p
    @inbounds for i in 1:cap.ntotal
        _is_interface_index(cap, i) || continue
        push!(interface_indices, i)

        Γi = cap.buf.Γ[i]
        nγi = cap.n_γ[i]
        xγi = cap.C_γ[i]
        pγi = _pressure_trace(model, pω, grad_p, i, xγi, pressure_reconstruction)
        fpvec = Γi .* (-pγi .* nγi)

        pressure[i] = pγi
        force_density[i] = fpvec
        Fp .+= fpvec

        push!(centers, xγi)
        push!(normals, nγi)
        push!(measure, Γi)
    end

    return (
        pressure=pressure,
        interface_indices=interface_indices,
        centers=centers,
        normals=normals,
        measure=measure,
        force_density=force_density,
        force=SVector{N,T}(Tuple(Fp)),
    )
end

function embedded_boundary_pressure(
    model::StokesModelMono{N,T},
    sys::LinearSystem{T};
    pressure_reconstruction::Symbol=:linear,
) where {N,T}
    return embedded_boundary_pressure(model, sys.x; pressure_reconstruction=pressure_reconstruction)
end

function _apply_force_convention(
    F::SVector{N,T},
    Fp::SVector{N,T},
    Fμ::SVector{N,T},
    convention::Symbol,
) where {N,T}
    if convention === :on_fluid
        return (force=F, force_pressure=Fp, force_viscous=Fμ)
    elseif convention === :on_body
        return (force=-F, force_pressure=-Fp, force_viscous=-Fμ)
    end
    throw(ArgumentError("convention must be :on_fluid or :on_body"))
end

function _nonzero_row_mask(A::SparseMatrixCSC{T,Int}) where {T}
    mask = falses(size(A, 1))
    rows, _, vals = findnz(A)
    @inbounds for k in eachindex(rows)
        if vals[k] != zero(T)
            mask[rows[k]] = true
        end
    end
    return mask
end

@inline function _force_convention_sign(::Type{T}, convention::Symbol) where {T}
    if convention === :on_fluid
        return one(T)
    elseif convention === :on_body
        return -one(T)
    end
    throw(ArgumentError("convention must be :on_fluid or :on_body"))
end

"""
    embedded_force_balance_density(model, x; mu=model.mu)

Return local embedded-boundary force contributions from the cut-boundary terms
of the discrete momentum balance. The result is a named tuple with `total`,
`pressure`, and `viscous`, each an `N`-tuple of vectors on the corresponding
staggered velocity component grid.

The viscous contribution is computed as the cut-face part of the assembled
diffusive flux,

```
mu * G_u[d]' * I_gamma[d] * Winv_u[d] * (G_u[d] * u_omega[d] + H_u[d] * u_gamma[d])
```

where `I_gamma[d]` keeps only face rows touched by `H_u[d]`. This is the exact
difference between the full viscous operator and the same operator with cut-face
flux rows removed.
"""
function embedded_force_balance_density(
    model::StokesModelMono{N,T},
    x::AbstractVector;
    mu::Real=model.mu,
) where {N,T}
    nsys = nunknowns(model.layout)
    length(x) == nsys || throw(DimensionMismatch("state length must be $nsys"))

    layout = model.layout
    nt = model.cap_p.ntotal
    μ = convert(T, mu)
    pω = Vector{T}(x[layout.pomega])

    r_pressure = ntuple(_ -> zeros(T, nt), N)
    r_viscous = ntuple(_ -> zeros(T, nt), N)
    r_total = ntuple(_ -> zeros(T, nt), N)

    @inbounds for d in 1:N
        opu = model.op_u[d]
        uω = Vector{T}(x[layout.uomega[d]])
        uγ = Vector{T}(x[layout.ugamma[d]])

        face_flux = opu.Winv * (opu.G * uω + opu.H * uγ)
        cut_rows = _nonzero_row_mask(opu.H)
        @inbounds for i in eachindex(face_flux)
            cut_rows[i] || (face_flux[i] = zero(T))
        end
        rμ = μ .* (opu.G' * face_flux)

        rows = ((d - 1) * nt + 1):(d * nt)
        Hp_d = model.op_p.H[rows, :]
        rp = -(Hp_d * pω)

        r_viscous[d] .= rμ
        r_pressure[d] .= rp
        r_total[d] .= rμ .+ rp
    end

    return (
        total=r_total,
        pressure=r_pressure,
        viscous=r_viscous,
    )
end

function embedded_force_balance_density(
    model::StokesModelMono{N,T},
    sys::LinearSystem{T};
    kwargs...,
) where {N,T}
    return embedded_force_balance_density(model, sys.x; kwargs...)
end

function integrated_embedded_torque_balance(
    model::StokesModelMono{2,T},
    q,
    origin::SVector{2,T},
    convention::Symbol,
) where {T}
    τp = zero(T)
    τμ = zero(T)

    @inbounds for i in eachindex(q.total[1])
        xγ = model.cap_u[1].C_γ[i]
        (isfinite(xγ[1]) && isfinite(xγ[2])) || continue
        y = xγ[2] - origin[2]
        τp -= y * q.pressure[1][i]
        τμ -= y * q.viscous[1][i]
    end

    @inbounds for i in eachindex(q.total[2])
        xγ = model.cap_u[2].C_γ[i]
        (isfinite(xγ[1]) && isfinite(xγ[2])) || continue
        xcoord = xγ[1] - origin[1]
        τp += xcoord * q.pressure[2][i]
        τμ += xcoord * q.viscous[2][i]
    end

    s = _force_convention_sign(T, convention)
    τ = τp + τμ
    return (
        torque=s * τ,
        torque_pressure=s * τp,
        torque_viscous=s * τμ,
    )
end

function integrated_embedded_torque_balance(
    model::StokesModelMono{3,T},
    q,
    origin::SVector{3,T},
    convention::Symbol,
) where {T}
    Mp = SVector{3,T}(zero(T), zero(T), zero(T))
    Mμ = SVector{3,T}(zero(T), zero(T), zero(T))

    @inbounds for d in 1:3
        ed = SVector{3,T}(ntuple(k -> k == d ? one(T) : zero(T), 3))
        for i in eachindex(q.total[d])
            xγ = model.cap_u[d].C_γ[i]
            (isfinite(xγ[1]) && isfinite(xγ[2]) && isfinite(xγ[3])) || continue
            r = SVector{3,T}(
                xγ[1] - origin[1],
                xγ[2] - origin[2],
                xγ[3] - origin[3],
            )
            Mp += cross(r, q.pressure[d][i] * ed)
            Mμ += cross(r, q.viscous[d][i] * ed)
        end
    end

    s = _force_convention_sign(T, convention)
    M = Mp + Mμ
    return (
        torque=s * M,
        torque_pressure=s * Mp,
        torque_viscous=s * Mμ,
    )
end

"""
    integrated_embedded_torque_balance(model, x; x0=nothing, convention=:on_body, mu=model.mu)

Return the first moment of the local balance-force density about `x0`. Force
component `d` is placed at the embedded-interface centroid on the corresponding
staggered velocity capacity `model.cap_u[d].C_γ`.
"""
function integrated_embedded_torque_balance(
    model::StokesModelMono{N,T},
    x::AbstractVector;
    x0=nothing,
    convention::Symbol=:on_body,
    mu::Real=model.mu,
) where {N,T}
    N == 2 || N == 3 || throw(ArgumentError("torque only implemented for N=2 or N=3"))
    q = embedded_force_balance_density(model, x; mu=mu)
    origin = _as_origin(x0, T, Val(N))
    return integrated_embedded_torque_balance(model, q, origin, convention)
end

function integrated_embedded_torque_balance(
    model::StokesModelMono{N,T},
    sys::LinearSystem{T};
    kwargs...,
) where {N,T}
    return integrated_embedded_torque_balance(model, sys.x; kwargs...)
end

"""
    integrated_embedded_force_balance(model, x; convention=:on_body, mu=model.mu, x0=nothing, torque_method=:balance)

Return the integrated embedded-boundary force from the cut-boundary terms in
the discrete momentum balance. For velocity component `d`, this sums

```
mu * G_u[d]' * I_gamma[d] * Winv_u[d] * (G_u[d] * u_omega[d] + H_u[d] * u_gamma[d])
    - H_p[d] * p_omega
```

over the velocity control volumes. `I_gamma[d]` keeps the operator rows whose
velocity-grid `H_u[d]` entry participates in the embedded-boundary closure, so
the viscous part includes both sides of the cut-face flux. This uses the same
`G`, `H`, and `Winv` operators as the Stokes assembly instead of reconstructing
`sigma*n` locally on the embedded boundary.

`convention=:on_fluid` returns the force contribution appearing in the fluid
momentum balance. `convention=:on_body` returns its opposite. With the default
`torque_method=:balance`, torque is the first moment of the same local balance
density on the staggered velocity interface grids.
"""
function integrated_embedded_force_balance(
    model::StokesModelMono{N,T},
    x::AbstractVector;
    convention::Symbol=:on_body,
    mu::Real=model.mu,
    x0=nothing,
    torque_method::Symbol=:balance,
) where {N,T}
    nsys = nunknowns(model.layout)
    length(x) == nsys || throw(DimensionMismatch("state length must be $nsys"))

    q = embedded_force_balance_density(model, x; mu=mu)
    Fp = zeros(T, N)
    Fμ = zeros(T, N)

    @inbounds for d in 1:N
        Fμ[d] = sum(q.viscous[d])
        Fp[d] = sum(q.pressure[d])
    end

    Fμv = SVector{N,T}(Tuple(Fμ))
    Fpv = SVector{N,T}(Tuple(Fp))
    Fv = Fμv + Fpv
    out = _apply_force_convention(Fv, Fpv, Fμv, convention)

    torque = if torque_method === :none
        N == 2 ? zero(T) : (N == 3 ? SVector{3,T}(zero(T), zero(T), zero(T)) : nothing)
    elseif torque_method === :balance
        origin = _as_origin(x0, T, Val(N))
        integrated_embedded_torque_balance(model, q, origin, convention).torque
    else
        throw(ArgumentError("torque_method must be :balance or :none"))
    end

    return (
        force=out.force,
        force_pressure=out.force_pressure,
        force_viscous=out.force_viscous,
        torque=torque,
    )
end

function integrated_embedded_force_balance(
    model::StokesModelMono{N,T},
    sys::LinearSystem{T};
    kwargs...,
) where {N,T}
    return integrated_embedded_force_balance(model, sys.x; kwargs...)
end

"""
    integrated_embedded_force(model, x; kwargs...)

Return integrated embedded-boundary force components and torque.
"""
function integrated_embedded_force(model::StokesModelMono, x::AbstractVector; kwargs...)
    return integrated_embedded_force_balance(model, x; kwargs...)
end

function integrated_embedded_force(model::StokesModelMono{N,T}, sys::LinearSystem{T}; kwargs...) where {N,T}
    return integrated_embedded_force(model, sys.x; kwargs...)
end
