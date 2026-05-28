mutable struct GraphLevelSet{N,T,AT}
    axis::Int
    grid::CartesianGrid{N,T}
    xf0::AT
    xf1::AT
    t0::T
    dt::T
    interp::Symbol
    periodic_transverse::Bool
end

function _interp_graph_position(ls::GraphLevelSet{2,T}, xf, x::SVector{2,T}) where {T}
    transverse_axis = ls.axis == 1 ? 2 : 1
    coords = grid1d(ls.grid, transverse_axis)
    s = x[transverse_axis]

    if s <= first(coords)
        return xf[1]
    elseif s >= last(coords)
        return xf[end]
    end

    j = searchsortedlast(coords, s)
    j = min(max(j, 1), length(coords) - 1)
    λ = (s - coords[j]) / (coords[j + 1] - coords[j])
    return (one(T) - λ) * xf[j] + λ * xf[j + 1]
end

function _graph_levelset_value(ls::GraphLevelSet{2,T}, x::SVector{2,T}, t::T) where {T}
    τ = iszero(ls.dt) ? one(T) : clamp((t - ls.t0) / ls.dt, zero(T), one(T))
    h0 = _interp_graph_position(ls, ls.xf0, x)
    h1 = _interp_graph_position(ls, ls.xf1, x)
    h = (one(T) - τ) * h0 + τ * h1
    return x[ls.axis] - h
end

function (ls::GraphLevelSet{N,T})(x...) where {N,T}
    length(x) == N || length(x) == N + 1 ||
        throw(ArgumentError("expected $N space coordinates, optionally followed by time"))
    xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
    t = length(x) == N ? ls.t0 + ls.dt : convert(T, x[N + 1])
    return _graph_levelset_value(ls, xs, t)
end

function update_graph_levelset!(
    ls::GraphLevelSet{N,T},
    xf0,
    xf1,
    t0::T,
    dt::T,
) where {N,T}
    size(xf0) == size(ls.xf0) || throw(ArgumentError("start graph shape mismatch"))
    size(xf1) == size(ls.xf1) || throw(ArgumentError("end graph shape mismatch"))
    ls.xf0 .= T.(xf0)
    ls.xf1 .= T.(xf1)
    ls.t0 = t0
    ls.dt = dt
    return ls
end

mutable struct GlobalHFRep{N,T,AT} <: AbstractInterfaceRep
    grid::CartesianGrid{N,T}
    axis::Int
    phi::Array{T,N}
    xf::AT
    periodic_transverse::Bool
    interp::Symbol
    xf_prev::AT
    dt_prev::T
    body::GraphLevelSet{N,T,AT}
end

function _normalize_ghf_interp(interp::Symbol)
    if interp === :linear || interp === :cubic
        return interp
    elseif interp === :pwlinear
        return :linear
    elseif interp === :cubic_spline
        return :cubic
    end
    throw(ArgumentError("unsupported interpolation `$interp`; expected :linear or :cubic"))
end

function _ghf_phi_from_initializer(phi0::AbstractArray, grid::CartesianGrid{N,T}) where {N,T}
    size(phi0) == grid.n || throw(DimensionMismatch("phi0 size must match grid.n"))
    return convert(Array{T,N}, phi0)
end

function _ghf_phi_from_initializer(phi0::Function, grid::CartesianGrid{N,T}) where {N,T}
    xyz = ntuple(d -> collect(T, grid1d(grid, d)), N)
    phi = Array{T,N}(undef, grid.n...)
    @inbounds for I in CartesianIndices(phi)
        x = ntuple(d -> xyz[d][I[d]], N)
        phi[I] = convert(T, phi0(x...))
    end
    return phi
end

"""
    GlobalHFRep(grid, phi0; axis=1, periodic_transverse=false, interp=:linear, method=:zero_crossing)

Global height-function interface representation following the
`GlobalHeightFunctions.jl` conventions used by PenguinStefan.jl. `rep.phi`
stores the grid signed field, `rep.xf` stores graph positions, and `rep.body`
is the callable level-set object passed to moving Stokes models.
"""
function GlobalHFRep(
    grid::CartesianGrid{N,T},
    phi0;
    axis::Union{Int,Symbol,Val}=1,
    periodic_transverse::Bool=false,
    interp::Symbol=:linear,
    method::Symbol=:zero_crossing,
) where {N,T}
    N == 2 || throw(ArgumentError("GlobalHFRep currently supports 2D graph interfaces in PenguinStokes"))
    axis_idx = GlobalHeightFunctions.axis_to_index(axis, Val(N))
    interp_mode = _normalize_ghf_interp(interp)
    phi = _ghf_phi_from_initializer(phi0, grid)
    xf = GlobalHeightFunctions.xf_from_sdf(phi, grid; axis=axis_idx, method=method)
    periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf)
    xf_prev = copy(xf)
    body = GraphLevelSet{N,T,typeof(xf)}(
        axis_idx,
        grid,
        copy(xf_prev),
        copy(xf),
        zero(T),
        zero(T),
        interp_mode,
        periodic_transverse,
    )
    return GlobalHFRep{N,T,typeof(xf)}(
        grid,
        axis_idx,
        phi,
        xf,
        periodic_transverse,
        interp_mode,
        xf_prev,
        zero(T),
        body,
    )
end

function GlobalHFRep(
    grid::CartesianGrid{N,T},
    axis::Integer,
    xf;
    xf_prev=xf,
    dt_prev::Real=zero(T),
    periodic_transverse::Bool=false,
    interp::Symbol=:linear,
) where {N,T}
    N == 2 || throw(ArgumentError("GlobalHFRep currently supports 2D graph interfaces in PenguinStokes"))
    axis_idx = GlobalHeightFunctions.axis_to_index(axis, Val(N))
    interp_mode = _normalize_ghf_interp(interp)
    xfc = convert(Array{T}, copy(xf))
    xfpc = convert(Array{T}, copy(xf_prev))
    size(xfc) == size(xfpc) || throw(ArgumentError("xf and xf_prev must have the same shape"))
    periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xfc)
    periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xfpc)
    phi = GlobalHeightFunctions.phi_from_xf(
        xfc,
        grid;
        axis=axis_idx,
        interp=interp_mode,
        periodic=periodic_transverse,
    )
    body = GraphLevelSet{N,T,typeof(xfc)}(
        axis_idx,
        grid,
        copy(xfpc),
        copy(xfc),
        zero(T),
        zero(T),
        interp_mode,
        periodic_transverse,
    )
    return GlobalHFRep{N,T,typeof(xfc)}(
        grid,
        axis_idx,
        phi,
        xfc,
        periodic_transverse,
        interp_mode,
        xfpc,
        convert(T, dt_prev),
        body,
    )
end

function predict_xf(rep::GlobalHFRep{N,T}, dt::T) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    xf_guess = copy(rep.xf)
    if rep.dt_prev > zero(T)
        ratio = dt / rep.dt_prev
        xf_guess .= rep.xf .+ ratio .* (rep.xf .- rep.xf_prev)
    end
    rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(xf_guess)
    return xf_guess
end

function commit!(rep::GlobalHFRep{N,T}, xf_new, dt::T) where {N,T}
    size(xf_new) == size(rep.xf) || throw(ArgumentError("graph position shape mismatch"))
    rep.xf_prev .= rep.xf
    rep.xf .= T.(xf_new)
    rep.periodic_transverse && GlobalHeightFunctions.ensure_periodic!(rep.xf)
    rep.dt_prev = dt
    rep.phi .= GlobalHeightFunctions.phi_from_xf(
        rep.xf,
        rep.grid;
        axis=rep.axis,
        interp=rep.interp,
        periodic=rep.periodic_transverse,
    )
    update_graph_levelset!(rep.body, rep.xf, rep.xf, zero(T), zero(T))
    return rep
end

function _set_slab_graph!(rep::GlobalHFRep{N,T}, xf_end, t::T, dt::T) where {N,T}
    update_graph_levelset!(rep.body, rep.xf, xf_end, t, dt)
    rep.phi .= GlobalHeightFunctions.phi_from_xf(
        xf_end,
        rep.grid;
        axis=rep.axis,
        interp=rep.interp,
        periodic=rep.periodic_transverse,
    )
    return rep.body
end

function _column_sum_profile(R::AbstractVector{T}, grid::CartesianGrid{N,T}, axis::Int) where {N,T}
    Rarr = GlobalHeightFunctions.reshape_V(R, grid)
    return GlobalHeightFunctions.column_profile(Rarr; axis=axis)
end

function _column_jacobian(rep::GlobalHFRep{N,T}) where {N,T}
    Δ = meshsize(rep.grid)
    scale = one(T)
    @inbounds for d in 1:N
        d == rep.axis && continue
        scale *= convert(T, Δ[d])
    end
    return fill(scale, size(rep.xf))
end
