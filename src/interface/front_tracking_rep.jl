mutable struct FrontTrackingLevelSet{T}
    points0::Vector{SVector{2,T}}
    points1::Vector{SVector{2,T}}
    t0::T
    dt::T
end

function FrontTrackingLevelSet(points::AbstractVector{<:SVector{2,T}}; t0::T=zero(T), dt::T=zero(T)) where {T}
    pts = Vector{SVector{2,T}}(points)
    return FrontTrackingLevelSet{T}(copy(pts), copy(pts), t0, dt)
end

function _ft_segment_distance(x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    ab = b - a
    den = dot(ab, ab)
    iszero(den) && return norm(x - a)
    θ = clamp(dot(x - a, ab) / den, zero(T), one(T))
    return norm(x - (a + θ * ab))
end

function _ft_point_inside_polygon(x::SVector{2,T}, pts::Vector{SVector{2,T}}) where {T}
    inside = false
    n = length(pts)
    j = n
    @inbounds for i in 1:n
        pi = pts[i]
        pj = pts[j]
        crosses = (pi[2] > x[2]) != (pj[2] > x[2])
        if crosses
            xcross = (pj[1] - pi[1]) * (x[2] - pi[2]) / (pj[2] - pi[2]) + pi[1]
            x[1] < xcross && (inside = !inside)
        end
        j = i
    end
    return inside
end

function _ft_signed_distance(x::SVector{2,T}, pts::Vector{SVector{2,T}}) where {T}
    n = length(pts)
    n >= 3 || throw(ArgumentError("closed front level-set requires at least 3 markers"))
    dmin = typemax(T)
    @inbounds for i in 1:n
        d = _ft_segment_distance(x, pts[i], pts[mod1(i + 1, n)])
        d < dmin && (dmin = d)
    end
    return _ft_point_inside_polygon(x, pts) ? -dmin : dmin
end

function (body::FrontTrackingLevelSet{T})(x::T, y::T) where {T}
    return _ft_signed_distance(SVector{2,T}(x, y), body.points1)
end

function (body::FrontTrackingLevelSet{T})(x::T, y::T, t::T) where {T}
    pt = SVector{2,T}(x, y)
    if iszero(body.dt)
        return _ft_signed_distance(pt, body.points1)
    end
    θ = clamp((t - body.t0) / body.dt, zero(T), one(T))
    n = length(body.points0)
    dmin = typemax(T)
    inside = false
    j = n
    @inbounds for i in 1:n
        pi = muladd(θ, body.points1[i] - body.points0[i], body.points0[i])
        pj = muladd(θ, body.points1[j] - body.points0[j], body.points0[j])
        d = _ft_segment_distance(pt, pi, muladd(θ, body.points1[mod1(i + 1, n)] - body.points0[mod1(i + 1, n)], body.points0[mod1(i + 1, n)]))
        d < dmin && (dmin = d)
        crosses = (pi[2] > pt[2]) != (pj[2] > pt[2])
        if crosses
            xcross = (pj[1] - pi[1]) * (pt[2] - pi[2]) / (pj[2] - pi[2]) + pi[1]
            pt[1] < xcross && (inside = !inside)
        end
        j = i
    end
    return inside ? -dmin : dmin
end

function update_front_levelset!(
    body::FrontTrackingLevelSet{T},
    points0::AbstractVector{<:SVector{2,T}},
    points1::AbstractVector{<:SVector{2,T}},
    t0::T,
    dt::T,
) where {T}
    body.points0 = Vector{SVector{2,T}}(points0)
    body.points1 = Vector{SVector{2,T}}(points1)
    body.t0 = t0
    body.dt = dt
    return body
end

struct FrontTrackingRep{T,S} <: AbstractInterfaceRep
    grid::CartesianGrid{2,T}
    state::S
    body::FrontTrackingLevelSet{T}
    coupling::Symbol
    last_normal_speed::Vector{T}
end

function FrontTrackingRep(
    grid::CartesianGrid{2,T},
    front;
    coupling::Symbol=:ft_redistribute,
    t::Real=0,
) where {T}
    coupling in (:ft_redistribute, :ft_lm, :explicit) ||
        throw(ArgumentError("coupling must be :ft_redistribute, :ft_lm, or :explicit"))
    state = front isa FrontTrackingMethods.FrontState ? front : FrontTrackingMethods.FrontState(front; t=t)
    pts = SVector{2,T}.(FrontTrackingMethods.vertex_coordinates(state))
    body = FrontTrackingLevelSet(pts)
    return FrontTrackingRep{T,typeof(state)}(grid, state, body, coupling, zeros(T, length(pts)))
end

front_points(rep::FrontTrackingRep{T}) where {T} =
    SVector{2,T}.(FrontTrackingMethods.vertex_coordinates(rep.state))

function _set_front_points!(rep::FrontTrackingRep{T}, pts::Vector{SVector{2,T}}; t=nothing) where {T}
    FrontTrackingMethods.set_vertex_coordinates!(rep.state, pts)
    FrontTrackingMethods.refresh_geometry!(rep.state)
    t === nothing || (rep.state.t = Float64(t))
    return rep
end

function _front_phi_on_grid(grid::CartesianGrid{2,T}, pts::Vector{SVector{2,T}}) where {T}
    xs = grid1d(grid)
    phi = Array{T}(undef, grid.n...)
    @inbounds for I in CartesianIndices(phi)
        x = SVector{2,T}(xs[1][I[1]], xs[2][I[2]])
        phi[I] = _ft_signed_distance(x, pts)
    end
    return phi
end

front_phi(rep::FrontTrackingRep{T}) where {T} = _front_phi_on_grid(rep.grid, front_points(rep))
