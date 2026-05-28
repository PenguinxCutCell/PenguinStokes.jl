mutable struct StokesGHFCouplingHistory{T}
    iter::Vector{Int}
    gcl_norm::Vector{T}
    kin_norm::Vector{T}
    div_norm::Vector{T}
    step_norm::Vector{T}
    damping::Vector{T}
end

StokesGHFCouplingHistory(::Type{T}) where {T} =
    StokesGHFCouplingHistory{T}(Int[], T[], T[], T[], T[], T[])

function _push_history!(
    history::StokesGHFCouplingHistory{T};
    iter::Int,
    gcl::T,
    kin::T,
    div::T,
    step::T,
    damping::T,
) where {T}
    push!(history.iter, iter)
    push!(history.gcl_norm, gcl)
    push!(history.kin_norm, kin)
    push!(history.div_norm, div)
    push!(history.step_norm, step)
    push!(history.damping, damping)
    return history
end
