struct StokesGCLTerms{T}
    ΔV::Vector{T}
    Fω::Vector{T}
    FΓ::Vector{T}
    R_gcl::Vector{T}
    R_kin::Vector{T}
    R_div::Vector{T}
end

function _stokes_gcl_terms(
    op_p_slab::DiffusionOps{N,T},
    Vn1::AbstractVector{T},
    x_new::AbstractVector,
    Vn::AbstractVector,
    uomega::NTuple{N,UnitRange{Int}},
    ugamma::NTuple{N,UnitRange{Int}},
    nt::Int,
) where {N,T}
    length(Vn) == nt || throw(ArgumentError("Vn length ($(length(Vn))) must match pressure ntotal ($nt)"))
    length(Vn1) == nt || throw(ArgumentError("Vn1 length ($(length(Vn1))) must match pressure ntotal ($nt)"))

    ΔV = Vector{T}(undef, nt)
    @inbounds for i in 1:nt
        ΔV[i] = (isfinite(Vn1[i]) && isfinite(Vn[i])) ? Vn1[i] - convert(T, Vn[i]) : zero(T)
    end
    Fω = zeros(T, nt)
    FΓ = zeros(T, nt)

    @inbounds for d in 1:N
        rows = ((d - 1) * nt + 1):(d * nt)
        Gd = op_p_slab.G[rows, :]
        Hd = op_p_slab.H[rows, :]

        uωd = view(x_new, uomega[d])
        uγd = view(x_new, ugamma[d])

        Fω .+= -(transpose(Gd) + transpose(Hd)) * uωd
        FΓ .+= sparse(transpose(Hd)) * uγd
    end

    R_gcl = ΔV .+ Fω
    R_kin = ΔV .- FΓ
    R_div = Fω .+ FΓ
    return StokesGCLTerms(ΔV, Fω, FΓ, R_gcl, R_kin, R_div)
end

"""
    stokes_gcl_terms_mono(model, x_new, Vn)

Compute the moving monophasic conservative GCL residual with the same slab
pressure-row convention used by `assemble_unsteady_moving!`.
"""
function stokes_gcl_terms_mono(
    model::MovingStokesModelMono{N,T},
    x_new::AbstractVector,
    Vn::AbstractVector,
) where {N,T}
    op_p_slab = something(model.op_p_slab)
    cap_p_end = something(model.cap_p_end)
    layout = model.layout
    return _stokes_gcl_terms(
        op_p_slab,
        cap_p_end.buf.V,
        x_new,
        Vn,
        layout.uomega,
        layout.ugamma,
        layout.nt,
    )
end

"""
    stokes_gcl_terms_diph(model, x_new, V1n, V2n)

Compute phase-wise moving two-phase conservative GCL residuals. Use one phase
to drive geometry updates and the other as a diagnostic.
"""
function stokes_gcl_terms_diph(
    model::MovingStokesModelTwoPhase{N,T},
    x_new::AbstractVector,
    V1n::AbstractVector,
    V2n::AbstractVector,
) where {N,T}
    layout = model.layout
    phase1 = _stokes_gcl_terms(
        something(model.op_p1_slab),
        something(model.cap_p1_end).buf.V,
        x_new,
        V1n,
        layout.uomega1,
        layout.ugamma1,
        layout.nt,
    )
    phase2 = _stokes_gcl_terms(
        something(model.op_p2_slab),
        something(model.cap_p2_end).buf.V,
        x_new,
        V2n,
        layout.uomega2,
        layout.ugamma2,
        layout.nt,
    )
    return (; phase1, phase2)
end

function _pressure_volume_at_time(model::MovingStokesModelMono{N,T}, t::T) where {N,T}
    xyz_p = grid1d(model.gridp)
    body_t = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    moms = geometric_moments(body_t, xyz_p, T, nan; method=model.geom_method)
    return Vector{T}(moms.V)
end

function _pressure_volumes_at_time(model::MovingStokesModelTwoPhase{N,T}, t::T) where {N,T}
    xyz_p = grid1d(model.gridp)
    body1 = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    body2 = (x...) -> -body1(x...)
    moms1 = geometric_moments(body1, xyz_p, T, nan; method=model.geom_method)
    moms2 = geometric_moments(body2, xyz_p, T, nan; method=model.geom_method)
    return Vector{T}(moms1.V), Vector{T}(moms2.V)
end

"""
    stokes_pressure_volume(model, t)

Return pressure-cell volumes for a moving model at time `t`, using the model's
current phase convention.
"""
stokes_pressure_volume(model::MovingStokesModelMono{N,T}, t::T) where {N,T} =
    _pressure_volume_at_time(model, t)

stokes_pressure_volume(model::MovingStokesModelTwoPhase{N,T}, t::T) where {N,T} =
    _pressure_volumes_at_time(model, t)

function _pressure_active_from_system(
    model::MovingStokesModelMono{N,T},
    A::SparseMatrixCSC{T,Int},
) where {N,T}
    active = _stokes_row_activity(model, A)
    return active[collect(model.layout.pomega)]
end

function _pressure_active_from_system(
    model::MovingStokesModelTwoPhase{N,T},
    A::SparseMatrixCSC{T,Int};
    phase::Int=1,
) where {N,T}
    active = _stokes_row_activity(model, A)
    rows = phase == 1 ? model.layout.pomega1 : phase == 2 ? model.layout.pomega2 :
        throw(ArgumentError("phase must be 1 or 2"))
    return active[collect(rows)]
end

"""
    mask_inactive_pressure_cells(model, R, A; phase=1)

Return a copy of pressure-cell residual `R` with pressure cells that were
identity-regularized in the moving Stokes solve set to zero.
"""
function mask_inactive_pressure_cells(
    model::MovingStokesModelMono{N,T},
    R::AbstractVector,
    A::SparseMatrixCSC{T,Int},
) where {N,T}
    mask = _pressure_active_from_system(model, A)
    out = Vector{T}(R)
    @inbounds for i in eachindex(out)
        mask[i] || (out[i] = zero(T))
    end
    return out
end

function mask_inactive_pressure_cells(
    model::MovingStokesModelTwoPhase{N,T},
    R::AbstractVector,
    A::SparseMatrixCSC{T,Int};
    phase::Int=1,
) where {N,T}
    mask = _pressure_active_from_system(model, A; phase=phase)
    out = Vector{T}(R)
    @inbounds for i in eachindex(out)
        mask[i] || (out[i] = zero(T))
    end
    return out
end
