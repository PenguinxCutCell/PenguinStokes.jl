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

psip_cn(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 0.5
psim_cn(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 0.5
psip_be(Vn, Vn1) = (iszero(Vn) && iszero(Vn1)) ? 0.0 : 1.0
psim_be(Vn, Vn1) = 0.0

function _psi_functions(scheme)
    if scheme isa Symbol
        if scheme === :CN
            return psip_cn, psim_cn
        elseif scheme === :BE
            return psip_be, psim_be
        end
    elseif scheme isa Real
        θ = Float64(scheme)
        psip = (Vn, Vn1) -> (iszero(Vn) && iszero(Vn1) ? 0.0 : θ)
        psim = (Vn, Vn1) -> (iszero(Vn) && iszero(Vn1) ? 0.0 : (1.0 - θ))
        return psip, psim
    end
    throw(ArgumentError("moving scheme must be :BE, :CN, or numeric θ"))
end

function _eval_levelset_time(body, x::SVector{N,T}, t::T) where {N,T}
    if applicable(body, x..., t)
        return convert(T, body(x..., t))
    elseif applicable(body, x...)
        return convert(T, body(x...))
    end
    throw(ArgumentError("level-set callback must accept (x...) or (x..., t)"))
end

function _space_moments_at_time(
    model::MovingStokesModelMono{N,T},
    xyz_space::NTuple{N,AbstractVector{T}},
    t::T,
) where {N,T}
    body_t = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    return geometric_moments(body_t, xyz_space, T, nan; method=model.geom_method)
end

function _slice_spacetime_to_space(
    vec_st::AbstractVector,
    nn_space::NTuple{N,Int},
    nt::Int,
    it::Int,
) where {N}
    dims_st = (nn_space..., nt)
    li_st = LinearIndices(dims_st)
    li_sp = LinearIndices(nn_space)
    out = similar(vec_st, prod(nn_space))
    @inbounds for I in CartesianIndices(nn_space)
        out[li_sp[I]] = vec_st[li_st[Tuple(I)..., it]]
    end
    return out
end

function reduce_slab_to_space(
    m_st::GeometricMoments{N1,T},
    nn_space::NTuple{N,Int},
) where {N1,N,T}
    N1 == N + 1 || throw(ArgumentError("expected slab moments dimension $(N + 1), got $N1"))
    nt = length(m_st.xyz[N1])
    nt == 2 || throw(ArgumentError("space-time reduction expects 2 time nodes, got $nt"))

    V = _slice_spacetime_to_space(m_st.V, nn_space, nt, 1)
    Γ = _slice_spacetime_to_space(m_st.interface_measure, nn_space, nt, 1)
    ctype = _slice_spacetime_to_space(m_st.cell_type, nn_space, nt, 1)
    A = ntuple(d -> _slice_spacetime_to_space(m_st.A[d], nn_space, nt, 1), N)
    B = ntuple(d -> _slice_spacetime_to_space(m_st.B[d], nn_space, nt, 1), N)
    W = ntuple(d -> _slice_spacetime_to_space(m_st.W[d], nn_space, nt, 1), N)

    bary_st = _slice_spacetime_to_space(m_st.barycenter, nn_space, nt, 1)
    baryγ_st = _slice_spacetime_to_space(m_st.barycenter_interface, nn_space, nt, 1)
    nγ_st = _slice_spacetime_to_space(m_st.interface_normal, nn_space, nt, 1)

    bary = Vector{SVector{N,T}}(undef, length(V))
    baryγ = Vector{SVector{N,T}}(undef, length(V))
    nγ = Vector{SVector{N,T}}(undef, length(V))
    @inbounds for i in eachindex(V)
        bi = bary_st[i]
        bγi = baryγ_st[i]
        ni = nγ_st[i]
        bary[i] = SVector{N,T}(ntuple(d -> bi[d], N))
        baryγ[i] = SVector{N,T}(ntuple(d -> bγi[d], N))
        nγ[i] = SVector{N,T}(ntuple(d -> ni[d], N))
    end

    face_bary_st = ntuple(d -> _slice_spacetime_to_space(m_st.face_barycenter[d], nn_space, nt, 1), N)
    face_bary = ntuple(d -> begin
        fb_d = Vector{SVector{N,T}}(undef, length(V))
        fb_st = face_bary_st[d]
        @inbounds for i in eachindex(V)
            fbi = fb_st[i]
            fb_d[i] = SVector{N,T}(ntuple(dd -> fbi[dd], N))
        end
        fb_d
    end, N)

    xyz = ntuple(d -> collect(T, m_st.xyz[d]), N)
    return GeometricMoments(V, bary, Γ, ctype, baryγ, nγ, A, B, W, face_bary, xyz)
end

function _build_moving_slab!(
    model::MovingStokesModelMono{N,T},
    t::T,
    dt::T,
) where {N,T}
    xyz_p = grid1d(model.gridp)
    body_n = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    body_n1 = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t + dt)
    moms_p_n1 = _space_moments_at_time(model, xyz_p, t + dt)

    stgrid_p = SpaceTimeCartesianGrid(model.gridp, T[t, t + dt])
    xyz_st_p = grid1d(stgrid_p)
    body_st = (x...) -> begin
        xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
        _eval_levelset_time(model.body, xs, convert(T, x[N + 1]))
    end
    moms_p_st = geometric_moments(body_st, xyz_st_p, T, nan; method=model.geom_method)
    moms_p_slab = reduce_slab_to_space(moms_p_st, model.gridp.n)
    cap_p_slab = assembled_capacity(moms_p_slab; bc=zero(T))
    op_p_slab = DiffusionOps(cap_p_slab; periodic=model.periodic)
    cap_p_end = assembled_capacity(moms_p_n1; bc=zero(T))
    op_p_end = DiffusionOps(cap_p_end; periodic=model.periodic)

    ugeom = ntuple(d -> begin
        xyz_u = grid1d(model.gridu[d])
        moms_u_n = geometric_moments(body_n, xyz_u, T, nan; method=model.geom_method)
        moms_u_n1 = geometric_moments(body_n1, xyz_u, T, nan; method=model.geom_method)
        stgrid_u = SpaceTimeCartesianGrid(model.gridu[d], T[t, t + dt])
        xyz_st_u = grid1d(stgrid_u)
        moms_u_st = geometric_moments(body_st, xyz_st_u, T, nan; method=model.geom_method)
        moms_u_slab = reduce_slab_to_space(moms_u_st, model.gridu[d].n)
        (
            cap_slab=assembled_capacity(moms_u_slab; bc=zero(T)),
            cap_end=assembled_capacity(moms_u_n1; bc=zero(T)),
            Vn=Vector{T}(moms_u_n.V),
            Vn1=Vector{T}(moms_u_n1.V),
        )
    end, N)
    cap_u_slab = ntuple(d -> ugeom[d].cap_slab, N)
    op_u_slab = ntuple(d -> DiffusionOps(cap_u_slab[d]; periodic=model.periodic), N)
    cap_u_end = ntuple(d -> ugeom[d].cap_end, N)
    Vun = ntuple(d -> ugeom[d].Vn, N)
    Vun1 = ntuple(d -> ugeom[d].Vn1, N)

    model.cap_p_slab = cap_p_slab
    model.op_p_slab = op_p_slab
    model.cap_p_end = cap_p_end
    model.op_p_end = op_p_end
    model.cap_u_slab = cap_u_slab
    model.op_u_slab = op_u_slab
    model.cap_u_end = cap_u_end
    model.Vun = Vun
    model.Vun1 = Vun1
    return model
end

function _build_moving_phase_geometry(
    gridp::CartesianGrid{N,T},
    gridu::NTuple{N,CartesianGrid{N,T}},
    periodic::NTuple{N,Bool},
    geom_method::Symbol,
    body_n,
    body_n1,
    body_st,
    t::T,
    dt::T,
) where {N,T}
    xyz_p = grid1d(gridp)
    moms_p_n1 = geometric_moments(body_n1, xyz_p, T, nan; method=geom_method)

    stgrid_p = SpaceTimeCartesianGrid(gridp, T[t, t + dt])
    xyz_st_p = grid1d(stgrid_p)
    moms_p_st = geometric_moments(body_st, xyz_st_p, T, nan; method=geom_method)
    moms_p_slab = reduce_slab_to_space(moms_p_st, gridp.n)

    cap_p_slab = assembled_capacity(moms_p_slab; bc=zero(T))
    op_p_slab = DiffusionOps(cap_p_slab; periodic=periodic)
    cap_p_end = assembled_capacity(moms_p_n1; bc=zero(T))
    op_p_end = DiffusionOps(cap_p_end; periodic=periodic)

    ugeom = ntuple(d -> begin
        xyz_u = grid1d(gridu[d])
        moms_u_n = geometric_moments(body_n, xyz_u, T, nan; method=geom_method)
        moms_u_n1 = geometric_moments(body_n1, xyz_u, T, nan; method=geom_method)
        stgrid_u = SpaceTimeCartesianGrid(gridu[d], T[t, t + dt])
        xyz_st_u = grid1d(stgrid_u)
        moms_u_st = geometric_moments(body_st, xyz_st_u, T, nan; method=geom_method)
        moms_u_slab = reduce_slab_to_space(moms_u_st, gridu[d].n)
        (
            cap_slab=assembled_capacity(moms_u_slab; bc=zero(T)),
            cap_end=assembled_capacity(moms_u_n1; bc=zero(T)),
            Vn=Vector{T}(moms_u_n.V),
            Vn1=Vector{T}(moms_u_n1.V),
        )
    end, N)

    cap_u_slab = ntuple(d -> ugeom[d].cap_slab, N)
    op_u_slab = ntuple(d -> DiffusionOps(cap_u_slab[d]; periodic=periodic), N)
    cap_u_end = ntuple(d -> ugeom[d].cap_end, N)
    op_u_end = ntuple(d -> DiffusionOps(cap_u_end[d]; periodic=periodic), N)
    Vun = ntuple(d -> ugeom[d].Vn, N)
    Vun1 = ntuple(d -> ugeom[d].Vn1, N)

    return (
        cap_p_slab=cap_p_slab,
        op_p_slab=op_p_slab,
        cap_p_end=cap_p_end,
        op_p_end=op_p_end,
        cap_u_slab=cap_u_slab,
        op_u_slab=op_u_slab,
        cap_u_end=cap_u_end,
        op_u_end=op_u_end,
        Vun=Vun,
        Vun1=Vun1,
    )
end

function _build_moving_slab!(
    model::MovingStokesModelTwoPhase{N,T},
    t::T,
    dt::T,
) where {N,T}
    body1_n = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t)
    body1_n1 = (x...) -> _eval_levelset_time(model.body, SVector{N,T}(x), t + dt)
    body1_st = (x...) -> begin
        xs = SVector{N,T}(ntuple(d -> convert(T, x[d]), N))
        _eval_levelset_time(model.body, xs, convert(T, x[N + 1]))
    end

    body2_n = (x...) -> -body1_n(x...)
    body2_n1 = (x...) -> -body1_n1(x...)
    body2_st = (x...) -> -body1_st(x...)

    phase1 = _build_moving_phase_geometry(
        model.gridp,
        model.gridu,
        model.periodic,
        model.geom_method,
        body1_n,
        body1_n1,
        body1_st,
        t,
        dt,
    )
    phase2 = _build_moving_phase_geometry(
        model.gridp,
        model.gridu,
        model.periodic,
        model.geom_method,
        body2_n,
        body2_n1,
        body2_st,
        t,
        dt,
    )

    if model.check_interface
        _check_two_phase_interface_consistency(phase1.cap_p_end, phase2.cap_p_end)
        _check_two_phase_interface_consistency(phase1.cap_p_slab, phase2.cap_p_slab)
    end

    model.cap_p1_slab = phase1.cap_p_slab
    model.op_p1_slab = phase1.op_p_slab
    model.cap_p1_end = phase1.cap_p_end
    model.op_p1_end = phase1.op_p_end
    model.cap_u1_slab = phase1.cap_u_slab
    model.op_u1_slab = phase1.op_u_slab
    model.cap_u1_end = phase1.cap_u_end
    model.op_u1_end = phase1.op_u_end
    model.Vu1n = phase1.Vun
    model.Vu1n1 = phase1.Vun1

    model.cap_p2_slab = phase2.cap_p_slab
    model.op_p2_slab = phase2.op_p_slab
    model.cap_p2_end = phase2.cap_p_end
    model.op_p2_end = phase2.op_p_end
    model.cap_u2_slab = phase2.cap_u_slab
    model.op_u2_slab = phase2.op_u_slab
    model.cap_u2_end = phase2.cap_u_end
    model.op_u2_end = phase2.op_u_end
    model.Vu2n = phase2.Vun
    model.Vu2n1 = phase2.Vun1
    return model
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

function _expand_prev_state(model::StokesModelTwoPhase{N,T}, x_prev::AbstractVector) where {N,T}
    nsys = nunknowns(model.layout)
    length(x_prev) == nsys ||
        throw(DimensionMismatch("x_prev length must be $nsys for two-phase full state"))
    return Vector{T}(x_prev)
end

function _expand_prev_state(model::MovingStokesModelMono{N,T}, x_prev::AbstractVector) where {N,T}
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

function _expand_prev_state(model::MovingStokesModelTwoPhase{N,T}, x_prev::AbstractVector) where {N,T}
    nsys = nunknowns(model.layout)
    length(x_prev) == nsys ||
        throw(DimensionMismatch("x_prev length must be $nsys for moving two-phase full state"))
    return Vector{T}(x_prev)
end

