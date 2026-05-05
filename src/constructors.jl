function StokesModelMono(
    cap_p::AssembledCapacity{N,T},
    op_p::DiffusionOps{N,T},
    cap_u::NTuple{N,AssembledCapacity{N,T}},
    op_u::NTuple{N,DiffusionOps{N,T}},
    mu::Real,
    rho::Real;
    force=_default_force(T, Val(N)),
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    bc_cut::Union{AbstractBoundary,NTuple{N,AbstractBoundary}}=ntuple(_ -> Dirichlet(zero(T)), N),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:prebuilt,
    body=nothing,
) where {N,T}
    nt = cap_p.ntotal
    @inbounds for d in 1:N
        cap_u[d].ntotal == nt || throw(ArgumentError("all velocity capacities must have same ntotal as pressure"))
    end

    pflags = _periodic_velocity_flags(bc_u)
    pflags == periodic_flags(bc_u[1], N) || throw(ArgumentError("invalid periodic flags"))
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)

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
    bc_cut_tuple = bc_cut isa NTuple ? bc_cut : _normalize_cut_bc_tuple(bc_cut, Val(N))
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
        bc_p,
        bc_cut_tuple,
        gauge,
        strong_wall_bc,
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
    bc_p::Union{Nothing,BorderConditions}=nothing,
    bc_cut::Union{AbstractBoundary,NTuple{N,AbstractBoundary}}=ntuple(_ -> Dirichlet(zero(T)), N),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:vofijul,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
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
    bc_cut_tuple = bc_cut isa NTuple ? bc_cut : _normalize_cut_bc_tuple(bc_cut, Val(N))

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
        bc_p,
        bc_cut_tuple,
        gauge,
        strong_wall_bc,
        layout,
        pflags,
        geom_method,
        body,
    )
end

"""
    MovingStokesModelMono(gridp, body, mu, rho; kwargs...)

Construct a moving-boundary monophasic Stokes model.

`body(x..., t)` defines moving geometry and `bc_cut_u` prescribes per-component
embedded-boundary velocity at `t_{n+1}` in `assemble_unsteady_moving!`.
"""
function MovingStokesModelMono(
    gridp::CartesianGrid{N,T},
    body,
    mu::Real,
    rho::Real;
    force=_default_force(T, Val(N)),
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    bc_cut_u::Union{AbstractBoundary,NTuple{N,AbstractBoundary}}=ntuple(_ -> Dirichlet(zero(T)), N),
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:vofijul,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    gridu = staggered_velocity_grids(gridp)
    nt = prod(gridp.n)
    layout = StokesLayout(nt, Val(N))
    bc_cut_tuple = bc_cut_u isa NTuple ? bc_cut_u : _normalize_cut_bc_tuple(bc_cut_u, Val(N))

    return MovingStokesModelMono{N,T,typeof(force),typeof(body)}(
        gridp,
        gridu,
        body,
        convert(T, mu),
        convert(T, rho),
        force,
        bc_u,
        bc_p,
        bc_cut_tuple,
        gauge,
        strong_wall_bc,
        pflags,
        geom_method,
        layout,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

"""
    MovingStokesModelTwoPhase(gridp, body, mu1, mu2; kwargs...)

Construct a moving-interface two-phase Stokes model.

`body(x..., t)` defines phase-1 geometry; phase-2 geometry uses `-body`.
Interface conditions are configured through `interface_jump`, `interface_force`,
and optional `bc_interface`, following `StokesModelTwoPhase`.
"""
function MovingStokesModelTwoPhase(
    gridp::CartesianGrid{N,T},
    body,
    mu1::Real,
    mu2::Real;
    rho1::Real=one(T),
    rho2::Real=one(T),
    force1=_default_force(T, Val(N)),
    force2=_default_force(T, Val(N)),
    interface_jump=_default_interface_jump(T, Val(N)),
    interface_force=_default_interface_force(T, Val(N)),
    bc_interface::Union{Nothing,InterfaceConditions,NTuple{N,InterfaceConditions}}=nothing,
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:vofijul,
    check_interface::Bool=true,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    bc_interface_tuple = _normalize_interface_bc(bc_interface, Val(N))
    _validate_stokes_interface_bcs!(bc_interface_tuple)

    gridu = staggered_velocity_grids(gridp)
    layout = StokesLayoutTwoPhase(prod(gridp.n), Val(N))

    return MovingStokesModelTwoPhase{
        N,T,typeof(force1),typeof(force2),typeof(interface_force),typeof(interface_jump),typeof(body)
    }(
        gridp,
        gridu,
        body,
        convert(T, mu1),
        convert(T, mu2),
        convert(T, rho1),
        convert(T, rho2),
        force1,
        force2,
        interface_force,
        interface_jump,
        bc_interface_tuple,
        bc_u,
        bc_p,
        gauge,
        strong_wall_bc,
        pflags,
        geom_method,
        check_interface,
        layout,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function _check_two_phase_interface_consistency(
    cap_p1::AssembledCapacity{N,T},
    cap_p2::AssembledCapacity{N,T},
) where {N,T}
    cap_p1.ntotal == cap_p2.ntotal ||
        throw(ArgumentError("two-phase pressure capacities must have same ntotal"))
    cap_p1.nnodes == cap_p2.nnodes ||
        throw(ArgumentError("two-phase pressure capacities must share nnodes"))
    tol_Γ_abs = sqrt(eps(T))
    tol_Γ_rel = convert(T, 1e-6)
    tol_n = convert(T, 1e-6)
    @inbounds for i in 1:cap_p1.ntotal
        Γ1 = cap_p1.buf.Γ[i]
        Γ2 = cap_p2.buf.Γ[i]
        has1 = isfinite(Γ1) && Γ1 > zero(T)
        has2 = isfinite(Γ2) && Γ2 > zero(T)
        has1 == has2 || throw(ArgumentError("inconsistent interface support between phases at cell index $i"))
        has1 || continue
        isapprox(Γ1, Γ2; atol=tol_Γ_abs, rtol=tol_Γ_rel) ||
            throw(ArgumentError("inconsistent interface measure between phases at cell index $i"))
        n1 = cap_p1.n_γ[i]
        n2 = cap_p2.n_γ[i]
        norm(n1 + n2) <= tol_n ||
            throw(ArgumentError("phase-interface normals are not opposite at cell index $i"))
    end
    return nothing
end

"""
    StokesModelTwoPhase(gridp, body, mu1, mu2; kwargs...)
    StokesModelTwoPhase(cap_p1, op_p1, cap_u1, op_u1, cap_p2, op_p2, cap_u2, op_u2, mu1, mu2, rho1, rho2; kwargs...)

Construct a fixed-interface two-phase Stokes model with phase-wise interface
trace unknowns and phase-wise pressure blocks.

Key keywords:
- `rho1`, `rho2`: phase densities (for unsteady terms)
- `force1`, `force2`: phase body forces
- `interface_jump`: interface velocity jump forcing callback/value for `[u] = f`
- `interface_force`: interface traction-jump forcing callback/value for `[traction] = g`
- `bc_interface`: optional per-component `InterfaceConditions` (`ScalarJump`/`FluxJump`)
- `bc_u`, `bc_p`, `gauge`, `strong_wall_bc`: box BC and gauge controls
"""
function StokesModelTwoPhase(
    cap_p1::AssembledCapacity{N,T},
    op_p1::DiffusionOps{N,T},
    cap_u1::NTuple{N,AssembledCapacity{N,T}},
    op_u1::NTuple{N,DiffusionOps{N,T}},
    cap_p2::AssembledCapacity{N,T},
    op_p2::DiffusionOps{N,T},
    cap_u2::NTuple{N,AssembledCapacity{N,T}},
    op_u2::NTuple{N,DiffusionOps{N,T}},
    mu1::Real,
    mu2::Real,
    rho1::Real,
    rho2::Real;
    force1=_default_force(T, Val(N)),
    force2=_default_force(T, Val(N)),
    interface_jump=_default_interface_jump(T, Val(N)),
    interface_force=_default_interface_force(T, Val(N)),
    bc_interface::Union{Nothing,InterfaceConditions,NTuple{N,InterfaceConditions}}=nothing,
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:prebuilt,
    body=nothing,
    check_interface::Bool=true,
) where {N,T}
    nt = cap_p1.ntotal
    cap_p2.ntotal == nt || throw(ArgumentError("phase-2 pressure capacity must have same ntotal as phase-1"))
    @inbounds for d in 1:N
        cap_u1[d].ntotal == nt || throw(ArgumentError("all phase-1 velocity capacities must match pressure ntotal"))
        cap_u2[d].ntotal == nt || throw(ArgumentError("all phase-2 velocity capacities must match pressure ntotal"))
    end

    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    bc_interface_tuple = _normalize_interface_bc(bc_interface, Val(N))
    _validate_stokes_interface_bcs!(bc_interface_tuple)

    check_interface && _check_two_phase_interface_consistency(cap_p1, cap_p2)

    gridp = CartesianGrid(
        ntuple(d -> cap_p1.xyz[d][1], N),
        ntuple(d -> cap_p1.xyz[d][end], N),
        cap_p1.nnodes,
    )
    gridu = ntuple(d -> CartesianGrid(
        ntuple(k -> cap_u1[d].xyz[k][1], N),
        ntuple(k -> cap_u1[d].xyz[k][end], N),
        cap_u1[d].nnodes,
    ), N)

    layout = StokesLayoutTwoPhase(nt, Val(N))
    return StokesModelTwoPhase{
        N,T,typeof(force1),typeof(force2),typeof(interface_force),typeof(interface_jump),typeof(body)
    }(
        gridp,
        gridu,
        cap_p1,
        cap_u1,
        op_p1,
        op_u1,
        cap_p2,
        cap_u2,
        op_p2,
        op_u2,
        convert(T, mu1),
        convert(T, mu2),
        convert(T, rho1),
        convert(T, rho2),
        force1,
        force2,
        interface_force,
        interface_jump,
        bc_interface_tuple,
        bc_u,
        bc_p,
        gauge,
        strong_wall_bc,
        pflags,
        geom_method,
        body,
        layout,
    )
end

function StokesModelTwoPhase(
    gridp::CartesianGrid{N,T},
    body,
    mu1::Real,
    mu2::Real;
    rho1::Real=one(T),
    rho2::Real=one(T),
    force1=_default_force(T, Val(N)),
    force2=_default_force(T, Val(N)),
    interface_jump=_default_interface_jump(T, Val(N)),
    interface_force=_default_interface_force(T, Val(N)),
    bc_interface::Union{Nothing,InterfaceConditions,NTuple{N,InterfaceConditions}}=nothing,
    bc_u::NTuple{N,BorderConditions}=ntuple(_ -> BorderConditions(), N),
    bc_p::Union{Nothing,BorderConditions}=nothing,
    gauge::AbstractPressureGauge=PinPressureGauge(),
    strong_wall_bc::Bool=true,
    geom_method::Symbol=:vofijul,
    check_interface::Bool=true,
) where {N,T}
    pflags = _periodic_velocity_flags(bc_u)
    bc_p = _validate_pressure_bc(bc_p, pflags)
    _validate_stokes_box_bcs!(bc_u, bc_p, pflags, T)
    gridu = staggered_velocity_grids(gridp)

    body2 = (x...) -> -body(x...)

    mp1 = geometric_moments(body, grid1d(gridp), T, nan; method=geom_method)
    mp2 = geometric_moments(body2, grid1d(gridp), T, nan; method=geom_method)
    mom_u1 = ntuple(d -> geometric_moments(body, grid1d(gridu[d]), T, nan; method=geom_method), N)
    mom_u2 = ntuple(d -> geometric_moments(body2, grid1d(gridu[d]), T, nan; method=geom_method), N)

    cap_p1 = assembled_capacity(mp1; bc=zero(T))
    cap_p2 = assembled_capacity(mp2; bc=zero(T))
    cap_u1 = ntuple(d -> assembled_capacity(mom_u1[d]; bc=zero(T)), N)
    cap_u2 = ntuple(d -> assembled_capacity(mom_u2[d]; bc=zero(T)), N)

    op_p1 = DiffusionOps(cap_p1; periodic=pflags)
    op_p2 = DiffusionOps(cap_p2; periodic=pflags)
    op_u1 = ntuple(d -> DiffusionOps(cap_u1[d]; periodic=pflags), N)
    op_u2 = ntuple(d -> DiffusionOps(cap_u2[d]; periodic=pflags), N)

    return StokesModelTwoPhase(
        cap_p1,
        op_p1,
        cap_u1,
        op_u1,
        cap_p2,
        op_p2,
        cap_u2,
        op_u2,
        mu1,
        mu2,
        rho1,
        rho2;
        force1=force1,
        force2=force2,
        interface_jump=interface_jump,
        interface_force=interface_force,
        bc_interface=bc_interface,
        bc_u=bc_u,
        bc_p=bc_p,
        gauge=gauge,
        strong_wall_bc=strong_wall_bc,
        geom_method=geom_method,
        body=body,
        check_interface=check_interface,
    )
end
