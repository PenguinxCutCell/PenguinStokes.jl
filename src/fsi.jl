function _promote_moving_model_body_to_any(
    model::MovingStokesModelMono{N,T,FT,Any},
) where {N,T,FT}
    return model
end

function _promote_moving_model_body_to_any(
    model::MovingStokesModelMono{N,T,FT,BT},
) where {N,T,FT,BT}
    return MovingStokesModelMono{N,T,FT,Any}(
        model.gridp,
        model.gridu,
        model.body,
        model.mu,
        model.rho,
        model.force,
        model.bc_u,
        model.bc_p,
        model.bc_cut_u,
        model.gauge,
        model.strong_wall_bc,
        model.periodic,
        model.geom_method,
        model.layout,
        model.cap_p_slab,
        model.op_p_slab,
        model.cap_p_end,
        model.op_p_end,
        model.cap_u_slab,
        model.op_u_slab,
        model.cap_u_end,
        model.Vun,
        model.Vun1,
    )
end

"""
    endtime_static_model(model)

Build a static `StokesModelMono` using the end-time capacities/operators cached by
`MovingStokesModelMono` after `assemble_unsteady_moving!`/`solve_unsteady_moving!`.
"""
function endtime_static_model(model::MovingStokesModelMono{N,T}) where {N,T}
    cap_p = something(model.cap_p_end)
    op_p = something(model.op_p_end)
    cap_u = something(model.cap_u_end)
    op_u = ntuple(d -> DiffusionOps(cap_u[d]; periodic=model.periodic), N)
    return StokesModelMono(
        cap_p,
        op_p,
        cap_u,
        op_u,
        model.mu,
        model.rho;
        force=model.force,
        bc_u=model.bc_u,
        bc_p=model.bc_p,
        bc_cut=Dirichlet(zero(T)),
        gauge=model.gauge,
        strong_wall_bc=model.strong_wall_bc,
        geom_method=:prebuilt,
        body=model.body,
    )
end

"""
    StokesFSIProblem

Dimension-generic rigid-body FSI wrapper around `MovingStokesModelMono`.
"""
mutable struct StokesFSIProblem{N,T,MT,ST,PT}
    model::MT
    state::ST
    params::PT
    xprev::Vector{T}
    pressure_reconstruction::Symbol
    force_sign::T
    torque_sign::T
end

const StokesFSIProblem2D{T,MT,ST,PT} = StokesFSIProblem{2,T,MT,ST,PT}

function StokesFSIProblem(
    model::MovingStokesModelMono{N,T,FT},
    state,
    params;
    xprev::Union{Nothing,AbstractVector{T}}=nothing,
    pressure_reconstruction::Symbol=:linear,
    force_sign::Real=one(T),
    torque_sign::Real=one(T),
) where {N,T,FT}
    model_any = _promote_moving_model_body_to_any(model)
    length(state_position(state)) == N || throw(ArgumentError("state dimension mismatch with model dimension $N"))

    nsys = last(model_any.layout.pomega)
    x0 = isnothing(xprev) ? zeros(T, nsys) : collect(xprev)
    length(x0) == nsys || throw(ArgumentError("xprev length mismatch: got $(length(x0)), expected $nsys"))

    return StokesFSIProblem{N,T,typeof(model_any),typeof(state),typeof(params)}(
        model_any,
        state,
        params,
        x0,
        pressure_reconstruction,
        convert(T, force_sign),
        convert(T, torque_sign),
    )
end

StokesFSIProblem2D(args...; kwargs...) = StokesFSIProblem(args...; kwargs...)

function _predict_state(state::RigidBodyState{N,T}, t::T, tau::T) where {N,T}
    return RigidBodyState{N,T}(state.X + (tau - t) * state.V, state.V)
end

function _predict_state(state::RigidBodyState2D{T}, t::T, tau::T) where {T}
    return RigidBodyState2D(
        state.X + (tau - t) * state.V,
        state.V;
        theta=state.theta + (tau - t) * state.omega,
        omega=state.omega,
    )
end

function _predict_state(state::RigidBodyState3D{T}, t::T, tau::T) where {T}
    dt = tau - t
    ori = advance_orientation(Orientation3D{T}(state.Q), state.Omega, dt)
    return RigidBodyState3D{T}(
        state.X + dt * state.V,
        state.V,
        ori.Q,
        state.Omega,
    )
end

function _predict_state_from_guess(state_n::RigidBodyState{N,T}, state_guess::RigidBodyState{N,T}, t::T, tau::T) where {N,T}
    return RigidBodyState{N,T}(state_n.X + (tau - t) * state_guess.V, state_guess.V)
end

function _predict_state_from_guess(state_n::RigidBodyState2D{T}, state_guess::RigidBodyState2D{T}, t::T, tau::T) where {T}
    return RigidBodyState2D(
        state_n.X + (tau - t) * state_guess.V,
        state_guess.V;
        theta=state_n.theta + (tau - t) * state_guess.omega,
        omega=state_guess.omega,
    )
end

function _predict_state_from_guess(state_n::RigidBodyState3D{T}, state_guess::RigidBodyState3D{T}, t::T, tau::T) where {T}
    dt = tau - t
    ori = advance_orientation(Orientation3D{T}(state_n.Q), state_guess.Omega, dt)
    return RigidBodyState3D{T}(state_n.X + dt * state_guess.V, state_guess.V, ori.Q, state_guess.Omega)
end

@inline function _extract_force(q, ::Val{N}, ::Type{T}) where {N,T}
    return SVector{N,T}(ntuple(i -> convert(T, q.force[i]), N))
end

@inline _extract_torque_hydro(::RigidBodyState, q, torque_sign, ::Type{T}) where {T} = nothing
@inline _extract_torque_hydro(::RigidBodyState2D, q, torque_sign, ::Type{T}) where {T} = convert(T, torque_sign) * convert(T, q.torque)
@inline function _extract_torque_hydro(::RigidBodyState3D, q, torque_sign, ::Type{T}) where {T}
    return convert(T, torque_sign) * SVector{3,T}(ntuple(i -> convert(T, q.torque[i]), 3))
end

function _apply_inertia_inverse(I::T, tau::SVector{3,T}) where {T}
    return tau / I
end

function _apply_inertia_inverse(I::SMatrix{3,3,T,9}, tau::SVector{3,T}) where {T}
    return I \ tau
end

function _advance_state!(
    state::RigidBodyState{N,T},
    params::RigidBodyParams{N,T},
    Fhydro::SVector{N,T},
    tau_hydro,
    dt::T;
    t::T=zero(T),
    ode_scheme::Symbol=:symplectic_euler,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    Fext = external_force(params)
    Xn = state.X
    Vn = state.V

    if ode_scheme == :symplectic_euler
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
        Xn1 = Xn + dt * Vn1
    elseif ode_scheme == :forward_euler
        Xn1 = Xn + dt * Vn
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
    else
        throw(ArgumentError("unsupported ODE scheme `$ode_scheme` (use :symplectic_euler or :forward_euler)"))
    end

    state.X = Xn1
    state.V = Vn1
    return (X=state.X, V=state.V, Fext=Fext, Fhydro=Fhydro)
end

function _advance_state!(
    state::RigidBodyState2D{T},
    params::RigidBodyParams2D{T},
    Fhydro::SVector{2,T},
    tau_hydro::T,
    dt::T;
    t::T=zero(T),
    ode_scheme::Symbol=:symplectic_euler,
) where {T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    Fext = external_force(params)
    tau_ext = external_torque(params, state, t)

    Xn = state.X
    Vn = state.V
    thetan = state.theta
    omegan = state.omega

    if ode_scheme == :symplectic_euler
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
        Xn1 = Xn + dt * Vn1
        omegan1 = omegan + (dt / params.I) * (tau_ext + tau_hydro)
        thetan1 = thetan + dt * omegan1
    elseif ode_scheme == :forward_euler
        Xn1 = Xn + dt * Vn
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
        thetan1 = thetan + dt * omegan
        omegan1 = omegan + (dt / params.I) * (tau_ext + tau_hydro)
    else
        throw(ArgumentError("unsupported ODE scheme `$ode_scheme` (use :symplectic_euler or :forward_euler)"))
    end

    state.X = Xn1
    state.V = Vn1
    state.theta = thetan1
    state.omega = omegan1

    return (
        X=state.X,
        V=state.V,
        theta=state.theta,
        omega=state.omega,
        Fext=Fext,
        tau_ext=tau_ext,
        Fhydro=Fhydro,
        tau_hydro=tau_hydro,
    )
end

function _advance_state!(
    state::RigidBodyState3D{T},
    params::RigidBodyParams3D{T},
    Fhydro::SVector{3,T},
    tau_hydro::SVector{3,T},
    dt::T;
    t::T=zero(T),
    ode_scheme::Symbol=:symplectic_euler,
) where {T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    Fext = external_force(params)
    tau_ext = external_torque(params, state, t)

    Xn = state.X
    Vn = state.V
    Qn = state.Q
    Omeg = state.Omega

    if ode_scheme == :symplectic_euler
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
        Xn1 = Xn + dt * Vn1
        Omeg1 = Omeg + dt * _apply_inertia_inverse(params.I, tau_ext + tau_hydro)
        Qn1 = advance_orientation(Orientation3D{T}(Qn), Omeg1, dt).Q
    elseif ode_scheme == :forward_euler
        Xn1 = Xn + dt * Vn
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
        Qn1 = advance_orientation(Orientation3D{T}(Qn), Omeg, dt).Q
        Omeg1 = Omeg + dt * _apply_inertia_inverse(params.I, tau_ext + tau_hydro)
    else
        throw(ArgumentError("unsupported ODE scheme `$ode_scheme` (use :symplectic_euler or :forward_euler)"))
    end

    state.X = Xn1
    state.V = Vn1
    state.Q = Qn1
    state.Omega = Omeg1

    return (
        X=state.X,
        V=state.V,
        Q=state.Q,
        Omega=state.Omega,
        Fext=Fext,
        tau_ext=tau_ext,
        Fhydro=Fhydro,
        tau_hydro=tau_hydro,
    )
end

function _set_moving_state!(fsi::StokesFSIProblem{N,T}, statefun) where {N,T}
    fsi.model.body = rigid_body_levelset(fsi.params.shape, statefun)
    fsi.model.bc_cut_u = rigid_cut_bc_tuple(statefun, Val(N))
    return nothing
end

function _solve_fluid_for_statefun!(
    fsi::StokesFSIProblem{N,T},
    statefun,
    xprev::AbstractVector{T},
    t::T,
    dt::T,
    fluid_scheme::Symbol,
) where {N,T}
    _set_moving_state!(fsi, statefun)
    return solve_unsteady_moving!(fsi.model, xprev; t=t, dt=dt, scheme=fluid_scheme)
end

"""
    step_fsi!(fsi; t, dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)

Advance one rigid-body FSI step with one-pass split coupling.
"""
function step_fsi!(
    fsi::StokesFSIProblem{N,T};
    t::T,
    dt::T,
    fluid_scheme::Symbol=:CN,
    ode_scheme::Symbol=:symplectic_euler,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    state_n = state_copy(fsi.state)
    statefun = tau -> _predict_state(state_n, t, convert(T, tau))

    sys = _solve_fluid_for_statefun!(fsi, statefun, fsi.xprev, t, dt, fluid_scheme)

    tnext = t + dt
    snext_pred = statefun(tnext)
    sm = endtime_static_model(fsi.model)
    q = integrated_embedded_force(
        sm,
        sys;
        pressure_reconstruction=fsi.pressure_reconstruction,
        x0=Tuple(state_position(snext_pred)),
    )

    Fhydro = convert(T, fsi.force_sign) * _extract_force(q, Val(N), T)
    tau_hydro = _extract_torque_hydro(fsi.state, q, fsi.torque_sign, T)

    ode = _advance_state!(
        fsi.state,
        fsi.params,
        Fhydro,
        tau_hydro,
        dt;
        t=t,
        ode_scheme=ode_scheme,
    )

    fsi.xprev .= sys.x

    return merge((sys=sys, force=q, t=tnext), ode)
end

function step_fsi_rotation!(
    fsi::StokesFSIProblem{2,T,MT,<:RigidBodyState2D,PT};
    kwargs...
) where {T,MT,PT}
    return step_fsi!(fsi; kwargs...)
end

"""
    simulate_fsi!(fsi; t0=0, dt, nsteps, fluid_scheme=:CN, ode_scheme=:symplectic_euler, callback=nothing)

Run repeated `step_fsi!` updates and return a lightweight history vector.
"""
function simulate_fsi!(
    fsi::StokesFSIProblem{N,T};
    t0::T=zero(T),
    dt::T,
    nsteps::Integer,
    fluid_scheme::Symbol=:CN,
    ode_scheme::Symbol=:symplectic_euler,
    callback=nothing,
) where {N,T}
    nsteps >= 0 || throw(ArgumentError("nsteps must be nonnegative"))
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    history = Vector{NamedTuple}(undef, nsteps)
    t = t0
    for step in 1:nsteps
        out = step_fsi!(fsi; t=t, dt=dt, fluid_scheme=fluid_scheme, ode_scheme=ode_scheme)
        kkeep = Tuple(filter(k -> k != :sys && k != :force && k != :t, keys(out)))
        vkeep = map(k -> getproperty(out, k), kkeep)
        extra = NamedTuple{kkeep}(vkeep)
        rec = merge(
            (
                step=step,
                t=out.t,
                residual=norm(out.sys.A * out.sys.x - out.sys.b),
                force=out.force.force,
                torque=out.force.torque,
            ),
            extra,
        )
        history[step] = rec
        callback === nothing || callback(fsi, rec, out)
        t = out.t
    end
    return history
end

function simulate_fsi_rotation!(
    fsi::StokesFSIProblem{2,T,MT,<:RigidBodyState2D,PT};
    kwargs...
) where {T,MT,PT}
    return simulate_fsi!(fsi; kwargs...)
end
