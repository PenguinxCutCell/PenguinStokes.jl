function _state_vector(state::RigidBodyState{N,T}) where {N,T}
    return vcat(Vector{T}(state.X), Vector{T}(state.V))
end

function _state_vector(state::RigidBodyState2D{T}) where {T}
    return T[state.X[1], state.X[2], state.V[1], state.V[2], state.theta, state.omega]
end

function _state_vector(state::RigidBodyState3D{T}) where {T}
    return vcat(
        Vector{T}(state.X),
        Vector{T}(state.V),
        Vector{T}(state.Omega),
        Vector{T}(vec(Matrix(state.Q))),
    )
end

function _set_state_from_vector!(state::RigidBodyState{N,T}, vec::AbstractVector{T}) where {N,T}
    length(vec) == 2N || throw(DimensionMismatch("translation rigid state vector must have length $(2N)"))
    state.X = SVector{N,T}(ntuple(i -> vec[i], N))
    state.V = SVector{N,T}(ntuple(i -> vec[N + i], N))
    return state
end

function _set_state_from_vector!(state::RigidBodyState2D{T}, vec::AbstractVector{T}) where {T}
    length(vec) == 6 || throw(DimensionMismatch("2D rigid state vector must have length 6"))
    state.X = SVector{2,T}(vec[1], vec[2])
    state.V = SVector{2,T}(vec[3], vec[4])
    state.theta = vec[5]
    state.omega = vec[6]
    return state
end

function _set_state_from_vector!(state::RigidBodyState3D{T}, vec::AbstractVector{T}) where {T}
    length(vec) == 18 || throw(DimensionMismatch("3D rigid state vector must have length 18"))
    state.X = SVector{3,T}(vec[1], vec[2], vec[3])
    state.V = SVector{3,T}(vec[4], vec[5], vec[6])
    state.Omega = SVector{3,T}(vec[7], vec[8], vec[9])
    Qraw = SMatrix{3,3,T,9}(reshape(vec[10:18], 3, 3))
    state.Q = orthonormalize_rotation(Qraw)
    return state
end

function _state_error_norm(vec_old::AbstractVector{T}, vec_new::AbstractVector{T}) where {T}
    delta = vec_new - vec_old
    absres = norm(delta)
    relres = absres / max(norm(vec_new), one(T))
    return absres, relres, delta
end

function _aitken_relaxation(alpha_prev::T, r_prev::AbstractVector{T}, r_curr::AbstractVector{T}) where {T}
    dr = r_curr - r_prev
    denom = dot(dr, dr)
    if !(isfinite(denom) && denom > eps(T))
        return alpha_prev
    end
    alpha = -alpha_prev * dot(r_prev, dr) / denom
    return clamp(alpha, convert(T, 0.05), convert(T, 1.5))
end

"""
    step_fsi_strong!(fsi; ...)

Strong partitioned coupling using fixed-point iterations with optional Aitken relaxation.
"""
function step_fsi_strong!(
    fsi::StokesFSIProblem{N,T};
    t::T,
    dt::T,
    fluid_scheme::Symbol=:CN,
    ode_scheme::Symbol=:symplectic_euler,
    maxiter::Integer=12,
    atol::Real=1e-10,
    rtol::Real=1e-8,
    relaxation=:aitken,
    omega_relax::Real=0.8,
    verbose::Bool=false,
    allow_nonconverged::Bool=false,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1"))

    atolT = convert(T, atol)
    rtolT = convert(T, rtol)
    omega0 = convert(T, omega_relax)

    state_n = state_copy(fsi.state)
    xprev_n = copy(fsi.xprev)
    tnext = t + dt

    state_guess = _predict_state(state_n, t, tnext)
    vec_guess = _state_vector(state_guess)

    sys_last = nothing
    q_last = nothing
    ode_last = nothing
    Fhydro_last = nothing
    tau_hydro_last = nothing
    coupled_residual = typemax(T)

    alpha_prev = omega0
    r_prev = nothing
    converged = false
    nit = 0

    for k in 1:maxiter
        nit = k
        statefun = tau -> _predict_state_from_guess(state_n, state_guess, t, convert(T, tau))
        sys = _solve_fluid_for_statefun!(fsi, statefun, xprev_n, t, dt, fluid_scheme)

        sref = statefun(tnext)
        sm = endtime_static_model(fsi.model)
        q = integrated_embedded_force(
            sm,
            sys;
            pressure_reconstruction=fsi.pressure_reconstruction,
            x0=Tuple(state_position(sref)),
        )

        Fhydro = convert(T, fsi.force_sign) * _extract_force(q, Val(N), T)
        tau_hydro = _extract_torque_hydro(state_guess, q, fsi.torque_sign, T)

        state_tilde = state_copy(state_n)
        ode = _advance_state!(
            state_tilde,
            fsi.params,
            Fhydro,
            tau_hydro,
            dt;
            t=t,
            ode_scheme=ode_scheme,
        )

        vec_tilde = _state_vector(state_tilde)
        r_curr = vec_tilde - vec_guess

        alpha = begin
            if relaxation == :none
                one(T)
            elseif relaxation == :constant
                clamp(omega0, convert(T, 0.05), one(T))
            elseif relaxation == :aitken
                r_prev === nothing ? clamp(omega0, convert(T, 0.05), one(T)) : _aitken_relaxation(alpha_prev, r_prev, r_curr)
            else
                throw(ArgumentError("unknown relaxation `$relaxation`; expected :none, :constant, or :aitken"))
            end
        end

        vec_new = vec_guess + alpha * r_curr
        absres, relres, _ = _state_error_norm(vec_guess, vec_new)
        coupled_residual = max(absres, relres)

        _set_state_from_vector!(state_guess, vec_new)
        vec_guess = vec_new

        sys_last = sys
        q_last = q
        ode_last = ode
        Fhydro_last = Fhydro
        tau_hydro_last = tau_hydro

        if verbose
            @info "FSI strong iteration" iter=k alpha=alpha absres=absres relres=relres
        end

        if absres <= atolT || relres <= rtolT
            converged = true
            break
        end

        r_prev = r_curr
        alpha_prev = alpha
    end

    if !converged && !allow_nonconverged
        throw(ErrorException("strong FSI coupling did not converge in $maxiter iterations (residual=$coupled_residual)"))
    end

    fsi.state = state_guess
    fsi.xprev .= sys_last.x

    out = merge(
        (sys=sys_last, force=q_last, t=tnext, iterations=nit, converged=converged, coupling_residual=coupled_residual),
        ode_last,
    )

    return out
end
