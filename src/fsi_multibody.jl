# Multi-body FSI: N rigid bodies sharing one fluid solve.
#
# Each body has its own RigidBodyState and RigidBodyParams.  The fluid sees a
# single combined level-set (union of all solid regions) and a BC dispatcher
# that assigns the velocity of the nearest body to each cut-cell.  After the
# fluid solve, the hydrodynamic force on each body is extracted by partitioning
# the interface cells by proximity to each body centre.

# ── Multi-body level-set ────────────────────────────────────────────────────

"""
    multi_body_levelset(shapes, statefuns)

Build a combined level-set callback `body(x..., t)` representing the union of
N rigid bodies.  Each body contributes its own SDF; the union is the
cell-wise maximum (positive = inside any solid).
"""
function multi_body_levelset(shapes, statefuns)
    return function (args...)
        nargs = length(args)
        nargs >= 2 || throw(ArgumentError("body callback expects spatial coords plus time"))
        N = nargs - 1
        tau = args[end]
        T = Float64
        xlab = SVector{N,T}(ntuple(i -> convert(T, args[i]), N))
        val = typemin(T)
        for k in eachindex(shapes)
            s = statefuns[k](tau)
            T2 = eltype(state_position(s))
            xl = SVector{N,T2}(ntuple(i -> convert(T2, args[i]), N))
            X = state_position(s)
            xi = if rotation_affects_geometry(shapes[k])
                rotate_to_body_frame(xl, X, state_orientation(s))
            else
                xl - X
            end
            val = max(val, sdf(shapes[k], xi))
        end
        return val
    end
end

"""
    multi_body_cut_bc_tuple(shapes, statefuns, Val(N))

Build per-component cut-boundary Dirichlet BC tuple for N bodies.
At each interface point the velocity of the nearest body (by SDF value) is
assigned — whichever body's level-set is largest (least negative) owns the
point.
"""
function multi_body_cut_bc_tuple(shapes, statefuns, ::Val{N}) where {N}
    return ntuple(d -> Dirichlet((args...) -> begin
        length(args) == N + 1 || throw(ArgumentError("cut BC expects $N coords plus time"))
        tau = args[end]
        T = Float64
        xlab = SVector{N,T}(ntuple(i -> convert(T, args[i]), N))
        # Find the body with the largest SDF value (= closest or inside).
        best_val = typemin(T)
        best_k   = 1
        for k in eachindex(shapes)
            s = statefuns[k](tau)
            T2 = eltype(state_position(s))
            xl = SVector{N,T2}(ntuple(i -> convert(T2, args[i]), N))
            X = state_position(s)
            xi = if rotation_affects_geometry(shapes[k])
                rotate_to_body_frame(xl, X, state_orientation(s))
            else
                xl - X
            end
            v = sdf(shapes[k], xi)
            if v > best_val
                best_val = v
                best_k   = k
            end
        end
        s = statefuns[best_k](tau)
        T2 = eltype(state_position(s))
        xl = SVector{N,T2}(ntuple(i -> convert(T2, args[i]), N))
        return rigid_boundary_velocity(xl, s)[d]
    end), N)
end

# ── Multi-body FSI struct ────────────────────────────────────────────────────

"""
    MultiBodyFSIProblem{Nbodies,N,T,MT}

FSI problem with `Nbodies` rigid bodies sharing one `MovingStokesModelMono`.
Each body has its own state and params; the fluid solve is shared.

Fields:
- `model`   — shared `MovingStokesModelMono`
- `states`  — `Vector` of mutable rigid body states
- `params`  — `Vector` of rigid body params
- `shapes`  — `Vector` of `AbstractRigidShape`
- `xprev`   — previous fluid solution vector
- `pressure_reconstruction` — `:linear` or `:mean`
- `force_signs`  — per-body drag sign (±1)
- `torque_signs` — per-body torque sign (±1)
"""
mutable struct MultiBodyFSIProblem{Nbodies,N,T,MT}
    model::MT
    states::Vector{Any}
    params::Vector{Any}
    shapes::Vector{Any}
    xprev::Vector{T}
    pressure_reconstruction::Symbol
    force_signs::Vector{T}
    torque_signs::Vector{T}
end

"""
    MultiBodyFSIProblem(model, states, params, shapes; kwargs...)

Construct a `MultiBodyFSIProblem`.

- `model`  — `MovingStokesModelMono{N,T}` (body and bc_cut_u will be overwritten)
- `states` — vector of `RigidBodyState` objects (one per body)
- `params` — vector of `RigidBodyParams` objects (one per body)
- `shapes` — vector of `AbstractRigidShape` (one per body)

Keyword arguments:
- `xprev`  — initial fluid state (zeros if not provided)
- `pressure_reconstruction` — `:linear` (default)
- `force_signs`  — per-body ±1 vector (default: all 1)
- `torque_signs` — per-body ±1 vector (default: all 1)
"""
function MultiBodyFSIProblem(
    model::MovingStokesModelMono{N,T,FT},
    states::AbstractVector,
    params::AbstractVector,
    shapes::AbstractVector;
    xprev::Union{Nothing,AbstractVector{T}}=nothing,
    pressure_reconstruction::Symbol=:linear,
    force_signs::AbstractVector{<:Real}=ones(T, length(states)),
    torque_signs::AbstractVector{<:Real}=ones(T, length(states)),
) where {N,T,FT}
    Nbodies = length(states)
    Nbodies == length(params) == length(shapes) || throw(ArgumentError("states, params, shapes must have equal length"))
    Nbodies >= 1 || throw(ArgumentError("need at least one body"))

    model_any = _promote_moving_model_body_to_any(model)
    nsys = last(model_any.layout.pomega)
    x0 = isnothing(xprev) ? zeros(T, nsys) : collect(xprev)

    return MultiBodyFSIProblem{Nbodies,N,T,typeof(model_any)}(
        model_any,
        collect(Any, states),
        collect(Any, params),
        collect(Any, shapes),
        x0,
        pressure_reconstruction,
        convert(Vector{T}, force_signs),
        convert(Vector{T}, torque_signs),
    )
end

# ── Internal helpers ─────────────────────────────────────────────────────────

function _multi_predict_statefuns(states_n, t::T, tau::T) where {T}
    return [(tau2 -> _predict_state(states_n[k], t, convert(T, tau2))) for k in eachindex(states_n)]
end

function _set_multi_moving_state!(fsi::MultiBodyFSIProblem{Nbodies,N,T}, statefuns) where {Nbodies,N,T}
    fsi.model.body     = multi_body_levelset(fsi.shapes, statefuns)
    fsi.model.bc_cut_u = multi_body_cut_bc_tuple(fsi.shapes, statefuns, Val(N))
    return nothing
end

"""
    _per_body_forces(fsi, sm, sys, statefuns, tnext)

Partition embedded-boundary force contributions among bodies by assigning each
interface centroid to the body with the largest SDF value at that point.
Returns a vector of `(force, torque)` NamedTuples, one per body.
"""
function _per_body_forces(
    fsi::MultiBodyFSIProblem{Nbodies,N,T},
    sm::StokesModelMono{N,T},
    sys,
    statefuns,
    tnext::T,
) where {Nbodies,N,T}
    q_all = embedded_boundary_quantities(
        sm, sys;
        pressure_reconstruction=fsi.pressure_reconstruction,
        x0=nothing,   # torque computed per-body below
    )

    cap = sm.cap_p

    # Per-body accumulators.
    F  = [zeros(T, N) for _ in 1:Nbodies]
    τ2 = zeros(T, Nbodies)                          # 2D torque
    τ3 = [zeros(T, 3) for _ in 1:Nbodies]           # 3D torque

    for i in q_all.interface_indices
        xγ = cap.C_γ[i]
        # Assign cell to the body with largest SDF at the interface centroid.
        best_val = typemin(T)
        best_k   = 1
        for k in 1:Nbodies
            s  = statefuns[k](tnext)
            T2 = eltype(state_position(s))
            xl = SVector{N,T2}(ntuple(d -> convert(T2, xγ[d]), N))
            X  = state_position(s)
            xi = if rotation_affects_geometry(fsi.shapes[k])
                rotate_to_body_frame(xl, X, state_orientation(s))
            else
                xl - X
            end
            v = sdf(fsi.shapes[k], xi)
            if v > best_val
                best_val = v
                best_k   = k
            end
        end

        fv = q_all.force_density[i]
        for d in 1:N
            F[best_k][d] += fv[d]
        end

        # Torque contribution relative to body centre.
        s_k = statefuns[best_k](tnext)
        X_k = SVector{N,T}(state_position(s_k))
        r   = SVector{N,T}(ntuple(d -> convert(T, xγ[d]) - X_k[d], N))
        fvs = SVector{N,T}(ntuple(d -> convert(T, fv[d]), N))
        if N == 2
            τ2[best_k] += r[1] * fvs[2] - r[2] * fvs[1]
        elseif N == 3
            rf  = SVector{3,T}(r[1], r[2], r[3])
            fvf = SVector{3,T}(fvs[1], fvs[2], fvs[3])
            c   = rf × fvf
            τ3[best_k] .+= c
        end
    end

    return [
        (
            force   = SVector{N,T}(F[k]),
            torque  = N == 2 ? τ2[k] : SVector{3,T}(τ3[k]),
        )
        for k in 1:Nbodies
    ]
end

# ── step_multi_fsi! ──────────────────────────────────────────────────────────

"""
    step_multi_fsi!(fsi; t, dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)

Advance one time step of the multi-body FSI problem.

1. Predict each body's state over [t, t+dt] (frozen-body extrapolation).
2. Solve the shared fluid problem with the combined level-set and BC.
3. Partition embedded-boundary forces among bodies by interface-centroid proximity.
4. Advance each body's rigid-body ODE independently.

Returns a NamedTuple with fields:
- `sys`     — LinearSystem from the fluid solve
- `t`       — new time (t + dt)
- `states`  — vector of per-body NamedTuples (X, V, Fhydro, …)
- `forces`  — vector of per-body raw force NamedTuples
"""
function step_multi_fsi!(
    fsi::MultiBodyFSIProblem{Nbodies,N,T};
    t::T,
    dt::T,
    fluid_scheme::Symbol=:CN,
    ode_scheme::Symbol=:symplectic_euler,
) where {Nbodies,N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    # 1. Snapshot current states and build prediction closures.
    states_n  = [state_copy(fsi.states[k]) for k in 1:Nbodies]
    statefuns = [(tau -> _predict_state(states_n[k], t, convert(T, tau))) for k in 1:Nbodies]

    # 2. Solve shared fluid problem.
    _set_multi_moving_state!(fsi, statefuns)
    sys = solve_unsteady_moving!(fsi.model, fsi.xprev; t=t, dt=dt, scheme=fluid_scheme)

    # 3. Partition forces.
    tnext = t + dt
    sm    = endtime_static_model(fsi.model)
    per_body = _per_body_forces(fsi, sm, sys, statefuns, convert(T, tnext))

    # 4. Advance each rigid body.
    ode_results = Vector{Any}(undef, Nbodies)
    for k in 1:Nbodies
        q_k      = per_body[k]
        Fhydro_k = convert(T, fsi.force_signs[k]) * _extract_force(q_k, Val(N), T)
        tau_hydro_k = _extract_torque_hydro(fsi.states[k], q_k, fsi.torque_signs[k], T)
        ode_results[k] = _advance_state!(
            fsi.states[k],
            fsi.params[k],
            Fhydro_k,
            tau_hydro_k,
            dt;
            t=t,
            ode_scheme=ode_scheme,
        )
    end

    fsi.xprev .= sys.x

    return (sys=sys, t=tnext, states=ode_results, forces=per_body)
end
