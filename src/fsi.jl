abstract type AbstractRigidShape end

"""
    Circle(R)

2D rigid circular shape with radius `R`.
"""
struct Circle{T<:Real} <: AbstractRigidShape
    R::T
    function Circle{T}(R::T) where {T<:Real}
        R > zero(T) || throw(ArgumentError("circle radius must be positive"))
        return new{T}(R)
    end
end
Circle(R::Real) = Circle{typeof(R)}(R)

"""
    Sphere(R)

3D rigid spherical shape with radius `R`.
"""
struct Sphere{T<:Real} <: AbstractRigidShape
    R::T
    function Sphere{T}(R::T) where {T<:Real}
        R > zero(T) || throw(ArgumentError("sphere radius must be positive"))
        return new{T}(R)
    end
end
Sphere(R::Real) = Sphere{typeof(R)}(R)

"""
    volume(shape)

Geometric volume (2D area for `Circle`, 3D volume for `Sphere`).
"""
volume(shape::Circle{T}) where {T} = convert(T, pi) * shape.R^2
volume(shape::Sphere{T}) where {T} = (convert(T, 4) / convert(T, 3)) * convert(T, pi) * shape.R^3

"""
    sdf(shape, x, y, X0)

Signed-distance-like level set used by moving-boundary examples.
Positive values are inside the rigid body.
"""
function sdf(shape::Circle, x::Real, y::Real, X0::NTuple{2,<:Real})
    T = promote_type(typeof(shape.R), typeof(x), typeof(y), typeof(X0[1]), typeof(X0[2]))
    return convert(T, shape.R) - hypot(convert(T, x) - convert(T, X0[1]), convert(T, y) - convert(T, X0[2]))
end

function sdf(shape::Sphere, x::Real, y::Real, z::Real, X0::NTuple{3,<:Real})
    T = promote_type(typeof(shape.R), typeof(x), typeof(y), typeof(z), typeof(X0[1]), typeof(X0[2]), typeof(X0[3]))
    dx = convert(T, x) - convert(T, X0[1])
    dy = convert(T, y) - convert(T, X0[2])
    dz = convert(T, z) - convert(T, X0[3])
    return convert(T, shape.R) - sqrt(dx^2 + dy^2 + dz^2)
end

@inline _sdf_eval(shape::Circle, x::NTuple{2,T}, X0::NTuple{2,T}) where {T} = sdf(shape, x[1], x[2], X0)
@inline _sdf_eval(shape::Sphere, x::NTuple{3,T}, X0::NTuple{3,T}) where {T} = sdf(shape, x[1], x[2], x[3], X0)

function _sdf_eval(shape::AbstractRigidShape, x::NTuple{N,T}, X0::NTuple{N,T}) where {N,T}
    throw(ArgumentError("no SDF evaluation defined for shape $(typeof(shape)) in $N dimensions"))
end

"""
    RigidBodyState{N,T}

Translation-only rigid body state.
"""
mutable struct RigidBodyState{N,T}
    X::SVector{N,T}
    V::SVector{N,T}
end

function RigidBodyState(
    X::NTuple{N,TX},
    V::NTuple{N,TV},
) where {N,TX<:Real,TV<:Real}
    T = promote_type(TX, TV)
    return RigidBodyState{N,T}(SVector{N,T}(X), SVector{N,T}(V))
end

"""
    RigidBodyParams{N,T}

Rigid-body parameters for translation-only FSI.
"""
struct RigidBodyParams{N,T}
    m::T
    rho_body::T
    shape::AbstractRigidShape
    g::SVector{N,T}
    rho_fluid::T
    buoyancy::Bool
end

function RigidBodyParams(
    m::Real,
    rho_body::Real,
    shape::AbstractRigidShape,
    g::SVector{N,T};
    rho_fluid::Real=one(T),
    buoyancy::Bool=true,
) where {N,T<:Real}
    TT = promote_type(T, typeof(m), typeof(rho_body), typeof(rho_fluid))
    return RigidBodyParams{N,TT}(
        convert(TT, m),
        convert(TT, rho_body),
        shape,
        SVector{N,TT}(Tuple(g)),
        convert(TT, rho_fluid),
        buoyancy,
    )
end

"""
    external_force(params)

Gravity + optional buoyancy contribution.
"""
function external_force(p::RigidBodyParams{N,T}) where {N,T}
    Vshape = convert(T, volume(p.shape))
    coeff = p.buoyancy ? (p.m - p.rho_fluid * Vshape) : p.m
    return coeff * p.g
end

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

Translation-only rigid-body FSI wrapper around `MovingStokesModelMono`.
"""
mutable struct StokesFSIProblem{N,T,FT}
    model::MovingStokesModelMono{N,T,FT,Any}
    state::RigidBodyState{N,T}
    params::RigidBodyParams{N,T}
    xprev::Vector{T}
    pressure_reconstruction::Symbol
    force_sign::T
end

function StokesFSIProblem(
    model::MovingStokesModelMono{N,T,FT},
    state::RigidBodyState{N,T},
    params::RigidBodyParams{N,T};
    xprev::Union{Nothing,AbstractVector{T}}=nothing,
    pressure_reconstruction::Symbol=:linear,
    force_sign::Real=one(T),
) where {N,T,FT}
    model_any = _promote_moving_model_body_to_any(model)
    nsys = last(model_any.layout.pomega)
    x0 = isnothing(xprev) ? zeros(T, nsys) : collect(xprev)
    length(x0) == nsys || throw(ArgumentError("xprev length mismatch: got $(length(x0)), expected $nsys"))
    return StokesFSIProblem{N,T,FT}(
        model_any,
        state,
        params,
        x0,
        pressure_reconstruction,
        convert(T, force_sign),
    )
end

function _advance_rigid_translation!(
    state::RigidBodyState{N,T},
    params::RigidBodyParams{N,T},
    Fhydro::SVector{N,T},
    dt::T;
    ode_scheme::Symbol=:symplectic_euler,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))
    Fext = external_force(params)
    Xn = state.X
    Vn = state.V

    if ode_scheme == :symplectic_euler
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
        Xn1 = Xn + dt * Vn1
        state.V = Vn1
        state.X = Xn1
    elseif ode_scheme == :forward_euler
        Xn1 = Xn + dt * Vn
        Vn1 = Vn + (dt / params.m) * (Fext + Fhydro)
        state.V = Vn1
        state.X = Xn1
    else
        throw(ArgumentError("unsupported ODE scheme `$ode_scheme` (use :symplectic_euler or :forward_euler)"))
    end
    return (X=state.X, V=state.V, Fext=Fext, Fhydro=Fhydro)
end

"""
    step_fsi!(fsi; t, dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)

Advance one rigid-body FSI step (translation-only):
1. predict body trajectory over slab,
2. solve moving-boundary unsteady Stokes,
3. integrate hydrodynamic force,
4. update rigid-body ODE.
"""
function step_fsi!(
    fsi::StokesFSIProblem{N,T};
    t::T,
    dt::T,
    fluid_scheme::Symbol=:CN,
    ode_scheme::Symbol=:symplectic_euler,
) where {N,T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    Xn = fsi.state.X
    Vn = fsi.state.V
    shape = fsi.params.shape

    Xpred(τ::T) = Xn + (τ - t) * Vn
    body = function (args...)
        length(args) == N + 1 || throw(ArgumentError("body closure expected $N spatial coordinates plus time"))
        τ = convert(T, args[end])
        xT = ntuple(i -> convert(T, args[i]), N)
        XT = Tuple(Xpred(τ))
        return _sdf_eval(shape, xT, XT)
    end

    fsi.model.body = body
    fsi.model.bc_cut_u = ntuple(d -> Dirichlet((args...) -> Vn[d]), N)

    sys = solve_unsteady_moving!(fsi.model, fsi.xprev; t=t, dt=dt, scheme=fluid_scheme)

    tnext = t + dt
    Xnext_pred = Xpred(tnext)
    sm = endtime_static_model(fsi.model)
    q = integrated_embedded_force(
        sm,
        sys;
        pressure_reconstruction=fsi.pressure_reconstruction,
        x0=Tuple(Xnext_pred),
    )

    Fhydro = fsi.force_sign * SVector{N,T}(Tuple(q.force))
    ode = _advance_rigid_translation!(fsi.state, fsi.params, Fhydro, dt; ode_scheme=ode_scheme)

    fsi.xprev .= sys.x

    return (
        sys=sys,
        force=q,
        X=ode.X,
        V=ode.V,
        Fhydro=ode.Fhydro,
        Fext=ode.Fext,
        t=tnext,
    )
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
        res = norm(out.sys.A * out.sys.x - out.sys.b)
        rec = (
            step=step,
            t=out.t,
            X=out.X,
            V=out.V,
            Fhydro=out.Fhydro,
            Fext=out.Fext,
            force=out.force.force,
            torque=out.force.torque,
            residual=res,
        )
        history[step] = rec
        callback === nothing || callback(fsi, rec, out)
        t = out.t
    end
    return history
end

# Placeholders for future FSI extensions (no behavior change in v0):
# - rigid-body rotation (omega, inertia tensor)
# - multiple rigid bodies
# - contact/collision handling
