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
    Ellipse(a, b)

2D rigid ellipse with semi-axes `a` and `b`.
"""
struct Ellipse{T<:Real} <: AbstractRigidShape
    a::T
    b::T
    function Ellipse{T}(a::T, b::T) where {T<:Real}
        a > zero(T) || throw(ArgumentError("ellipse semi-axis `a` must be positive"))
        b > zero(T) || throw(ArgumentError("ellipse semi-axis `b` must be positive"))
        return new{T}(a, b)
    end
end
Ellipse(a::Real, b::Real) = Ellipse{promote_type(typeof(a), typeof(b))}(promote(a, b)...)

"""
    volume(shape)

Geometric volume (2D area for planar shapes, 3D volume for sphere).
"""
volume(shape::Circle{T}) where {T} = convert(T, pi) * shape.R^2
volume(shape::Ellipse{T}) where {T} = convert(T, pi) * shape.a * shape.b
volume(shape::Sphere{T}) where {T} = (convert(T, 4) / convert(T, 3)) * convert(T, pi) * shape.R^3

body_volume(shape::AbstractRigidShape) = volume(shape)

"""
    body_inertia(shape, rho_body)

Planar scalar moment of inertia (about center, per unit thickness) for 2D shapes,
and standard scalar inertia for a 3D sphere.
"""
function body_inertia(shape::Circle{T}, rho_body::Real) where {T}
    TT = promote_type(T, typeof(rho_body))
    return convert(TT, 0.5) * convert(TT, rho_body) * convert(TT, pi) * convert(TT, shape.R)^4
end

function body_inertia(shape::Ellipse{T}, rho_body::Real) where {T}
    TT = promote_type(T, typeof(rho_body))
    a = convert(TT, shape.a)
    b = convert(TT, shape.b)
    m = convert(TT, rho_body) * convert(TT, pi) * a * b
    return convert(TT, 0.25) * m * (a^2 + b^2)
end

function body_inertia(shape::Sphere{T}, rho_body::Real) where {T}
    TT = promote_type(T, typeof(rho_body))
    m = convert(TT, rho_body) * convert(TT, volume(shape))
    return convert(TT, 2 // 5) * m * convert(TT, shape.R)^2
end

rotation_affects_geometry(::Circle) = false
rotation_affects_geometry(::Sphere) = false
rotation_affects_geometry(::Ellipse) = true
rotation_affects_geometry(::AbstractRigidShape) = true

@inline function rotmat(theta::T) where {T}
    c = cos(theta)
    s = sin(theta)
    return SMatrix{2,2,T}(c, -s, s, c)
end

@inline function to_body_frame(x::SVector{2,T}, X::SVector{2,T}, theta::T) where {T}
    return rotmat(-theta) * (x - X)
end

"""
    sdf(shape, ...)

Level-set convention used in moving-boundary examples (`>0` inside).
"""
sdf(shape::Circle{T}, xi::SVector{2,T}) where {T} = convert(T, shape.R) - norm(xi)
sdf(shape::Ellipse{T}, xi::SVector{2,T}) where {T} = one(T) - sqrt((xi[1] / convert(T, shape.a))^2 + (xi[2] / convert(T, shape.b))^2)
sdf(shape::Sphere{T}, xi::SVector{3,T}) where {T} = convert(T, shape.R) - norm(xi)

function sdf(shape::Circle, x::Real, y::Real, X0::NTuple{2,<:Real})
    T = promote_type(typeof(shape.R), typeof(x), typeof(y), typeof(X0[1]), typeof(X0[2]))
    return sdf(shape, SVector{2,T}(convert(T, x) - convert(T, X0[1]), convert(T, y) - convert(T, X0[2])))
end

function sdf(shape::Ellipse, x::Real, y::Real, X0::NTuple{2,<:Real})
    T = promote_type(typeof(shape.a), typeof(shape.b), typeof(x), typeof(y), typeof(X0[1]), typeof(X0[2]))
    xi = SVector{2,T}(convert(T, x) - convert(T, X0[1]), convert(T, y) - convert(T, X0[2]))
    return sdf(shape, xi)
end

function sdf(shape::Sphere, x::Real, y::Real, z::Real, X0::NTuple{3,<:Real})
    T = promote_type(typeof(shape.R), typeof(x), typeof(y), typeof(z), typeof(X0[1]), typeof(X0[2]), typeof(X0[3]))
    xi = SVector{3,T}(
        convert(T, x) - convert(T, X0[1]),
        convert(T, y) - convert(T, X0[2]),
        convert(T, z) - convert(T, X0[3]),
    )
    return sdf(shape, xi)
end

@inline _sdf_eval(shape::Circle, x::NTuple{2,T}, X0::NTuple{2,T}) where {T} = sdf(shape, x[1], x[2], X0)
@inline _sdf_eval(shape::Ellipse, x::NTuple{2,T}, X0::NTuple{2,T}) where {T} = sdf(shape, x[1], x[2], X0)
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
    RigidBodyState2D{T}

2D rigid body state with translation and scalar rotation.
"""
mutable struct RigidBodyState2D{T}
    X::SVector{2,T}
    V::SVector{2,T}
    theta::T
    omega::T
end

function RigidBodyState2D(
    X::SVector{2,T},
    V::SVector{2,T};
    theta::T=zero(T),
    omega::T=zero(T),
) where {T}
    return RigidBodyState2D{T}(X, V, theta, omega)
end

function RigidBodyState2D(
    X::NTuple{2,TX},
    V::NTuple{2,TV};
    theta::Real=0,
    omega::Real=0,
) where {TX<:Real,TV<:Real}
    T = promote_type(TX, TV, typeof(theta), typeof(omega))
    return RigidBodyState2D{T}(
        SVector{2,T}(X),
        SVector{2,T}(V),
        convert(T, theta),
        convert(T, omega),
    )
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
    RigidBodyParams2D{T,S}

2D rigid-body parameters with scalar inertia `I`.
"""
struct RigidBodyParams2D{T,S<:AbstractRigidShape}
    m::T
    I::T
    rho_body::T
    rho_fluid::T
    g::SVector{2,T}
    shape::S
    buoyancy::Bool
end

function RigidBodyParams2D(
    m::Real,
    I::Real,
    rho_body::Real,
    shape::S,
    g::SVector{2,T};
    rho_fluid::Real=one(T),
    buoyancy::Bool=true,
) where {T<:Real,S<:AbstractRigidShape}
    TT = promote_type(T, typeof(m), typeof(I), typeof(rho_body), typeof(rho_fluid))
    return RigidBodyParams2D{TT,S}(
        convert(TT, m),
        convert(TT, I),
        convert(TT, rho_body),
        convert(TT, rho_fluid),
        SVector{2,TT}(Tuple(g)),
        shape,
        buoyancy,
    )
end

function RigidBodyParams2D(
    m::Real,
    rho_body::Real,
    shape::S,
    g::SVector{2,T};
    I::Union{Nothing,Real}=nothing,
    rho_fluid::Real=one(T),
    buoyancy::Bool=true,
) where {T<:Real,S<:AbstractRigidShape}
    inertia = isnothing(I) ? body_inertia(shape, rho_body) : I
    return RigidBodyParams2D(m, inertia, rho_body, shape, g; rho_fluid=rho_fluid, buoyancy=buoyancy)
end

"""
    external_force(params)

Gravity + optional buoyancy contribution.
"""
function external_force(p::RigidBodyParams{N,T}) where {N,T}
    Vshape = convert(T, body_volume(p.shape))
    coeff = p.buoyancy ? (p.m - p.rho_fluid * Vshape) : p.m
    return coeff * p.g
end

function external_force(p::RigidBodyParams2D{T}) where {T}
    Vshape = convert(T, body_volume(p.shape))
    coeff = p.buoyancy ? (p.m - p.rho_fluid * Vshape) : p.m
    return coeff * p.g
end

"""
    external_torque(params, state, t)

Default external torque (zero). Override with a method for custom forcing.
"""
external_torque(p::RigidBodyParams2D{T}, state::RigidBodyState2D{T}, t::T) where {T} = zero(T)

@inline function rigid_velocity_2d(
    x::SVector{2,T},
    X::SVector{2,T},
    V::SVector{2,T},
    omega::T,
) where {T}
    r = x - X
    return SVector{2,T}(V[1] - omega * r[2], V[2] + omega * r[1])
end

rigid_velocity(x::SVector{2,T}, state::RigidBodyState2D{T}) where {T} =
    rigid_velocity_2d(x, state.X, state.V, state.omega)

# TODO(3D): rigid_velocity_3d(x, X, V, Omega) = V + cross(Omega, x - X)

function rigid_cut_bc_tuple_2d(statefun)
    return (
        Dirichlet((x, y, t) -> begin
            s = statefun(t)
            rigid_velocity_2d(SVector(x, y), s.X, s.V, s.omega)[1]
        end),
        Dirichlet((x, y, t) -> begin
            s = statefun(t)
            rigid_velocity_2d(SVector(x, y), s.X, s.V, s.omega)[2]
        end),
    )
end

function rigid_body_levelset(shape::AbstractRigidShape, statefun)
    if shape isa Circle
        cshape = shape
        return function (x, y, t)
            s = statefun(t)
            return sdf(cshape, x, y, (s.X[1], s.X[2]))
        end
    end

    return function (x, y, t)
        s = statefun(t)
        xlab = SVector(x, y)
        if rotation_affects_geometry(shape)
            xi = to_body_frame(xlab, s.X, s.theta)
        else
            xi = xlab - s.X
        end
        return sdf(shape, xi)
    end
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

mutable struct StokesFSIProblem2D{T,FT,S<:AbstractRigidShape}
    model::MovingStokesModelMono{2,T,FT,Any}
    state::RigidBodyState2D{T}
    params::RigidBodyParams2D{T,S}
    xprev::Vector{T}
    pressure_reconstruction::Symbol
    force_sign::T
    torque_sign::T
end

function StokesFSIProblem2D(
    model::MovingStokesModelMono{2,T,FT},
    state::RigidBodyState2D{T},
    params::RigidBodyParams2D{T,S};
    xprev::Union{Nothing,AbstractVector{T}}=nothing,
    pressure_reconstruction::Symbol=:linear,
    force_sign::Real=one(T),
    torque_sign::Real=one(T),
) where {T,FT,S<:AbstractRigidShape}
    model_any = _promote_moving_model_body_to_any(model)
    nsys = last(model_any.layout.pomega)
    x0 = isnothing(xprev) ? zeros(T, nsys) : collect(xprev)
    length(x0) == nsys || throw(ArgumentError("xprev length mismatch: got $(length(x0)), expected $nsys"))
    return StokesFSIProblem2D{T,FT,S}(
        model_any,
        state,
        params,
        x0,
        pressure_reconstruction,
        convert(T, force_sign),
        convert(T, torque_sign),
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

function _advance_rigid_body_2d!(
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

    Xpred(tau::T) = Xn + (tau - t) * Vn
    body = function (args...)
        length(args) == N + 1 || throw(ArgumentError("body closure expected $N spatial coordinates plus time"))
        tau = convert(T, args[end])
        xT = ntuple(i -> convert(T, args[i]), N)
        XT = Tuple(Xpred(tau))
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
    step_fsi_rotation!(fsi; t, dt, fluid_scheme=:CN, ode_scheme=:symplectic_euler)

Advance one 2D rigid-body FSI step with translation + rotation.
"""
function step_fsi_rotation!(
    fsi::StokesFSIProblem2D{T};
    t::T,
    dt::T,
    fluid_scheme::Symbol=:CN,
    ode_scheme::Symbol=:symplectic_euler,
) where {T}
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    sn = fsi.state
    statefun = function (tau::Real)
        taum = convert(T, tau)
        return RigidBodyState2D(
            sn.X + (taum - t) * sn.V,
            sn.V;
            theta=sn.theta + (taum - t) * sn.omega,
            omega=sn.omega,
        )
    end

    fsi.model.body = rigid_body_levelset(fsi.params.shape, statefun)
    fsi.model.bc_cut_u = rigid_cut_bc_tuple_2d(statefun)

    sys = solve_unsteady_moving!(fsi.model, fsi.xprev; t=t, dt=dt, scheme=fluid_scheme)

    tnext = t + dt
    snext_pred = statefun(tnext)
    sm = endtime_static_model(fsi.model)
    q = integrated_embedded_force(
        sm,
        sys;
        pressure_reconstruction=fsi.pressure_reconstruction,
        x0=(snext_pred.X[1], snext_pred.X[2]),
    )

    Fhydro = fsi.force_sign * SVector{2,T}(Tuple(q.force))
    tau_hydro = fsi.torque_sign * convert(T, q.torque)
    ode = _advance_rigid_body_2d!(
        fsi.state,
        fsi.params,
        Fhydro,
        tau_hydro,
        dt;
        t=t,
        ode_scheme=ode_scheme,
    )

    fsi.xprev .= sys.x

    return (
        sys=sys,
        force=q,
        X=ode.X,
        V=ode.V,
        theta=ode.theta,
        omega=ode.omega,
        Fhydro=ode.Fhydro,
        tau_hydro=ode.tau_hydro,
        Fext=ode.Fext,
        tau_ext=ode.tau_ext,
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

"""
    simulate_fsi_rotation!(fsi; t0=0, dt, nsteps, fluid_scheme=:CN, ode_scheme=:symplectic_euler, callback=nothing)

Run repeated `step_fsi_rotation!` updates and return a lightweight history vector.
"""
function simulate_fsi_rotation!(
    fsi::StokesFSIProblem2D{T};
    t0::T=zero(T),
    dt::T,
    nsteps::Integer,
    fluid_scheme::Symbol=:CN,
    ode_scheme::Symbol=:symplectic_euler,
    callback=nothing,
) where {T}
    nsteps >= 0 || throw(ArgumentError("nsteps must be nonnegative"))
    dt > zero(T) || throw(ArgumentError("dt must be positive"))

    history = Vector{NamedTuple}(undef, nsteps)
    t = t0
    for step in 1:nsteps
        out = step_fsi_rotation!(fsi; t=t, dt=dt, fluid_scheme=fluid_scheme, ode_scheme=ode_scheme)
        res = norm(out.sys.A * out.sys.x - out.sys.b)
        rec = (
            step=step,
            t=out.t,
            X=out.X,
            V=out.V,
            theta=out.theta,
            omega=out.omega,
            Fhydro=out.Fhydro,
            tau_hydro=out.tau_hydro,
            Fext=out.Fext,
            tau_ext=out.tau_ext,
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

# Placeholders for future FSI extensions:
# - 3D orientation/rotation (matrix/quaternion)
# - multiple bodies
# - contact/collision handling
