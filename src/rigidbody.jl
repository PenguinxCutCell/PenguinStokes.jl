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
    Ellipsoid(a, b, c)

3D rigid ellipsoid with semi-axes `a`, `b`, and `c`.
"""
struct Ellipsoid{T<:Real} <: AbstractRigidShape
    a::T
    b::T
    c::T
    function Ellipsoid{T}(a::T, b::T, c::T) where {T<:Real}
        a > zero(T) || throw(ArgumentError("ellipsoid semi-axis `a` must be positive"))
        b > zero(T) || throw(ArgumentError("ellipsoid semi-axis `b` must be positive"))
        c > zero(T) || throw(ArgumentError("ellipsoid semi-axis `c` must be positive"))
        return new{T}(a, b, c)
    end
end
Ellipsoid(a::Real, b::Real, c::Real) = Ellipsoid{promote_type(typeof(a), typeof(b), typeof(c))}(promote(a, b, c)...)

"""
    volume(shape)

Geometric volume (2D area for planar shapes, 3D volume for 3D shapes).
"""
volume(shape::Circle{T}) where {T} = convert(T, pi) * shape.R^2
volume(shape::Ellipse{T}) where {T} = convert(T, pi) * shape.a * shape.b
volume(shape::Sphere{T}) where {T} = (convert(T, 4) / convert(T, 3)) * convert(T, pi) * shape.R^3
volume(shape::Ellipsoid{T}) where {T} = (convert(T, 4) / convert(T, 3)) * convert(T, pi) * shape.a * shape.b * shape.c

body_volume(shape::AbstractRigidShape) = volume(shape)

"""
    body_inertia(shape, rho_body)

Return inertia for rigid-body dynamics.
- 2D shapes: scalar out-of-plane moment of inertia (per unit thickness)
- 3D sphere: scalar moment of inertia
- 3D ellipsoid: inertia tensor about center in body frame
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

function body_inertia(shape::Ellipsoid{T}, rho_body::Real) where {T}
    TT = promote_type(T, typeof(rho_body))
    a = convert(TT, shape.a)
    b = convert(TT, shape.b)
    c = convert(TT, shape.c)
    m = convert(TT, rho_body) * convert(TT, volume(shape))
    d11 = (m / convert(TT, 5)) * (b^2 + c^2)
    d22 = (m / convert(TT, 5)) * (a^2 + c^2)
    d33 = (m / convert(TT, 5)) * (a^2 + b^2)
    return @SMatrix [d11 zero(TT) zero(TT); zero(TT) d22 zero(TT); zero(TT) zero(TT) d33]
end

rotation_affects_geometry(::Circle) = false
rotation_affects_geometry(::Sphere) = false
rotation_affects_geometry(::Ellipse) = true
rotation_affects_geometry(::Ellipsoid) = true
rotation_affects_geometry(::AbstractRigidShape) = true

"""
    sdf(shape, xi)

Level-set convention used in moving-boundary examples (`>0` inside).
"""
sdf(shape::Circle{T}, xi::SVector{2,T}) where {T} = convert(T, shape.R) - norm(xi)
sdf(shape::Ellipse{T}, xi::SVector{2,T}) where {T} = one(T) - sqrt((xi[1] / convert(T, shape.a))^2 + (xi[2] / convert(T, shape.b))^2)
sdf(shape::Sphere{T}, xi::SVector{3,T}) where {T} = convert(T, shape.R) - norm(xi)
sdf(shape::Ellipsoid{T}, xi::SVector{3,T}) where {T} = one(T) - sqrt((xi[1] / convert(T, shape.a))^2 + (xi[2] / convert(T, shape.b))^2 + (xi[3] / convert(T, shape.c))^2)

function sdf(shape::Circle, x::Real, y::Real, X0::NTuple{2,<:Real})
    T = promote_type(typeof(shape.R), typeof(x), typeof(y), typeof(X0[1]), typeof(X0[2]))
    return sdf(shape, SVector{2,T}(convert(T, x) - convert(T, X0[1]), convert(T, y) - convert(T, X0[2])))
end

function sdf(shape::Ellipse, x::Real, y::Real, X0::NTuple{2,<:Real})
    T = promote_type(typeof(shape.a), typeof(shape.b), typeof(x), typeof(y), typeof(X0[1]), typeof(X0[2]))
    return sdf(shape, SVector{2,T}(convert(T, x) - convert(T, X0[1]), convert(T, y) - convert(T, X0[2])))
end

function sdf(shape::Sphere, x::Real, y::Real, z::Real, X0::NTuple{3,<:Real})
    T = promote_type(typeof(shape.R), typeof(x), typeof(y), typeof(z), typeof(X0[1]), typeof(X0[2]), typeof(X0[3]))
    return sdf(shape, SVector{3,T}(convert(T, x) - convert(T, X0[1]), convert(T, y) - convert(T, X0[2]), convert(T, z) - convert(T, X0[3])))
end

function sdf(shape::Ellipsoid, x::Real, y::Real, z::Real, X0::NTuple{3,<:Real})
    T = promote_type(typeof(shape.a), typeof(shape.b), typeof(shape.c), typeof(x), typeof(y), typeof(z), typeof(X0[1]), typeof(X0[2]), typeof(X0[3]))
    return sdf(shape, SVector{3,T}(convert(T, x) - convert(T, X0[1]), convert(T, y) - convert(T, X0[2]), convert(T, z) - convert(T, X0[3])))
end

@inline _sdf_eval(shape::Circle, x::NTuple{2,T}, X0::NTuple{2,T}) where {T} = sdf(shape, x[1], x[2], X0)
@inline _sdf_eval(shape::Ellipse, x::NTuple{2,T}, X0::NTuple{2,T}) where {T} = sdf(shape, x[1], x[2], X0)
@inline _sdf_eval(shape::Sphere, x::NTuple{3,T}, X0::NTuple{3,T}) where {T} = sdf(shape, x[1], x[2], x[3], X0)
@inline _sdf_eval(shape::Ellipsoid, x::NTuple{3,T}, X0::NTuple{3,T}) where {T} = sdf(shape, x[1], x[2], x[3], X0)

function _sdf_eval(shape::AbstractRigidShape, x::NTuple{N,T}, X0::NTuple{N,T}) where {N,T}
    throw(ArgumentError("no SDF evaluation defined for shape $(typeof(shape)) in $N dimensions"))
end

"""
    RigidBodyState{N,T}

Translation-only rigid-body state with center position `X` and velocity `V`.
"""
mutable struct RigidBodyState{N,T}
    X::SVector{N,T}
    V::SVector{N,T}
end

function RigidBodyState(X::NTuple{N,TX}, V::NTuple{N,TV}) where {N,TX<:Real,TV<:Real}
    T = promote_type(TX, TV)
    return RigidBodyState{N,T}(SVector{N,T}(X), SVector{N,T}(V))
end

"""
    RigidBodyState2D{T}

2D rigid-body state with translation (`X`, `V`) and scalar rotation (`theta`, `omega`).
"""
mutable struct RigidBodyState2D{T}
    X::SVector{2,T}
    V::SVector{2,T}
    theta::T
    omega::T
end

function RigidBodyState2D(X::SVector{2,T}, V::SVector{2,T}; theta::T=zero(T), omega::T=zero(T)) where {T}
    return RigidBodyState2D{T}(X, V, theta, omega)
end

function RigidBodyState2D(X::NTuple{2,TX}, V::NTuple{2,TV}; theta::Real=0, omega::Real=0) where {TX<:Real,TV<:Real}
    T = promote_type(TX, TV, typeof(theta), typeof(omega))
    return RigidBodyState2D{T}(SVector{2,T}(X), SVector{2,T}(V), convert(T, theta), convert(T, omega))
end

"""
    RigidBodyState3D{T}

3D rigid-body state with translation (`X`, `V`) and rotational state (`Q`, `Omega`).
`Q` maps body-frame coordinates to lab-frame coordinates.
"""
mutable struct RigidBodyState3D{T}
    X::SVector{3,T}
    V::SVector{3,T}
    Q::SMatrix{3,3,T,9}
    Omega::SVector{3,T}
end

function RigidBodyState3D(
    X::SVector{3,T},
    V::SVector{3,T};
    Q::SMatrix{3,3,T,9}=rotation_matrix(identity_orientation(Val(3), T)),
    Omega::SVector{3,T}=SVector{3,T}(zero(T), zero(T), zero(T)),
) where {T}
    return RigidBodyState3D{T}(X, V, Q, Omega)
end

function RigidBodyState3D(
    X::NTuple{3,TX},
    V::NTuple{3,TV};
    Q::Union{Nothing,AbstractMatrix}=nothing,
    Omega::NTuple{3,<:Real}=(0.0, 0.0, 0.0),
) where {TX<:Real,TV<:Real}
    T = promote_type(TX, TV, map(typeof, Omega)...)
    QQ = isnothing(Q) ? rotation_matrix(identity_orientation(Val(3), T)) : SMatrix{3,3,T,9}(Q)
    return RigidBodyState3D{T}(SVector{3,T}(X), SVector{3,T}(V), QQ, SVector{3,T}(Omega))
end

"""
    RigidBodyParams{N,T}

Translation-only rigid-body parameters.
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
    RigidBodyParams3D{T,S}

3D rigid-body parameters with isotropic (`I::T`) or tensor (`I::SMatrix`) inertia.
"""
struct RigidBodyParams3D{T,S<:AbstractRigidShape}
    m::T
    I::Union{T,SMatrix{3,3,T,9}}
    rho_body::T
    rho_fluid::T
    g::SVector{3,T}
    shape::S
    buoyancy::Bool
end

function RigidBodyParams3D(
    m::Real,
    I::Union{Real,SMatrix{3,3,<:Real,9},AbstractMatrix},
    rho_body::Real,
    shape::S,
    g::SVector{3,T};
    rho_fluid::Real=one(T),
    buoyancy::Bool=true,
) where {T<:Real,S<:AbstractRigidShape}
    TT = promote_type(T, typeof(m), typeof(rho_body), typeof(rho_fluid))
    I3 = I isa Real ? convert(TT, I) : SMatrix{3,3,TT,9}(I)
    return RigidBodyParams3D{TT,S}(
        convert(TT, m),
        I3,
        convert(TT, rho_body),
        convert(TT, rho_fluid),
        SVector{3,TT}(Tuple(g)),
        shape,
        buoyancy,
    )
end

function RigidBodyParams3D(
    m::Real,
    rho_body::Real,
    shape::S,
    g::SVector{3,T};
    I::Union{Nothing,Real,SMatrix{3,3,<:Real,9},AbstractMatrix}=nothing,
    rho_fluid::Real=one(T),
    buoyancy::Bool=true,
) where {T<:Real,S<:AbstractRigidShape}
    inertia = isnothing(I) ? body_inertia(shape, rho_body) : I
    return RigidBodyParams3D(m, inertia, rho_body, shape, g; rho_fluid=rho_fluid, buoyancy=buoyancy)
end

state_position(s::RigidBodyState) = s.X
state_position(s::RigidBodyState2D) = s.X
state_position(s::RigidBodyState3D) = s.X

state_velocity(s::RigidBodyState) = s.V
state_velocity(s::RigidBodyState2D) = s.V
state_velocity(s::RigidBodyState3D) = s.V

state_orientation(s::RigidBodyState{N,T}) where {N,T} = identity_orientation(Val(N), T)
state_orientation(s::RigidBodyState2D{T}) where {T} = Orientation2D{T}(s.theta)
state_orientation(s::RigidBodyState3D{T}) where {T} = Orientation3D{T}(s.Q)

state_spin(s::RigidBodyState{N,T}) where {N,T} = SVector{N,T}(ntuple(_ -> zero(T), N))
state_spin(s::RigidBodyState2D{T}) where {T} = s.omega
state_spin(s::RigidBodyState3D{T}) where {T} = s.Omega

function state_copy(s::RigidBodyState{N,T}) where {N,T}
    return RigidBodyState{N,T}(s.X, s.V)
end

function state_copy(s::RigidBodyState2D{T}) where {T}
    return RigidBodyState2D{T}(s.X, s.V, s.theta, s.omega)
end

function state_copy(s::RigidBodyState3D{T}) where {T}
    return RigidBodyState3D{T}(s.X, s.V, s.Q, s.Omega)
end

"""
    external_force(params)

Return gravitational (and optional buoyancy-corrected) external force applied to the rigid body.
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

function external_force(p::RigidBodyParams3D{T}) where {T}
    Vshape = convert(T, body_volume(p.shape))
    coeff = p.buoyancy ? (p.m - p.rho_fluid * Vshape) : p.m
    return coeff * p.g
end

"""
    external_torque(params, state, t)

Return external torque applied to the rigid body at time `t`.
Defaults to zero for built-in parameter/state types.
"""
external_torque(p::RigidBodyParams2D{T}, s::RigidBodyState2D{T}, t::T) where {T} = zero(T)
external_torque(p::RigidBodyParams3D{T}, s::RigidBodyState3D{T}, t::T) where {T} = SVector{3,T}(zero(T), zero(T), zero(T))

"""
    rigid_boundary_velocity(x, X, V, omega_or_Omega)
    rigid_boundary_velocity(x, state)

Rigid-body boundary velocity.

- 2D: `u_b = V + ω * k × (x - X)`
- 3D: `u_b = V + Ω × (x - X)`
"""
@inline function rigid_boundary_velocity(
    x::SVector{2,T},
    X::SVector{2,T},
    V::SVector{2,T},
    omega::T,
) where {T}
    r = x - X
    return SVector{2,T}(V[1] - omega * r[2], V[2] + omega * r[1])
end

@inline function rigid_boundary_velocity(
    x::SVector{3,T},
    X::SVector{3,T},
    V::SVector{3,T},
    Omega::SVector{3,T},
) where {T}
    return V + cross(Omega, x - X)
end

@inline rigid_boundary_velocity(x::SVector{N,T}, s::RigidBodyState{N,T}) where {N,T} = s.V
@inline rigid_boundary_velocity(x::SVector{2,T}, s::RigidBodyState2D{T}) where {T} = rigid_boundary_velocity(x, s.X, s.V, s.omega)
@inline rigid_boundary_velocity(x::SVector{3,T}, s::RigidBodyState3D{T}) where {T} = rigid_boundary_velocity(x, s.X, s.V, s.Omega)

"""
    rigid_velocity_2d(x, X, V, omega)
    rigid_velocity(x, state2d)

2D convenience wrappers for `rigid_boundary_velocity`.
"""
@inline rigid_velocity_2d(x::SVector{2,T}, X::SVector{2,T}, V::SVector{2,T}, omega::T) where {T} = rigid_boundary_velocity(x, X, V, omega)

"""
    rigid_velocity(x, state2d)

2D convenience wrapper for `rigid_boundary_velocity(x, state2d)`.
"""
@inline rigid_velocity(x::SVector{2,T}, s::RigidBodyState2D{T}) where {T} = rigid_boundary_velocity(x, s)

"""
    rigid_body_levelset(shape, statefun)

Build a moving level-set callback `body(x..., t)` from a rigid shape and a
state callback `statefun(t)`.
"""
function rigid_body_levelset(shape::AbstractRigidShape, statefun)
    return function (args...)
        nargs = length(args)
        nargs >= 2 || throw(ArgumentError("body callback expects spatial coordinates plus time"))
        N = nargs - 1
        tau = args[end]
        s = statefun(tau)
        T = eltype(state_position(s))
        xlab = SVector{N,T}(ntuple(i -> convert(T, args[i]), N))
        X = state_position(s)
        xi = if rotation_affects_geometry(shape)
            rotate_to_body_frame(xlab, X, state_orientation(s))
        else
            xlab - X
        end
        return sdf(shape, xi)
    end
end

"""
    rigid_cut_bc_tuple(statefun, Val(N))

Build per-component cut-boundary Dirichlet BC tuple for rigid-body motion,
compatible with `MovingStokesModelMono(...; bc_cut_u=...)`.
"""
function rigid_cut_bc_tuple(statefun, ::Val{N}) where {N}
    return ntuple(d -> Dirichlet((args...) -> begin
        length(args) == N + 1 || throw(ArgumentError("cut BC callback expects $N spatial coordinates plus time"))
        s = statefun(args[end])
        T = eltype(state_position(s))
        xlab = SVector{N,T}(ntuple(i -> convert(T, args[i]), N))
        return rigid_boundary_velocity(xlab, s)[d]
    end), N)
end

"""
    rigid_cut_bc_tuple_2d(statefun)

2D convenience wrapper for `rigid_cut_bc_tuple(statefun, Val(2))`.
"""
rigid_cut_bc_tuple_2d(statefun) = rigid_cut_bc_tuple(statefun, Val(2))
