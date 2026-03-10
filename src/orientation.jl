abstract type AbstractOrientation end

"""
    Orientation2D(theta)

Planar orientation represented by a scalar angle `theta` (radians).
"""
struct Orientation2D{T} <: AbstractOrientation
    theta::T
end

"""
    Orientation3D(Q)

3D orientation represented by a rotation matrix `Q` (body -> lab frame).
"""
struct Orientation3D{T} <: AbstractOrientation
    Q::SMatrix{3,3,T,9}
end

@inline function _I2(::Type{T}) where {T}
    return @SMatrix [one(T) zero(T); zero(T) one(T)]
end

@inline function _I3(::Type{T}) where {T}
    return @SMatrix [
        one(T) zero(T) zero(T)
        zero(T) one(T) zero(T)
        zero(T) zero(T) one(T)
    ]
end

identity_orientation(::Val{2}, ::Type{T}) where {T} = Orientation2D{T}(zero(T))
identity_orientation(::Val{3}, ::Type{T}) where {T} = Orientation3D{T}(_I3(T))

@inline function rotation_matrix(ori::Orientation2D{T}) where {T}
    c = cos(ori.theta)
    s = sin(ori.theta)
    return @SMatrix [c -s; s c]
end

@inline rotation_matrix(ori::Orientation3D{T}) where {T} = ori.Q

@inline function rotate_to_body_frame(x::SVector{2,T}, X::SVector{2,T}, ori::Orientation2D{T}) where {T}
    return transpose(rotation_matrix(ori)) * (x - X)
end

@inline function rotate_to_body_frame(x::SVector{3,T}, X::SVector{3,T}, ori::Orientation3D{T}) where {T}
    return transpose(rotation_matrix(ori)) * (x - X)
end

function orthonormalize_rotation(Q::SMatrix{3,3,T,9}) where {T}
    F = qr(Matrix(Q))
    Qm = Matrix(F.Q)
    if det(Qm) < zero(T)
        Qm[:, 3] .*= -one(T)
    end
    return SMatrix{3,3,T,9}(Qm)
end

@inline function _skew(v::SVector{3,T}) where {T}
    return @SMatrix [
        zero(T) -v[3] v[2]
        v[3] zero(T) -v[1]
        -v[2] v[1] zero(T)
    ]
end

function _exp_so3(wdt::SVector{3,T}) where {T}
    theta = norm(wdt)
    K = _skew(wdt)
    I3 = _I3(T)
    if theta <= sqrt(eps(T))
        return I3 + K
    end
    theta2 = theta * theta
    s = sin(theta) / theta
    c = (one(T) - cos(theta)) / theta2
    return I3 + s * K + c * (K * K)
end

@inline function advance_orientation(ori::Orientation2D{T}, omega::T, dt::T) where {T}
    return Orientation2D{T}(ori.theta + dt * omega)
end

function advance_orientation(ori::Orientation3D{T}, Omega::SVector{3,T}, dt::T) where {T}
    R = _exp_so3(Omega * dt)
    Qnext = orthonormalize_rotation(R * ori.Q)
    return Orientation3D{T}(Qnext)
end
