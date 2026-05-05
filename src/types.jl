"""
    AbstractPressureGauge

Abstract supertype for pressure nullspace constraints in Stokes systems.
Concrete options are `PinPressureGauge` and `MeanPressureGauge`.
"""
abstract type AbstractPressureGauge end

"""
    PinPressureGauge(; index=nothing)

Pressure gauge that replaces one pressure equation by a pointwise pinning
constraint `p[index] = 0` (or the first active pressure DOF when `index=nothing`).
"""
struct PinPressureGauge <: AbstractPressureGauge
    index::Union{Nothing,Int}
end
PinPressureGauge(; index::Union{Nothing,Int}=nothing) = PinPressureGauge(index)

"""
    MeanPressureGauge()

Pressure gauge that replaces one pressure equation by a zero-mean pressure
constraint over active pressure cells.
"""
struct MeanPressureGauge <: AbstractPressureGauge end

"""
    StokesLayout{N}

Unknown ordering for monophasic Stokes:
`[uomega_1; ugamma_1; ...; uomega_N; ugamma_N; pomega]`.
"""
struct StokesLayout{N}
    nt::Int
    uomega::NTuple{N,UnitRange{Int}}
    ugamma::NTuple{N,UnitRange{Int}}
    pomega::UnitRange{Int}
end

"""
    StokesLayoutTwoPhase{N}

Unknown ordering for fixed-interface two-phase Stokes:
`[uomega1_1; ...; uomega1_N; uomega2_1; ...; uomega2_N; ugamma1_1; ...; ugamma1_N; ugamma2_1; ...; ugamma2_N; pomega1; pomega2]`.
"""
struct StokesLayoutTwoPhase{N}
    nt::Int
    uomega1::NTuple{N,UnitRange{Int}}
    uomega2::NTuple{N,UnitRange{Int}}
    ugamma1::NTuple{N,UnitRange{Int}}
    ugamma2::NTuple{N,UnitRange{Int}}
    pomega1::UnitRange{Int}
    pomega2::UnitRange{Int}
end

@inline function Base.getproperty(layout::StokesLayoutTwoPhase, s::Symbol)
    if s === :ugamma
        return getfield(layout, :ugamma1)
    end
    return getfield(layout, s)
end

function StokesLayout(nt::Int, ::Val{N}) where {N}
    nt > 0 || throw(ArgumentError("nt must be positive"))
    start = 1
    uomega = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    ugamma = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    pomega = start:(start + nt - 1)
    return StokesLayout{N}(nt, uomega, ugamma, pomega)
end

nunknowns(layout::StokesLayout) = last(layout.pomega)
nunknowns(layout::StokesLayoutTwoPhase) = last(layout.pomega2)

function StokesLayoutTwoPhase(nt::Int, ::Val{N}) where {N}
    nt > 0 || throw(ArgumentError("nt must be positive"))
    start = 1
    uomega1 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    uomega2 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    ugamma1 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    ugamma2 = ntuple(d -> begin
        r = start:(start + nt - 1)
        start += nt
        r
    end, N)
    pomega1 = start:(start + nt - 1)
    start += nt
    pomega2 = start:(start + nt - 1)
    return StokesLayoutTwoPhase{N}(nt, uomega1, uomega2, ugamma1, ugamma2, pomega1, pomega2)
end

"""
    StokesModelMono{N,T}

Monophasic cut-cell Stokes model on a staggered MAC grid.

Use constructors
`StokesModelMono(gridp, body, mu, rho; ...)` or
`StokesModelMono(cap_p, op_p, cap_u, op_u, mu, rho; ...)`.
"""
mutable struct StokesModelMono{N,T,FT,BT}
    gridp::CartesianGrid{N,T}
    gridu::NTuple{N,CartesianGrid{N,T}}
    cap_p::AssembledCapacity{N,T}
    cap_u::NTuple{N,AssembledCapacity{N,T}}
    op_p::DiffusionOps{N,T}
    op_u::NTuple{N,DiffusionOps{N,T}}
    mu::T
    rho::T
    force::FT
    bc_u::NTuple{N,BorderConditions}
    bc_p::Union{Nothing,BorderConditions}
    bc_cut::NTuple{N,AbstractBoundary}
    gauge::AbstractPressureGauge
    strong_wall_bc::Bool
    layout::StokesLayout{N}
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    body::BT
end

"""
    StokesModelTwoPhase{N,T}

Fixed-interface two-phase Stokes model with phase-wise interface velocity traces
`ugamma1`, `ugamma2` and one pressure block per phase.

Use constructors
`StokesModelTwoPhase(gridp, body, mu1, mu2; ...)` or
`StokesModelTwoPhase(cap_p1, op_p1, ..., cap_p2, op_p2, ...; ...)`.
"""
mutable struct StokesModelTwoPhase{N,T,FT1,FT2,IFT,IJT,BT}
    gridp::CartesianGrid{N,T}
    gridu::NTuple{N,CartesianGrid{N,T}}

    cap_p1::AssembledCapacity{N,T}
    cap_u1::NTuple{N,AssembledCapacity{N,T}}
    op_p1::DiffusionOps{N,T}
    op_u1::NTuple{N,DiffusionOps{N,T}}

    cap_p2::AssembledCapacity{N,T}
    cap_u2::NTuple{N,AssembledCapacity{N,T}}
    op_p2::DiffusionOps{N,T}
    op_u2::NTuple{N,DiffusionOps{N,T}}

    mu1::T
    mu2::T
    rho1::T
    rho2::T
    force1::FT1
    force2::FT2
    interface_force::IFT
    interface_jump::IJT
    bc_interface::Union{Nothing,NTuple{N,InterfaceConditions}}
    bc_u::NTuple{N,BorderConditions}
    bc_p::Union{Nothing,BorderConditions}
    gauge::AbstractPressureGauge
    strong_wall_bc::Bool
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    body::BT
    layout::StokesLayoutTwoPhase{N}
end

"""
    MovingStokesModelMono{N,T}

Unsteady monophasic moving-boundary Stokes model with prescribed embedded
boundary velocity through `bc_cut_u`.
"""
mutable struct MovingStokesModelMono{N,T,FT,BT}
    gridp::CartesianGrid{N,T}
    gridu::NTuple{N,CartesianGrid{N,T}}
    body::BT
    mu::T
    rho::T
    force::FT
    bc_u::NTuple{N,BorderConditions}
    bc_p::Union{Nothing,BorderConditions}
    bc_cut_u::NTuple{N,AbstractBoundary}
    gauge::AbstractPressureGauge
    strong_wall_bc::Bool
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    layout::StokesLayout{N}
    cap_p_slab::Union{Nothing,AssembledCapacity{N,T}}
    op_p_slab::Union{Nothing,DiffusionOps{N,T}}
    cap_p_end::Union{Nothing,AssembledCapacity{N,T}}
    op_p_end::Union{Nothing,DiffusionOps{N,T}}
    cap_u_slab::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    op_u_slab::Union{Nothing,NTuple{N,DiffusionOps{N,T}}}
    cap_u_end::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    Vun::Union{Nothing,NTuple{N,Vector{T}}}
    Vun1::Union{Nothing,NTuple{N,Vector{T}}}
end

"""
    MovingStokesModelTwoPhase{N,T}

Unsteady moving-interface two-phase Stokes model with fixed jump/traction
interface conditions and prescribed interface geometry `body(x..., t)`.
"""
mutable struct MovingStokesModelTwoPhase{N,T,FT1,FT2,IFT,IJT,BT}
    gridp::CartesianGrid{N,T}
    gridu::NTuple{N,CartesianGrid{N,T}}
    body::BT
    mu1::T
    mu2::T
    rho1::T
    rho2::T
    force1::FT1
    force2::FT2
    interface_force::IFT
    interface_jump::IJT
    bc_interface::Union{Nothing,NTuple{N,InterfaceConditions}}
    bc_u::NTuple{N,BorderConditions}
    bc_p::Union{Nothing,BorderConditions}
    gauge::AbstractPressureGauge
    strong_wall_bc::Bool
    periodic::NTuple{N,Bool}
    geom_method::Symbol
    check_interface::Bool
    layout::StokesLayoutTwoPhase{N}

    cap_p1_slab::Union{Nothing,AssembledCapacity{N,T}}
    op_p1_slab::Union{Nothing,DiffusionOps{N,T}}
    cap_p1_end::Union{Nothing,AssembledCapacity{N,T}}
    op_p1_end::Union{Nothing,DiffusionOps{N,T}}
    cap_u1_slab::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    op_u1_slab::Union{Nothing,NTuple{N,DiffusionOps{N,T}}}
    cap_u1_end::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    op_u1_end::Union{Nothing,NTuple{N,DiffusionOps{N,T}}}
    Vu1n::Union{Nothing,NTuple{N,Vector{T}}}
    Vu1n1::Union{Nothing,NTuple{N,Vector{T}}}

    cap_p2_slab::Union{Nothing,AssembledCapacity{N,T}}
    op_p2_slab::Union{Nothing,DiffusionOps{N,T}}
    cap_p2_end::Union{Nothing,AssembledCapacity{N,T}}
    op_p2_end::Union{Nothing,DiffusionOps{N,T}}
    cap_u2_slab::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    op_u2_slab::Union{Nothing,NTuple{N,DiffusionOps{N,T}}}
    cap_u2_end::Union{Nothing,NTuple{N,AssembledCapacity{N,T}}}
    op_u2_end::Union{Nothing,NTuple{N,DiffusionOps{N,T}}}
    Vu2n::Union{Nothing,NTuple{N,Vector{T}}}
    Vu2n1::Union{Nothing,NTuple{N,Vector{T}}}
end
