using Test
using LinearAlgebra
using SparseArrays
using StaticArrays: SVector

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, DoNothing, PressureOutlet, Traction
using PenguinSolverCore: LinearSystem
using PenguinStokes

full_body_dbg(args...) = -1.0

function _row_dense(A::SparseMatrixCSC{T,Int}, row::Int) where {T}
    return Array(A[row, :])
end

function _assemble(model)
    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)
    return sys
end

@testset "Traction row algebra debug (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (17, 17))
    pout = 1.7

    bcx_po = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(pout),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_po = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(pout),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcx_tr = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction(SVector(-pout, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_tr = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction(SVector(-pout, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcx_dn = BorderConditions(
        ; left=Dirichlet(0.0), right=DoNothing(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_dn = BorderConditions(
        ; left=Dirichlet(0.0), right=DoNothing(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcx_t0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction(SVector(0.0, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_t0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction(SVector(0.0, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    m_po = StokesModelMono(grid, full_body_dbg, 1.0, 1.0; bc_u=(bcx_po, bcy_po), force=(0.0, 0.0))
    m_tr = StokesModelMono(grid, full_body_dbg, 1.0, 1.0; bc_u=(bcx_tr, bcy_tr), force=(0.0, 0.0))
    m_dn = StokesModelMono(grid, full_body_dbg, 1.0, 1.0; bc_u=(bcx_dn, bcy_dn), force=(0.0, 0.0))
    m_t0 = StokesModelMono(grid, full_body_dbg, 1.0, 1.0; bc_u=(bcx_t0, bcy_t0), force=(0.0, 0.0))

    s_po = _assemble(m_po)
    s_tr = _assemble(m_tr)
    s_dn = _assemble(m_dn)
    s_t0 = _assemble(m_t0)

    li = LinearIndices(m_po.cap_u[1].nnodes)
    I = CartesianIndex(m_po.cap_u[1].nnodes[1] - 1, 4)
    i = li[I]

    rux_po = m_po.layout.uomega[1][i]
    ruy_po = m_po.layout.uomega[2][i]
    rux_tr = m_tr.layout.uomega[1][i]
    ruy_tr = m_tr.layout.uomega[2][i]
    rux_dn = m_dn.layout.uomega[1][i]
    ruy_dn = m_dn.layout.uomega[2][i]
    rux_t0 = m_t0.layout.uomega[1][i]
    ruy_t0 = m_t0.layout.uomega[2][i]

    # PressureOutlet(pout) == Traction((-pout, 0)).
    @test _row_dense(s_po.A, rux_po) ≈ _row_dense(s_tr.A, rux_tr) atol=1e-13 rtol=0.0
    @test _row_dense(s_po.A, ruy_po) ≈ _row_dense(s_tr.A, ruy_tr) atol=1e-13 rtol=0.0
    @test isapprox(s_po.b[rux_po], s_tr.b[rux_tr]; atol=1e-13, rtol=0.0)
    @test isapprox(s_po.b[ruy_po], s_tr.b[ruy_tr]; atol=1e-13, rtol=0.0)

    # DoNothing == Traction(0).
    @test _row_dense(s_dn.A, rux_dn) ≈ _row_dense(s_t0.A, rux_t0) atol=1e-13 rtol=0.0
    @test _row_dense(s_dn.A, ruy_dn) ≈ _row_dense(s_t0.A, ruy_t0) atol=1e-13 rtol=0.0
    @test isapprox(s_dn.b[rux_dn], s_t0.b[rux_t0]; atol=1e-13, rtol=0.0)
    @test isapprox(s_dn.b[ruy_dn], s_t0.b[ruy_t0]; atol=1e-13, rtol=0.0)

    # Right-boundary y-momentum traction row should not couple pressure directly.
    prow = _row_dense(s_po.A, ruy_po)[m_po.layout.pomega]
    @test maximum(abs, prow) == 0.0
    # And should include cross-coupling through u_x (∂_t u_n term).
    uxblk = _row_dense(s_po.A, ruy_po)[m_po.layout.uomega[1]]
    @test maximum(abs, uxblk) > 0.0
end
