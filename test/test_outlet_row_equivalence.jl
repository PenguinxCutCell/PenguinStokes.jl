using Test
using SparseArrays
using StaticArrays: SVector

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, DoNothing, PressureOutlet, Traction
using PenguinSolverCore: LinearSystem
using PenguinStokes

full_body_outlet_eq(args...) = -1.0

function _assemble_outlet_eq(model)
    nsys = last(model.layout.pomega)
    sys = LinearSystem(spzeros(Float64, nsys, nsys), zeros(Float64, nsys))
    assemble_steady!(sys, model, 0.0)
    return sys
end

function _right_boundary_rows(model, comp::Int)
    cap = model.cap_u[comp]
    li = LinearIndices(cap.nnodes)
    iwall = cap.nnodes[1] - 1
    local_idx = Int[]
    rows = Int[]
    for j in 1:(cap.nnodes[2] - 1)
        I = CartesianIndex(iwall, j)
        i = li[I]
        V = cap.buf.V[i]
        Aface = cap.buf.A[1][i]
        if isfinite(V) && V > 0.0 && isfinite(Aface) && Aface > 0.0
            push!(local_idx, i)
            push!(rows, model.layout.uomega[comp][i])
        end
    end
    return local_idx, rows
end

@testset "Outlet row equivalence special cases (2D)" begin
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (21, 19))
    pout = 2.3

    bcx_A = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction((x, y) -> SVector(-pout, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_A = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction((x, y) -> SVector(-pout, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcx_B = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(pout),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_B = BorderConditions(
        ; left=Dirichlet(0.0), right=PressureOutlet(pout),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcx_C = BorderConditions(
        ; left=Dirichlet(0.0), right=DoNothing(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_C = BorderConditions(
        ; left=Dirichlet(0.0), right=DoNothing(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcx_D = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction((x, y) -> SVector(0.0, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
    bcy_D = BorderConditions(
        ; left=Dirichlet(0.0), right=Traction((x, y) -> SVector(0.0, 0.0)),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    mA = StokesModelMono(grid, full_body_outlet_eq, 1.0, 1.0; bc_u=(bcx_A, bcy_A), force=(0.0, 0.0))
    mB = StokesModelMono(grid, full_body_outlet_eq, 1.0, 1.0; bc_u=(bcx_B, bcy_B), force=(0.0, 0.0))
    mC = StokesModelMono(grid, full_body_outlet_eq, 1.0, 1.0; bc_u=(bcx_C, bcy_C), force=(0.0, 0.0))
    mD = StokesModelMono(grid, full_body_outlet_eq, 1.0, 1.0; bc_u=(bcx_D, bcy_D), force=(0.0, 0.0))

    sA = _assemble_outlet_eq(mA)
    sB = _assemble_outlet_eq(mB)
    sC = _assemble_outlet_eq(mC)
    sD = _assemble_outlet_eq(mD)

    for comp in 1:2
        idxA, rowsA = _right_boundary_rows(mA, comp)
        idxB, rowsB = _right_boundary_rows(mB, comp)
        idxC, rowsC = _right_boundary_rows(mC, comp)
        idxD, rowsD = _right_boundary_rows(mD, comp)
        @test idxA == idxB
        @test idxC == idxD

        for (rA, rB) in zip(rowsA, rowsB)
            @test Array(sA.A[rA, :]) ≈ Array(sB.A[rB, :]) atol=1e-13 rtol=0.0
            @test isapprox(sA.b[rA], sB.b[rB]; atol=1e-13, rtol=0.0)
        end
        for (rC, rD) in zip(rowsC, rowsD)
            @test Array(sC.A[rC, :]) ≈ Array(sD.A[rD, :]) atol=1e-13 rtol=0.0
            @test isapprox(sC.b[rC], sD.b[rD]; atol=1e-13, rtol=0.0)
        end
    end
end
