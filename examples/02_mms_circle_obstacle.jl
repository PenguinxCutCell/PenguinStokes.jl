using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Circle obstacle cut-cell showcase with no-slip cut boundary.
grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (41, 41))
xc, yc, r = 0.0, 0.0, 0.35
body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r

bc = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

model = StokesModelMono(
    grid,
    body,
    1.0,
    1.0;
    bc_u=(bc, bc),
    bc_cut=Dirichlet(0.0),
    force=(0.0, 0.0),
)

sys = solve_steady!(model)
println("Solved cut-cell Stokes system with ", length(sys.x), " unknowns")
println("Residual norm = ", norm(sys.A * sys.x - sys.b))
