using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Qualitative flow-past-circle setup with Dirichlet walls (simple and robust).
grid = CartesianGrid((-2.0, -1.0), (6.0, 1.0), (97, 49))
xc, yc, r = 0.0, 0.0, 0.2
body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r

bcx = BorderConditions(
    ; left=Dirichlet(1.0), right=Dirichlet(1.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)
bcy = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

model = StokesModelMono(
    grid,
    body,
    1.0,
    1.0;
    bc_u=(bcx, bcy),
    bc_cut=Dirichlet(0.0),
    force=(0.0, 0.0),
)

sys = solve_steady!(model)
println("Solved steady flow-past-circle prototype")
println("Residual norm = ", norm(sys.A * sys.x - sys.b))
