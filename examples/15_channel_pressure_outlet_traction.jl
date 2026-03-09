using LinearAlgebra

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, PressureOutlet
using PenguinStokes

full_body(args...) = -1.0

grid = CartesianGrid((0.0, 0.0), (4.0, 1.0), (97, 49))

uin(x, y) = 4.0 * y * (1.0 - y)

bcux = BorderConditions(
    ; left=Dirichlet((x, y) -> uin(x, y)),
    right=PressureOutlet(0.0),
    bottom=Dirichlet(0.0),
    top=Dirichlet(0.0),
)
bcuy = BorderConditions(
    ; left=Dirichlet(0.0),
    right=PressureOutlet(0.0),
    bottom=Dirichlet(0.0),
    top=Dirichlet(0.0),
)

model = StokesModelMono(
    grid,
    full_body,
    1.0,
    1.0;
    bc_u=(bcux, bcuy),
    bc_p=nothing,
    bc_cut=Dirichlet(0.0),
    force=(0.0, 0.0),
    gauge=MeanPressureGauge(),
)

sys = solve_steady!(model)
res = norm(sys.A * sys.x - sys.b)

layout = model.layout
blocks = PenguinStokes._stokes_blocks(model)
uo = ntuple(d -> sys.x[layout.uomega[d]], 2)
ug = ntuple(d -> sys.x[layout.ugamma[d]], 2)
div = blocks.div_omega[1] * uo[1] + blocks.div_gamma[1] * ug[1] +
      blocks.div_omega[2] * uo[2] + blocks.div_gamma[2] * ug[2]
div_l2 = norm(div) / sqrt(length(div))

println("Steady Stokes channel with right PressureOutlet(0)")
println("  ||A*x - b||_2 = ", res)
println("  divergence L2  = ", div_l2)
println("  ux range       = (", minimum(uo[1]), ", ", maximum(uo[1]), ")")
