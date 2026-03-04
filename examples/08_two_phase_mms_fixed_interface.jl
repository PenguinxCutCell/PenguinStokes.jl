using LinearAlgebra
using StaticArrays
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Two-phase fixed-interface manufactured equilibrium:
#   u1 = u2 = 0, p1 = 0, p2 = 1
# Interface traction forcing is fΓ = n1.

grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (65, 65))
xc, yc, r = 0.5, 0.5, 0.2
body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - r

bc0 = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

interface_force(x, y) = begin
    dx = x - xc
    dy = y - yc
    rr = sqrt(dx^2 + dy^2)
    rr == 0.0 ? SVector(0.0, 0.0) : SVector(dx / rr, dy / rr)
end

model = StokesModelTwoPhase(
    grid,
    body,
    1.0,
    10.0;
    bc_u=(bc0, bc0),
    force1=(0.0, 0.0),
    force2=(0.0, 0.0),
    interface_force=interface_force,
)

sys = solve_steady!(model)
r = sys.A * sys.x - sys.b

u1x = sys.x[model.layout.uomega1[1]]
u1y = sys.x[model.layout.uomega1[2]]
u2x = sys.x[model.layout.uomega2[1]]
u2y = sys.x[model.layout.uomega2[2]]

iface = findall(PenguinStokes._pressure_activity(model.cap_p1) .& (model.cap_p1.buf.Γ .> 0.0))
rowsx = model.layout.ugamma[1]
rowsy = model.layout.ugamma[2]

p1 = sys.x[model.layout.pomega1]
p2 = sys.x[model.layout.pomega2]
idxp1 = findall(PenguinStokes._pressure_activity(model.cap_p1))
idxp2 = findall(PenguinStokes._pressure_activity(model.cap_p2))
p1mean = sum(p1[idxp1]) / length(idxp1)
p2mean = sum(p2[idxp2]) / length(idxp2)

println("Two-phase fixed-interface solve")
println("||Ax-b|| = ", norm(r))
println("max|u1| = ", max(maximum(abs, u1x), maximum(abs, u1y)))
println("max|u2| = ", max(maximum(abs, u2x), maximum(abs, u2y)))
if !isempty(iface)
    println("max traction-row residual = ", max(maximum(abs, r[rowsx][iface]), maximum(abs, r[rowsy][iface])))
end
println("mean pressure jump p2-p1 = ", p2mean - p1mean)
