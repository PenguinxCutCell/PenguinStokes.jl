using LinearAlgebra

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, PressureOutlet
using PenguinStokes

full_body(args...) = -1.0

μ = 1.0
H = 1.0
L = 6.0
p_in = 1.0
p_out = 0.0
G = (p_in - p_out) / L

# Fully-developed Stokes/Poiseuille profile for the chosen pressure gradient.
u_exact(y) = (G / (2 * μ)) * y * (H - y)

grid = CartesianGrid((0.0, 0.0), (L, H), (193, 65))

bcux = BorderConditions(
    ; left=Dirichlet((x, y) -> u_exact(y)),
    right=PressureOutlet(p_out),
    bottom=Dirichlet(0.0),
    top=Dirichlet(0.0),
)
bcuy = BorderConditions(
    ; left=Dirichlet(0.0),
    right=PressureOutlet(p_out),
    bottom=Dirichlet(0.0),
    top=Dirichlet(0.0),
)

model = StokesModelMono(
    grid,
    full_body,
    μ,
    1.0;
    bc_u=(bcux, bcuy),
    bc_p=nothing,
    bc_cut=Dirichlet(0.0),
    force=(0.0, 0.0),
    gauge=MeanPressureGauge(),
)

sys = solve_steady!(model)
res = norm(sys.A * sys.x - sys.b)

u = sys.x[model.layout.uomega[1]]
v = sys.x[model.layout.uomega[2]]

# Compare against Poiseuille profile on a mid-channel slice (x ≈ L/2),
# away from inlet/outlet boundary layers.
xmid = L / 2
let
    err2 = 0.0
    w = 0.0
    vmax = 0.0
    for i in 1:model.cap_u[1].ntotal
        V = model.cap_u[1].buf.V[i]
        if !(isfinite(V) && V > 0.0)
            continue
        end
        xw = model.cap_u[1].C_ω[i]
        if abs(xw[1] - xmid) <= L / 40
            e = u[i] - u_exact(xw[2])
            err2 += V * e^2
            w += V
        end
        vmax = max(vmax, abs(v[i]))
    end

    u_mid_l2 = sqrt(err2 / max(w, eps()))

    println("Channel Poiseuille with right PressureOutlet")
    println("  ||A*x - b||_2             = ", res)
    println("  mid-channel u L2 error     = ", u_mid_l2)
    println("  max |v|                    = ", vmax)
end
