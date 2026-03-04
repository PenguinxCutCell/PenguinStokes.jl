using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

function two_layer_couette_profile(mu1, mu2, h, H, U)
    τ = U / (h / mu1 + (H - h) / mu2)
    a1 = τ / mu1
    a2 = τ / mu2
    b2 = τ * h * (inv(mu1) - inv(mu2))
    u(y) = y <= h ? (a1 * y) : (a2 * y + b2)
    return u
end

function velocity_errors(model, sys, u_exact)
    u1 = sys.x[model.layout.uomega1[1]]
    u2 = sys.x[model.layout.uomega2[1]]
    v1 = sys.x[model.layout.uomega1[2]]
    v2 = sys.x[model.layout.uomega2[2]]

    e1 = 0.0
    e2 = 0.0
    w1 = 0.0
    w2 = 0.0
    vInf = 0.0

    for i in 1:model.cap_u1[1].ntotal
        V = model.cap_u1[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u1[1].C_ω[i][2]
            e = u1[i] - u_exact(y)
            e1 += V * e^2
            w1 += V
            vInf = max(vInf, abs(v1[i]))
        end
    end

    for i in 1:model.cap_u2[1].ntotal
        V = model.cap_u2[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u2[1].C_ω[i][2]
            e = u2[i] - u_exact(y)
            e2 += V * e^2
            w2 += V
            vInf = max(vInf, abs(v2[i]))
        end
    end

    return (u1L2=sqrt(e1 / w1), u2L2=sqrt(e2 / w2), vInf=vInf)
end

mu1 = 1.0
mu2 = 5.0
h = 0.45
H = 1.0
U = 1.0
ns = (33, 65, 97)

u_exact = two_layer_couette_profile(mu1, mu2, h, H, U)
body(x, y) = y - h

bcx = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Dirichlet(0.0), top=Dirichlet(U),
)
bcy = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

hs = Float64[]
e1 = Float64[]
e2 = Float64[]

for n in ns
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    model = StokesModelTwoPhase(
        grid,
        body,
        mu1,
        mu2;
        bc_u=(bcx, bcy),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=(0.0, 0.0),
    )
    sys = solve_steady!(model)
    m = velocity_errors(model, sys, u_exact)

    push!(hs, 1.0 / (n - 1))
    push!(e1, m.u1L2)
    push!(e2, m.u2L2)

    println("n=$n  u1L2=$(m.u1L2)  u2L2=$(m.u2L2)  vInf=$(m.vInf)  ||Ax-b||=$(norm(sys.A * sys.x - sys.b))")
end

for k in 1:(length(ns) - 1)
    o1 = log(e1[k] / e1[k + 1]) / log(hs[k] / hs[k + 1])
    o2 = log(e2[k] / e2[k + 1]) / log(hs[k] / hs[k + 1])
    println("order $(ns[k])->$(ns[k+1]): phase1=$o1  phase2=$o2")
end
