using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

const MU = 1.0

# Divergence-free polynomial MMS with homogeneous Dirichlet velocity and p_exact = 0.
hfun(x) = x^2 * (1 - x)^2
hp(x) = 2x - 6x^2 + 4x^3
hpp(x) = 2 - 12x + 12x^2
h3(x) = -12 + 24x
g(y) = y^2 * (1 - y)^2
gp(y) = 2y - 6y^2 + 4y^3
gpp(y) = 2 - 12y + 12y^2
g3(y) = -12 + 24y

u_exact(x, y) = hfun(x) * gp(y)
v_exact(x, y) = -hp(x) * g(y)
lap_u(x, y) = hpp(x) * gp(y) + hfun(x) * g3(y)
lap_v(x, y) = -(h3(x) * g(y) + hp(x) * gpp(y))
fx(x, y) = -MU * lap_u(x, y)
fy(x, y) = -MU * lap_v(x, y)

body(x...) = -1.0

bc = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

function velocity_l2_errors(model, sys)
    u = sys.x[model.layout.uomega[1]]
    v = sys.x[model.layout.uomega[2]]

    li_u = LinearIndices(model.cap_u[1].nnodes)
    li_v = LinearIndices(model.cap_u[2].nnodes)

    eu2 = 0.0
    ev2 = 0.0
    wu = 0.0
    wv = 0.0

    for I in CartesianIndices(model.cap_u[1].nnodes)
        i = li_u[I]
        if I[1] < model.cap_u[1].nnodes[1] && I[2] < model.cap_u[1].nnodes[2]
            V = model.cap_u[1].buf.V[i]
            if V > 0.0
                x = model.cap_u[1].C_ω[i]
                eu2 += V * (u[i] - u_exact(x[1], x[2]))^2
                wu += V
            end
        end
    end

    for I in CartesianIndices(model.cap_u[2].nnodes)
        i = li_v[I]
        if I[1] < model.cap_u[2].nnodes[1] && I[2] < model.cap_u[2].nnodes[2]
            V = model.cap_u[2].buf.V[i]
            if V > 0.0
                x = model.cap_u[2].C_ω[i]
                ev2 += V * (v[i] - v_exact(x[1], x[2]))^2
                wv += V
            end
        end
    end

    return sqrt(eu2 / wu), sqrt(ev2 / wv)
end

function main()
    ns = (17, 33, 65)
    hs = Float64[]
    uerrs = Float64[]
    verrs = Float64[]

    for n in ns
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = StokesModelMono(
            grid,
            body,
            MU,
            1.0;
            bc_u=(bc, bc),
            bc_cut=Dirichlet(0.0),
            force=(fx, fy),
        )
        sys = solve_steady!(model)
        uL2, vL2 = velocity_l2_errors(model, sys)
        h = 1.0 / (n - 1)
        push!(hs, h)
        push!(uerrs, uL2)
        push!(verrs, vL2)
        println("n=$n  h=$h  uL2=$uL2  vL2=$vL2  ||Ax-b||=$(norm(sys.A * sys.x - sys.b))")
    end

    for k in 1:(length(ns) - 1)
        ou = log(uerrs[k] / uerrs[k + 1]) / log(hs[k] / hs[k + 1])
        ov = log(verrs[k] / verrs[k + 1]) / log(hs[k] / hs[k + 1])
        println("order $(ns[k])->$(ns[k+1]): u=$ou  v=$ov")
    end
end

main()
