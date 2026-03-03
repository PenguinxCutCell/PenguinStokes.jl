using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

function main()
    # MMS setup with zero exact pressure and divergence-free velocity from a streamfunction.
    mu = 1.0

    h(x) = x^2 * (1 - x)^2
    hp(x) = 2x - 6x^2 + 4x^3
    hpp(x) = 2 - 12x + 12x^2
    h3(x) = -12 + 24x
    g(y) = y^2 * (1 - y)^2
    gp(y) = 2y - 6y^2 + 4y^3
    gpp(y) = 2 - 12y + 12y^2
    g3(y) = -12 + 24y

    u_exact(x, y) = h(x) * gp(y)
    v_exact(x, y) = -hp(x) * g(y)
    lap_u(x, y) = hpp(x) * gp(y) + h(x) * g3(y)
    lap_v(x, y) = -(h3(x) * g(y) + hp(x) * gpp(y))
    fx(x, y) = -mu * lap_u(x, y)
    fy(x, y) = -mu * lap_v(x, y)

    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
    body(x...) = -1.0
    bc = BorderConditions(; left=Dirichlet(0.0), right=Dirichlet(0.0), bottom=Dirichlet(0.0), top=Dirichlet(0.0))
    model = StokesModelMono(grid, body, mu, 1.0; bc_u=(bc, bc), bc_cut=Dirichlet(0.0), force=(fx, fy))
    sys = solve_steady!(model)

    u = sys.x[model.layout.uomega[1]]
    v = sys.x[model.layout.uomega[2]]
    p = sys.x[model.layout.pomega]

    li_u = LinearIndices(model.cap_u[1].nnodes)
    err_u = 0.0
    cnt_u = 0
    for I in CartesianIndices(model.cap_u[1].nnodes)
        if I[1] < model.cap_u[1].nnodes[1] && I[2] < model.cap_u[1].nnodes[2]
            i = li_u[I]
            if model.cap_u[1].buf.V[i] > 0.0
                x = model.cap_u[1].C_ω[i]
                err_u += (u[i] - u_exact(x[1], x[2]))^2
                cnt_u += 1
            end
        end
    end

    li_v = LinearIndices(model.cap_u[2].nnodes)
    err_v = 0.0
    cnt_v = 0
    for I in CartesianIndices(model.cap_u[2].nnodes)
        if I[1] < model.cap_u[2].nnodes[1] && I[2] < model.cap_u[2].nnodes[2]
            i = li_v[I]
            if model.cap_u[2].buf.V[i] > 0.0
                x = model.cap_u[2].C_ω[i]
                err_v += (v[i] - v_exact(x[1], x[2]))^2
                cnt_v += 1
            end
        end
    end

    println("u L2 error = ", sqrt(err_u / cnt_u))
    println("v L2 error = ", sqrt(err_v / cnt_v))
    println("pressure min/max = ", minimum(p), " / ", maximum(p))
    println("||Ax-b||_2 = ", norm(sys.A * sys.x - sys.b))
end

main()
