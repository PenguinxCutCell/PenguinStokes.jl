using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

periodic_2d_bc() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

function slab_force(U::SVector{2,Float64}; n=41, dt=0.02, R=0.12)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    bc = periodic_2d_bc()
    X0 = SVector(0.5, 0.5)
    shape = Circle(R)
    body(x, y, t) = shape.R - hypot(x - X0[1], y - X0[2])

    model = MovingStokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet(U[1]), Dirichlet(U[2])),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys = solve_unsteady_moving!(model, xprev; t=0.0, dt=dt, scheme=:CN)
    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=Tuple(X0))
    return SVector{2,Float64}(Tuple(q.force))
end

function main()
    U = SVector(0.0, 1.0)
    F1 = slab_force(U)
    F2 = slab_force(2 .* U)
    Fm = slab_force(-U)

    force_sign = F1[2] <= 0 ? 1.0 : -1.0
    zeta = -(force_sign * F1[2])

    println("FSI drag calibration (circle, periodic box)")
    println("F(U)   = ", Tuple(F1))
    println("F(2U)  = ", Tuple(F2), "   ratio = ", Tuple(F2 ./ F1))
    println("F(-U)  = ", Tuple(Fm), "   odd check = ", Tuple(Fm .+ F1))
    println("force_sign = ", force_sign)
    println("zeta_num (from Fy at Uy=1) = ", zeta)
end

main()
