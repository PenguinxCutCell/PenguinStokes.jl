using LinearAlgebra: norm
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs
using PenguinStokes

function box_noslip_3d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
end

function run_drag_for_speed(Uz::Float64; n::Int=17, dt::Float64=0.03, nsteps::Int=8)
    mu = 1.0
    rho = 1.0
    R = 0.12
    center = SVector(0.5, 0.5, 0.5)

    grid = CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (n, n, n))
    bc = box_noslip_3d()
    shape = Sphere(R)

    body(x, y, z, t) = shape.R - sqrt((x - center[1])^2 + (y - center[2])^2 + (z - center[3])^2)

    model = MovingStokesModelMono(
        grid,
        body,
        mu,
        rho;
        bc_u=(bc, bc, bc),
        force=(0.0, 0.0, 0.0),
        bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0), Dirichlet(Uz)),
    )

    xprev = zeros(Float64, last(model.layout.pomega))
    sys = nothing
    t = 0.0
    for _ in 1:nsteps
        sys = solve_unsteady_moving!(model, xprev; t=t, dt=dt, scheme=:CN)
        xprev .= sys.x
        t += dt
    end

    sm = endtime_static_model(model)
    q = integrated_embedded_force(sm, sys; x0=Tuple(center))
    Fz = abs(q.force[3])
    Fref = 6 * pi * mu * R * abs(Uz)
    return (Fz=Fz, Fref=Fref, ratio=(Fref > 0 ? Fz / Fref : NaN), q=q, residual=norm(sys.A * sys.x - sys.b))
end

function main()
    println("3D rigid sphere drag benchmark")
    println("Reference: F = 6*pi*mu*R*U (unbounded creeping flow)")
    println("Columns: Uz, |Fz|_num, F_ref, ratio, residual")

    for Uz in (0.1, 0.2, 0.4)
        out = run_drag_for_speed(Uz)
        println(Uz, ", ", out.Fz, ", ", out.Fref, ", ", out.ratio, ", ", out.residual)
    end

    println("Note: ratios include finite-box confinement; refine/enlarge domain for closer agreement with unbounded Stokes law.")
end

main()
