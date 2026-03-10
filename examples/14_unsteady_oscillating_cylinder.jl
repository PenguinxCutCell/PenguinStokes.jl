using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

function box_noslip_2d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )
end

function main()
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (61, 61))

    μ = 1.0
    ρ = 1.0

    R = 0.12
    xc0, yc0 = 0.5, 0.5
    amp = 0.06
    ω = 2 * pi

    xc(t) = xc0 + amp * sin(ω * t)
    uw(t) = amp * ω * cos(ω * t)

    body(x, y, t) = R - sqrt((x - xc(t))^2 + (y - yc0)^2)

    bc = box_noslip_2d()
    model = MovingStokesModelMono(
        grid,
        body,
        μ,
        ρ;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(
            Dirichlet((x, y, t) -> uw(t)),
            Dirichlet(0.0),
        ),
    )

    nsys = last(model.layout.pomega)
    xprev = zeros(Float64, nsys)

    dt = 0.01
    nsteps = 30
    t = 0.0

    println("Unsteady moving-cylinder Stokes (prescribed oscillation)")
    println("grid=$(grid.n), dt=$dt, steps=$nsteps")
    println("Columns: step, t, uw, Fx, Fy, torque, ||Ax-b||")

    for step in 1:nsteps
        sys = solve_unsteady_moving!(model, xprev; t=t, dt=dt, scheme=:CN)
        tnext = t + dt
        sm = endtime_static_model(model)
        q = integrated_embedded_force(sm, sys; pressure_reconstruction=:linear, x0=(xc(tnext), yc0))
        res = norm(sys.A * sys.x - sys.b)
        println(step, ", ", tnext, ", ", uw(tnext), ", ", q.force[1], ", ", q.force[2], ", ", q.torque, ", ", res)
        xprev .= sys.x
        t = tnext
    end
end

main()
