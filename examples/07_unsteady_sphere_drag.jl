using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

const MU = 1.0
const RHO = 1.0
const BODY_FORCE_X = 1.0
const RADIUS = 0.15
const CENTER = (0.5, 0.5, 0.5)

# Negative outside sphere => fluid outside obstacle in current cut-cell convention.
sphere_levelset(x, y, z) = RADIUS - sqrt((x - CENTER[1])^2 + (y - CENTER[2])^2 + (z - CENTER[3])^2)

periodic_3d_bc() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
    backward=Periodic(), forward=Periodic(),
)

function bulk_mean_ux(model, x)
    u = x[model.layout.uomega[1]]
    cap = model.cap_u[1]
    acc = 0.0
    vol = 0.0
    for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if isfinite(V) && V > 0.0
            acc += V * u[i]
            vol += V
        end
    end
    return acc / vol
end

function main()
    grid = CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (15, 15, 15))

    bc = periodic_3d_bc()
    model = StokesModelMono(
        grid,
        sphere_levelset,
        MU,
        RHO;
        bc_u=(bc, bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(BODY_FORCE_X, 0.0, 0.0),
    )

    nsys = last(model.layout.pomega)
    xprev = zeros(Float64, nsys)

    dt = 0.05
    nsteps = 8
    t = 0.0

    println("Unsteady 3D Stokes around a sphere (periodic box, body-force driven)")
    println("grid=$(grid.n), dt=$dt, steps=$nsteps")

    for step in 1:nsteps
        sys = solve_unsteady!(model, xprev; t=t, dt=dt, scheme=:BE)
        q = integrated_embedded_force(model, sys; x0=CENTER)
        ubar = bulk_mean_ux(model, sys.x)
        drag_ref = 6 * pi * MU * RADIUS * abs(ubar)
        ratio = drag_ref > 0 ? abs(q.force[1]) / drag_ref : NaN
        fres = norm(sys.A * sys.x - sys.b)
        println(
            "step=$step  t=$(t + dt)  F=$(q.force)  torque=$(q.torque)  ",
            "ūx=$ubar  |Fx|/(6πμRŪ)=$ratio  ||Ax-b||=$fres",
        )
        xprev .= sys.x
        t += dt
    end
end

main()
