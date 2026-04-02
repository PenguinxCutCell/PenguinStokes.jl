using CartesianGrids
using PenguinBCs
using PenguinStokes

periodic_2d_bc() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

function mean_ux(model::MovingStokesModelTwoPhase{2,T}, x::AbstractVector{T}) where {T}
    cap_u1_end = something(model.cap_u1_end)
    cap_u2_end = something(model.cap_u2_end)
    ux1 = x[model.layout.uomega1[1]]
    ux2 = x[model.layout.uomega2[1]]

    acc = zero(T)
    vol = zero(T)
    @inbounds for i in 1:cap_u1_end[1].ntotal
        V1 = cap_u1_end[1].buf.V[i]
        if isfinite(V1) && V1 > zero(T)
            acc += V1 * ux1[i]
            vol += V1
        end
        V2 = cap_u2_end[1].buf.V[i]
        if isfinite(V2) && V2 > zero(T)
            acc += V2 * ux2[i]
            vol += V2
        end
    end
    return acc / vol
end

function solve_mean(tf, dt, scheme)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (25, 25))
    bc = periodic_2d_bc()

    R = 0.18
    xc0, yc0 = 0.5, 0.5
    amp = 0.06
    ω = 2 * pi

    xc(t) = xc0 + amp * sin(ω * t)
    uexact(t) = sin(t)
    duexact(t) = cos(t)
    body(x, y, t) = R - sqrt((x - xc(t))^2 + (y - yc0)^2)

    model = MovingStokesModelTwoPhase(
        grid,
        body,
        1.0,
        3.0;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc, bc),
        bc_p=bc,
        force1=((x, y, t) -> duexact(t), 0.0),
        force2=((x, y, t) -> duexact(t), 0.0),
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
    )

    x = zeros(Float64, last(model.layout.pomega2))
    t = 0.0
    while t < tf - 1e-12
        sys = solve_unsteady_moving!(model, x; t=t, dt=dt, scheme=scheme)
        x .= sys.x
        t += dt
    end

    return mean_ux(model, x)
end

function report(scheme)
    tf = 0.4
    exact = sin(tf)

    u1 = solve_mean(tf, 0.1, scheme)
    u2 = solve_mean(tf, 0.05, scheme)
    e1 = abs(u1 - exact)
    e2 = abs(u2 - exact)
    ratio = e1 / e2

    println("scheme=$scheme")
    println("  dt=0.1   error=$e1")
    println("  dt=0.05  error=$e2")
    println("  ratio e(dt)/e(dt/2)=$ratio")
end

println("Moving two-phase MMS temporal sanity")
report(:BE)
report(:CN)
