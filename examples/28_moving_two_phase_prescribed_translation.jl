using LinearAlgebra
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

function max_interface_jump_error(model::MovingStokesModelTwoPhase{2,T}, x::AbstractVector{T}, jumpx::T, jumpy::T) where {T}
    cap_p1_end = something(model.cap_p1_end)
    ug1x = x[model.layout.ugamma1[1]]
    ug1y = x[model.layout.ugamma1[2]]
    ug2x = x[model.layout.ugamma2[1]]
    ug2y = x[model.layout.ugamma2[2]]

    ex = zero(T)
    ey = zero(T)
    cnt = 0
    @inbounds for i in 1:cap_p1_end.ntotal
        Γ = cap_p1_end.buf.Γ[i]
        if isfinite(Γ) && Γ > zero(T)
            cnt += 1
            ex = max(ex, abs((ug1x[i] - ug2x[i]) - jumpx))
            ey = max(ey, abs((ug1y[i] - ug2y[i]) - jumpy))
        end
    end
    return ex, ey, cnt
end

function main()
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (41, 41))
    bc = periodic_2d_bc()

    R = 0.16
    xc0, yc0 = 0.5, 0.5
    amp = 0.08
    ω = 2 * pi

    xc(t) = xc0 + amp * sin(ω * t)
    uexact(t) = amp * ω * cos(ω * t)
    duexact(t) = -amp * ω^2 * sin(ω * t)
    body(x, y, t) = R - sqrt((x - xc(t))^2 + (y - yc0)^2)

    ρ1 = 1.0
    ρ2 = 2.0
    model = MovingStokesModelTwoPhase(
        grid,
        body,
        1.0,
        5.0;
        rho1=ρ1,
        rho2=ρ2,
        bc_u=(bc, bc),
        bc_p=bc,
        force1=((x, y, t) -> ρ1 * duexact(t), 0.0),
        force2=((x, y, t) -> ρ2 * duexact(t), 0.0),
        interface_jump=(0.0, 0.0),
        interface_force=(0.0, 0.0),
        gauge=PinPressureGauge(),
    )

    nsys = last(model.layout.pomega2)
    xprev = zeros(Float64, nsys)

    dt = 0.02
    nsteps = 4
    t = 0.0

    println("Unsteady moving-interface two-phase Stokes (prescribed translation)")
    println("grid=$(grid.n), dt=$dt, steps=$nsteps")

    for step in 1:nsteps
        sys = solve_unsteady_moving!(model, xprev; t=t, dt=dt, scheme=:CN)
        tnext = t + dt

        ubar = mean_ux(model, sys.x)
        uref = uexact(tnext)
        ejx, ejy, niface = max_interface_jump_error(model, sys.x, 0.0, 0.0)
        res = norm(sys.A * sys.x - sys.b)

        println(
            "step=$step  t=$tnext  ubar=$ubar  uref=$uref  ",
            "|ubar-uref|=$(abs(ubar - uref))  jump_err=($ejx,$ejy)  ",
            "iface_cells=$niface  ||Ax-b||=$res",
        )

        xprev .= sys.x
        t = tnext
    end
end

main()
