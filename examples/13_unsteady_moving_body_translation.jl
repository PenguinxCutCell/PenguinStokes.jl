using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

periodic_2d_bc() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

function mean_ux(model::MovingStokesModelMono{2,T}, x::AbstractVector{T}) where {T}
    cap_u_end = something(model.cap_u_end)
    cap = cap_u_end[1]
    ux = x[model.layout.uomega[1]]
    acc = zero(T)
    vol = zero(T)
    @inbounds for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if isfinite(V) && V > zero(T)
            acc += V * ux[i]
            vol += V
        end
    end
    return acc / vol
end

function max_interface_trace_error(model::MovingStokesModelMono{2,T}, x::AbstractVector{T}, uwall::T) where {T}
    cap_u_end = something(model.cap_u_end)
    ugx = x[model.layout.ugamma[1]]
    ugy = x[model.layout.ugamma[2]]
    ex = zero(T)
    ey = zero(T)
    cnt = 0
    @inbounds for i in 1:cap_u_end[1].ntotal
        Γ = cap_u_end[1].buf.Γ[i]
        if isfinite(Γ) && Γ > zero(T)
            cnt += 1
            ex = max(ex, abs(ugx[i] - uwall))
            ey = max(ey, abs(ugy[i]))
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
    uwall(t) = amp * ω * cos(ω * t)
    body(x, y, t) = R - sqrt((x - xc(t))^2 + (y - yc0)^2)

    model = MovingStokesModelMono(
        grid,
        body,
        1.0,
        1.0;
        bc_u=(bc, bc),
        force=(0.0, 0.0),
        bc_cut_u=(Dirichlet((x, y, t) -> uwall(t)), Dirichlet(0.0)),
    )

    nsys = last(model.layout.pomega)
    xprev = zeros(Float64, nsys)

    dt = 0.05
    nsteps = 12
    t = 0.0

    println("Unsteady moving-body one-phase Stokes (prescribed translation)")
    println("grid=$(grid.n), dt=$dt, steps=$nsteps")

    for step in 1:nsteps
        sys = solve_unsteady_moving!(model, xprev; t=t, dt=dt, scheme=:CN)
        tnext = t + dt
        uw = uwall(tnext)
        ubar = mean_ux(model, sys.x)
        ex, ey, cnt = max_interface_trace_error(model, sys.x, uw)
        res = norm(sys.A * sys.x - sys.b)
        println(
            "step=$step  t=$tnext  uw=$uw  ubar=$ubar  ",
            "trace_err=($ex,$ey)  iface_cells=$cnt  ||Ax-b||=$res",
        )
        xprev .= sys.x
        t = tnext
    end
end

main()
