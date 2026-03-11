using LinearAlgebra
using StaticArrays: SVector
using CartesianGrids
using PenguinBCs
using PenguinStokes

function periodic_2d_bc()
    return BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Periodic(), top=Periodic(),
    )
end

function mean_and_l2_ux(model::MovingStokesModelMono{2,T}, x::AbstractVector{T}, uref::T) where {T}
    cap_u_end = something(model.cap_u_end)
    cap = cap_u_end[1]
    ux = x[model.layout.uomega[1]]
    acc = zero(T)
    vol = zero(T)
    e2 = zero(T)
    @inbounds for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if isfinite(V) && V > zero(T)
            ui = ux[i]
            acc += V * ui
            vol += V
            e2 += V * (ui - uref)^2
        end
    end
    return acc / vol, sqrt(e2 / vol)
end

rate(eh, e2h, h, h2) = log(eh / e2h) / log(h / h2)

function run_scheme(; scheme=:CN, dts=(0.1, 0.05, 0.025), tf=0.4)
    ω = 2pi
    U(t) = sin(ω * t)
    dU(t) = ω * cos(ω * t)
    shift(t) = (1 - cos(ω * t)) / ω

    R = 0.18
    xc0 = 0.5
    yc = 0.5

    body(x, y, t) = R - sqrt((x - (xc0 + shift(t)))^2 + (y - yc)^2)

    bc = periodic_2d_bc()
    fx(x, y, t) = dU(t)

    errs_mean = Float64[]
    errs_l2 = Float64[]

    for dt in dts
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
        model = MovingStokesModelMono(
            grid,
            body,
            1.0,
            1.0;
            bc_u=(bc, bc),
            force=(fx, 0.0),
            bc_cut_u=(Dirichlet((x, y, t) -> U(t)), Dirichlet(0.0)),
        )

        x = zeros(Float64, last(model.layout.pomega))
        t = 0.0
        sys_last = nothing
        while t < tf - 1e-12
            sys = solve_unsteady_moving!(model, x; t=t, dt=dt, scheme=scheme)
            x .= sys.x
            t += dt
            sys_last = sys
        end

        uex = U(tf)
        umean, ul2 = mean_and_l2_ux(model, x, uex)
        emean = abs(umean - uex)

        push!(errs_mean, emean)
        push!(errs_l2, ul2)

        linres = isnothing(sys_last) ? NaN : norm(sys_last.A * sys_last.x - sys_last.b)
        println("scheme=$(scheme)  dt=$dt  mean_err=$emean  l2_err=$ul2  ||Ax-b||=$linres")
    end

    println("orders for scheme=$(scheme):")
    for k in 1:(length(dts) - 1)
        println("  dt $(dts[k])->$(dts[k+1]): mean order = ", rate(errs_mean[k], errs_mean[k + 1], dts[k], dts[k + 1]),
                "  l2 order = ", rate(errs_l2[k], errs_l2[k + 1], dts[k], dts[k + 1]))
    end

    return errs_mean, errs_l2
end

function run_scheme_no_body(; scheme=:CN, dts=(0.1, 0.05, 0.025), tf=0.4)
    ω = 2pi
    U(t) = sin(ω * t)
    dU(t) = ω * cos(ω * t)

    bc = periodic_2d_bc()
    body(x, y, t) = -1.0
    fx(x, y, t) = dU(t)

    errs_mean = Float64[]
    errs_l2 = Float64[]

    for dt in dts
        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (33, 33))
        model = MovingStokesModelMono(
            grid,
            body,
            1.0,
            1.0;
            bc_u=(bc, bc),
            force=(fx, 0.0),
            bc_cut_u=(Dirichlet(0.0), Dirichlet(0.0)),
        )

        x = zeros(Float64, last(model.layout.pomega))
        t = 0.0
        sys_last = nothing
        while t < tf - 1e-12
            sys = solve_unsteady_moving!(model, x; t=t, dt=dt, scheme=scheme)
            x .= sys.x
            t += dt
            sys_last = sys
        end

        uex = U(tf)
        umean, ul2 = mean_and_l2_ux(model, x, uex)
        emean = abs(umean - uex)

        push!(errs_mean, emean)
        push!(errs_l2, ul2)

        linres = isnothing(sys_last) ? NaN : norm(sys_last.A * sys_last.x - sys_last.b)
        println("[no-body] scheme=$(scheme)  dt=$dt  mean_err=$emean  l2_err=$ul2  ||Ax-b||=$linres")
    end

    println("[no-body] orders for scheme=$(scheme):")
    for k in 1:(length(dts) - 1)
        println("  dt $(dts[k])->$(dts[k+1]): mean order = ", rate(errs_mean[k], errs_mean[k + 1], dts[k], dts[k + 1]),
                "  l2 order = ", rate(errs_l2[k], errs_l2[k + 1], dts[k], dts[k + 1]))
    end

    return errs_mean, errs_l2
end

function main()
    println("Moving monophasic MMS (embedded moving body, periodic box)")
    println("Exact solution: u(x,t) = (sin(2πt), 0), p = const")
    println("Body center follows x_c(t)=x_c0 + (1-cos(2πt))/(2π), cut BC matches exact rigid translation")

    dts = (0.1, 0.05, 0.025)

    run_scheme(; scheme=:BE, dts=dts)
    run_scheme(; scheme=:CN, dts=dts)
    run_scheme(; scheme=0.75, dts=dts)

    println("\nMoving monophasic MMS baseline (no embedded body)")
    println("Same exact solution/time forcing, body=-1")
    run_scheme_no_body(; scheme=:BE, dts=dts)
    run_scheme_no_body(; scheme=:CN, dts=dts)
    run_scheme_no_body(; scheme=0.75, dts=dts)
end

main()
