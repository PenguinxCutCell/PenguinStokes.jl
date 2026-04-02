using LinearAlgebra
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

function manufactured_fields()
    ω = 2pi
    U(t) = sin(ω * t)
    dU(t) = ω * cos(ω * t)
    shift(t) = (1 - cos(ω * t)) / ω
    shiftdot(t) = sin(ω * t)
    return (; ω, U, dU, shift, shiftdot)
end

function check_analytical_consistency(; tf=0.02, nsamp=9)
    f = manufactured_fields()
    ts = range(0.0, tf; length=nsamp)

    max_body_speed_mismatch = maximum(abs(f.shiftdot(t) - f.U(t)) for t in ts)
    fx(x, y, t) = f.dU(t)
    max_force_mismatch = maximum(abs(f.dU(t) - fx(0.37, 0.61, t)) for t in ts)

    println("Analytical consistency checks on [0,$tf]:")
    println("  max |ẋc(t)-U(t)| = $max_body_speed_mismatch")
    println("  max |∂t U(t)-fx(t)| = $max_force_mismatch")
end

function run_scheme_mesh(; scheme=:CN, nlevels=(17, 33, 65), tf=0.02, cdt=0.5, with_body=true)
    f = manufactured_fields()
    U = f.U
    dU = f.dU
    shift = f.shift

    R = 0.18
    xc0 = 0.5
    yc = 0.5

    body_moving(x, y, t) = R - sqrt((x - (xc0 + shift(t)))^2 + (y - yc)^2)
    body_no_embedded(args...) = -1.0
    body = with_body ? body_moving : body_no_embedded

    bc = periodic_2d_bc()
    fx(x, y, t) = dU(t)
    bc_cut_u = with_body ? (Dirichlet((x, y, t) -> U(t)), Dirichlet(0.0)) : (Dirichlet(0.0), Dirichlet(0.0))
    label = with_body ? "" : "[no-body] "

    hs = Float64[]
    dts = Float64[]
    errs_mean = Float64[]
    errs_l2 = Float64[]

    for n in nlevels
        h = 1.0 / (n - 1)
        dt_target = cdt * h^2
        nsteps = max(1, ceil(Int, tf / dt_target))
        dt = tf / nsteps

        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = MovingStokesModelMono(
            grid,
            body,
            1.0,
            1.0;
            bc_u=(bc, bc),
            force=(fx, 0.0),
            bc_cut_u=bc_cut_u,
        )

        x = zeros(Float64, last(model.layout.pomega))
        t = 0.0
        sys_last = nothing
        for _ in 1:nsteps
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
        push!(hs, h)
        push!(dts, dt)

        linres = isnothing(sys_last) ? NaN : norm(sys_last.A * sys_last.x - sys_last.b)
        println("$(label)scheme=$(scheme)  n=$n  h=$h  dt=$dt  nsteps=$nsteps  mean_err=$emean  l2_err=$ul2  ||Ax-b||=$linres")
    end

    println("$(label)mesh orders for scheme=$(scheme):")
    for k in 1:(length(nlevels) - 1)
        println(
            "  n $(nlevels[k])->$(nlevels[k+1]): mean order = ",
            rate(errs_mean[k], errs_mean[k + 1], hs[k], hs[k + 1]),
            "  l2 order = ",
            rate(errs_l2[k], errs_l2[k + 1], hs[k], hs[k + 1]),
            "  (dt $(dts[k])->$(dts[k+1]))",
        )
    end

    return errs_mean, errs_l2
end

function main()
    println("Moving monophasic MMS mesh convergence (periodic box)")
    println("Exact solution: u(x,t) = (sin(2πt), 0), p = const")
    println("Body center follows x_c(t)=x_c0 + (1-cos(2πt))/(2π), with ẋ_c(t)=sin(2πt)=U(t)")
    println("Mesh sweep uses dt≈0.5*h^2 with exact t_f hit by dt=t_f/nsteps")

    tf = 0.02
    nlevels = (33, 49, 65, 97)

    check_analytical_consistency(; tf=tf)

    run_scheme_mesh(; scheme=:BE, nlevels=nlevels, tf=tf, cdt=1.0, with_body=true)
    run_scheme_mesh(; scheme=:CN, nlevels=nlevels, tf=tf, cdt=1.0, with_body=true)
    run_scheme_mesh(; scheme=0.75, nlevels=nlevels, tf=tf, cdt=1.0, with_body=true)

    println("\nMoving monophasic MMS baseline (no embedded body)")
    println("Same exact solution/forcing with body=-1")
    run_scheme_mesh(; scheme=:BE, nlevels=nlevels, tf=tf, cdt=1.0, with_body=false)
    run_scheme_mesh(; scheme=:CN, nlevels=nlevels, tf=tf, cdt=1.0, with_body=false)
    run_scheme_mesh(; scheme=0.75, nlevels=nlevels, tf=tf, cdt=1.0, with_body=false)
end

main()
