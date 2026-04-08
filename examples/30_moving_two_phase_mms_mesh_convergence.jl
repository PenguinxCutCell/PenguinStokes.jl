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

function mean_and_l2_ux(model::MovingStokesModelTwoPhase{2,T}, x::AbstractVector{T}, uref::T) where {T}
    cap_u1_end = something(model.cap_u1_end)
    cap_u2_end = something(model.cap_u2_end)
    ux1 = x[model.layout.uomega1[1]]
    ux2 = x[model.layout.uomega2[1]]
    acc = zero(T); vol = zero(T); e2 = zero(T)
    @inbounds for i in 1:cap_u1_end[1].ntotal
        V1 = cap_u1_end[1].buf.V[i]
        if isfinite(V1) && V1 > zero(T)
            acc += V1 * ux1[i]; vol += V1; e2 += V1 * (ux1[i] - uref)^2
        end
        V2 = cap_u2_end[1].buf.V[i]
        if isfinite(V2) && V2 > zero(T)
            acc += V2 * ux2[i]; vol += V2; e2 += V2 * (ux2[i] - uref)^2
        end
    end
    return acc / vol, sqrt(e2 / vol)
end

rate(eh, e2h, h, h2) = log(eh / e2h) / log(h / h2)

function run_mesh_convergence(; scheme=:BE, nlevels=(17, 33, 65), tf=0.02, cdt=1.0)
    ω = 2pi
    U(t) = sin(ω * t)
    dU(t) = ω * cos(ω * t)
    shift(t) = (1 - cos(ω * t)) / ω

    R = 0.18; xc0 = 0.5; yc = 0.5
    body(x, y, t) = R - sqrt((x - (xc0 + shift(t)))^2 + (y - yc)^2)

    bc = periodic_2d_bc()
    fx(x, y, t) = dU(t)

    hs = Float64[]; dts = Float64[]
    errs_mean = Float64[]; errs_l2 = Float64[]

    for n in nlevels
        println("  Running n = $n ...")
        h = 1.0 / (n - 1)
        dt_target = cdt * h^2
        nsteps = max(1, ceil(Int, tf / dt_target))
        dt = tf / nsteps

        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        model = MovingStokesModelTwoPhase(
            grid, body, 1.0, 3.0;
            rho1=1.0, rho2=1.0,
            bc_u=(bc, bc), bc_p=bc,
            force1=(fx, 0.0), force2=(fx, 0.0),
            interface_jump=(0.0, 0.0),
            interface_force=(0.0, 0.0),
            gauge=MeanPressureGauge(),
        )

        x = zeros(Float64, last(model.layout.pomega2))
        t = 0.0
        sys_last = nothing
        for _ in 1:nsteps
            sys = solve_unsteady_moving!(model, x; t=t, dt=dt, scheme=scheme)
            x .= sys.x; t += dt; sys_last = sys
        end

        uex = U(tf)
        umean, ul2 = mean_and_l2_ux(model, x, uex)
        emean = abs(umean - uex)
        linres = isnothing(sys_last) ? NaN : norm(sys_last.A * sys_last.x - sys_last.b)

        push!(hs, h); push!(dts, dt)
        push!(errs_mean, emean); push!(errs_l2, ul2)

        println("scheme=$(scheme)  n=$n  h=$h  dt=$dt  nsteps=$nsteps  mean_err=$emean  l2_err=$ul2  ||Ax-b||=$linres")
    end

    println("mesh orders for scheme=$(scheme):")
    for k in 1:(length(nlevels) - 1)
        println(
            "  n $(nlevels[k])->$(nlevels[k+1]): mean order = ",
            rate(errs_mean[k], errs_mean[k+1], hs[k], hs[k+1]),
            "  l2 order = ",
            rate(errs_l2[k], errs_l2[k+1], hs[k], hs[k+1]),
            "  (dt $(dts[k])->$(dts[k+1]))",
        )
    end
    return errs_mean, errs_l2
end

function main()
    println("Moving two-phase MMS mesh convergence (periodic box)")
    println("Exact solution: u(x,t) = (sin(2πt), 0) in both phases, p = const")
    println("Body center: x_c(t) = 0.5 + (1-cos(2πt))/(2π), rho1=mu1=1, rho2=3, mu2=3")
    println("dt ≈ cdt*h^2, tf=0.02\n")

    tf = 0.01
    nlevels = (9, 13, 17, 25, 33, 49)

    for scheme in (:BE, :CN)
        println("Testing scheme: $(scheme)")
        run_mesh_convergence(; scheme=scheme, nlevels=nlevels, tf=tf, cdt=1.0)
        println()
    end
end

main()
