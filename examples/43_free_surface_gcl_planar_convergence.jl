using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

periodic_2d_bc() = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Periodic(), top=Periodic(),
)

rate(eh, e2h, h, h2) = log(eh / e2h) / log(h / h2)

function manufactured_solution()
    ω = 2π
    U(t) = sin(ω * t)
    dU(t) = ω * cos(ω * t)
    shift(t) = (1 - cos(ω * t)) / ω
    return (; U, dU, shift)
end

function velocity_errors(model::MovingStokesModelMono{2,T}, x::AbstractVector, uexact::T) where {T}
    cap = something(model.cap_u_end)[1]
    ux = view(x, model.layout.uomega[1])
    acc = zero(T)
    vol = zero(T)
    e2 = zero(T)

    @inbounds for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if isfinite(V) && V > zero(T)
            err = ux[i] - uexact
            acc += V * ux[i]
            vol += V
            e2 += V * err^2
        end
    end

    mean_err = abs(acc / vol - uexact)
    l2_err = sqrt(e2 / vol)
    return mean_err, l2_err
end

function run_moving_interface_mms_convergence(;
    nlevels=(17, 25, 33),
    tf=0.02,
    cdt=0.5,
    scheme=:CN,
)
    sol = manufactured_solution()
    R = 0.18
    xc0 = 0.5
    yc = 0.5

    hs = Float64[]
    dts = Float64[]
    errs_mean = Float64[]
    errs_l2 = Float64[]
    errs_gcl = Float64[]
    errs_kin = Float64[]

    println("Moving-interface GCL MMS convergence")
    println("Exact solution: u(x,t) = (sin(2πt), 0), p = 0")
    println("Exact material interface: circle center x_c(t) = x_c0 + (1-cos(2πt))/(2π)")
    println("Body force: f = (2π cos(2πt), 0)")

    for n in nlevels
        h = 1.0 / (n - 1)
        dt_target = cdt * h^2
        nsteps = max(1, ceil(Int, tf / dt_target))
        dt = tf / nsteps

        grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
        body(x, y, t) = R - sqrt((x - (xc0 + sol.shift(t)))^2 + (y - yc)^2)
        fx(x, y, t) = sol.dU(t)
        bc = periodic_2d_bc()

        model = MovingStokesModelMono(
            grid,
            body,
            1.0,
            1.0;
            bc_u=(bc, bc),
            force=(fx, 0.0),
            bc_cut_u=(Dirichlet((x, y, t) -> sol.U(t)), Dirichlet(0.0)),
        )

        x = zeros(Float64, last(model.layout.pomega))
        t = 0.0
        sys_last = nothing
        terms_last = nothing

        for _ in 1:nsteps
            Vn = stokes_pressure_volume(model, t)
            sys = solve_unsteady_moving!(model, x; t=t, dt=dt, scheme=scheme)
            terms = stokes_gcl_terms_mono(model, sys.x, Vn)
            x .= sys.x
            t += dt
            sys_last = sys
            terms_last = terms
        end

        mean_err, l2_err = velocity_errors(model, x, sol.U(tf))
        gcl_err = norm(mask_inactive_pressure_cells(model, terms_last.R_gcl, sys_last.A), Inf)
        kin_err = norm(terms_last.R_kin, Inf)

        push!(hs, h)
        push!(dts, dt)
        push!(errs_mean, mean_err)
        push!(errs_l2, l2_err)
        push!(errs_gcl, gcl_err)
        push!(errs_kin, kin_err)

        println(
            "n=$n  h=$h  dt=$dt  steps=$nsteps  ",
            "mean_u_err=$mean_err  l2_u_err=$l2_err  ",
            "masked_R_gcl=$gcl_err  R_kin=$kin_err",
        )
    end

    mean_orders = [rate(errs_mean[k], errs_mean[k + 1], hs[k], hs[k + 1]) for k in 1:(length(hs) - 1)]
    l2_orders = [rate(errs_l2[k], errs_l2[k + 1], hs[k], hs[k + 1]) for k in 1:(length(hs) - 1)]

    println("mesh orders:")
    for k in eachindex(l2_orders)
        println(
            "  n $(nlevels[k])->$(nlevels[k + 1]): ",
            "mean order=$(mean_orders[k])  l2 order=$(l2_orders[k])  ",
            "dt $(dts[k])->$(dts[k + 1])",
        )
    end
    println("minimum L2 velocity order = ", minimum(l2_orders))

    return (; hs, dts, errs_mean, errs_l2, errs_gcl, errs_kin, mean_orders, l2_orders)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_moving_interface_mms_convergence()
    minimum(result.l2_orders) > 1.4 || error("expected L2 velocity convergence order > 1.4")
end
