using LinearAlgebra
using Printf
using Statistics
using CairoMakie
using StaticArrays: SVector
using CartesianGrids
using PenguinBCs
using PenguinStokes

function active_velocity_indices(cap)
    return findall(isfinite.(cap.buf.V) .& (cap.buf.V .> 0.0))
end

function sample_indices(idxs::Vector{Int}, stride::Int)
    stride <= 1 && return idxs
    out = Int[]
    for (k, i) in enumerate(idxs)
        if k % stride == 1
            push!(out, i)
        end
    end
    return out
end

function velocity_l2_combined(model, u1x, u1y, u2x, u2y)
    e2 = 0.0
    w = 0.0

    for (cap, vals) in (
        (model.cap_u1[1], u1x),
        (model.cap_u1[2], u1y),
        (model.cap_u2[1], u2x),
        (model.cap_u2[2], u2y),
    )
        idx = active_velocity_indices(cap)
        @inbounds for i in idx
            Vi = cap.buf.V[i]
            v = vals[i]
            e2 += Vi * v * v
            w += Vi
        end
    end

    return sqrt(e2 / max(w, eps(Float64)))
end

function _gauge_symbol(gauge)
    gauge isa MeanPressureGauge && return :mean
    gauge isa PinPressureGauge && return :pin
    error("unsupported gauge type $(typeof(gauge))")
end

function run_case(
    ;
    n::Int=65,
    R::Float64=0.35,
    sigma::Float64=1.0,
    mu::Float64=1.0,
    rho::Float64=1.0,
    center::SVector{2,Float64}=SVector(0.0, 0.0),
    gauge::AbstractPressureGauge=MeanPressureGauge(),
)
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    xc, yc = center
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - R

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    dp_th = sigma / R

    # Sign convention retained explicitly:
    # body = r - R and interface_force = -(sigma/R) n,
    # so p_in - p_out = sigma/R.
    interface_force(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? SVector(0.0, 0.0) : -(dp_th) * SVector(dx / rr, dy / rr)
    end

    model = StokesModelTwoPhase(
        grid,
        body,
        mu,
        mu;
        rho1=rho,
        rho2=rho,
        bc_u=(bc0, bc0),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=interface_force,
        gauge=gauge,
    )

    sys = solve_steady!(model)

    u1x = sys.x[model.layout.uomega1[1]]
    u1y = sys.x[model.layout.uomega1[2]]
    u2x = sys.x[model.layout.uomega2[1]]
    u2y = sys.x[model.layout.uomega2[2]]

    idx_u1x = active_velocity_indices(model.cap_u1[1])
    idx_u1y = active_velocity_indices(model.cap_u1[2])
    idx_u2x = active_velocity_indices(model.cap_u2[1])
    idx_u2y = active_velocity_indices(model.cap_u2[2])

    u1_inf = max(maximum(abs, u1x[idx_u1x]), maximum(abs, u1y[idx_u1y]))
    u2_inf = max(maximum(abs, u2x[idx_u2x]), maximum(abs, u2y[idx_u2y]))
    uinf = max(u1_inf, u2_inf)
    ul2 = velocity_l2_combined(model, u1x, u1y, u2x, u2y)

    flat = PenguinStokes.pressure_flatness_report(model, sys; io=devnull, verbose=false)
    p_in = flat.phase1[:all].mean
    p_out = flat.phase2[:all].mean
    p1_std = flat.phase1[:all].std
    p2_std = flat.phase2[:all].std

    dp_num = p_in - p_out
    jump_relerr = abs(dp_num - dp_th) / max(abs(dp_th), eps(Float64))

    xeq = PenguinStokes.build_static_circle_equilibrium_state(
        model;
        sigma=sigma,
        R=R,
        gauge=_gauge_symbol(gauge),
        sys=sys,
    )
    eqaudit = PenguinStokes.exact_equilibrium_residual(sys, model, xeq)

    D = 2.0 * R
    La = rho * D * sigma / (mu * mu)
    Ca = uinf * mu / sigma

    return (
        n=n,
        R=R,
        D=D,
        sigma=sigma,
        mu=mu,
        rho=rho,
        La=La,
        Ca=Ca,
        u1_inf=u1_inf,
        u2_inf=u2_inf,
        uinf=uinf,
        ul2=ul2,
        p_in=p_in,
        p_out=p_out,
        p1_std=p1_std,
        p2_std=p2_std,
        dp_num=dp_num,
        dp_th=dp_th,
        jump_relerr=jump_relerr,
        residual=norm(sys.A * sys.x - sys.b),
        eq_residual=norm(eqaudit.residual),
        model=model,
        sys=sys,
        body=body,
    )
end

function print_single_case_summary(r)
    println("\nStatic circle fixed-interface benchmark (single mesh)")
    println("Domain=[-1,1]^2, no-slip walls, MeanPressureGauge")
    @printf("n=%d, R=%.4f, D=%.4f, sigma=%.4f, mu=%.4f, rho=%.4f\n", r.n, r.R, r.D, r.sigma, r.mu, r.rho)
    println("Sign convention: body=r-R and interface_force=-(sigma/R) n => p_in-p_out=sigma/R")
    @printf("u1_inf=%.6e, u2_inf=%.6e, uinf=%.6e, ul2=%.6e\n", r.u1_inf, r.u2_inf, r.uinf, r.ul2)
    @printf("p_in=%.12f, p_out=%.12f, p1_std=%.3e, p2_std=%.3e\n", r.p_in, r.p_out, r.p1_std, r.p2_std)
    @printf("dp_num=%.12f, dp_th=%.12f, rel_jump_err=%.3e\n", r.dp_num, r.dp_th, r.jump_relerr)
    @printf("La=%.6e, Ca=%.6e, ||Ax-b||=%.3e, ||Axeq-b||=%.3e\n", r.La, r.Ca, r.residual, r.eq_residual)
end

function velocity_arrow_data(r; stride::Int=4)
    model = r.model
    sys = r.sys
    layout = model.layout

    x = Float64[]
    y = Float64[]
    u = Float64[]
    v = Float64[]

    chunks = (
        (model.cap_u1[1], sys.x[layout.uomega1[1]], 1),
        (model.cap_u1[2], sys.x[layout.uomega1[2]], 2),
        (model.cap_u2[1], sys.x[layout.uomega2[1]], 1),
        (model.cap_u2[2], sys.x[layout.uomega2[2]], 2),
    )

    for (cap, vals, dir) in chunks
        idx = sample_indices(active_velocity_indices(cap), stride)
        @inbounds for i in idx
            c = cap.C_ω[i]
            push!(x, c[1])
            push!(y, c[2])
            if dir == 1
                push!(u, vals[i])
                push!(v, 0.0)
            else
                push!(u, 0.0)
                push!(v, vals[i])
            end
        end
    end

    return x, y, u, v
end

function pressure_map_data(r)
    model = r.model
    sys = r.sys
    body = r.body
    layout = model.layout

    p1 = sys.x[layout.pomega1]
    p2 = sys.x[layout.pomega2]

    active_p1 = PenguinStokes._pressure_activity(model.cap_p1)
    active_p2 = PenguinStokes._pressure_activity(model.cap_p2)

    idx_p2 = findall(active_p2)
    p_out_mean = mean(p2[idx_p2])

    coords = model.cap_p1.C_ω
    xp = getindex.(coords, 1)
    yp = getindex.(coords, 2)
    pv = zeros(length(coords))

    @inbounds for i in eachindex(coords)
        x = xp[i]
        y = yp[i]
        inside = body(x, y) <= 0.0

        if inside && active_p1[i]
            pv[i] = p1[i] - p_out_mean
        elseif !inside && active_p2[i]
            pv[i] = p2[i] - p_out_mean
        elseif inside
            pv[i] = p1[i] - p_out_mean
        else
            pv[i] = p2[i] - p_out_mean
        end
    end

    return xp, yp, pv
end

function plot_single_case(
    r;
    filename::AbstractString,
    arrow_stride::Int=4,
    velocity_scale::Float64=1.0e11,
    pressure_marker_size::Float64=9.0,
)
    xv, yv, uv, vv = velocity_arrow_data(r; stride=arrow_stride)
    xp, yp, pv = pressure_map_data(r)

    fig = Figure(size=(1200, 520), fontsize=15)

    axv = Axis(
        fig[1, 1],
        title=@sprintf("Velocity arrows (scaled by %.1e)", velocity_scale),
        xlabel="x",
        ylabel="y",
        aspect=DataAspect(),
    )
    arrows!(axv, xv, yv, velocity_scale .* uv, velocity_scale .* vv; color=:black, linewidth=1.0, arrowsize=8)

    theta = range(0.0, 2pi; length=400)
    lines!(axv, r.R .* cos.(theta), r.R .* sin.(theta); color=:dodgerblue3, linewidth=2)
    limits!(axv, -1, 1, -1, 1)

    axp = Axis(
        fig[1, 2],
        title="Pressure map (inside/outside)",
        xlabel="x",
        ylabel="y",
        aspect=DataAspect(),
    )
    hm = scatter!(axp, xp, yp; color=pv, colormap=:viridis, marker=:rect, markersize=pressure_marker_size)
    lines!(axp, r.R .* cos.(theta), r.R .* sin.(theta); color=:white, linewidth=2)
    limits!(axp, -1, 1, -1, 1)
    Colorbar(fig[1, 3], hm, label="p - mean(p_out)")

    summary = @sprintf(
        "uinf=%.3e, ul2=%.3e, Ca=%.3e\np_in=%.12f, p_out=%.12f\ndp_num=%.12f, dp_th=%.12f, rel_err=%.3e\np1_std=%.3e, p2_std=%.3e",
        r.uinf,
        r.ul2,
        r.Ca,
        r.p_in,
        r.p_out,
        r.dp_num,
        r.dp_th,
        r.jump_relerr,
        r.p1_std,
        r.p2_std,
    )
    Label(fig[2, 1:3], summary; tellheight=true)

    save(filename, fig)
end

function run_laplace_sweep(
    ;
    n::Int,
    R::Float64,
    sigma::Float64,
    rho::Float64,
    La_min::Float64=12.0,
    La_max::Float64=120000.0,
    npoints::Int=3,
    gauge::AbstractPressureGauge=MeanPressureGauge(),
)
    Las = exp10.(range(log10(La_min), log10(La_max), length=npoints))

    println("\nLaplace-number sweep on fixed mesh n=$(n)")
    println("La = rho*D*sigma/mu^2 with D=2R")
    println("Capillary number: Ca = ||u||_inf * mu / sigma")
    println(rpad("La", 14), rpad("mu", 14), rpad("uinf", 14), rpad("Ca", 14), "rel_jump_err")

    results = NamedTuple[]

    for La in Las
        mu = sqrt(rho * (2.0 * R) * sigma / La)
        r = run_case(; n=n, R=R, sigma=sigma, mu=mu, rho=rho, gauge=gauge)
        push!(results, (La=La, mu=mu, uinf=r.uinf, Ca=r.Ca, rel_jump_err=r.jump_relerr))
        @printf("%-14.6e%-14.6e%-14.6e%-14.6e%.6e\n", La, mu, r.uinf, r.Ca, r.jump_relerr)
    end

    return results
end

function plot_laplace_sweep(results; filename::AbstractString)
    La = [r.La for r in results]
    Ca = [max(r.Ca, eps(Float64)) for r in results]
    jump = [max(r.rel_jump_err, eps(Float64)) for r in results]

    fig = Figure(size=(980, 430), fontsize=15)

    ax1 = Axis(
        fig[1, 1],
        title="Capillary number vs Laplace number",
        xlabel="Laplace number La",
        ylabel="Ca = ||u||_inf * mu / sigma",
        xscale=log10,
        yscale=log10,
    )
    lines!(ax1, La, Ca; color=:dodgerblue3, linewidth=2)
    scatter!(ax1, La, Ca; color=:dodgerblue3, markersize=9)

    ax2 = Axis(
        fig[1, 2],
        title="Pressure jump relative error vs Laplace number",
        xlabel="Laplace number La",
        ylabel="|dp_num - dp_th| / |dp_th|",
        xscale=log10,
        yscale=log10,
    )
    lines!(ax2, La, jump; color=:firebrick3, linewidth=2)
    scatter!(ax2, La, jump; color=:firebrick3, markersize=9)

    save(filename, fig)
end

function main()
    n = parse(Int, get(ENV, "PS_N", "65"))
    R = parse(Float64, get(ENV, "PS_R", "0.35"))
    sigma = parse(Float64, get(ENV, "PS_SIGMA", "1.0"))
    rho = parse(Float64, get(ENV, "PS_RHO", "1.0"))

    arrow_stride = parse(Int, get(ENV, "PS_ARROW_STRIDE", "4"))
    velocity_scale = parse(Float64, get(ENV, "PS_ARROW_SCALE", "1e11"))
    pressure_marker_size = parse(Float64, get(ENV, "PS_PRESSURE_MARKER_SIZE", "9.0"))

    single_plot_file = get(ENV, "PS_SINGLE_FIG", "two_phase_static_circle_snapshot_n$(n).png")
    sweep_plot_file = get(ENV, "PS_SWEEP_FIG", "two_phase_static_circle_laplace_sweep_n$(n).png")

    sweep_points = parse(Int, get(ENV, "PS_LA_POINTS", "9"))
    La_min = parse(Float64, get(ENV, "PS_LA_MIN", "12.0"))
    La_max = parse(Float64, get(ENV, "PS_LA_MAX", "120000.0"))

    single = run_case(; n=n, R=R, sigma=sigma, mu=1.0, rho=rho, gauge=MeanPressureGauge())
    print_single_case_summary(single)

    plot_single_case(
        single;
        filename=single_plot_file,
        arrow_stride=arrow_stride,
        velocity_scale=velocity_scale,
        pressure_marker_size=pressure_marker_size,
    )
    println("Saved single-case figure: " * single_plot_file)

    sweep = run_laplace_sweep(
        ;
        n=n,
        R=R,
        sigma=sigma,
        rho=rho,
        La_min=La_min,
        La_max=La_max,
        npoints=sweep_points,
        gauge=MeanPressureGauge(),
    )

    plot_laplace_sweep(sweep; filename=sweep_plot_file)
    println("Saved Laplace-sweep figure: " * sweep_plot_file)

    return nothing
end

main()
