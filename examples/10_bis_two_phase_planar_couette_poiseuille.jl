using LinearAlgebra
using CairoMakie
using CartesianGrids
using PenguinBCs
using PenguinStokes

function two_layer_couette_poiseuille_profile(mu1, mu2, h, H, U, G)
    # PenguinStokes sign convention for f=(G,0), p_x=0: μ Δu = -G.
    C1 = -G / mu1
    C2 = -G / mu2

    M = [
        mu1 -mu2 0.0
        h -h -1.0
        0.0 H 1.0
    ]
    rhs = [
        0.0,
        0.5 * (C2 - C1) * h^2,
        U - 0.5 * C2 * H^2,
    ]
    A1, A2, B2 = M \ rhs

    u(y) = y <= h ? (0.5 * C1 * y^2 + A1 * y) : (0.5 * C2 * y^2 + A2 * y + B2)
    return u
end

function velocity_errors(model, sys, u_exact)
    u1 = sys.x[model.layout.uomega1[1]]
    u2 = sys.x[model.layout.uomega2[1]]
    v1 = sys.x[model.layout.uomega1[2]]
    v2 = sys.x[model.layout.uomega2[2]]

    e1 = 0.0
    e2 = 0.0
    w1 = 0.0
    w2 = 0.0
    vInf = 0.0

    for i in 1:model.cap_u1[1].ntotal
        V = model.cap_u1[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u1[1].C_ω[i][2]
            e = u1[i] - u_exact(y)
            e1 += V * e^2
            w1 += V
            vInf = max(vInf, abs(v1[i]))
        end
    end

    for i in 1:model.cap_u2[1].ntotal
        V = model.cap_u2[1].buf.V[i]
        if isfinite(V) && V > 0.0
            y = model.cap_u2[1].C_ω[i][2]
            e = u2[i] - u_exact(y)
            e2 += V * e^2
            w2 += V
            vInf = max(vInf, abs(v2[i]))
        end
    end

    return (u1L2=sqrt(e1 / w1), u2L2=sqrt(e2 / w2), vInf=vInf)
end

function active_velocity_indices(cap)
    return findall(isfinite.(cap.buf.V) .& (cap.buf.V .> 0.0))
end

function accumulate_component_y!(fld, cnt, cap, vals, n)
    idx = active_velocity_indices(cap)
    @inbounds for i in idx
        _, y = cap.C_ω[i]
        iy = clamp(round(Int, y * (n - 1)) + 1, 1, n)
        fld[iy] += vals[i]
        cnt[iy] += 1
    end
    return nothing
end

function build_velocity_maps(model, sys, n)
    uacc = zeros(n)
    vacc = zeros(n)
    ucnt = zeros(Int, n)
    vcnt = zeros(Int, n)

    u1 = sys.x[model.layout.uomega1[1]]
    v1 = sys.x[model.layout.uomega1[2]]
    u2 = sys.x[model.layout.uomega2[1]]
    v2 = sys.x[model.layout.uomega2[2]]

    accumulate_component_y!(uacc, ucnt, model.cap_u1[1], u1, n)
    accumulate_component_y!(uacc, ucnt, model.cap_u2[1], u2, n)
    accumulate_component_y!(vacc, vcnt, model.cap_u1[2], v1, n)
    accumulate_component_y!(vacc, vcnt, model.cap_u2[2], v2, n)

    uprof = fill(NaN, n)
    vprof = fill(NaN, n)
    @inbounds for j in 1:n
        ucnt[j] > 0 && (uprof[j] = uacc[j] / ucnt[j])
        vcnt[j] > 0 && (vprof[j] = vacc[j] / vcnt[j])
    end

    fill_missing_by_linear_interp!(uprof)
    fill_missing_by_linear_interp!(vprof)

    umap = repeat(reshape(uprof, 1, n), n, 1)
    vmap = repeat(reshape(vprof, 1, n), n, 1)
    speed = sqrt.(umap .^ 2 .+ vmap .^ 2)
    return umap, vmap, speed
end

function fill_missing_by_linear_interp!(a)
    n = length(a)
    valid = findall(isfinite, a)
    isempty(valid) && return fill!(a, 0.0)
    firstv = first(valid)
    lastv = last(valid)
    @inbounds for i in 1:(firstv - 1)
        a[i] = a[firstv]
    end
    @inbounds for i in (lastv + 1):n
        a[i] = a[lastv]
    end
    prev = firstv
    for next in valid[2:end]
        if next > prev + 1
            y0 = a[prev]
            y1 = a[next]
            span = next - prev
            @inbounds for i in (prev + 1):(next - 1)
                t = (i - prev) / span
                a[i] = (1 - t) * y0 + t * y1
            end
        end
        prev = next
    end
    return a
end

function numerical_profile_samples_at_x(model, sys, xsec, dx_tol)
    ys = Float64[]
    us = Float64[]

    u1 = sys.x[model.layout.uomega1[1]]
    u2 = sys.x[model.layout.uomega2[1]]

    for (cap, vals) in ((model.cap_u1[1], u1), (model.cap_u2[1], u2))
        idx = active_velocity_indices(cap)
        @inbounds for i in idx
            x, y = cap.C_ω[i]
            if abs(x - xsec) <= dx_tol
                push!(ys, y)
                push!(us, vals[i])
            end
        end
    end

    p = sortperm(ys)
    return ys[p], us[p]
end

function plot_velocity_overlay(model, sys, u_exact, h, n; filename="10_bis_two_phase_planar_couette_poiseuille.png")
    umap, vmap, speed = build_velocity_maps(model, sys, n)
    xs = collect(range(0.0, 1.0, length=n))
    ys = collect(range(0.0, 1.0, length=n))

    fig = Figure(size=(1050, 500))
    ax = Axis(
        fig[1, 1];
        title="Two-phase Couette-Poiseuille ",
        xlabel="x",
        ylabel="y",
        aspect=DataAspect(),
    )

    hm = heatmap!(ax, xs, ys, speed; colormap=:viridis)
    Colorbar(fig[1, 2], hm; label="|u|")

    step = max(2, n ÷ 18)
    qx = Float64[]
    qy = Float64[]
    qu = Float64[]
    qv = Float64[]
    @inbounds for j in 1:step:n, i in 1:step:n
        uij = umap[i, j]
        vij = vmap[i, j]
        if isfinite(uij) && isfinite(vij)
            push!(qx, xs[i])
            push!(qy, ys[j])
            push!(qu, uij)
            push!(qv, vij)
        end
    end
    arrows2d!(ax, qx, qy, qu, qv; lengthscale=0.18, shaftwidth=1.5, tiplength=8, tipwidth=8, color=:white)

    xsec = 0.5
    yprof = ys
    y_num, u_num = numerical_profile_samples_at_x(model, sys, xsec, 0.5 / (n - 1))
    u_ex = [u_exact(y) for y in yprof]
    umax = max(maximum(abs.(u_num)), maximum(abs.(u_ex)), eps(Float64))
    profile_scale = 0.35 / umax

    lines!(ax, fill(xsec, n), yprof; color=(:white, 0.55), linestyle=:dash, linewidth=1.0)
    lines!(ax, xsec .+ profile_scale .* u_ex, yprof; color=:orange, linewidth=3, label="exact profile @ x=0.5")
    scatter!(ax, xsec .+ profile_scale .* u_num, y_num; color=:red, markersize=4.5, label="numerical profile @ x=0.5")
    hlines!(ax, [h]; color=(:black, 0.8), linestyle=:dot, linewidth=2, label="interface y=h")
    xlims!(ax, 0.0, 1.0)
    ylims!(ax, 0.0, 1.0)
    axislegend(ax; position=:lb)

    save(filename, fig)
    return filename
end

mu1 = 1.0
mu2 = 0.25
h = 0.4
H = 1.0
U = 0.5
G = 1.0
ns = (33, 65, 97)

u_exact = two_layer_couette_poiseuille_profile(mu1, mu2, h, H, U, G)
body(x, y) = y - h

bcx = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Dirichlet(0.0), top=Dirichlet(U),
)
bcy = BorderConditions(
    ; left=Periodic(), right=Periodic(),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

hs = Float64[]
e1 = Float64[]
e2 = Float64[]
last_model = Ref{Any}(nothing)
last_sys = Ref{Any}(nothing)
last_n = Ref(0)

for n in ns
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    model = StokesModelTwoPhase(
        grid,
        body,
        mu1,
        mu2;
        bc_u=(bcx, bcy),
        force1=(G, 0.0),
        force2=(G, 0.0),
        interface_force=(0.0, 0.0),
    )
    sys = solve_steady!(model)
    m = velocity_errors(model, sys, u_exact)

    push!(hs, 1.0 / (n - 1))
    push!(e1, m.u1L2)
    push!(e2, m.u2L2)
    last_model[] = model
    last_sys[] = sys
    last_n[] = n

    println("n=$n  u1L2=$(m.u1L2)  u2L2=$(m.u2L2)  vInf=$(m.vInf)  ||Ax-b||=$(norm(sys.A * sys.x - sys.b))")
end

for k in 1:(length(ns) - 1)
    o1 = log(e1[k] / e1[k + 1]) / log(hs[k] / hs[k + 1])
    o2 = log(e2[k] / e2[k + 1]) / log(hs[k] / hs[k + 1])
    println("order $(ns[k])->$(ns[k+1]): phase1=$o1  phase2=$o2")
end

plot_file = plot_velocity_overlay(last_model[], last_sys[], u_exact, h, last_n[])
println("saved plot: $plot_file")
