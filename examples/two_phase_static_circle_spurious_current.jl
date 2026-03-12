using LinearAlgebra
using Printf
using Statistics
using StaticArrays: SVector
using CartesianGrids
using PenguinBCs
using PenguinStokes

function active_velocity_indices(cap)
    return findall(isfinite.(cap.buf.V) .& (cap.buf.V .> 0.0))
end

function pressure_active_indices(cap)
    return findall(PenguinStokes._pressure_activity(cap))
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
            v = vals[i]
            Vi = cap.buf.V[i]
            e2 += Vi * v * v
            w += Vi
        end
    end

    return sqrt(e2 / max(w, eps(Float64)))
end

function run_case(; n::Int, R::Float64=0.35, sigma::Float64=1.0, mu1::Float64=1.0, mu2::Float64=1.0, center::SVector{2,Float64}=SVector(0.0, 0.0))
    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    xc, yc = center
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - R

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    dp_th = sigma / R

    # In this two-phase convention, phase-1 is the inside region for body = r-R.
    # Using fΓ = -(σ/R) n imposes p_in - p_out = σ/R at static equilibrium.
    interface_force(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? SVector(0.0, 0.0) : -(dp_th) * SVector(dx / rr, dy / rr)
    end

    model = StokesModelTwoPhase(
        grid,
        body,
        mu1,
        mu2;
        rho1=1.0,
        rho2=1.0,
        bc_u=(bc0, bc0),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=interface_force,
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

    umax = max(maximum(abs, u1x[idx_u1x]), maximum(abs, u2x[idx_u2x]))
    vmax = max(maximum(abs, u1y[idx_u1y]), maximum(abs, u2y[idx_u2y]))
    uinf = max(umax, vmax)
    ul2 = velocity_l2_combined(model, u1x, u1y, u2x, u2y)

    p1 = sys.x[model.layout.pomega1] # inside
    p2 = sys.x[model.layout.pomega2] # outside
    idx_p1 = pressure_active_indices(model.cap_p1)
    idx_p2 = pressure_active_indices(model.cap_p2)
    p_in = mean(p1[idx_p1])
    p_out = mean(p2[idx_p2])
    dp_num = p_in - p_out
    jump_relerr = abs(dp_num - dp_th) / abs(dp_th)

    h = 2.0 / (n - 1)
    residual = norm(sys.A * sys.x - sys.b)

    return (
        n=n,
        h=h,
        umax=umax,
        vmax=vmax,
        uinf=uinf,
        ul2=ul2,
        p_in=p_in,
        p_out=p_out,
        dp_num=dp_num,
        dp_th=dp_th,
        jump_relerr=jump_relerr,
        residual=residual,
        model=model,
        sys=sys,
        body=body,
    )
end

function print_refinement_table(results)
    println("\nStatic circle spurious-current benchmark (fixed two-phase interface)")
    println("Domain=[-1,1]^2, R=0.35, sigma=1.0, mu1=mu2=1, no-slip walls")
    println("Theoretical jump: dp_th = sigma/R")
    println("Expected exact velocity: u = 0")
    println()
    println(rpad("N", 8), rpad("h", 16), rpad("uinf", 16), rpad("ul2", 16), rpad("dp_num", 16), "rel_jump_err")
    for r in results
        @printf("%-8d%-16.8e%-16.8e%-16.8e%-16.8e%.8e\n", r.n, r.h, r.uinf, r.ul2, r.dp_num, r.jump_relerr)
    end
end

function _build_plot_fields(model, sys, bodyfun)
    nt = model.cap_p1.ntotal
    nn = model.cap_p1.nnodes

    p1 = sys.x[model.layout.pomega1]
    p2 = sys.x[model.layout.pomega2]
    u1x = sys.x[model.layout.uomega1[1]]
    u1y = sys.x[model.layout.uomega1[2]]
    u2x = sys.x[model.layout.uomega2[1]]
    u2y = sys.x[model.layout.uomega2[2]]

    pfield = fill(NaN, nt)
    speed = fill(NaN, nt)
    phi = fill(NaN, nt)

    @inbounds for i in 1:nt
        V1 = model.cap_p1.buf.V[i]
        V2 = model.cap_p2.buf.V[i]
        c = model.cap_p1.C_ω[i]
        phi[i] = bodyfun(c[1], c[2])

        if isfinite(V1) && V1 > 0.0
            pfield[i] = p1[i]
            speed[i] = hypot(u1x[i], u1y[i])
        elseif isfinite(V2) && V2 > 0.0
            pfield[i] = p2[i]
            speed[i] = hypot(u2x[i], u2y[i])
        end
    end

    x = model.cap_p1.xyz[1]
    y = model.cap_p1.xyz[2]
    pmat = reshape(pfield, nn...)
    smat = reshape(speed, nn...)
    phimat = reshape(phi, nn...)
    return x, y, pmat, smat, phimat
end

function main()
    ns = (32, 64, 128)
    results = map(n -> run_case(; n=n), ns)

    for r in results
        println(
            "n=$(r.n): umax=$(r.umax), vmax=$(r.vmax), uinf=$(r.uinf), ul2=$(r.ul2), ",
            "p_in=$(r.p_in), p_out=$(r.p_out), dp_num=$(r.dp_num), dp_th=$(r.dp_th), ",
            "rel_jump_err=$(r.jump_relerr), ||Ax-b||=$(r.residual)",
        )
    end

    print_refinement_table(results)
    return nothing
end

main()
