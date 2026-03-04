using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

function oscillatory_two_layer_amplitude(mu1, mu2, rho1, rho2, h, H, U0, ω)
    k1 = sqrt(im * ω * rho1 / mu1)
    k2 = sqrt(im * ω * rho2 / mu2)
    η = H - h

    M = ComplexF64[
        sinh(k1 * h) -sinh(k2 * η)
        mu1 * k1 * cosh(k1 * h) mu2 * k2 * cosh(k2 * η)
    ]
    rhs = ComplexF64[
        U0 * cosh(k2 * η)
        -mu2 * U0 * k2 * sinh(k2 * η)
    ]
    A, C = M \ rhs

    function Uhat(y)
        if y <= h
            return A * sinh(k1 * y)
        end
        return C * sinh(k2 * (H - y)) + U0 * cosh(k2 * (H - y))
    end

    return Uhat
end

function nearest_index_at_y(cap, y_target)
    best_i = 1
    best_d = Inf
    for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if !(isfinite(V) && V > 0.0)
            continue
        end
        y = cap.C_ω[i][2]
        d = abs(y - y_target)
        if d < best_d
            best_d = d
            best_i = i
        end
    end
    return best_i
end

function fit_harmonic(times, vals, ω)
    M = hcat(cos.(ω .* times), sin.(ω .* times))
    c = M \ vals
    A = c[1]
    B = c[2]
    Uhat = A - im * B
    return (amp=abs(Uhat), phase=angle(Uhat), Uhat=Uhat)
end

function run_case(; n=41, dt=0.02, t_end=2.0)
    H = 1.0
    h = 0.4375
    U0 = 1.0
    ω = 2pi
    mu1, mu2 = 1.0, 5.0
    rho1, rho2 = 1.0, 1.0

    grid = CartesianGrid((0.0, 0.0), (1.0, H), (n, n))
    body(x, y) = y - h

    bcx = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet((x, y, t) -> U0 * cos(ω * t)),
    )
    bcy = BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    model = StokesModelTwoPhase(
        grid,
        body,
        mu1,
        mu2;
        rho1=rho1,
        rho2=rho2,
        bc_u=(bcx, bcy),
        force1=(0.0, 0.0),
        force2=(0.0, 0.0),
        interface_force=(0.0, 0.0),
    )

    Uhat_exact = oscillatory_two_layer_amplitude(mu1, mu2, rho1, rho2, h, H, U0, ω)

    y1 = 0.25
    y2 = 0.75
    i1 = nearest_index_at_y(model.cap_u1[1], y1)
    i2 = nearest_index_at_y(model.cap_u2[1], y2)

    nsys = last(model.layout.pomega2)
    x = zeros(Float64, nsys)

    times = Float64[]
    u1_hist = Float64[]
    u2_hist = Float64[]

    t = 0.0
    nsteps = Int(round(t_end / dt))
    for _ in 1:nsteps
        sys = solve_unsteady!(model, x; t=t, dt=dt, scheme=:CN)
        x = copy(sys.x)
        t += dt

        if t >= (t_end - 1.0)  # keep last period samples
            push!(times, t)
            push!(u1_hist, x[model.layout.uomega1[1][i1]])
            push!(u2_hist, x[model.layout.uomega2[1][i2]])
        end
    end

    fit1 = fit_harmonic(times, u1_hist, ω)
    fit2 = fit_harmonic(times, u2_hist, ω)
    ex1 = Uhat_exact(model.cap_u1[1].C_ω[i1][2])
    ex2 = Uhat_exact(model.cap_u2[1].C_ω[i2][2])

    amp_err1 = abs(fit1.amp - abs(ex1)) / max(abs(ex1), 1e-12)
    amp_err2 = abs(fit2.amp - abs(ex2)) / max(abs(ex2), 1e-12)
    phase_err1 = abs(angle(exp(im * (fit1.phase - angle(ex1)))))
    phase_err2 = abs(angle(exp(im * (fit2.phase - angle(ex2)))))

    return (
        dt=dt,
        probe1=(y=model.cap_u1[1].C_ω[i1][2], amp=fit1.amp, amp_exact=abs(ex1), amp_relerr=amp_err1, phase=fit1.phase, phase_exact=angle(ex1), phase_err=phase_err1),
        probe2=(y=model.cap_u2[1].C_ω[i2][2], amp=fit2.amp, amp_exact=abs(ex2), amp_relerr=amp_err2, phase=fit2.phase, phase_exact=angle(ex2), phase_err=phase_err2),
    )
end

for dt in (0.04, 0.02)
    r = run_case(dt=dt)
    p1 = r.probe1
    p2 = r.probe2
    println("dt=$(r.dt)")
    println("  probe1 y=$(p1.y): amp=$(p1.amp), exact=$(p1.amp_exact), relerr=$(p1.amp_relerr), phase_err=$(p1.phase_err)")
    println("  probe2 y=$(p2.y): amp=$(p2.amp), exact=$(p2.amp_exact), relerr=$(p2.amp_relerr), phase_err=$(p2.phase_err)")
end
