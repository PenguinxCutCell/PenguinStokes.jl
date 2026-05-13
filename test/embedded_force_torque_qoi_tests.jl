using Test
using LinearAlgebra
using StaticArrays: SVector

using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet, Periodic
using PenguinStokes

function _periodic_bc_2d()
    return BorderConditions(
        ; left=Periodic(), right=Periodic(),
        bottom=Periodic(), top=Periodic(),
    )
end

function _noslip_bc_3d()
    return BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
        backward=Dirichlet(0.0), forward=Dirichlet(0.0),
    )
end

function _circle_model(n::Int; center=(0.5, 0.5), radius=0.2, mu=1.0)
    grid = CartesianGrid((0.0, 0.0), (1.0, 1.0), (n, n))
    xc, yc = center
    body(x, y) = radius - sqrt((x - xc)^2 + (y - yc)^2)
    bc = _periodic_bc_2d()
    return StokesModelMono(
        grid,
        body,
        mu,
        1.0;
        bc_u=(bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0),
        gauge=MeanPressureGauge(),
    )
end

function _linear_pressure_state(model, a, b, c)
    x = zeros(Float64, last(model.layout.pomega))
    p = view(x, model.layout.pomega)
    for i in 1:model.cap_p.ntotal
        xω = model.cap_p.C_ω[i]
        if isfinite(xω[1]) && isfinite(xω[2])
            p[i] = a * xω[1] + b * xω[2] + c
        end
    end
    return x
end

function _deterministic_state(model)
    x = zeros(Float64, last(model.layout.pomega))
    for i in eachindex(x)
        x[i] = sin(0.17 * i) + 0.31 * cos(0.11 * i)
    end
    return x
end

function _exact_state_from_callbacks(model, ufun, pfun)
    x = zeros(Float64, last(model.layout.pomega))
    for d in 1:length(model.layout.uomega)
        uω = view(x, model.layout.uomega[d])
        uγ = view(x, model.layout.ugamma[d])
        for i in eachindex(uω)
            xω = model.cap_u[d].C_ω[i]
            if all(isfinite, Tuple(xω))
                uω[i] = ufun(xω)[d]
            end
            xγ = model.cap_u[d].C_γ[i]
            if all(isfinite, Tuple(xγ))
                uγ[i] = ufun(xγ)[d]
            end
        end
    end
    pω = view(x, model.layout.pomega)
    for i in eachindex(pω)
        xω = model.cap_p.C_ω[i]
        if all(isfinite, Tuple(xω))
            pω[i] = pfun(xω)
        end
    end
    return x
end

function _geometry_moments(model)
    cap = model.cap_p
    nsum = zeros(2)
    moment_n = 0.0
    perimeter = 0.0
    fluid_area = 0.0
    first_surface = zeros(2)
    for i in 1:cap.ntotal
        V = cap.buf.V[i]
        if isfinite(V) && V > 0.0
            fluid_area += V
        end
        Γ = cap.buf.Γ[i]
        if !(isfinite(Γ) && Γ > 0.0)
            continue
        end
        xγ = cap.C_γ[i]
        nγ = cap.n_γ[i]
        (isfinite(xγ[1]) && isfinite(xγ[2]) && isfinite(nγ[1]) && isfinite(nγ[2])) || continue
        nsum .+= Γ .* collect(nγ)
        moment_n += (xγ[1] - 0.5) * Γ * nγ[2] - (xγ[2] - 0.5) * Γ * nγ[1]
        first_surface .+= Γ .* collect(xγ)
        perimeter += Γ
    end
    return (; nsum, moment_n, perimeter, fluid_area, first_surface)
end

function _manufactured_2d_fields(; μ=1.0)
    ufun = x -> begin
        X, Y = x[1], x[2]
        SVector(
            π * sin(π * X) * cos(π * Y),
            -π * cos(π * X) * sin(π * Y),
        )
    end
    pfun = x -> 0.3 * sin(2π * x[1]) + 0.2 * cos(π * x[2])
    gradfun = x -> begin
        X, Y = x[1], x[2]
        SVector(
            π^2 * cos(π * X) * cos(π * Y),
            -π^2 * sin(π * X) * sin(π * Y),
            π^2 * sin(π * X) * sin(π * Y),
            -π^2 * cos(π * X) * cos(π * Y),
        )
    end
    stressfun = x -> begin
        G = gradfun(x)
        p = pfun(x)
        SVector(
            -p + 2μ * G[1],
            μ * (G[2] + G[3]),
            μ * (G[2] + G[3]),
            -p + 2μ * G[4],
        )
    end
    return ufun, pfun, stressfun
end

function _circle_surface_force_torque(center, radius, stressfun; nq=20000)
    xc = SVector(center...)
    F = zeros(2)
    τ = 0.0
    ds = 2π * radius / nq
    for k in 0:(nq - 1)
        θ = 2π * (k + 0.5) / nq
        n = SVector(cos(θ), sin(θ))
        x = xc + radius * n
        σ = stressfun(x)
        t = SVector(
            σ[1] * n[1] + σ[2] * n[2],
            σ[3] * n[1] + σ[4] * n[2],
        )
        F .+= ds .* collect(t)
        r = x - xc
        τ += ds * (r[1] * t[2] - r[2] * t[1])
    end
    return (force=SVector(F...), torque=τ)
end

function _sphere_model(n::Int; center=(0.5, 0.5, 0.5), radius=0.16, mu=1.0)
    grid = CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (n, n, n))
    C = SVector(center...)
    body(x, y, z) = radius - norm(SVector(x, y, z) - C)
    bc = _noslip_bc_3d()
    return StokesModelMono(
        grid,
        body,
        mu,
        1.0;
        bc_u=(bc, bc, bc),
        bc_cut=Dirichlet(0.0),
        force=(0.0, 0.0, 0.0),
        gauge=MeanPressureGauge(),
    )
end

function _sphere_translation_state(model, U::SVector{3,Float64}; center=SVector(0.5, 0.5, 0.5), radius=0.16, mu=1.0)
    a = radius
    ufun = x -> begin
        r = x - center
        ρ = norm(r)
        if ρ <= eps(Float64)
            return U
        end
        rhat = r / ρ
        term1 = (3a / (4ρ)) .* (U + rhat * dot(U, rhat))
        term2 = (a^3 / (4ρ^3)) .* (U - 3 * rhat * dot(U, rhat))
        term1 + term2
    end
    pfun = x -> begin
        r = x - center
        ρ = norm(r)
        ρ <= eps(Float64) && return 0.0
        rhat = r / ρ
        (3 * mu * a / (2 * ρ^2)) * dot(U, rhat)
    end
    return _exact_state_from_callbacks(model, ufun, pfun)
end

function _sphere_rotation_state(model, Ω::SVector{3,Float64}; center=SVector(0.5, 0.5, 0.5), radius=0.16)
    a = radius
    ufun = x -> begin
        r = x - center
        ρ = norm(r)
        ρ <= eps(Float64) && return cross(Ω, r)
        (a^3 / ρ^3) .* cross(Ω, r)
    end
    pfun = _ -> 0.0
    return _exact_state_from_callbacks(model, ufun, pfun)
end

@testset "Embedded force/torque QoI: geometry identities" begin
    radius = 0.2
    model = _circle_model(65; radius=radius)
    geom = _geometry_moments(model)

    @test norm(geom.nsum) < 1e-12
    @test abs(geom.moment_n) < 1e-12
    @test abs(geom.perimeter - 2π * radius) < 2e-2
    @test abs((1.0 - geom.fluid_area) - π * radius^2) < 4e-3
    @test norm(geom.first_surface / geom.perimeter - SVector(0.5, 0.5)) < 1e-12
end

@testset "Embedded force/torque QoI: algebraic split" begin
    model = _circle_model(25; mu=1.7)
    x = _deterministic_state(model)
    q = embedded_force_balance_density(model, x)
    pω = Vector{Float64}(x[model.layout.pomega])
    nt = model.cap_p.ntotal

    for d in 1:2
        opu = model.op_u[d]
        uω = Vector{Float64}(x[model.layout.uomega[d]])
        uγ = Vector{Float64}(x[model.layout.ugamma[d]])

        full_flux = opu.Winv * (opu.G * uω + opu.H * uγ)
        bulk_flux = copy(full_flux)
        cut_rows = PenguinStokes._nonzero_row_mask(opu.H)
        for i in eachindex(bulk_flux)
            cut_rows[i] && (bulk_flux[i] = 0.0)
        end
        viscous_cut = model.mu .* (opu.G' * (full_flux - bulk_flux))

        rows = ((d - 1) * nt + 1):(d * nt)
        pressure_full = -((model.op_p.G[rows, :] + model.op_p.H[rows, :]) * pω)
        pressure_bulk = -(model.op_p.G[rows, :] * pω)
        pressure_cut = pressure_full - pressure_bulk

        @test norm(q.viscous[d] - viscous_cut) < 1e-11
        @test norm(q.pressure[d] - pressure_cut) < 1e-11
        @test norm(q.total[d] - (viscous_cut + pressure_cut)) < 1e-11
    end
end

@testset "Embedded force/torque QoI: exact pressure moments" begin
    radius = 0.2
    xc, yc = 0.5, 0.5
    a, b, c = 1.0, -0.35, 0.7
    x0 = (0.15, 0.8)
    area = π * radius^2
    exact_body_force = -area .* SVector(a, b)

    errs = Float64[]
    for n in (33, 65, 129)
        model = _circle_model(n; radius=radius)
        x = _linear_pressure_state(model, a, b, c)
        f_body = integrated_embedded_force_balance(model, x; convention=:on_body, torque_method=:none)
        τ_center = integrated_embedded_torque_balance(model, x; convention=:on_body, x0=(xc, yc))
        τ_off = integrated_embedded_torque_balance(model, x; convention=:on_body, x0=x0)

        τ_exact_from_discrete_force =
            (xc - x0[1]) * f_body.force[2] - (yc - x0[2]) * f_body.force[1]

        push!(errs, norm(f_body.force_pressure - exact_body_force))
        @test norm(f_body.force_viscous) < 1e-12
        @test abs(τ_center.torque) < 2e-2
        @test abs(τ_center.torque_viscous) < 1e-12
        @test abs(τ_off.torque - τ_exact_from_discrete_force) < 2e-2
        @test abs(τ_off.torque_viscous) < 1e-12
    end

    @test errs[2] < errs[1]
    @test errs[3] < errs[2]
    @test log(errs[1] / errs[2]) / log(2) > 0.7
    @test log(errs[2] / errs[3]) / log(2) > 0.7
end

@testset "Embedded force/torque QoI: manufactured Stokes traction" begin
    center = (0.5, 0.5)
    radius = 0.2
    μ = 1.0
    ufun, pfun, stressfun = _manufactured_2d_fields(; μ=μ)
    exact = _circle_surface_force_torque(center, radius, stressfun)

    errs_F = Float64[]
    for n in (33, 65, 129)
        model = _circle_model(n; center=center, radius=radius, mu=μ)
        x = _exact_state_from_callbacks(model, ufun, pfun)
        q = integrated_embedded_force_balance(model, x; convention=:on_body, x0=center)
        push!(errs_F, norm(q.force - exact.force))
        @test norm(q.force_pressure + q.force_viscous - q.force) < 1e-12
    end

    @test errs_F[2] < errs_F[1]
    @test errs_F[3] < errs_F[2]

    # A pure shear manufactured field has nonzero local viscous traction on the
    # circle but zero net force by symmetry. Torque is validated below with
    # pressure moments and exact rotating-sphere fields.
    shear_u = x -> SVector(x[2] - center[2], 0.0)
    zero_p = _ -> 0.0
    model = _circle_model(65; center=center, radius=radius, mu=μ)
    x_shear = _exact_state_from_callbacks(model, shear_u, zero_p)
    q_shear = integrated_embedded_force_balance(model, x_shear; convention=:on_body, x0=center)
    @test norm(q_shear.force) < 1e-10
    @test norm(q_shear.force_viscous) < 1e-10
end

@testset "Embedded force/torque QoI: exact rigid-body sphere fields" begin
    radius = 0.16
    μ = 1.0
    model = _sphere_model(13; radius=radius, mu=μ)

    U = SVector(0.0, 0.0, 0.1)
    x_trans = _sphere_translation_state(model, U; radius=radius, mu=μ)
    q_trans = integrated_embedded_force_balance(model, x_trans; convention=:on_body, x0=(0.5, 0.5, 0.5))
    drag_ref = 6π * μ * radius * norm(U)

    @test abs(norm(q_trans.force) - drag_ref) / drag_ref < 0.85
    @test abs(abs(q_trans.force[3]) - norm(q_trans.force)) / drag_ref < 0.25
    @test norm(q_trans.force[1:2]) / drag_ref < 0.35
    @test norm(q_trans.torque) / drag_ref < 0.25

    Ω = SVector(0.0, 0.0, 0.3)
    x_rot = _sphere_rotation_state(model, Ω; radius=radius)
    q_rot = integrated_embedded_force_balance(model, x_rot; convention=:on_body, x0=(0.5, 0.5, 0.5))
    torque_ref = 8π * μ * radius^3 * norm(Ω)

    @test norm(q_rot.force) / torque_ref < 0.35
    @test abs(norm(q_rot.torque) - torque_ref) / torque_ref < 0.75
    @test abs(abs(q_rot.torque[3]) - norm(q_rot.torque)) / torque_ref < 0.25
end

@testset "Embedded force/torque QoI: constant fields and linearity" begin
    model = _circle_model(65)

    x_const_p = zeros(Float64, last(model.layout.pomega))
    view(x_const_p, model.layout.pomega) .= 2.3
    f_const = integrated_embedded_force_balance(model, x_const_p; convention=:on_body, torque_method=:none)
    τ_const = integrated_embedded_torque_balance(model, x_const_p; convention=:on_body, x0=(0.15, 0.8))
    @test norm(f_const.force_pressure) < 1e-10
    @test norm(f_const.force_viscous) < 1e-12
    @test abs(τ_const.torque_pressure) < 1e-10
    @test abs(τ_const.torque_viscous) < 1e-12

    x_const_u = zeros(Float64, last(model.layout.pomega))
    view(x_const_u, model.layout.uomega[1]) .= 1.25
    view(x_const_u, model.layout.uomega[2]) .= -0.5
    view(x_const_u, model.layout.ugamma[1]) .= 1.25
    view(x_const_u, model.layout.ugamma[2]) .= -0.5
    f_u = integrated_embedded_force_balance(model, x_const_u; convention=:on_body, torque_method=:none)
    τ_u = integrated_embedded_torque_balance(model, x_const_u; convention=:on_body, x0=(0.15, 0.8))
    @test norm(f_u.force_viscous) < 1e-10
    @test norm(f_u.force_pressure) < 1e-12
    @test abs(τ_u.torque_viscous) < 1e-10
    @test abs(τ_u.torque_pressure) < 1e-12

    x = _deterministic_state(model)
    q1 = integrated_embedded_force_balance(model, x; convention=:on_body, x0=(0.2, 0.7))
    q2 = integrated_embedded_force_balance(model, 2 .* x; convention=:on_body, x0=(0.2, 0.7))
    qm = integrated_embedded_force_balance(model, -x; convention=:on_body, x0=(0.2, 0.7))

    @test norm(q2.force - 2 .* q1.force) < 1e-10
    @test norm(q2.force_pressure - 2 .* q1.force_pressure) < 1e-10
    @test norm(q2.force_viscous - 2 .* q1.force_viscous) < 1e-10
    @test abs(q2.torque - 2 * q1.torque) < 1e-10

    @test norm(qm.force + q1.force) < 1e-10
    @test norm(qm.force_pressure + q1.force_pressure) < 1e-10
    @test norm(qm.force_viscous + q1.force_viscous) < 1e-10
    @test abs(qm.torque + q1.torque) < 1e-10
end

@testset "Embedded force/torque QoI: resistance matrix structure" begin
    radius = 0.16
    μ = 1.0
    model = _sphere_model(13; radius=radius, mu=μ)
    center = (0.5, 0.5, 0.5)

    qTx = integrated_embedded_force_balance(
        model,
        _sphere_translation_state(model, SVector(0.1, 0.0, 0.0); radius=radius, mu=μ);
        convention=:on_body,
        x0=center,
    )
    qTy = integrated_embedded_force_balance(
        model,
        _sphere_translation_state(model, SVector(0.0, 0.1, 0.0); radius=radius, mu=μ);
        convention=:on_body,
        x0=center,
    )
    qTz = integrated_embedded_force_balance(
        model,
        _sphere_translation_state(model, SVector(0.0, 0.0, 0.1); radius=radius, mu=μ);
        convention=:on_body,
        x0=center,
    )
    qRz = integrated_embedded_force_balance(
        model,
        _sphere_rotation_state(model, SVector(0.0, 0.0, 0.3); radius=radius);
        convention=:on_body,
        x0=center,
    )
    qRz2 = integrated_embedded_force_balance(
        model,
        _sphere_rotation_state(model, SVector(0.0, 0.0, 0.6); radius=radius);
        convention=:on_body,
        x0=center,
    )
    qRzm = integrated_embedded_force_balance(
        model,
        _sphere_rotation_state(model, SVector(0.0, 0.0, -0.3); radius=radius);
        convention=:on_body,
        x0=center,
    )

    drag = (norm(qTx.force) + norm(qTy.force) + norm(qTz.force)) / 3
    @test abs(norm(qTx.force) - drag) / drag < 0.35
    @test abs(norm(qTy.force) - drag) / drag < 0.35
    @test abs(norm(qTz.force) - drag) / drag < 0.35
    @test norm(qTx.torque) / drag < 0.35
    @test norm(qTy.torque) / drag < 0.35
    @test norm(qTz.torque) / drag < 0.35

    @test norm(qRz2.torque - 2 .* qRz.torque) / max(norm(qRz.torque), eps()) < 1e-10
    @test norm(qRzm.torque + qRz.torque) / max(norm(qRz.torque), eps()) < 1e-10
    @test norm(qRz2.force - 2 .* qRz.force) / max(norm(qRz.torque), eps()) < 1e-10
    @test dot(SVector(0.0, 0.0, 0.3), qRz.torque) > 0.0 || dot(SVector(0.0, 0.0, 0.3), -qRz.torque) > 0.0
end
