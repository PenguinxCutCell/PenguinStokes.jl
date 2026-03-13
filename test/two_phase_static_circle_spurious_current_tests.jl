using Test
using StaticArrays: SVector
using CartesianGrids: CartesianGrid
using PenguinBCs: BorderConditions, Dirichlet
using PenguinStokes

function _active_velocity_indices(cap)
    return findall(isfinite.(cap.buf.V) .& (cap.buf.V .> 0.0))
end

function _avg(vals, idx)
    s = 0.0
    @inbounds for i in idx
        s += vals[i]
    end
    return s / length(idx)
end

@testset "Two-phase static circle spurious-current regression (2D)" begin
    n = 36
    R = 0.35
    sigma = 1.0
    dp_th = sigma / R
    xc, yc = 0.0, 0.0

    grid = CartesianGrid((-1.0, -1.0), (1.0, 1.0), (n, n))
    body(x, y) = sqrt((x - xc)^2 + (y - yc)^2) - R

    bc0 = BorderConditions(
        ; left=Dirichlet(0.0), right=Dirichlet(0.0),
        bottom=Dirichlet(0.0), top=Dirichlet(0.0),
    )

    interface_force(x, y) = begin
        dx = x - xc
        dy = y - yc
        rr = hypot(dx, dy)
        rr == 0.0 ? SVector(0.0, 0.0) : -(dp_th) * SVector(dx / rr, dy / rr)
    end

    model = StokesModelTwoPhase(
        grid,
        body,
        1.0,
        1.0;
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

    idx_u1x = _active_velocity_indices(model.cap_u1[1])
    idx_u1y = _active_velocity_indices(model.cap_u1[2])
    idx_u2x = _active_velocity_indices(model.cap_u2[1])
    idx_u2y = _active_velocity_indices(model.cap_u2[2])

    umax = max(maximum(abs, u1x[idx_u1x]), maximum(abs, u2x[idx_u2x]))
    vmax = max(maximum(abs, u1y[idx_u1y]), maximum(abs, u2y[idx_u2y]))
    uinf = max(umax, vmax)

    p1 = sys.x[model.layout.pomega1]
    p2 = sys.x[model.layout.pomega2]
    idx_p1 = findall(PenguinStokes._pressure_activity(model.cap_p1))
    idx_p2 = findall(PenguinStokes._pressure_activity(model.cap_p2))
    dp_num = _avg(p1, idx_p1) - _avg(p2, idx_p2)
    jump_relerr = abs(dp_num - dp_th) / abs(dp_th)

    @test uinf < 0.2
    @test jump_relerr < 0.15
end
