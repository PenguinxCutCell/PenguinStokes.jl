using LinearAlgebra
using StaticArrays
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Fixed spherical drop in uniform flow (drop frame), using
# Hadamard-Rybczynski analytical streamfunctions:
#   ψ_out(r,θ) = 1/4 U d^2 sin^2(θ) [ (μ*/(μ+μ*)) (d/r)
#               - ((2μ+3μ*)/(μ+μ*)) (r/d) + 2 (r/d)^2 ]
#   ψ_in (r,θ) = 1/4 U sin^2(θ) (μ/(μ+μ*)) r^2 [ (r/d)^2 - 1 ].
# Drag and terminal-velocity formulas:
#   Fdrag = 2π μ d U (2μ + 3μ*)/(μ + μ*)
#   Uterm = 2 d^2 g /(3ν) (1 - ρ*/ρ) (μ + μ*)/(2μ + 3μ*), ν = μ/ρ.

function hadamard_rybczynski_velocity(
    x,
    y,
    z;
    xc,
    yc,
    zc,
    d,
    U,
    mu_out,
    mu_in,
)
    X = x - xc
    Y = y - yc
    Z = z - zc
    s = sqrt(X^2 + Y^2)
    r = sqrt(X^2 + Y^2 + Z^2)
    if r == 0.0
        γ = mu_out / (mu_out + mu_in)
        return SVector(0.0, 0.0, -0.5 * U * γ)
    end

    sinθ = s / r
    cosθ = Z / r

    if r >= d
        α = mu_in / (mu_out + mu_in)
        β = (2 * mu_out + 3 * mu_in) / (mu_out + mu_in)
        F = α * (d / r) - β * (r / d) + 2 * (r / d)^2
        Fp = -α * d / r^2 - β / d + 4 * r / d^2
        ur = (U * d^2 * cosθ / (2 * r^2)) * F
        uθ = -(U * d^2 * sinθ / (4 * r)) * Fp
    else
        γ = mu_out / (mu_out + mu_in)
        ur = 0.5 * U * γ * cosθ * ((r / d)^2 - 1)
        uθ = -0.5 * U * γ * sinθ * (2 * (r / d)^2 - 1)
    end

    us = ur * sinθ + uθ * cosθ
    uz = ur * cosθ - uθ * sinθ
    if s > 0
        ux = us * X / s
        uy = us * Y / s
    else
        ux = 0.0
        uy = 0.0
    end
    return SVector(ux, uy, uz)
end

function hadamard_rybczynski_drag(mu_out, mu_in, d, U)
    return 2pi * mu_out * d * U * (2 * mu_out + 3 * mu_in) / (mu_out + mu_in)
end

function hadamard_rybczynski_terminal_velocity(mu_out, mu_in, rho_out, rho_in, d, g)
    ν = mu_out / rho_out
    return 2 * d^2 * g / (3 * ν) * (1 - rho_in / rho_out) * (mu_out + mu_in) / (2 * mu_out + 3 * mu_in)
end

mu_out = 1.0   # outside viscosity μ
mu_in = 10.0   # inside viscosity μ*
rho_out = 1.0  # outside density ρ
rho_in = 0.8   # inside density ρ*
d = 0.2        # drop radius
U = 1.0        # imposed uniform far-field speed (drop frame)
g = 9.81       # optional gravity for terminal-velocity formula

grid = CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (15, 15, 15))
xc, yc, zc = 0.5, 0.5, 0.5
body(x, y, z) = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2) - d

u_ref(x, y, z) = hadamard_rybczynski_velocity(
    x,
    y,
    z;
    xc=xc,
    yc=yc,
    zc=zc,
    d=d,
    U=U,
    mu_out=mu_out,
    mu_in=mu_in,
)

bcx = BorderConditions(
    ; left=Dirichlet((x, y, z) -> u_ref(x, y, z)[1]),
    right=Dirichlet((x, y, z) -> u_ref(x, y, z)[1]),
    bottom=Dirichlet((x, y, z) -> u_ref(x, y, z)[1]),
    top=Dirichlet((x, y, z) -> u_ref(x, y, z)[1]),
    backward=Dirichlet((x, y, z) -> u_ref(x, y, z)[1]),
    forward=Dirichlet((x, y, z) -> u_ref(x, y, z)[1]),
)
bcy = BorderConditions(
    ; left=Dirichlet((x, y, z) -> u_ref(x, y, z)[2]),
    right=Dirichlet((x, y, z) -> u_ref(x, y, z)[2]),
    bottom=Dirichlet((x, y, z) -> u_ref(x, y, z)[2]),
    top=Dirichlet((x, y, z) -> u_ref(x, y, z)[2]),
    backward=Dirichlet((x, y, z) -> u_ref(x, y, z)[2]),
    forward=Dirichlet((x, y, z) -> u_ref(x, y, z)[2]),
)
bcz = BorderConditions(
    ; left=Dirichlet((x, y, z) -> u_ref(x, y, z)[3]),
    right=Dirichlet((x, y, z) -> u_ref(x, y, z)[3]),
    bottom=Dirichlet((x, y, z) -> u_ref(x, y, z)[3]),
    top=Dirichlet((x, y, z) -> u_ref(x, y, z)[3]),
    backward=Dirichlet((x, y, z) -> u_ref(x, y, z)[3]),
    forward=Dirichlet((x, y, z) -> u_ref(x, y, z)[3]),
)

function solve_drag(n::Int)
    gridn = CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (n, n, n))
    model = StokesModelTwoPhase(
        gridn,
        body,
        mu_in,   # phase-1 (inside drop)
        mu_out;  # phase-2 (outside)
        bc_u=(bcx, bcy, bcz),
        force1=(0.0, 0.0, 0.0),
        force2=(0.0, 0.0, 0.0),
        interface_force=(0.0, 0.0, 0.0),
        # With a zero interface traction-jump, the interface coupling is too weak
        # (O(h^2)) to anchor the outside-phase pressure level, so the default
        # phase-1-only gauge leaves phase 2 as a near-null mode that pollutes the
        # solution. Gauge each phase independently; the mean variant is robust to
        # degenerate cut cells (the pin variant can land on a tiny one).
        gauge=PerPhasePressureGauge(MeanPressureGauge()),
    )

    sys = solve_steady!(model)

    # Numerical drag from the outside-phase (phase 2) discrete momentum balance,
    # using the same cut-face flux operators as the assembly. This avoids the
    # ill-conditioned pointwise sigma*n reconstruction (which divides by
    # vanishing cut-cell volumes and does not converge under refinement).
    fb = integrated_embedded_force_balance(model, sys; phase=2, convention=:on_body)

    return (Fz=fb.force[3], res=norm(sys.A * sys.x - sys.b))
end

Fz_ref = hadamard_rybczynski_drag(mu_out, mu_in, d, U)
U_term = hadamard_rybczynski_terminal_velocity(mu_out, mu_in, rho_out, rho_in, d, g)

println("Hadamard-Rybczynski Fz_ref = ", Fz_ref)
println("Terminal speed (formula)   = ", U_term)
for n in (11, 15, 19, 23, 27, 31)
    r = solve_drag(n)
    println("n=$n  ||Ax-b||=$(r.res)  Fz_num=$(r.Fz)  ratio=$(r.Fz / Fz_ref)")
end
