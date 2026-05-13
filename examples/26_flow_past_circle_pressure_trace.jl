using LinearAlgebra
using CartesianGrids
using PenguinBCs
using PenguinStokes

# Steady flow past a circular embedded boundary.
grid = CartesianGrid((-2.0, -1.0), (6.0, 1.0), (97, 49))
xc, yc, r = 0.0, 0.0, 0.2
body(x, y) = r - sqrt((x - xc)^2 + (y - yc)^2)

bcx = BorderConditions(
    ; left=Dirichlet(1.0), right=Dirichlet(1.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)
bcy = BorderConditions(
    ; left=Dirichlet(0.0), right=Dirichlet(0.0),
    bottom=Dirichlet(0.0), top=Dirichlet(0.0),
)

model = StokesModelMono(
    grid,
    body,
    1.0,
    1.0;
    bc_u=(bcx, bcy),
    bc_cut=Dirichlet(0.0),
    force=(0.0, 0.0),
)

sys = solve_steady!(model)

pdata = embedded_boundary_pressure(model, sys; pressure_reconstruction=:linear)
fdata = integrated_embedded_force(model, sys)

idx = pdata.interface_indices
nsamp = length(idx)
theta = Vector{Float64}(undef, nsamp)
xgamma = Vector{Float64}(undef, nsamp)
ygamma = Vector{Float64}(undef, nsamp)
pgamma = Vector{Float64}(undef, nsamp)
Gamma = Vector{Float64}(undef, nsamp)
nx = Vector{Float64}(undef, nsamp)
ny = Vector{Float64}(undef, nsamp)

for k in 1:nsamp
    i = idx[k]
    c = pdata.centers[k]
    n = pdata.normals[k]
    theta[k] = atan(c[2] - yc, c[1] - xc)
    xgamma[k] = c[1]
    ygamma[k] = c[2]
    pgamma[k] = pdata.pressure[i]
    Gamma[k] = pdata.measure[k]
    nx[k] = n[1]
    ny[k] = n[2]
end

perm = sortperm(theta)
outfile = joinpath(@__DIR__, "26_flow_past_circle_pressure_trace.csv")
open(outfile, "w") do io
    println(io, "theta,xgamma,ygamma,pgamma,Gamma,nx,ny")
    for k in perm
        println(io, string(theta[k], ",", xgamma[k], ",", ygamma[k], ",", pgamma[k], ",", Gamma[k], ",", nx[k], ",", ny[k]))
    end
end

println("interface samples = ", nsamp)
println("pressure force (embedded_boundary_pressure) = ", pdata.force)
println("pressure force (balance integrated_embedded_force) = ", fdata.force_pressure)
println("||pressure-trace helper - balance pressure|| = ", norm(pdata.force - fdata.force_pressure))
println("wrote: ", outfile)
