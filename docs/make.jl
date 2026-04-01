using Documenter
using PenguinStokes

makedocs(
    modules = [PenguinStokes],
    authors = "PenguinxCutCell contributors",
    sitename = "PenguinStokes.jl",
    format = Documenter.HTML(
        canonical = "https://PenguinxCutCell.github.io/PenguinStokes.jl",
        repolink = "https://github.com/PenguinxCutCell/PenguinStokes.jl",
        collapselevel = 2,
    ),
    doctest = false,
    pages = [
        "Home" => "index.md",
        "Models and Equations" => "stokes.md",
        "Algorithms" => "algorithms.md",
        "Boundary Conditions and Gauges" => "boundary_conditions.md",
        "FSI" => "fsi.md",
        "Postprocessing" => "postprocessing.md",
        "API" => "api.md",
        "Examples and Verification" => "examples.md",
        "Feature Matrix" => "feature_matrix.md",
        "Developer Notes" => "developer_notes.md",
    ],
    pagesonly = true,
    warnonly = true,
    remotes = nothing,
)

if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/PenguinxCutCell/PenguinStokes.jl",
        push_preview = true,
    )
end
