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
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Examples" => "examples.md",
        "Algorithms" => "algorithms.md",
        "Stokes Model" => "stokes.md",
    ],
    pagesonly = true,
    warnonly = false,
    remotes = nothing,
)

if get(ENV, "CI", "") == "true"
    deploydocs(
        repo = "github.com/PenguinxCutCell/PenguinStokes.jl",
        push_preview = true,
    )
end
