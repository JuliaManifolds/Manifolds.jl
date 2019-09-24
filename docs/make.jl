using Manifolds, Documenter

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(prettyurls = false),
    modules = [Manifolds],
    sitename = "Manifolds",
    pages = [
        "Home" => "index.md",
        "Manifolds" => [
            "Basic manifolds" => [
                "Euclidean" => "manifolds/euclidean.md",
                "Rotations" => "manifolds/rotations.md",
                "Sphere" => "manifolds/sphere.md"
            ],
            "Combined manifolds" => [
                "Product manifold" => "manifolds/product.md",
                "Vector bundle" => "manifolds/vector_bundle.md"
            ],
            "Manifold decorators" => [
                "Array manifold" => "manifolds/array.md",
                "Metric manifold" => "manifolds/metric.md"
            ]
        ],
        "Distributions" => "distributions.md",
        "Library" => [
            "Public" => "lib/public.md",
            "Internals" => "lib/internals.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/JuliaNLSolvers/Manifolds.jl.git",
)
