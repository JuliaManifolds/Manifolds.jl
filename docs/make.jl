using Manifolds, ManifoldsBase, Documenter

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(prettyurls = false),
    modules = [Manifolds, ManifoldsBase],
    sitename = "Manifolds",
    pages = [
        "Home" => "index.md",
        "ManifoldsBase.jl" => "interface.md",
        "Manifolds" => [
            "Basic manifolds" => [
                "Euclidean" => "manifolds/euclidean.md",
                "Cholesky Space" => "manifolds/choleskyspace.md",
                "Grassmannian" => "manifolds/grassmann.md",
                "Rotations" => "manifolds/rotations.md",
                "Sphere" => "manifolds/sphere.md",
                "Symmetric Positive Definite" => "manifolds/symmetricpositivedefinite.md"
            ],
            "Combined manifolds" => [
                "Power manifold" => "manifolds/power.md",
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
