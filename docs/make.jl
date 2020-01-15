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
                "Cholesky Space" => "manifolds/choleskyspace.md",
                "Circle" => "manifolds/circle.md",
                "Euclidean" => "manifolds/euclidean.md",
                "Fixed Rank Matrices" => "manifolds/fixedrankmatrices.md",
                "Grassmannian" => "manifolds/grassmann.md",
                "Hyperbolic Space" => "manifolds/hyperbolic.md",
                "Rotations" => "manifolds/rotations.md",
                "Sphere" => "manifolds/sphere.md",
                "Stiefel" => "manifolds/stiefel.md",
                "Symmetric Matrices" => "manifolds/symmetric.md",
                "Symmetric Positive Definite" => "manifolds/symmetricpositivedefinite.md"
            ],
            "Combined manifolds" => [
                "Power manifold" => "manifolds/power.md",
                "Product manifold" => "manifolds/product.md",
                "Vector bundle" => "manifolds/vector_bundle.md"
            ],
            "Manifold decorators" => [
                "Array manifold" => "manifolds/array.md",
                "Metric manifold" => "manifolds/metric.md",
                "Group manifold" => "manifolds/group.md"
            ]
        ],
        "Statistics" => "statistics.md",
        "Distributions" => "distributions.md",
        "Orthonormal bases" => "orthonormal_bases.md",
        "Library" => [
            "Numbers" => "lib/numbers.md",
            "Public" => "lib/public.md",
            "Internals" => "lib/internals.md",
            "Automatic Differentiation" => "lib/autodiff.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/JuliaNLSolvers/Manifolds.jl.git",
    push_preview = true,
)
