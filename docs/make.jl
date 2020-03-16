using Manifolds, ManifoldsBase, Documenter

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(prettyurls = false),
    modules = [Manifolds, ManifoldsBase],
    authors = "Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename = "Manifolds.jl",
    pages = [
        "Home" => "index.md",
        "ManifoldsBase.jl" => "interface.md",
        "Manifolds" => [
            "Basic manifolds" => [
                "Cholesky space" => "manifolds/choleskyspace.md",
                "Circle" => "manifolds/circle.md",
                "Euclidean" => "manifolds/euclidean.md",
                "Fixed-rank matrices" => "manifolds/fixedrankmatrices.md",
                "Generalized Stiefel" => "manifolds/generalizedstiefel.md",
                "Generalized Grassmann" => "manifolds/generalizedstiefel.md",
                "Grassmann" => "manifolds/grassmann.md",
                "Hyperbolic space" => "manifolds/hyperbolic.md",
                "Lorentzian manifold" => "manifolds/lorentz.md",
                "Oblique manifold" => "manifolds/oblique.md",
                "Rotations" => "manifolds/rotations.md",
                "Skew-symmetric matrices" => "manifolds/skewsymmetric.md",
                "Sphere" => "manifolds/sphere.md",
                "Stiefel" => "manifolds/stiefel.md",
                "Symmetric matrices" => "manifolds/symmetric.md",
                "Symmetric positive definite" => "manifolds/symmetricpositivedefinite.md",
                "Torus" => "manifolds/torus.md"
            ],
            "Combined manifolds" => [
                "Graph manifold" => "manifolds/graph.md",
                "Power manifold" => "manifolds/power.md",
                "Product manifold" => "manifolds/product.md",
                "Vector bundle" => "manifolds/vector_bundle.md"
            ],
            "Manifold decorators" => [
                "Embedded manifold" => "manifolds/embedded.md",
                "Metric manifold" => "manifolds/metric.md",
                "Group manifold" => "manifolds/group.md"
            ]
        ],
        "Statistics" => "statistics.md",
        "Distributions" => "distributions.md",
        "Orthonormal bases" => "orthonormal_bases.md",
        "Notation" => "notation.md",
        "Library" => [
            "Number systems" => "lib/numbers.md",
            "Public" => "lib/public.md",
            "Internals" => "lib/internals.md",
            "Automatic differentiation" => "lib/autodiff.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/JuliaNLSolvers/Manifolds.jl.git",
    push_preview = true,
)
