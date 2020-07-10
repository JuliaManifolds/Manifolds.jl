using Manifolds, ManifoldsBase, Documenter

generated_path = joinpath(@__DIR__, "src", "generated")
isdir(generated_path) || mkdir(generated_path)
cp(
    joinpath(dirname(@__DIR__), "CONTRIBUTING.md"),
    joinpath(generated_path, "contributing.md");
    force = true,
)

makedocs(
    # for development, we disable prettyurls
    format = Documenter.HTML(prettyurls = false, assets = ["assets/favicon.ico"]),
    modules = [Manifolds, ManifoldsBase],
    authors = "Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename = "Manifolds.jl",
    pages = [
        "Home" => "index.md",
        "ManifoldsBase.jl" => "interface.md",
        "How to..." => ["implement a Manifold" => "tutorials/manifold.md"],
        "Manifolds" => [
            "Basic manifolds" => [
                "Centered matrices" => "manifolds/centeredmatrices.md",
                "Cholesky space" => "manifolds/choleskyspace.md",
                "Circle" => "manifolds/circle.md",
                "Euclidean" => "manifolds/euclidean.md",
                "Fixed-rank matrices" => "manifolds/fixedrankmatrices.md",
                "Generalized Stiefel" => "manifolds/generalizedstiefel.md",
                "Generalized Grassmann" => "manifolds/generalizedgrassmann.md",
                "Grassmann" => "manifolds/grassmann.md",
                "Hyperbolic space" => "manifolds/hyperbolic.md",
                "Lorentzian manifold" => "manifolds/lorentz.md",
                "Multinomial matrices" => "manifolds/multinomial.md",
                "Oblique manifold" => "manifolds/oblique.md",
                "Probability simplex" => "manifolds/probabilitysimplex.md",
                "Rotations" => "manifolds/rotations.md",
                "Skew-symmetric matrices" => "manifolds/skewsymmetric.md",
                "Sphere" => "manifolds/sphere.md",
                "Stiefel" => "manifolds/stiefel.md",
                "Symmetric matrices" => "manifolds/symmetric.md",
                "Symmetric positive definite" =>
                    "manifolds/symmetricpositivedefinite.md",
                "Torus" => "manifolds/torus.md",
                "Unit-norm symmetric matrices" =>
                    "manifolds/spheresymmetricmatrices.md",
            ],
            "Combined manifolds" => [
                "Graph manifold" => "manifolds/graph.md",
                "Power manifold" => "manifolds/power.md",
                "Product manifold" => "manifolds/product.md",
                "Vector bundle" => "manifolds/vector_bundle.md",
            ],
            "Manifold decorators" => [
                "Metric manifold" => "manifolds/metric.md",
                "Group manifold" => "manifolds/group.md",
            ],
        ],
        "Statistics" => "statistics.md",
        "Distributions" => "distributions.md",
        "About" => "about.md",
        "Contributing" => "generated/contributing.md",
        "Notation" => "notation.md",
        "Library" => [
            "Public" => "lib/public.md",
            "Internals" => "lib/internals.md",
            "Differentiation" => "lib/differentiation.md",
            "Maps" => "lib/maps.md",
        ],
    ],
)

deploydocs(repo = "github.com/JuliaManifolds/Manifolds.jl.git", push_preview = true)
