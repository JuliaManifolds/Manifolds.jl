using Plots, RecipesBase, Manifolds, ManifoldsBase, Documenter, PyPlot
# required for loading the Manifolds.ManifoldTests module
using Test, ForwardDiff, ReverseDiff
ENV["GKSwstype"] = "100"

generated_path = joinpath(@__DIR__, "src", "misc")
isdir(generated_path) || mkdir(generated_path)
cp(
    joinpath(dirname(@__DIR__), "CONTRIBUTING.md"),
    joinpath(generated_path, "contributing.md");
    force=true,
)

makedocs(
    # for development, we disable prettyurls
    format=Documenter.HTML(prettyurls=false, assets=["assets/favicon.ico"]),
    modules=[Manifolds, ManifoldsBase, Manifolds.ManifoldTests],
    authors="Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename="Manifolds.jl",
    pages=[
        "Home" => "index.md",
        "ManifoldsBase.jl" => "interface.md",
        "Examples" => ["How to implement a Manifold" => "examples/manifold.md"],
        "Manifolds" => [
            "Basic manifolds" => [
                "Centered matrices" => "manifolds/centeredmatrices.md",
                "Cholesky space" => "manifolds/choleskyspace.md",
                "Circle" => "manifolds/circle.md",
                "Elliptope" => "manifolds/elliptope.md",
                "Essential manifold" => "manifolds/essentialmanifold.md",
                "Euclidean" => "manifolds/euclidean.md",
                "Fixed-rank matrices" => "manifolds/fixedrankmatrices.md",
                "Generalized Stiefel" => "manifolds/generalizedstiefel.md",
                "Generalized Grassmann" => "manifolds/generalizedgrassmann.md",
                "Grassmann" => "manifolds/grassmann.md",
                "Hyperbolic space" => "manifolds/hyperbolic.md",
                "Lorentzian manifold" => "manifolds/lorentz.md",
                "Multinomial doubly stochastic matrices" =>
                    "manifolds/multinomialdoublystochastic.md",
                "Multinomial matrices" => "manifolds/multinomial.md",
                "Multinomial symmetric matrices" => "manifolds/multinomialsymmetric.md",
                "Oblique manifold" => "manifolds/oblique.md",
                "Probability simplex" => "manifolds/probabilitysimplex.md",
                "Positive numbers" => "manifolds/positivenumbers.md",
                "Projective space" => "manifolds/projectivespace.md",
                "Rotations" => "manifolds/rotations.md",
                "Skew-symmetric matrices" => "manifolds/skewsymmetric.md",
                "Spectrahedron" => "manifolds/spectrahedron.md",
                "Sphere" => "manifolds/sphere.md",
                "Stiefel" => "manifolds/stiefel.md",
                "Symmetric matrices" => "manifolds/symmetric.md",
                "Symmetric positive definite" =>
                    "manifolds/symmetricpositivedefinite.md",
                "Symmetric positive semidefinite fixed rank" =>
                    "manifolds/symmetricpsdfixedrank.md",
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
        "Features on Manifolds" => [
            "Differentiation" => "features/differentiation.md",
            "Distributions" => "features/distributions.md",
            "Statistics" => "features/statistics.md",
            "Testing" => "features/testing.md",
            "Utilities" => "features/utilities.md",
        ],
        "Miscellanea" => [
            "About" => "misc/about.md",
            "Contributing" => "misc/contributing.md",
            "Internals" => "misc/internals.md",
            "Notation" => "misc/notation.md",
        ],
    ],
)
deploydocs(repo="github.com/JuliaManifolds/Manifolds.jl.git", push_preview=true)
