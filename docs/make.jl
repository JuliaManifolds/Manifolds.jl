#!/usr/bin/env julia
#
#

#
# (a) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
    if "--quarto" ∈ ARGS
        Pkg.build("IJulia") # to activate the right kernel
    end
end

# (b) Did someone say render? Then we render!
if "--quarto" ∈ ARGS
    using CondaPkg
    CondaPkg.withenv() do
        @info "Rendering Quarto"
        tutorials_folder = (@__DIR__) * "/../tutorials"
        # instantiate the tutorials environment if necessary
        Pkg.activate(tutorials_folder)
        Pkg.resolve()
        Pkg.instantiate()
        Pkg.activate(@__DIR__) # but return to the docs one before
        run(`quarto render $(tutorials_folder)`)
        return nothing
    end
end

# (c) load necessary packages for the docs
using Plots, RecipesBase, Manifolds, ManifoldsBase, Documenter, PythonPlot
using DocumenterCitations
# required for loading methods that handle differential equation solving
using OrdinaryDiffEq, BoundaryValueDiffEq, DiffEqCallbacks
# required for loading the manifold tests functions
using Test, FiniteDifferences
ENV["GKSwstype"] = "100"

# (d) add contributing.md to docs
generated_path = joinpath(@__DIR__, "src", "misc")
base_url = "https://github.com/JuliaManifolds/Manifolds.jl/blob/master/"
isdir(generated_path) || mkdir(generated_path)
open(joinpath(generated_path, "contributing.md"), "w") do io
    # Point to source license file
    println(
        io,
        """
        ```@meta
        EditURL = "$(base_url)CONTRIBUTING.md"
        ```
        """,
    )
    # Write the contents out below the meta block
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        println(io, line)
    end
end

# (e) ...finally! make docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:alpha)
makedocs(
    bib;
    # for development, we disable prettyurls
    format=Documenter.HTML(prettyurls=false, assets=["assets/favicon.ico"]),
    modules=[
        Manifolds,
        isdefined(Base, :get_extension) ?
        Base.get_extension(Manifolds, :ManifoldsBoundaryValueDiffEqExt) :
        Manifolds.ManifoldsBoundaryValueDiffEqExt,
        isdefined(Base, :get_extension) ?
        Base.get_extension(Manifolds, :ManifoldsNLsolveExt) : Manifolds.ManifoldsNLsolveExt,
        isdefined(Base, :get_extension) ?
        Base.get_extension(Manifolds, :ManifoldsOrdinaryDiffEqDiffEqCallbacksExt) :
        Manifolds.ManifoldsOrdinaryDiffEqDiffEqCallbacksExt,
        isdefined(Base, :get_extension) ?
        Base.get_extension(Manifolds, :ManifoldsOrdinaryDiffEqExt) :
        Manifolds.ManifoldsOrdinaryDiffEqExt,
        isdefined(Base, :get_extension) ?
        Base.get_extension(Manifolds, :ManifoldsRecipesBaseExt) :
        Manifolds.ManifoldsRecipesBaseExt,
        isdefined(Base, :get_extension) ? Base.get_extension(Manifolds, :ManifoldsTestExt) :
        Manifolds.ManifoldsTestExt,
    ],
    authors="Seth Axen, Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename="Manifolds.jl",
    pages=[
        "Home" => "index.md",
        "How to..." => [
            "🚀 Get Started with `Manifolds.jl`" => "tutorials/getstarted.md",
            "work in charts" => "tutorials/working-in-charts.md",
            "perform Hand gesture analysis" => "tutorials/hand-gestures.md",
            "integrate on manifolds and handle probability densities" => "tutorials/integration.md",
        ],
        "Manifolds" => [
            "Basic manifolds" => [
                "Centered matrices" => "manifolds/centeredmatrices.md",
                "Cholesky space" => "manifolds/choleskyspace.md",
                "Circle" => "manifolds/circle.md",
                "Elliptope" => "manifolds/elliptope.md",
                "Essential manifold" => "manifolds/essentialmanifold.md",
                "Euclidean" => "manifolds/euclidean.md",
                "Fixed-rank matrices" => "manifolds/fixedrankmatrices.md",
                "Flag" => "manifolds/flag.md",
                "Generalized Stiefel" => "manifolds/generalizedstiefel.md",
                "Generalized Grassmann" => "manifolds/generalizedgrassmann.md",
                "Grassmann" => "manifolds/grassmann.md",
                "Hyperbolic space" => "manifolds/hyperbolic.md",
                "Lorentzian manifold" => "manifolds/lorentz.md",
                "Multinomial doubly stochastic matrices" => "manifolds/multinomialdoublystochastic.md",
                "Multinomial matrices" => "manifolds/multinomial.md",
                "Multinomial symmetric matrices" => "manifolds/multinomialsymmetric.md",
                "Oblique manifold" => "manifolds/oblique.md",
                "Probability simplex" => "manifolds/probabilitysimplex.md",
                "Positive numbers" => "manifolds/positivenumbers.md",
                "Projective space" => "manifolds/projectivespace.md",
                "Orthogonal and Unitary Matrices" => "manifolds/generalunitary.md",
                "Rotations" => "manifolds/rotations.md",
                "Shape spaces" => "manifolds/shapespace.md",
                "Skew-Hermitian matrices" => "manifolds/skewhermitian.md",
                "Spectrahedron" => "manifolds/spectrahedron.md",
                "Sphere" => "manifolds/sphere.md",
                "Stiefel" => "manifolds/stiefel.md",
                "Symmetric matrices" => "manifolds/symmetric.md",
                "Symmetric positive definite" => "manifolds/symmetricpositivedefinite.md",
                "SPD, fixed determinant" => "manifolds/spdfixeddeterminant.md",
                "Symmetric positive semidefinite fixed rank" => "manifolds/symmetricpsdfixedrank.md",
                "Symplectic" => "manifolds/symplectic.md",
                "Symplectic Stiefel" => "manifolds/symplecticstiefel.md",
                "Torus" => "manifolds/torus.md",
                "Tucker" => "manifolds/tucker.md",
                "Unit-norm symmetric matrices" => "manifolds/spheresymmetricmatrices.md",
            ],
            "Combined manifolds" => [
                "Graph manifold" => "manifolds/graph.md",
                "Power manifold" => "manifolds/power.md",
                "Product manifold" => "manifolds/product.md",
                "Vector bundle" => "manifolds/vector_bundle.md",
            ],
            "Manifold decorators" => [
                "Connection manifold" => "manifolds/connection.md",
                "Group manifold" => "manifolds/group.md",
                "Metric manifold" => "manifolds/metric.md",
                "Quotient manifold" => "manifolds/quotient.md",
            ],
        ],
        "Features on Manifolds" => [
            "Atlases and charts" => "features/atlases.md",
            "Differentiation" => "features/differentiation.md",
            "Distributions" => "features/distributions.md",
            "Integration" => "features/integration.md",
            "Statistics" => "features/statistics.md",
            "Testing" => "features/testing.md",
            "Utilities" => "features/utilities.md",
        ],
        "Miscellanea" => [
            "About" => "misc/about.md",
            "Contributing" => "misc/contributing.md",
            "Internals" => "misc/internals.md",
            "Notation" => "misc/notation.md",
            "References" => "misc/references.md",
        ],
    ],
)
deploydocs(repo="github.com/JuliaManifolds/Manifolds.jl.git", push_preview=true)
